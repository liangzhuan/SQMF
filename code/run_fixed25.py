"""Predict the fixed 25-degree view of all 20 COIL-20 objects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.io import imread

HERE = Path(__file__).resolve().parent

from baseline_adapter import project_baseline
from exact_span_cache import exact_span_cache
from methods import (fit_global_model, fit_pca_model, fit_quadratic_ls_model,
                     fit_spherelet_model, reconstruct_spherelet_training)
from real_method_grid import build_method_configs
from scqm_njit import pair_indices, warmup


DATA = HERE.parent / "data" / "coil-20-proc"
OUT = HERE.parent / "rerun_seed43"
MODEL_SEED = 43
OBJECTS = tuple(range(1, 21))
D_VALUES = tuple(range(2, 11))
LOSSES = ("p1", "p1.25", "p1.5", "p1.75", "p2", "l2")
METHODS = tuple(f"{kind}-{loss}" for kind in ("Q", "L") for loss in LOSSES) + (
    "SPH", "PCA-d", "Q-LS", "MFIT", "LOG-KDE")
FIELDS = ("object", "object_seed", "target_view_index", "target_angle_deg",
          "d", "method", "prediction_mse", "prediction_rmse", "success", "error",
          "training_count", "support_pixels", "anchor_view_indices",
          "training_iterations", "training_converged", "training_stop_reason",
          "final_relative_training_mse_change", "fit_time_s", "prediction_rule",
          "ridge", "local_k")


def read_gray(path: Path) -> np.ndarray:
    a = np.asarray(imread(path))
    if a.ndim == 3:
        if a.shape[-1] == 4:
            a = a[..., :3]
        a = rgb2gray(a)
    if a.ndim != 2:
        raise ValueError(f"Expected grayscale/RGB image: {path}; got shape={a.shape}")
    a = a.astype(np.float64)
    if a.max() > 1.0:
        a /= 255.0 if a.max() <= 255.0 else a.max()
    return np.clip(a, 0.0, 1.0)


def latent_decode(model, tau: np.ndarray) -> np.ndarray:
    result, mu, scale = model
    d = result.taus.shape[1]
    ii, jj = pair_indices(d)
    phi = tau[ii] * tau[jj]
    z = phi @ result.Theta
    return mu + scale * (result.c + result.Q[:, :d] @ tau +
                         result.Q[:, d:] @ z)


def sphere_decode(model, z: np.ndarray) -> np.ndarray:
    return model["mu"] + model["W"] @ z


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_manifest() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "image_manifest.csv"
    if path.exists():
        frame = pd.read_csv(path)
        if len(frame) != 1440:
            raise ValueError("Incomplete source manifest")
        return frame
    rows = []
    for obj in OBJECTS:
        for index in range(72):
            source = DATA / f"obj{obj}__{index}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append({"object": obj, "view_index": index,
                         "angle_deg": index * 5, "path": str(source.resolve()),
                         "sha256": digest(source)})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def prepare_targets() -> pd.DataFrame:
    path = OUT / "selected_targets.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        rows = []
        for obj in OBJECTS:
            index = 5
            rows.append({"object": obj, "seed": "not_applicable",
                         "target_view_index": index,
                         "target_angle_deg": index * 5,
                         "excluded_opposite_view_index": (index + 36) % 72})
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
    if len(frame) != 20 or not frame.groupby("object").size().eq(1).all():
        raise ValueError("Invalid target selection")
    if not frame.target_view_index.eq(5).all():
        raise ValueError("Saved target selection does not match requested fixed target")
    return frame


def load_object(manifest: pd.DataFrame, obj: int) -> dict[int, np.ndarray]:
    subset = manifest.loc[manifest.object.eq(obj)]
    images = {}
    for row in subset.itertuples():
        source = Path(row.path)
        if digest(source) != row.sha256:
            raise ValueError(f"Image checksum mismatch: {source}")
        images[int(row.view_index)] = read_gray(source)
    if len(images) != 72 or any(image.shape != (128, 128) for image in images.values()):
        raise ValueError(f"Unexpected image set for object {obj}")
    return images


def fold_data(images: dict[int, np.ndarray], target_index: int):
    ids = [(target_index + offset) % 72
           for offset in range(-35, 36) if offset != 0]
    support = np.any(np.stack([images[index] for index in ids]) > 8 / 255,
                     axis=0).ravel()
    X = np.stack([images[index].ravel()[support] for index in ids])
    target = images[target_index].ravel()[support]
    anchor_ids = ((target_index - 1) % 72, (target_index + 1) % 72)
    positions = {index: pos for pos, index in enumerate(ids)}
    anchors = (positions[anchor_ids[0]], positions[anchor_ids[1]])
    return X, target, support, anchor_ids, anchors


def evaluate(X, target, anchor_ids, anchors, d, method, config):
    started = time.perf_counter()
    iterations = 0
    converged = ""
    stop = ""
    final_change = ""
    rule = "decode_mean_anchor_latents"
    if method in config:
        model = fit_global_model(X, config[method])
        result = model[0]
        latent = result.taus
        tau = latent[list(anchors)].mean(axis=0)
        prediction = latent_decode(model, tau)
        iterations, converged, stop = result.iterations, result.converged, result.stop_reason
        final_change = result.final_relative_mse_change
    elif method == "PCA-d":
        mu, U = fit_pca_model(X, d)
        tau = ((X[list(anchors)] - mu) @ U).mean(axis=0)
        prediction = mu + U @ tau
        converged, stop = True, "closed_form"
    elif method == "Q-LS":
        model = fit_quadratic_ls_model(X, d, 1, ridge=10.0)
        tau = model[0].taus[list(anchors)].mean(axis=0)
        prediction = latent_decode(model, tau)
        converged, stop = True, "one_shot_fit"
    elif method == "SPH":
        model = fit_spherelet_model(X, d)
        projected = reconstruct_spherelet_training(model, X)
        tau = ((projected[list(anchors)] - model["mu"]) @ model["W"]).mean(axis=0)
        if model["valid"]:
            radial = tau - model["center"]
            length = np.linalg.norm(radial)
            if length > 1e-12:
                tau = model["center"] + model["radius"] * radial / length
        prediction = sphere_decode(model, tau)
        converged, stop = True, "algebraic_fit" if model["valid"] else "linear_fallback"
    else:
        rule = "project_pixel_mean_of_anchor_images"
        anchor_mean = X[list(anchors)].mean(axis=0)
        prediction = project_baseline(method, X, anchor_mean, d, 69)
        stop = "vendor_projection_max_49_updates_or_tolerance"
    fit_time = time.perf_counter() - started
    clipped = np.clip(prediction, 0, 1)
    mse = float(np.mean((clipped - target)**2))
    if not np.isfinite(mse):
        raise ValueError("Non-finite prediction MSE")
    return mse, iterations, converged, stop, final_change, fit_time, rule


def main() -> None:
    global OUT, DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--object-min", type=int, default=1)
    parser.add_argument("--object-max", type=int, default=20)
    parser.add_argument("--d-min", type=int, default=2)
    parser.add_argument("--d-max", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    OUT = args.out
    DATA = args.data
    if not (1 <= args.object_min <= args.object_max <= 20):
        parser.error("Invalid object range")
    if not (2 <= args.d_min <= args.d_max <= 10):
        parser.error("Invalid d range")
    manifest = prepare_manifest()
    targets = prepare_targets()
    protocol_path = OUT / "protocol.json"
    results_path = OUT / "all_results.csv"
    if results_path.exists():
        if not protocol_path.exists():
            raise ValueError("Existing results have no protocol; refusing to mix runs")
        prior = json.loads(protocol_path.read_text(encoding="utf-8"))
        if prior.get("model_seed") != MODEL_SEED or prior.get("target_sampling") != "fixed view index 5":
            raise ValueError("Existing results have a different model seed or target")
    protocol_path.write_text(json.dumps({
        "status": "exploratory", "objects": OBJECTS,
        "seed_rule": "no random target sampling",
        "targets_per_object": 1,
        "target_sampling": "fixed view index 5",
        "d_values": D_VALUES, "s_quadratic": 1, "model_seed": MODEL_SEED,
        "train_radius_each_side": 35, "train_count": 70,
        "prediction_anchors_each_side": 1,
        "support_rule": "pixel retained if any of 70 training images exceeds 8/255",
        "pixel_scaling": "original grayscale values divided by 255",
        "mse": "mean squared error on retained pixels after clipping prediction to [0,1]",
        "scqm_max_iter_safety_cap": 1200,
        "scqm_stop": "relative one-step scaled training MSE change <=1e-4 after >=8 iterations",
        "qls_ridge": 10.0, "mfit_logkde_local_k": 69,
        "method_names": METHODS,
        "prediction_note": "MFIT/LOG-KDE project anchor pixel mean; other methods decode anchor latent mean",
    }, indent=2), encoding="utf-8")
    path = results_path
    done = set()
    if path.exists():
        old = pd.read_csv(path)
        done = set(zip(old.object.astype(int), old.target_view_index.astype(int),
                       old.d.astype(int), old.method))
    warmup()
    for obj in range(args.object_min, args.object_max + 1):
        images = load_object(manifest, obj)
        object_targets = targets.loc[targets.object.eq(obj)]
        for selected in object_targets.itertuples():
            target_index = int(selected.target_view_index)
            X, target, support, anchor_ids, anchors = fold_data(images, target_index)
            if X.shape[0] != 70 or X.shape[1] < 10:
                raise ValueError(f"Unexpected fold shape for obj{obj}, view{target_index}")
            with exact_span_cache(X):
                for d in range(args.d_min, args.d_max + 1):
                    config = build_method_configs(
                        d=d, s=1, max_iter=1200, seed=MODEL_SEED,
                        eta_Q=.35, eta_Theta=.020, eta_c=.020,
                        eta_tau=.030, min_iter=8)
                    for method in args.methods:
                        key = (obj, target_index, d, method)
                        if key in done:
                            continue
                        try:
                            values = evaluate(X, target, anchor_ids, anchors,
                                              d, method, config)
                            mse, iterations, converged, stop, change, elapsed, rule = values
                            success, error = True, ""
                        except Exception as exc:
                            mse, iterations, converged, stop, change, elapsed = (
                                float("nan"), "", "", "", "", "")
                            rule = ("project_pixel_mean_of_anchor_images" if method in
                                    ("MFIT", "LOG-KDE") else "decode_mean_anchor_latents")
                            success, error = False, repr(exc)
                        row = {"object": obj, "object_seed": selected.seed,
                               "target_view_index": target_index,
                               "target_angle_deg": target_index * 5, "d": d,
                               "method": method, "prediction_mse": mse,
                               "prediction_rmse": float(np.sqrt(mse)),
                               "success": success, "error": error,
                               "training_count": 70, "support_pixels": int(support.sum()),
                               "anchor_view_indices": f"{anchor_ids[0]};{anchor_ids[1]}",
                               "training_iterations": iterations,
                               "training_converged": converged,
                               "training_stop_reason": stop,
                               "final_relative_training_mse_change": change,
                               "fit_time_s": elapsed, "prediction_rule": rule,
                               "ridge": 0.1 if method == "Q-LS" else "",
                               "local_k": 69 if method in ("MFIT", "LOG-KDE") else ""}
                        with path.open("a", newline="", encoding="utf-8") as stream:
                            writer = csv.DictWriter(stream, fieldnames=FIELDS)
                            if stream.tell() == 0:
                                writer.writeheader()
                            writer.writerow(row)
                        done.add(key)
                    print(f"obj={obj:2d} angle={target_index*5:3d} d={d:2d}: "
                          f"{len(done)} rows", flush=True)


if __name__ == "__main__":
    main()
