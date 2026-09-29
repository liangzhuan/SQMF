"""Validate and summarize the fixed-25-degree COIL-20 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_fixed25 import D_VALUES, METHODS, OBJECTS


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "recomputed_summary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "all_3060_runs.csv")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    canonical_targets = pd.read_csv(ROOT / "results" / "selected_targets.csv")
    canonical_manifest = pd.read_csv(ROOT / "results" / "image_manifest.csv")
    if len(canonical_targets) != 20 or not canonical_targets.target_view_index.eq(5).all():
        raise ValueError("Invalid target manifest")
    if len(canonical_manifest) != 1440:
        raise ValueError("Invalid image manifest")
    results = pd.read_csv(args.input)
    keys = ["object", "target_view_index", "d", "method"]
    if results.duplicated(keys).any():
        raise ValueError("Duplicate experiment cells")
    expected = {(obj, 5, d, method)
                for obj in OBJECTS for d in D_VALUES for method in METHODS}
    actual = set(zip(results.object, results.target_view_index,
                     results.d, results.method))
    if actual != expected or len(results) != len(expected):
        raise ValueError(f"Incomplete grid {len(results)}/{len(expected)}; "
                         f"missing={sorted(expected - actual)[:10]}; "
                         f"extra={sorted(actual - expected)[:10]}")
    if not results.target_angle_deg.eq(25).all():
        raise ValueError("Target angles differ from 25 degrees")
    if not results.training_count.eq(70).all():
        raise ValueError("Training count differs from 70")
    if not results.anchor_view_indices.eq("4;6").all():
        raise ValueError("Prediction anchors differ from 20 and 30 degrees")
    if not results.local_k.loc[results.method.isin(["MFIT", "LOG-KDE"])].eq(69).all():
        raise ValueError("Unexpected local K")
    if not results.ridge.loc[results.method.eq("Q-LS")].eq(10).all():
        raise ValueError("Unexpected Q-LS ridge")

    out.mkdir(parents=True, exist_ok=True)
    canonical_targets.to_csv(out / "selected_targets.csv", index=False)
    canonical_manifest.to_csv(out / "image_manifest.csv", index=False)
    results = results.sort_values(keys)
    results.to_csv(out / "all_3060_runs.csv", index=False)
    failed = results.loc[~results.success.astype(bool) |
                         ~np.isfinite(results.prediction_mse)]
    failed.to_csv(out / "failed_runs.csv", index=False)
    scqm = results.loc[results.method.isin(
        [f"{kind}-{loss}" for kind in ("Q", "L")
         for loss in ("p1", "p1.25", "p1.5", "p1.75", "p2", "l2")])].copy()
    unconverged = scqm.loc[~scqm.training_converged.astype(bool)]
    unconverged.to_csv(out / "scqm_not_converged.csv", index=False)

    wide = results.pivot(index=["object", "target_view_index", "target_angle_deg",
                                "d", "support_pixels"], columns="method",
                         values="prediction_mse").reset_index()
    wide.columns.name = None
    wide = wide[["object", "target_view_index", "target_angle_deg", "d",
                 "support_pixels", *METHODS]].sort_values(["object", "d"])
    wide.to_csv(out / "each_object_17_methods_mse.csv", index=False)

    summary = (results.groupby(["d", "method"], as_index=False)
               .agg(mean_mse=("prediction_mse", "mean"),
                    sd_mse=("prediction_mse", "std"),
                    n_finite=("prediction_mse", "count")))
    summary["n_expected"] = len(OBJECTS)
    summary.to_csv(out / "method_mean_sd_over_20_objects.csv", index=False)

    paired = []
    for loss in ("p1", "p1.25", "p1.5", "p1.75", "p2", "l2"):
        for _, row in wide.iterrows():
            q, l = row[f"Q-{loss}"], row[f"L-{loss}"]
            paired.append({"object": int(row.object), "d": int(row.d),
                           "loss": loss, "quadratic_mse": q, "linear_mse": l,
                           "quadratic_better": bool(q < l) if np.isfinite(q + l) else None,
                           "relative_gain_pct": 100 * (l - q) / l if np.isfinite(q + l) and l else np.nan})
    pd.DataFrame(paired).sort_values(["object", "d", "loss"]).to_csv(
        out / "quadratic_vs_linear_each_object.csv", index=False)
    print(json.dumps({"rows": len(results), "failed": len(failed),
                      "scqm_not_converged": len(unconverged),
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
