"""Supplied MFIT/LOG-KDE functions with an exact out-of-sample span adapter.

The source algorithm, including its 49-update budget, is retained. The old
adapter discarded held-out components orthogonal to the training affine span.
Here that component adds one orthonormal coordinate before projection.
"""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def _load_source():
    source = Path(__file__).parent / "vendor" / "run_mfit_lkde_baselines.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_lift", "_knn_sigma", "_normal_projector_local", "log_kde_project", "mfit_project"}
    definitions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"np": np, "cKDTree": cKDTree, "_pca_reduce_if_needed": exact_span}
    exec(compile(definitions, str(source), "exec"), namespace)
    return namespace


def exact_span(Data_rows, points_rows, d, threshold=64):
    X = np.asarray(Data_rows, dtype=float)
    P = np.asarray(points_rows, dtype=float)
    if X.shape[1] <= threshold:
        return X, P, None
    mu = X.mean(axis=0)
    centered = X - mu
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    W = vt[:rank].T
    for point in P:
        residual = point - mu - W @ (W.T @ (point - mu))
        length = np.linalg.norm(residual)
        if length > max(tolerance, 1e-10):
            W = np.column_stack((W, residual / length))
    if W.shape[1] <= d:
        raise ValueError("Training/query span has insufficient dimension")
    return centered @ W, (P - mu) @ W, (mu, W)


_functions = None


def project_baseline(name, X, point, d, k):
    global _functions
    if _functions is None:
        _functions = _load_source()
    function = _functions["mfit_project" if name == "MFIT" else "log_kde_project"]
    return function(X, point, d, k)
