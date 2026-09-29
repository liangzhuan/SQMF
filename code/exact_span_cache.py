"""Optional per-context cache of unchanged training work in baseline exact_span.

Use only around serial calls in one process. Vendor functions and their query
arithmetic are unchanged; the context restores the prior adapter namespace.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np

import baseline_adapter


class ExactSpanCache:
    """Keep one complete training decomposition, validating content on every call."""

    def __init__(self):
        self._key = None
        self._entry = None
        self.hits = 0
        self.decompositions = 0
        self.invalidations = 0
        self.bypasses = 0

    @staticmethod
    def _fingerprint(source, X):
        # ravel(K) is a view for contiguous inputs; avoid a full bytes allocation.
        digest = hashlib.sha256(memoryview(X.ravel(order="K"))).hexdigest()
        return (source.dtype.str, X.dtype.str, X.shape, X.strides, digest)

    def _training(self, Data_rows):
        source = np.asarray(Data_rows)
        X = np.asarray(Data_rows, dtype=float)
        key = self._fingerprint(source, X)
        if key == self._key:
            self.hits += 1
            return self._entry
        if self._entry is not None:
            self.invalidations += 1
        self.clear()
        mu = X.mean(axis=0)
        centered = X - mu
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        tolerance = max(centered.shape) * np.finfo(float).eps * singular[0]
        rank = int(np.sum(singular > tolerance))
        self._entry = (mu, centered, vt, tolerance, rank)
        self._key = key
        self.decompositions += 1
        return self._entry

    def precompute(self, Data_rows, threshold=64):
        X = np.asarray(Data_rows, dtype=float)
        if X.shape[1] > threshold:
            self._training(Data_rows)

    def __call__(self, Data_rows, points_rows, d, threshold=64):
        X = np.asarray(Data_rows, dtype=float)
        P = np.asarray(points_rows, dtype=float)
        if X.shape[1] <= threshold:
            self.bypasses += 1
            return X, P, None
        mu, centered, vt, tolerance, rank = self._training(Data_rows)
        W = vt[:rank].T
        for point in P:
            residual = point - mu - W @ (W.T @ (point - mu))
            length = np.linalg.norm(residual)
            if length > max(tolerance, 1e-10):
                W = np.column_stack((W, residual / length))
        if W.shape[1] <= d:
            raise ValueError("Training/query span has insufficient dimension")
        return centered @ W, (P - mu) @ W, (mu, W)

    def clear(self):
        self._key = None
        self._entry = None

    def diagnostics(self):
        return {
            "decompositions": self.decompositions,
            "cache_hits": self.hits,
            "input_invalidations": self.invalidations,
            "low_dimension_bypasses": self.bypasses,
            "cached_training_sha256": self._key[-1] if self._key is not None else None,
            "cached_training_shape": list(self._key[2]) if self._key is not None else None,
            "cache_policy": "one full training SVD; shape/dtype/strides/content checked per query; no query span reuse",
        }


@contextmanager
def exact_span_cache(Data_rows=None):
    """Enable exact training reuse for serial project_baseline calls in this scope.

    The namespace swap is process-local and must not overlap baseline calls in
    other threads. Nested scopes and exceptions restore the original object.
    """
    previous = baseline_adapter._functions
    cache = ExactSpanCache()
    if Data_rows is not None:
        cache.precompute(Data_rows)
    functions = baseline_adapter._load_source()
    functions["_pca_reduce_if_needed"] = cache
    baseline_adapter._functions = functions
    try:
        yield cache
    finally:
        baseline_adapter._functions = previous
        cache.clear()
