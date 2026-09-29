"""General-dimensional ERJBA / NJIT implementation of robust SCQM.

The model is
    f(tau) = c + U tau + V Theta.T vech(tau tau.T),
with Q=[U,V] on the Stiefel manifold.  The implementation keeps the
ERJBA residual-increment idea: Armijo candidates for Q, Theta and c are
evaluated by updating the current residual rather than rebuilding the
whole model.  Hot element-wise, quadratic-feature, latent-gradient and
projection kernels are compiled with numba.njit.

The code is intentionally self-contained and deterministic.  It supports
arbitrary D, d and s, unlike the uploaded sphere-specialized implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
from typing import Any

import numpy as np
from numba import njit

LOSS_LP = 0
LOSS_L2 = 1
LOSS_L2_SQUARED = 2
LOSS_HUBER = 3


def loss_code(name: str) -> int:
    key = name.lower().replace("-", "_")
    table = {
        "lp": LOSS_LP,
        "lp_p": LOSS_LP,
        "l1": LOSS_LP,
        "l2": LOSS_L2,
        "euclidean": LOSS_L2,
        "l2_squared": LOSS_L2_SQUARED,
        "squared": LOSS_L2_SQUARED,
        "sqmf": LOSS_L2_SQUARED,
        "huber": LOSS_HUBER,
    }
    if key not in table:
        raise ValueError(f"Unsupported loss {name!r}; choose {sorted(table)}")
    return table[key]


@njit(cache=True)
def _loss_value_grad_rows(R: np.ndarray, code: int, p: float, delta: float):
    n, D = R.shape
    G = np.empty_like(R)
    total = 0.0
    eps = 1e-15
    for i in range(n):
        if code == LOSS_LP:
            for j in range(D):
                r = R[i, j]
                ar = abs(r)
                total += ar ** p
                if ar <= eps:
                    G[i, j] = 0.0
                elif r > 0.0:
                    G[i, j] = p * ar ** (p - 1.0)
                else:
                    G[i, j] = -p * ar ** (p - 1.0)
        elif code == LOSS_L2:
            ss = 0.0
            for j in range(D):
                ss += R[i, j] * R[i, j]
            nr = math.sqrt(ss)
            total += nr
            if nr <= eps:
                for j in range(D):
                    G[i, j] = 0.0
            else:
                inv = 1.0 / nr
                for j in range(D):
                    G[i, j] = R[i, j] * inv
        elif code == LOSS_L2_SQUARED:
            for j in range(D):
                r = R[i, j]
                total += r * r
                G[i, j] = 2.0 * r
        else:  # radial Huber
            ss = 0.0
            for j in range(D):
                ss += R[i, j] * R[i, j]
            nr = math.sqrt(ss)
            if nr <= delta:
                total += 0.5 * ss
                for j in range(D):
                    G[i, j] = R[i, j]
            else:
                total += delta * nr - 0.5 * delta * delta
                scale = delta / max(nr, eps)
                for j in range(D):
                    G[i, j] = scale * R[i, j]
    return total, G


@njit(cache=True)
def _loss_value_rows(R: np.ndarray, code: int, p: float, delta: float) -> float:
    n, D = R.shape
    total = 0.0
    for i in range(n):
        if code == LOSS_LP:
            for j in range(D):
                total += abs(R[i, j]) ** p
        elif code == LOSS_L2:
            ss = 0.0
            for j in range(D):
                ss += R[i, j] * R[i, j]
            total += math.sqrt(ss)
        elif code == LOSS_L2_SQUARED:
            for j in range(D):
                total += R[i, j] * R[i, j]
        else:
            ss = 0.0
            for j in range(D):
                ss += R[i, j] * R[i, j]
            nr = math.sqrt(ss)
            if nr <= delta:
                total += 0.5 * ss
            else:
                total += delta * nr - 0.5 * delta * delta
    return total


@njit(cache=True)
def _build_phi(taus: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    n = taus.shape[0]
    m = pair_i.size
    Phi = np.empty((n, m), dtype=np.float64)
    for r in range(n):
        for q in range(m):
            Phi[r, q] = taus[r, pair_i[q]] * taus[r, pair_j[q]]
    return Phi


@njit(cache=True)
def _tau_gradient(
    taus: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    Theta: np.ndarray,
    G: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    """Gradient wrt all latent coordinates without materializing Jacobians."""
    n, d = taus.shape
    D = G.shape[1]
    s = V.shape[1]
    m = pair_i.size
    out = np.zeros((n, d), dtype=np.float64)
    vg = np.empty(s, dtype=np.float64)
    for r in range(n):
        # U.T g
        for a in range(d):
            val = 0.0
            for h in range(D):
                val += U[h, a] * G[r, h]
            out[r, a] = val
        # V.T g
        for k in range(s):
            val = 0.0
            for h in range(D):
                val += V[h, k] * G[r, h]
            vg[k] = val
        # Chain rule through z_k = sum_q Theta[q,k] tau_i tau_j.
        for q in range(m):
            a = pair_i[q]
            b = pair_j[q]
            coeff = 0.0
            for k in range(s):
                coeff += vg[k] * Theta[q, k]
            if a == b:
                out[r, a] += 2.0 * coeff * taus[r, a]
            else:
                out[r, a] += coeff * taus[r, b]
                out[r, b] += coeff * taus[r, a]
    return out


@njit(cache=True)
def _single_value_grad_tau(
    x: np.ndarray,
    tau: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    c: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    code: int,
    p: float,
    delta: float,
):
    D = Q.shape[0]
    d = tau.size
    s = Theta.shape[1]
    m = pair_i.size
    z = np.zeros(s, dtype=np.float64)
    phi = np.empty(m, dtype=np.float64)
    for q in range(m):
        v = tau[pair_i[q]] * tau[pair_j[q]]
        phi[q] = v
        for k in range(s):
            z[k] += Theta[q, k] * v
    residual = np.empty((1, D), dtype=np.float64)
    for h in range(D):
        y = c[h]
        for a in range(d):
            y += Q[h, a] * tau[a]
        for k in range(s):
            y += Q[h, d + k] * z[k]
        residual[0, h] = y - x[h]
    value, G = _loss_value_grad_rows(residual, code, p, delta)
    grad = np.zeros(d, dtype=np.float64)
    # U.T g
    for a in range(d):
        for h in range(D):
            grad[a] += Q[h, a] * G[0, h]
    # chain through quadratic coordinates
    for k in range(s):
        vg = 0.0
        for h in range(D):
            vg += Q[h, d + k] * G[0, h]
        for q in range(m):
            a = pair_i[q]
            b = pair_j[q]
            coeff = vg * Theta[q, k]
            if a == b:
                grad[a] += 2.0 * coeff * tau[a]
            else:
                grad[a] += coeff * tau[b]
                grad[b] += coeff * tau[a]
    return value, grad


@njit(cache=True)
def _decode_tau(
    tau: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    c: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    D = Q.shape[0]
    d = tau.size
    s = Theta.shape[1]
    y = c.copy()
    for h in range(D):
        for a in range(d):
            y[h] += Q[h, a] * tau[a]
    for k in range(s):
        z = 0.0
        for q in range(pair_i.size):
            z += Theta[q, k] * tau[pair_i[q]] * tau[pair_j[q]]
        for h in range(D):
            y[h] += Q[h, d + k] * z
    return y


@njit(cache=True)
def _project_point_kernel(
    x: np.ndarray,
    tau0: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    c: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    code: int,
    p: float,
    delta: float,
    max_iter: int,
    eta0: float,
    tol: float,
    armijo_alpha: float,
    armijo_beta: float,
    max_backtracks: int,
):
    tau = tau0.copy()
    for _ in range(max_iter):
        old, grad = _single_value_grad_tau(
            x, tau, Q, Theta, c, pair_i, pair_j, code, p, delta
        )
        gn2 = 0.0
        for a in range(grad.size):
            gn2 += grad[a] * grad[a]
        if gn2 <= tol * tol:
            break
        eta = eta0
        accepted = False
        for _ in range(max_backtracks):
            candidate = tau - eta * grad
            new, _ = _single_value_grad_tau(
                x, candidate, Q, Theta, c, pair_i, pair_j, code, p, delta
            )
            if math.isfinite(new) and new <= old - armijo_alpha * eta * gn2:
                tau = candidate
                accepted = True
                break
            eta *= armijo_beta
        if not accepted:
            break
    return _decode_tau(tau, Q, Theta, c, pair_i, pair_j), tau


def pair_indices(d: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(d)
    return ii.astype(np.int64), jj.astype(np.int64)


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _retract(Q: np.ndarray) -> np.ndarray:
    Qn, R = np.linalg.qr(Q, mode="reduced")
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    return Qn * signs.reshape(1, -1)


def _clip_gradient(G: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None or max_norm <= 0:
        return G
    norm = float(np.linalg.norm(G))
    if norm > max_norm:
        return G * (max_norm / norm)
    return G


@dataclass
class SCQMConfig:
    d: int
    s: int
    loss: str = "l2"
    p: float = 1.5
    huber_delta: float = 1.0
    max_iter: int = 100
    tol: float = 1e-7
    eta_Q: float = 1.0
    eta_Theta: float = 1e-2
    eta_c: float = 1e-2
    eta_tau: float = 1e-2
    tau_strategy: str = "armijo"  # armijo or fixed
    armijo_alpha: float = 1e-4
    armijo_beta: float = 0.5
    max_backtracks: int = 20
    theta_init_scale: float = 0.0
    random_state: int = 43
    gradient_clip: float | None = 1e4
    refresh_gradient_each_block: bool = True
    min_iter: int = 5
    mse_relative_change_tol: float = 1e-4


@dataclass
class SCQMResult:
    Q: np.ndarray
    Theta: np.ndarray
    c: np.ndarray
    taus: np.ndarray
    losses: np.ndarray
    elapsed_s: float
    iterations: int
    converged: bool
    backtracks_Q: int
    backtracks_Theta: int
    backtracks_c: int
    backtracks_tau: int
    orthogonality_error: float
    final_gradient_norm: float
    config: dict[str, Any]
    mse_history: np.ndarray | None = None
    final_relative_mse_change: float = float("nan")
    stop_reason: str = ""


def _residual(
    X_rows: np.ndarray,
    taus: np.ndarray,
    c: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
):
    Phi = _build_phi(taus, pair_i, pair_j)
    Z = Phi @ Theta if Theta.shape[1] else np.empty((taus.shape[0], 0))
    H = np.concatenate((taus, Z), axis=1)
    R = c.reshape(1, -1) + H @ Q.T - X_rows
    return np.ascontiguousarray(R), Phi, Z, H


def fit_scqm(X: np.ndarray, config: SCQMConfig) -> SCQMResult:
    """Fit SCQM to X.

    Parameters
    ----------
    X : array, shape (n_samples, D)
        This release requires row-major samples explicitly.
    """
    start = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    # This release uses an explicit row-major contract: one sample per row.
    # Automatic orientation guessing is intentionally avoided because real-data
    # class/object subsets may have fewer samples than ambient features.
    X_rows = np.ascontiguousarray(X)
    n, D = X_rows.shape
    k = config.d + config.s
    if config.d < 1 or config.s < 0 or k > D:
        raise ValueError(f"Require 1 <= d and d+s <= D; got d={config.d}, s={config.s}, D={D}")
    if n < k + 1:
        raise ValueError(f"Need at least d+s+1 samples; got n={n}, d+s={k}")
    code = loss_code(config.loss)
    if code == LOSS_LP and config.p < 1.0:
        raise ValueError("The implemented smooth/subgradient solver requires p >= 1")

    pair_i, pair_j = pair_indices(config.d)
    m = pair_i.size
    rng = np.random.default_rng(config.random_state)
    c = X_rows.mean(axis=0)
    centered = X_rows - c
    # PCA/SVD initialization; right singular vectors span the ambient subspace.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    Q = np.ascontiguousarray(Vt[:k].T)
    taus = np.ascontiguousarray(centered @ Q[:, : config.d])
    Theta = np.zeros((m, config.s), dtype=np.float64)
    if config.s and config.theta_init_scale > 0:
        Theta = rng.normal(scale=config.theta_init_scale, size=(m, config.s))

    losses: list[float] = []
    mse_history: list[float] = []
    bt_Q = bt_T = bt_c = bt_tau = 0
    eta_Q_state = float(config.eta_Q)
    converged = False
    final_grad_norm = math.inf
    final_relative_mse_change = math.inf
    stop_reason = "maximum_iterations"

    for iteration in range(config.max_iter):
        R, Phi, Z, H = _residual(X_rows, taus, c, Q, Theta, pair_i, pair_j)
        old_loss, G = _loss_value_grad_rows(R, code, config.p, config.huber_delta)

        # ----- Stiefel block Q -----
        grad_Q = G.T @ H
        grad_Q = grad_Q - Q @ _sym(Q.T @ grad_Q)
        grad_Q = _clip_gradient(grad_Q, config.gradient_clip)
        gn2 = float(np.sum(grad_Q * grad_Q))
        eta = eta_Q_state
        accepted = False
        for b in range(config.max_backtracks):
            Qn = _retract(Q - eta * grad_Q)
            Rn = R + H @ (Qn - Q).T
            new_loss = _loss_value_rows(np.ascontiguousarray(Rn), code, config.p, config.huber_delta)
            if np.isfinite(new_loss) and new_loss <= old_loss - config.armijo_alpha * eta * gn2:
                Q, R, old_loss = Qn, np.ascontiguousarray(Rn), float(new_loss)
                eta_Q_state = eta
                bt_Q += b
                accepted = True
                break
            eta *= config.armijo_beta
        if not accepted:
            bt_Q += config.max_backtracks

        if config.refresh_gradient_each_block:
            old_loss, G = _loss_value_grad_rows(R, code, config.p, config.huber_delta)

        # ----- Euclidean block Theta -----
        if config.s:
            V = Q[:, config.d :]
            grad_Theta = Phi.T @ (G @ V)
            grad_Theta = _clip_gradient(grad_Theta, config.gradient_clip)
            gn2 = float(np.sum(grad_Theta * grad_Theta))
            eta = float(config.eta_Theta)
            accepted = False
            for b in range(config.max_backtracks):
                Tn = Theta - eta * grad_Theta
                Rn = R + (Phi @ (Tn - Theta)) @ V.T
                new_loss = _loss_value_rows(np.ascontiguousarray(Rn), code, config.p, config.huber_delta)
                if np.isfinite(new_loss) and new_loss <= old_loss - config.armijo_alpha * eta * gn2:
                    Theta, R, old_loss = Tn, np.ascontiguousarray(Rn), float(new_loss)
                    bt_T += b
                    accepted = True
                    break
                eta *= config.armijo_beta
            if not accepted:
                bt_T += config.max_backtracks
            if config.refresh_gradient_each_block:
                old_loss, G = _loss_value_grad_rows(R, code, config.p, config.huber_delta)

        # ----- Euclidean block c -----
        grad_c = G.sum(axis=0)
        grad_c = _clip_gradient(grad_c, config.gradient_clip)
        gn2 = float(np.sum(grad_c * grad_c))
        eta = float(config.eta_c)
        accepted = False
        for b in range(config.max_backtracks):
            cn = c - eta * grad_c
            Rn = R + (cn - c).reshape(1, -1)
            new_loss = _loss_value_rows(np.ascontiguousarray(Rn), code, config.p, config.huber_delta)
            if np.isfinite(new_loss) and new_loss <= old_loss - config.armijo_alpha * eta * gn2:
                c, R, old_loss = cn, np.ascontiguousarray(Rn), float(new_loss)
                bt_c += b
                accepted = True
                break
            eta *= config.armijo_beta
        if not accepted:
            bt_c += config.max_backtracks

        # ----- Latent block -----
        if config.refresh_gradient_each_block:
            old_loss, G = _loss_value_grad_rows(R, code, config.p, config.huber_delta)
        U = Q[:, : config.d]
        V = Q[:, config.d :]
        grad_tau = _tau_gradient(taus, U, V, Theta, G, pair_i, pair_j)
        grad_tau = _clip_gradient(grad_tau, config.gradient_clip)
        if config.tau_strategy == "fixed":
            taus = np.ascontiguousarray(taus - config.eta_tau * grad_tau)
        elif config.tau_strategy == "armijo":
            gn2 = float(np.sum(grad_tau * grad_tau))
            eta = float(config.eta_tau)
            accepted = False
            for b in range(config.max_backtracks):
                tn = np.ascontiguousarray(taus - eta * grad_tau)
                Rn, _, _, _ = _residual(X_rows, tn, c, Q, Theta, pair_i, pair_j)
                new_loss = _loss_value_rows(Rn, code, config.p, config.huber_delta)
                if np.isfinite(new_loss) and new_loss <= old_loss - config.armijo_alpha * eta * gn2:
                    taus = tn
                    bt_tau += b
                    accepted = True
                    break
                eta *= config.armijo_beta
            if not accepted:
                bt_tau += config.max_backtracks
        else:
            raise ValueError("tau_strategy must be 'fixed' or 'armijo'")

        Rf, _, _, Hf = _residual(X_rows, taus, c, Q, Theta, pair_i, pair_j)
        current, Gf = _loss_value_grad_rows(Rf, code, config.p, config.huber_delta)
        losses.append(float(current))
        mse_history.append(float(np.mean(Rf * Rf)))
        # Stationarity diagnostic at current iterate (without another line search).
        grad_Q_f = Gf.T @ Hf
        grad_Q_f = grad_Q_f - Q @ _sym(Q.T @ grad_Q_f)
        final_grad_norm = float(np.linalg.norm(grad_Q_f))
        if len(mse_history) >= max(config.min_iter, 2):
            final_relative_mse_change = (
                abs(mse_history[-2] - mse_history[-1]) /
                max(abs(mse_history[-2]), 1e-12)
            )
            if final_relative_mse_change <= config.mse_relative_change_tol:
                converged = True
                stop_reason = "relative_training_mse_change_tolerance"
                break

    elapsed = time.perf_counter() - start
    orth = float(np.linalg.norm(Q.T @ Q - np.eye(k), ord="fro"))
    return SCQMResult(
        Q=Q,
        Theta=Theta,
        c=c,
        taus=taus,
        losses=np.asarray(losses),
        elapsed_s=float(elapsed),
        iterations=len(losses),
        converged=converged,
        backtracks_Q=bt_Q,
        backtracks_Theta=bt_T,
        backtracks_c=bt_c,
        backtracks_tau=bt_tau,
        orthogonality_error=orth,
        final_gradient_norm=final_grad_norm,
        config=asdict(config),
        mse_history=np.asarray(mse_history),
        final_relative_mse_change=float(final_relative_mse_change),
        stop_reason=stop_reason,
    )


def project_point(
    x: np.ndarray,
    result: SCQMResult,
    tau0: np.ndarray | None = None,
    *,
    max_iter: int = 100,
    eta0: float = 0.1,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = result.config
    d = int(cfg["d"])
    code = loss_code(str(cfg["loss"]))
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if tau0 is None:
        tau0 = result.Q[:, :d].T @ (x - result.c)
    ii, jj = pair_indices(d)
    return _project_point_kernel(
        x,
        np.asarray(tau0, dtype=np.float64),
        np.asarray(result.Q, dtype=np.float64),
        np.asarray(result.Theta, dtype=np.float64),
        np.asarray(result.c, dtype=np.float64),
        ii,
        jj,
        code,
        float(cfg.get("p", 1.5)),
        float(cfg.get("huber_delta", 1.0)),
        int(max_iter),
        float(eta0),
        float(tol),
        float(cfg.get("armijo_alpha", 1e-4)),
        float(cfg.get("armijo_beta", 0.5)),
        int(cfg.get("max_backtracks", 20)),
    )


def decode(result: SCQMResult, taus: np.ndarray | None = None) -> np.ndarray:
    """Decode latent coordinates to row-major ambient samples."""
    T = result.taus if taus is None else np.asarray(taus, dtype=np.float64)
    d = int(result.config["d"])
    ii, jj = pair_indices(d)
    Phi = _build_phi(np.ascontiguousarray(T), ii, jj)
    Z = Phi @ result.Theta if result.Theta.shape[1] else np.empty((T.shape[0], 0))
    H = np.concatenate((T, Z), axis=1)
    return result.c.reshape(1, -1) + H @ result.Q.T


def warmup() -> float:
    """Compile all NJIT kernels outside benchmark timing."""
    start = time.perf_counter()
    rng = np.random.default_rng(43)
    R = rng.normal(size=(4, 5))
    for code in (LOSS_LP, LOSS_L2, LOSS_L2_SQUARED, LOSS_HUBER):
        _loss_value_grad_rows(R, code, 1.5, 1.0)
        _loss_value_rows(R, code, 1.5, 1.0)
    taus = rng.normal(size=(4, 2))
    ii, jj = pair_indices(2)
    Phi = _build_phi(taus, ii, jj)
    U, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    Vraw = rng.normal(size=(5, 1))
    Vraw -= U @ (U.T @ Vraw)
    V = Vraw / np.linalg.norm(Vraw)
    Theta = rng.normal(size=(3, 1))
    G = rng.normal(size=(4, 5))
    _tau_gradient(taus, U, V, Theta, G, ii, jj)
    Q = np.concatenate((U, V), axis=1)
    c = np.zeros(5)
    _project_point_kernel(
        rng.normal(size=5), taus[0], Q, Theta, c, ii, jj,
        LOSS_LP, 1.5, 1.0, 2, 0.05, 1e-8, 1e-4, 0.5, 3,
    )
    return time.perf_counter() - start
