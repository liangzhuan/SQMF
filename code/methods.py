"""Baselines and local/global wrappers used by the SCQM experiments."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from scqm_njit import SCQMConfig, SCQMResult, fit_scqm, project_point, decode


def robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    X=np.asarray(X,dtype=float)
    mu=X.mean(axis=0)
    radii=np.linalg.norm(X-mu,axis=1)
    scale=float(np.median(radii))
    if not np.isfinite(scale) or scale < 1e-10:
        scale=float(np.sqrt(np.mean((X-mu)**2)))
    if not np.isfinite(scale) or scale < 1e-10:
        scale=1.0
    return (X-mu)/scale, mu, scale


def lpca_project(X_local: np.ndarray, x: np.ndarray, d: int) -> np.ndarray:
    X=np.asarray(X_local,dtype=float)
    x=np.asarray(x,dtype=float)
    mu=X.mean(axis=0)
    _,_,vt=np.linalg.svd(X-mu,full_matrices=False)
    U=vt[:d].T
    return mu + U@(U.T@(x-mu))


def local_quadratic_ls_project(
    X_local: np.ndarray,
    x: np.ndarray,
    d: int,
    s: int,
    ridge: float=1e-5,
    projection_iter: int=80,
) -> np.ndarray:
    """One-shot local polynomial/MLS-style quadratic fit.

    PCA fixes U and V.  Normal coordinates are then regressed on quadratic
    tangent features by ridge least squares.  Projection uses Euclidean loss.
    """
    Xs,mu,scale=robust_scale(X_local)
    xs=(np.asarray(x)-mu)/scale
    c=Xs.mean(axis=0)
    Y=Xs-c
    _,_,vt=np.linalg.svd(Y,full_matrices=False)
    if d+s > Xs.shape[1]:
        s=max(0,Xs.shape[1]-d)
    Q=vt[:d+s].T
    U=Q[:,:d]
    V=Q[:,d:d+s]
    taus=Y@U
    ii,jj=np.triu_indices(d)
    Phi=np.column_stack([taus[:,a]*taus[:,b] for a,b in zip(ii,jj)])
    if s:
        normal=Y@V
        A=Phi.T@Phi + ridge*np.eye(Phi.shape[1])
        Theta=np.linalg.solve(A,Phi.T@normal)
    else:
        Theta=np.zeros((len(ii),0))
    cfg=SCQMConfig(d=d,s=s,loss='l2',max_iter=1)
    result=SCQMResult(
        Q=Q,Theta=Theta,c=c,taus=taus,losses=np.array([]),elapsed_s=0.0,
        iterations=0,converged=True,backtracks_Q=0,backtracks_Theta=0,
        backtracks_c=0,backtracks_tau=0,orthogonality_error=float(np.linalg.norm(Q.T@Q-np.eye(d+s))),
        final_gradient_norm=np.nan,config=cfg.__dict__.copy())
    y,_=project_point(xs,result,max_iter=projection_iter,eta0=.2)
    return mu+scale*y


def fit_project_local(
    X_local: np.ndarray,
    x: np.ndarray,
    config: SCQMConfig,
    *,
    target_position: int|None=None,
    projection_iter: int=80,
    projection_eta: float=.2,
) -> tuple[np.ndarray, SCQMResult]:
    Xs,mu,scale=robust_scale(X_local)
    xs=(np.asarray(x,dtype=float)-mu)/scale
    res=fit_scqm(Xs,config)
    tau0=None
    if target_position is not None and 0 <= target_position < res.taus.shape[0]:
        tau0=res.taus[target_position]
    y,_=project_point(xs,res,tau0=tau0,max_iter=projection_iter,eta0=projection_eta)
    return mu+scale*y,res


def fit_global_model(X: np.ndarray, config: SCQMConfig):
    Xs,mu,scale=robust_scale(X)
    res=fit_scqm(Xs,config)
    return res,mu,scale


def project_global_model(x: np.ndarray, model, projection_iter=100, projection_eta=.15):
    res,mu,scale=model
    xs=(np.asarray(x)-mu)/scale
    y,tau=project_point(xs,res,max_iter=projection_iter,eta0=projection_eta)
    return mu+scale*y,tau


def reconstruct_training(model):
    res,mu,scale=model
    return mu+scale*decode(res)


def local_sphere_project(X_local: np.ndarray, x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Algebraic local sphere fit (SPH/spherelet-style baseline).

    Returns projected point and unit surface normal at that point. Falls back to
    LPCA when the sphere system is ill-conditioned.
    """
    X=np.asarray(X_local,dtype=float); x=np.asarray(x,dtype=float)
    # Solve ||z||^2 + a^T z + b = 0.
    A=np.c_[X,np.ones(len(X))]; rhs=-np.sum(X*X,axis=1)
    try:
        coef,_,rank,_=np.linalg.lstsq(A,rhs,rcond=None)
        center=-.5*coef[:-1]
        rad2=float(np.dot(center,center)-coef[-1])
        if rank < X.shape[1]+1 or rad2 <= 1e-12: raise np.linalg.LinAlgError
        radius=np.sqrt(rad2); v=x-center; nv=np.linalg.norm(v)
        if nv<1e-12: raise np.linalg.LinAlgError
        y=center+radius*v/nv; normal=(y-center)/radius
        return y,normal
    except np.linalg.LinAlgError:
        y=lpca_project(X,x,min(2,X.shape[1]-1))
        _,_,vt=np.linalg.svd(X-X.mean(0),full_matrices=False)
        normal=vt[-1]
        return y,normal/np.linalg.norm(normal)


def fit_pca_model(X: np.ndarray, d: int):
    X=np.asarray(X,float); mu=X.mean(0); _,_,vt=np.linalg.svd(X-mu,full_matrices=False)
    return mu,vt[:d].T


def project_pca_model(x: np.ndarray, model):
    mu,U=model; return mu+U@(U.T@(np.asarray(x)-mu))


def fit_quadratic_ls_model(X: np.ndarray,d: int,s: int,ridge:float=1e-5):
    Xs,mu,scale=robust_scale(X); c=Xs.mean(0);Y=Xs-c
    _,_,vt=np.linalg.svd(Y,full_matrices=False);s=min(s,Xs.shape[1]-d);Q=vt[:d+s].T
    U=Q[:,:d];V=Q[:,d:d+s];taus=Y@U;ii,jj=np.triu_indices(d)
    Phi=np.column_stack([taus[:,a]*taus[:,b] for a,b in zip(ii,jj)])
    if s:
        normal=Y@V;Theta=np.linalg.solve(Phi.T@Phi+ridge*np.eye(Phi.shape[1]),Phi.T@normal)
    else:Theta=np.zeros((Phi.shape[1],0))
    cfg=SCQMConfig(d=d,s=s,loss='l2',max_iter=1)
    result=SCQMResult(Q=Q,Theta=Theta,c=c,taus=taus,losses=np.array([]),elapsed_s=0.,iterations=0,converged=True,
      backtracks_Q=0,backtracks_Theta=0,backtracks_c=0,backtracks_tau=0,
      orthogonality_error=float(np.linalg.norm(Q.T@Q-np.eye(d+s))),final_gradient_norm=np.nan,config=cfg.__dict__.copy())
    return result,mu,scale


def fit_quadratic_ls_local_model(X_local: np.ndarray, d: int, s: int, ridge: float = 1e-5):
    """Fit the transparent one-shot local quadratic baseline and return its model."""
    Xs, mu, scale = robust_scale(X_local)
    c = Xs.mean(axis=0)
    Y = Xs - c
    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    s = min(s, Xs.shape[1] - d)
    Q = vt[: d + s].T
    U = Q[:, :d]
    V = Q[:, d : d + s]
    taus = Y @ U
    ii, jj = np.triu_indices(d)
    Phi = np.column_stack([taus[:, a] * taus[:, b] for a, b in zip(ii, jj)])
    if s:
        normal = Y @ V
        Theta = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ normal)
    else:
        Theta = np.zeros((Phi.shape[1], 0))
    cfg = SCQMConfig(d=d, s=s, loss="l2", max_iter=1)
    result = SCQMResult(
        Q=Q,
        Theta=Theta,
        c=c,
        taus=taus,
        losses=np.array([]),
        elapsed_s=0.0,
        iterations=0,
        converged=True,
        backtracks_Q=0,
        backtracks_Theta=0,
        backtracks_c=0,
        backtracks_tau=0,
        orthogonality_error=float(np.linalg.norm(Q.T @ Q - np.eye(d + s))),
        final_gradient_norm=np.nan,
        config=cfg.__dict__.copy(),
    )
    return result, mu, scale


def quadratic_jacobian(result: SCQMResult, tau: np.ndarray) -> np.ndarray:
    """Ambient Jacobian df/dtau of the fitted quadratic mapping at ``tau``."""
    tau = np.asarray(tau, dtype=float).reshape(-1)
    d = int(result.config["d"])
    if tau.size != d:
        raise ValueError(f"Expected tau dimension {d}, got {tau.size}")
    U = result.Q[:, :d]
    V = result.Q[:, d:]
    if V.shape[1] == 0:
        return U.copy()
    ii, jj = np.triu_indices(d)
    dz = np.zeros((V.shape[1], d), dtype=float)
    for q, (a, b) in enumerate(zip(ii, jj)):
        coeff = result.Theta[q]
        if a == b:
            dz[:, a] += 2.0 * coeff * tau[a]
        else:
            dz[:, a] += coeff * tau[b]
            dz[:, b] += coeff * tau[a]
    return U + V @ dz


def normal_from_jacobian(J: np.ndarray) -> np.ndarray | None:
    """Return a unit normal for a two-dimensional surface in R^3."""
    J = np.asarray(J, dtype=float)
    if J.shape != (3, 2):
        return None
    n = np.cross(J[:, 0], J[:, 1])
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return n / norm

# ---------------------------------------------------------------------------
# Unified SPH (spherelet-style) baseline.
# The ambient data are first reduced to a (d+1)-dimensional affine PCA space,
# then an algebraic sphere is fitted there.  This is numerically meaningful
# even for high-dimensional images with only a few local samples.
# ---------------------------------------------------------------------------
def _spherelet_fit_core(X: np.ndarray, d: int = 2):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("SPH requires a 2D sample matrix with at least two rows")
    mu = X.mean(axis=0)
    Y = X - mu
    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    q = max(1, min(int(d) + 1, vt.shape[0], X.shape[1], len(X) - 1))
    W = vt[:q].T
    Z = Y @ W
    A = np.c_[Z, np.ones(len(Z))]
    rhs = -np.sum(Z * Z, axis=1)
    try:
        coef, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
        center = -0.5 * coef[:-1]
        rad2 = float(np.dot(center, center) - coef[-1])
        valid = bool(rank >= q + 1 and np.isfinite(rad2) and rad2 > 1e-12)
    except np.linalg.LinAlgError:
        center = np.zeros(q, dtype=float)
        rad2 = -1.0
        valid = False
    radius = float(np.sqrt(rad2)) if valid else float("nan")
    return mu, W, center, radius, valid


def local_sphere_project(X_local: np.ndarray, x: np.ndarray, d: int = 2) -> tuple[np.ndarray, np.ndarray | None]:
    """SPH/spherelet projection in a local (d+1)-dimensional affine PCA space."""
    X = np.asarray(X_local, dtype=float)
    x = np.asarray(x, dtype=float)
    mu, W, center, radius, valid = _spherelet_fit_core(X, d=d)
    if not valid:
        y = lpca_project(X, x, min(d, max(1, X.shape[1] - 1)))
        return y, None
    z = (x - mu) @ W
    v = z - center
    nv = float(np.linalg.norm(v))
    if not np.isfinite(nv) or nv < 1e-12:
        y = lpca_project(X, x, min(d, max(1, X.shape[1] - 1)))
        return y, None
    zhat = center + radius * v / nv
    y = mu + W @ zhat
    # In R^3 with d=2, W @ radial direction is the ambient sphere normal.
    normal = W @ ((zhat - center) / radius)
    nn = float(np.linalg.norm(normal))
    if not np.isfinite(nn) or nn < 1e-12:
        normal = None
    else:
        normal = normal / nn
    return y, normal


def fit_spherelet_model(X: np.ndarray, d: int = 2):
    """Fit an SPH model for direct reconstruction of all rows of X."""
    X = np.asarray(X, dtype=float)
    mu, W, center, radius, valid = _spherelet_fit_core(X, d=d)
    return {"mu": mu, "W": W, "center": center, "radius": radius, "valid": valid, "d": int(d)}


def reconstruct_spherelet_training(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = model["mu"]
    W = model["W"]
    if not model["valid"]:
        # Stable fallback to the d-dimensional PCA reconstruction.
        d = min(int(model["d"]), W.shape[1])
        U = W[:, :d]
        return mu + ((X - mu) @ U) @ U.T
    center = model["center"]
    radius = float(model["radius"])
    Z = (X - mu) @ W
    V = Z - center
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Zhat = center + radius * V / norms
    return mu + Zhat @ W.T
