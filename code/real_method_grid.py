"""Prespecified Table-II-style method grid for all real-data experiments.

The grid mirrors the paper's organization:
  * quadratic SCQM: lp^p, p in {1,1.25,1.5,1.75,2}, and unsquared l2;
  * linear counterpart (Theta fixed to zero): the same six losses;
  * three comparison baselines (SPH, LPCA/PCA-d, Q-LS) are added by each experiment runner.

No test-set result is used to choose p or a learning rate.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Iterable

from scqm_njit import SCQMConfig

P_VALUES: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)


def method_name(quadratic: bool, *, p: float | None = None, l2: bool = False) -> str:
    prefix = "Q" if quadratic else "L"
    if l2:
        return f"{prefix}-l2"
    if p is None:
        raise ValueError("p is required unless l2=True")
    txt = str(int(p)) if float(p).is_integer() else str(p).rstrip("0").rstrip(".")
    return f"{prefix}-p{txt}"


def method_metadata(name: str) -> dict[str, object]:
    if name in {"SPH", "LPCA", "PCA-d", "Q-LS"}:
        return {"family": "baseline", "quadratic": None, "loss": None, "p": None}
    quadratic = name.startswith("Q-")
    if name.endswith("-l2"):
        return {"family": "SCQM", "quadratic": quadratic, "loss": "l2", "p": None}
    p = float(name.split("p", 1)[1])
    return {"family": "SCQM", "quadratic": quadratic, "loss": "lp_p", "p": p}


def build_method_configs(
    *,
    d: int,
    s: int,
    max_iter: int,
    seed: int,
    eta_Q: float,
    eta_Theta: float,
    eta_c: float,
    eta_tau: float,
    tol: float = 1e-6,
    min_iter: int = 8,
    max_backtracks: int = 18,
    theta_init_scale: float = 0.0,
    gradient_clip: float = 1e6,
) -> "OrderedDict[str, SCQMConfig]":
    """Create the fixed 12-model comparison grid.

    The p=2 model is exactly the elementwise squared Euclidean objective and is
    the SQMF-style comparator.  A slightly smaller quadratic-parameter step is
    used for p=2 because its gradients scale linearly with large residuals.
    """
    if d < 1 or s < 1:
        raise ValueError("Real-data grid requires d>=1 and s>=1")
    common = dict(
        d=d,
        max_iter=max_iter,
        tol=tol,
        eta_Q=eta_Q,
        eta_Theta=eta_Theta,
        eta_c=eta_c,
        eta_tau=eta_tau,
        tau_strategy="armijo",
        max_backtracks=max_backtracks,
        min_iter=min_iter,
        random_state=seed,
        theta_init_scale=theta_init_scale,
        gradient_clip=gradient_clip,
        refresh_gradient_each_block=True,
    )
    out: "OrderedDict[str, SCQMConfig]" = OrderedDict()
    for quadratic in (True, False):
        s_here = s if quadratic else 0
        for p in P_VALUES:
            theta_step = eta_Theta * (0.5 if quadratic and p == 2.0 else 1.0)
            out[method_name(quadratic, p=p)] = SCQMConfig(
                s=s_here,
                loss="lp_p",
                p=p,
                eta_Theta=theta_step,
                **{k: v for k, v in common.items() if k != "eta_Theta"},
            )
        out[method_name(quadratic, l2=True)] = SCQMConfig(
            s=s_here,
            loss="l2",
            **common,
        )
    return out


def ordered_method_names(image_baseline: bool = False) -> list[str]:
    names = [method_name(True, p=p) for p in P_VALUES] + [method_name(True, l2=True)]
    names += [method_name(False, p=p) for p in P_VALUES] + [method_name(False, l2=True)]
    names += ["SPH", "PCA-d" if image_baseline else "LPCA", "Q-LS"]
    return names
