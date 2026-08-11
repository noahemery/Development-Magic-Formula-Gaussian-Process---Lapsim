"""Saving/loading fitted results.

Previously the fitted tire models (lat_160X75_R20_70, etc.) were plain local
variables that vanished the moment magic.py finished running. This is the
fix for that -- everything computed by the pipeline gets written to disk so
it can actually be reused (by the GP layer, by lap-sim tooling, by the next
person who wants to load a model without re-fitting).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib

from .pipeline import TireFitResult


@dataclass
class TireModel:
    """The final, persisted artifact for one tire spec: the Magic Formula
    fit plus (once Stage 2 runs) its residual GP. `p_params`/`F_z0` alone
    are the plain, portable Pacejka coefficients -- anything that only
    wants the standard MF can ignore `gp` entirely."""
    name: str
    direction: str
    F_z0: float
    p_params: object          # np.ndarray
    lambda_mu: float
    bcde_params: object = None       # np.ndarray, kept for diagnostics
    gp: Optional[object] = None      # magic.gp_residual.ResidualGP, once fit
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_fit_result(cls, fit_result: TireFitResult, **metadata) -> "TireModel":
        return cls(
            name=fit_result.name,
            direction=fit_result.direction,
            F_z0=fit_result.F_z0,
            p_params=fit_result.p_params,
            lambda_mu=fit_result.lambda_mu,
            bcde_params=fit_result.bcde_params,
            metadata=metadata,
        )


def save_fit_results(results: dict, path) -> Path:
    """results: {spec_name: TireFitResult}. Used for the Stage 1 output
    (mf_fits.joblib), which still carries `cases` -- needed by Stage 2 to
    build the GP residual dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, path)
    return path


def load_fit_results(path) -> dict:
    return joblib.load(path)


def save_tire_models(models: dict, path) -> Path:
    """models: {spec_name: TireModel}. The final MF+GP artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    return path


def load_tire_models(path) -> dict:
    return joblib.load(path)
