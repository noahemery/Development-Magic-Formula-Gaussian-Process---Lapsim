"""Hybrid residual Gaussian Process layer.

The Magic Formula (magic.pacejka) is a fixed analytic curve shape -- it gets
the overall S-curve right but can't represent everything real rig data does
(local asymmetries, peak-region deviations, load/camber effects the
P-parameter interpolation smooths over). ResidualGP fits a Gaussian Process
on what's left over after the MF: `residual = measured_force - MF_prediction`,
as a function of (F_z, slip, IA). It corrects the MF's systematic misfit and,
just as importantly, gives a calibrated uncertainty band the MF alone can't
provide.

The MF's own coefficients (TireFitResult.p_params / TireModel.p_params) are
left untouched by this -- anything that only wants the standard, portable
Pacejka numbers can ignore this module entirely.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from .pacejka import tm_lat, tm_long


@dataclass
class ResidualGPConfig:
    length_scale_init: tuple = (1.0, 1.0, 1.0)   # one per feature: [F_z, slip, IA], in standardized space
    length_scale_bounds: tuple = (0.2, 1e2)       # floor raised from an earlier 1e-2 after real data showed it collapsing:
                                                    # 3 of 6 tire specs pinned a length-scale (usually IA) at 1e-2, which
                                                    # made the GP hypersensitive to small real IA jitter within a sweep --
                                                    # visually, spiky, non-physical corrections instead of smooth ones.
                                                    # 0.2 (20% of a standardized std-dev) still lets a feature matter a lot
                                                    # without letting the optimizer chase sample-to-sample noise.
    constant_value_bounds: tuple = (1e-3, 1e3)
    noise_level_init: float = 1.0
    noise_level_bounds: tuple = (1e-5, 1e1)       # WhiteKernel models the real noise floor
    n_restarts_optimizer: int = 5
    alpha: float = 1e-10                           # numerical jitter ONLY -- not the noise model, that's WhiteKernel's job
    max_points_per_segment: int = 200              # per-segment ceiling; build_gp_dataset's max_total_points is what actually bounds the pooled dataset
    random_state: int = 0                          # fixed so teammates re-running this get the same numbers


class ResidualGP:
    """StandardScaler + GaussianProcessRegressor(anisotropic RBF + WhiteKernel).

    Feature scaling matters here: F_z is ~O(100-2000) N while slip/IA are
    ~O(+/-15 deg) -- an unscaled isotropic kernel would be dominated by F_z
    alone and effectively ignore slip and camber.
    """

    def __init__(self, config: ResidualGPConfig = None):
        self.config = config or ResidualGPConfig()
        self.scaler_ = None
        self.gpr_ = None

    def fit(self, X_raw: np.ndarray, residual: np.ndarray) -> "ResidualGP":
        cfg = self.config
        self.scaler_ = StandardScaler().fit(X_raw)
        X = self.scaler_.transform(X_raw)

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=cfg.constant_value_bounds)
            * RBF(length_scale=list(cfg.length_scale_init), length_scale_bounds=cfg.length_scale_bounds)
            + WhiteKernel(noise_level=cfg.noise_level_init, noise_level_bounds=cfg.noise_level_bounds)
        )
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=cfg.n_restarts_optimizer,
            alpha=cfg.alpha,
            random_state=cfg.random_state,
        )
        self.gpr_.fit(X, residual)
        return self

    def predict(self, X_raw: np.ndarray, return_std: bool = True):
        X = self.scaler_.transform(np.atleast_2d(X_raw))
        return self.gpr_.predict(X, return_std=return_std)

    @property
    def fitted_length_scales_(self):
        """Fitted length-scales in standardized feature space, order
        [F_z, slip, IA]. A length-scale that's blown up relative to the
        others means the GP decided that feature doesn't matter -- worth
        checking, not assuming away."""
        return self.gpr_.kernel_.k1.k2.length_scale


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def build_gp_dataset(cases: list, fit_result, direction: str,
                      max_points_per_segment: int = 200, max_total_points: int = 3000):
    """Build (X, residual, segment_id) for fitting/evaluating a ResidualGP.

    X columns: [F_z (positive-loaded), slip (SA in degrees / SL ratio, raw
    units -- not pre-transformed), IA (degrees)].

    residual = measured_force - MF_prediction, computed via tm_lat/tm_long
    (the SAME function, with the SAME sign convention, the MF was actually
    fit with) so this can't silently drift from what second_pass_y/x fit
    against -- that mismatch is exactly the bug class this is designed to
    avoid (see magic/pacejka.py docstring).

    segment_id: one integer per row, constant within a case -- required for
    GroupKFold so validation never splits a single sweep across train/val.

    max_total_points bounds the POOLED dataset, not just each segment --
    some tire specs here have 45-90 accepted segments, so a flat per-segment
    cap alone (e.g. 200) can still pool into 9,000-18,000 points, which is
    not tractable for an exact GP (O(n^3), refit on every hyperparameter
    optimizer step x n_restarts_optimizer). The effective per-segment cap is
    derived from the total budget divided across however many segments this
    spec actually has, floored at 10 points/segment so sparse specs still
    get reasonable slip coverage.
    """
    if direction not in ("lateral", "longitudinal"):
        raise ValueError(f"unknown direction {direction!r}")

    force_key = "FY" if direction == "lateral" else "FX"
    slip_key = "SA" if direction == "lateral" else "SL"
    measured_sign = -1.0 if direction == "lateral" else 1.0   # matches first_pass_y's `-data["FY"]` / first_pass_x's `data["FX"]`

    effective_cap = max(10, min(max_points_per_segment, max_total_points // max(1, len(cases))))

    X_rows, residual_rows, seg_ids = [], [], []

    for seg_id, case in enumerate(cases):
        F_z = np.atleast_1d(np.asarray(-case["FZ"])).ravel()
        slip = np.atleast_1d(np.asarray(case[slip_key])).ravel()
        IA = np.atleast_1d(np.asarray(case["IA"])).ravel()
        measured = measured_sign * np.atleast_1d(np.asarray(case[force_key])).ravel()

        idx = _subsample_indices(len(F_z), effective_cap)
        F_z_s, slip_s, IA_s, measured_s = F_z[idx], slip[idx], IA[idx], measured[idx]

        if direction == "lateral":
            mf_pred = tm_lat(F_z_s, fit_result.F_z0, slip_s, IA_s, fit_result.lambda_mu, fit_result.p_params)
        else:
            # Longitudinal MF has no camber term (matches first_pass_x/second_pass_x/tm_long) --
            # IA is still recorded as a GP feature since real Fx does vary with camber even though the MF ignores it.
            mf_pred = tm_long(F_z_s, fit_result.F_z0, slip_s, fit_result.lambda_mu, fit_result.p_params)

        residual = measured_s - mf_pred

        X_rows.append(np.column_stack([F_z_s, slip_s, IA_s]))
        residual_rows.append(residual)
        seg_ids.append(np.full(len(idx), seg_id))

    X = np.vstack(X_rows)
    residual = np.concatenate(residual_rows)
    segment_id = np.concatenate(seg_ids)
    return X, residual, segment_id
