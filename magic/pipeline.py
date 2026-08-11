"""Config-driven replacement for the hand-duplicated per-tire-spec blocks.

fit_tire_spec_lateral(spec) / fit_tire_spec_longitudinal(spec) replace the
2x / 4x copy-pasted segmentation+fit blocks that used to live directly in
magic.py, one per tire spec. All fitting numerics (bounds, x0 seeding,
least_squares settings) are carried over unchanged from the original script;
only the "typed out N times" part is collapsed into a loop over
magic.config.LATERAL_SPECS / LONGITUDINAL_SPECS.
"""

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import scipy.io
from scipy.optimize import least_squares

from .config import TireSpecConfig, SegmentFileConfig
from .segmentation import sort, bound
from .pacejka import first_pass_y, second_pass_y, first_pass_x, second_pass_x

_LSQ_TOL_KWARGS = dict(ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16, max_nfev=int(1e8), verbose=1)


@dataclass
class TireFitResult:
    name: str
    direction: str
    F_z0: float
    p_params: np.ndarray
    bcde_params: np.ndarray
    cases: list
    lambda_mu: float = 1.0


def load_segments(file_cfg: SegmentFileConfig) -> list:
    """loadmat -> sort() -> bound() -> ET-duration / small-SA filtering, for one raw file."""
    raw = scipy.io.loadmat(file_cfg.path)
    blocks = sort(raw, load_key=file_cfg.sort_load_key,
                  window=file_cfg.sort_window, threshold_factor=file_cfg.sort_threshold_factor)
    trimmed = bound(blocks, slip_key=file_cfg.bound_slip_key, threshold_factor=file_cfg.bound_threshold_factor)

    cases = []
    for seg in trimmed:
        et_span = seg["ET"].max() - seg["ET"].min()
        if not (file_cfg.et_min < et_span < file_cfg.et_max):
            continue
        if file_cfg.filter_small_sa and not (np.abs(seg["SA"]).mean() < 1e-1):
            continue
        cases.append(seg)
    return cases


def compute_fz0(spec: TireSpecConfig) -> float:
    """Mean load across spec.fz0_files -- NOT necessarily the same files that
    get segmented (see config.py).

    Raw FZ is negative under load (rig convention). Every other use of load
    in this pipeline flips it positive (`F_z = -case["FZ"]`, see pacejka.py),
    so F_z0 is negated here too to match -- otherwise df_z = (F_z - F_z0)/F_z0
    compares a positive F_z against a negative F_z0 and never gets near zero
    even at the reference load (verified against real data: df_z sat around
    -1.9 on average with the unnegated F_z0, vs ~-0.09 -- a proper small
    deviation around zero -- once the signs match). Fixed here rather than
    left as found.
    """
    arrays = [-scipy.io.loadmat(path)["FZ"] for path in spec.fz0_files]
    return np.vstack(arrays).mean()


def build_cases(spec: TireSpecConfig) -> list:
    cases = []
    for seg_cfg in spec.segments:
        cases.extend(load_segments(seg_cfg))
    return cases


def fit_bcde_per_segment(cases: list, first_pass_fn, spec: TireSpecConfig) -> np.ndarray:
    """Per-segment BCDE fit (pass 1). C is warm-started from the previous
    segment's fit for i>0; the D upper bound is a per-segment margin added
    to that segment's own max |force| -- both reproduced exactly as in the
    original hardcoded blocks."""
    n = len(cases)
    bcde_params = np.zeros((n, 6))

    for i in range(n):
        x0 = [0, 1.45, 500, 0, 0, 0] if i == 0 else [0, bcde_params[i - 1, 1], 500, 0, 0, 0]

        upper = list(spec.bcde_upper_base)
        upper[2] = spec.bcde_upper_base[2] + np.abs(cases[i][spec.force_key]).max()

        fit_func = lambda x, seg=cases[i]: first_pass_fn(seg, x)
        result = least_squares(fit_func, x0, jac='3-point', method='trf',
                                bounds=(spec.bcde_lower, upper), **_LSQ_TOL_KWARGS)
        bcde_params[i] = result.x

    return bcde_params


def _fit_tire_spec(spec: TireSpecConfig, first_pass_fn, second_pass_fn) -> TireFitResult:
    F_z0 = compute_fz0(spec)
    cases = build_cases(spec)
    bcde_params = fit_bcde_per_segment(cases, first_pass_fn, spec)

    p_x0 = list(spec.p_x0_template)
    p_x0[spec.p_x0_c_index] = bcde_params[:, 1].mean()

    fit_func = lambda x: second_pass_fn(cases, F_z0, spec.lambda_mu, bcde_params, x)
    lsq_kwargs = dict(jac='3-point', method='lm', **_LSQ_TOL_KWARGS)
    if spec.x_scale_jac:
        lsq_kwargs['x_scale'] = 'jac'
    result = least_squares(fit_func, p_x0, **lsq_kwargs)

    return TireFitResult(
        name=spec.name,
        direction=spec.direction,
        F_z0=F_z0,
        p_params=result.x,
        bcde_params=bcde_params,
        cases=cases,
        lambda_mu=spec.lambda_mu,
    )


fit_tire_spec_lateral = partial(_fit_tire_spec, first_pass_fn=first_pass_y, second_pass_fn=second_pass_y)
fit_tire_spec_longitudinal = partial(_fit_tire_spec, first_pass_fn=first_pass_x, second_pass_fn=second_pass_x)
