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

from .config import TireSpecConfig, SegmentFileConfig, CombinedSlipSpecConfig, CombinedSlipSegmentConfig
from .segmentation import sort, bound
from .pacejka import first_pass_y, second_pass_y, first_pass_x, second_pass_x
from .combined_slip import first_pass_gx, second_pass_gx, R_PARAM_NAMES

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

    # x_scale='jac' unconditionally: verified against the installed scipy source
    # (check_x_scale in scipy/optimize/_lsq/least_squares.py) that x_scale=None
    # already resolves to 'jac' whenever method='lm' -- which is every spec here.
    # spec.x_scale_jac (config.py) has therefore been a no-op the whole time;
    # this makes the actual behavior explicit instead of relying on an
    # undocumented, scipy-version-dependent default.
    lsq_kwargs = dict(jac='3-point', method='lm', x_scale='jac', **_LSQ_TOL_KWARGS)
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


def refit_second_pass(fit_result: TireFitResult, second_pass_fn, p_x0=None, x_scale='jac'):
    """Re-run ONLY the second-pass P-parameter fit, reusing fit_result's
    already-computed cases/bcde_params/F_z0/lambda_mu -- skips the expensive
    per-segment first pass entirely (~29s/spec instead of the full fit).

    Returns the full scipy OptimizeResult (.x, .jac, .fun, .cost, ...)
    instead of just .x -- _fit_tire_spec above discards everything but
    result.x, but parameter-uncertainty work needs the Jacobian at the
    solution.

    Warm-starts from fit_result.p_params by default (not the generic
    p_x0_template): we want the Jacobian evaluated AT the already-reported
    solution, not wherever a fresh optimization happens to land -- some
    specs are known ill-conditioned enough for that to matter (see
    DECISIONS.md).
    """
    if p_x0 is None:
        p_x0 = fit_result.p_params
    fit_func = lambda x: second_pass_fn(fit_result.cases, fit_result.F_z0, fit_result.lambda_mu, fit_result.bcde_params, x)
    return least_squares(fit_func, p_x0, jac='3-point', method='lm', x_scale=x_scale, **_LSQ_TOL_KWARGS)


# ---------------------------------------------------------------------------
# Combined-slip (G_x) fitting -- see magic/combined_slip.py and
# magic/config.py's CombinedSlipSpecConfig/COMBINED_SLIP_SPECS.
# ---------------------------------------------------------------------------

@dataclass
class CombinedSlipFitResult:
    base_spec_name: str
    r_params: np.ndarray          # the 7 fitted RBX*/RCX1/REX*/RHX1 params, see combined_slip.R_PARAM_NAMES
    bces_params: np.ndarray       # per-segment B_gx/C_gx/E_gx/S_hgx, kept for diagnostics
    cases: list


def load_combined_slip_segments(seg_cfg: CombinedSlipSegmentConfig) -> list:
    """Same sort()->bound() pipeline as load_segments (magic.pipeline), but
    keeps segments WITH meaningful slip angle (mean(|SA|) > sa_threshold)
    instead of filtering them out -- these are the combined-slip cases."""
    raw = scipy.io.loadmat(seg_cfg.path)
    blocks = sort(raw, load_key=seg_cfg.sort_load_key,
                  window=seg_cfg.sort_window, threshold_factor=seg_cfg.sort_threshold_factor)
    trimmed = bound(blocks, slip_key=seg_cfg.bound_slip_key, threshold_factor=seg_cfg.bound_threshold_factor)

    cases = []
    for seg in trimmed:
        et_span = seg["ET"].max() - seg["ET"].min()
        if not (seg_cfg.et_min < et_span < seg_cfg.et_max):
            continue
        if not (np.abs(seg["SA"]).mean() > seg_cfg.sa_threshold):
            continue
        cases.append(seg)
    return cases


def fit_combined_slip_spec(cs_spec: CombinedSlipSpecConfig, base_fit_result: TireFitResult) -> CombinedSlipFitResult:
    """Fit the G_x combined-slip correction for one longitudinal spec,
    layered on top of its already-fitted pure-slip model (base_fit_result,
    from mf_fits.joblib -- NOT refit here, reused as-is).
    """
    cases = []
    for seg_cfg in cs_spec.segments:
        cases.extend(load_combined_slip_segments(seg_cfg))

    F_z0 = base_fit_result.F_z0
    long_params = base_fit_result.p_params
    n = len(cases)
    bces_params = np.zeros((n, 4))

    for i in range(n):
        fit_func = lambda x, seg=cases[i]: first_pass_gx(seg, x, F_z0, long_params)
        result = least_squares(fit_func, list(cs_spec.bces_x0), jac='3-point', method='trf',
                                bounds=(cs_spec.bces_lower, cs_spec.bces_upper),
                                ftol=None, xtol=2.3e-16, gtol=2.3e-16, max_nfev=int(1e8), verbose=1)
        bces_params[i] = result.x

    r_x0 = list(cs_spec.r_x0_template)
    r_x0[3] = bces_params[:, 1].mean()  # RCX1 <- mean fitted C_gx
    r_x0[6] = bces_params[:, 3].mean()  # RHX1 <- mean fitted S_hgx

    fit_func = lambda x: second_pass_gx(cases, F_z0, base_fit_result.lambda_mu, bces_params, x)
    result = least_squares(fit_func, r_x0, jac='3-point', method='lm', x_scale='jac', **_LSQ_TOL_KWARGS)

    return CombinedSlipFitResult(
        base_spec_name=cs_spec.base_spec_name,
        r_params=result.x,
        bces_params=bces_params,
        cases=cases,
    )
