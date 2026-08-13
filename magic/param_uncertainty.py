"""Asymptotic parameter-level uncertainty for the fitted Pacejka P-parameters.

Different question from magic.gp_residual: that module gives uncertainty on
the PREDICTED FORCE at a given input. This module gives uncertainty on the
P-PARAMETERS themselves (how confident are we that PDY1 is really 2.9, not
somewhere between 1 and 5?) -- what William asked for specifically.

Method: the standard asymptotic covariance for nonlinear least squares
(Seber & Wild; the same formula scipy.optimize.curve_fit uses internally),
cov ~= sigma^2 * pinv(J^T J), evaluated at the fitted solution's Jacobian
(J = result.jac from magic.pipeline.refit_second_pass) and residual variance
(sigma^2 = RSS / degrees_of_freedom). This is a LINEARIZED approximation
(assumes the residual surface is locally quadratic near the solution) --
cheap (no refitting), but not as trustworthy as a bootstrap (resample +
refit many times, look at the empirical spread) would be. Bootstrap is
deferred -- see DECISIONS.md for the cost estimate (hours, not minutes).

Two real reasons this can come out singular or wildly ill-conditioned, and
both are handled explicitly rather than treated as a fitting failure:

1. Lateral's second-pass x-vector has a documented spare, unused 21st
   parameter (x[20]) -- its Jacobian column is EXACTLY zero, which alone
   makes J^T J rank-deficient. Drop it before computing (see `drop_cols`).
2. Genuine non-identifiability: some P-parameter combinations may have no
   real, identifiable effect on the fit given a spec's actual data
   (independently confirmed for 160X75_R20_70 -- see DECISIONS.md). This
   is a real finding about the physics/data, not a bug to hide, so pinv is
   used (degrades gracefully to large/NaN std) instead of inv (would raise
   LinAlgError and stop the whole report).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ParamUncertainty:
    name: str
    direction: str
    param_names: tuple
    p_params: np.ndarray          # values AT this Jacobian (post drop_cols), matches param_names 1:1
    std: np.ndarray                # NaN where the pinv diagonal came out negative (ill-conditioned)
    cov: np.ndarray
    relative_std_pct: np.ndarray
    cond_number: float              # cond(J^T J) -- large means ill-conditioned/near-singular
    rank: int
    n_params: int
    dof: int
    residual_variance: float
    well_identified: np.ndarray      # bool per param; relative_std_pct < WELL_IDENTIFIED_THRESHOLD_PCT
    dropped_param_names: tuple = ()  # e.g. lateral's unused x[20] -- recorded, never silently discarded


WELL_IDENTIFIED_THRESHOLD_PCT = 50.0   # rough, explicitly-approximate cutoff for a same-day report, not a rigorous test
RCOND_DEFAULT = 1e-8                    # looser than pinv's tiny numpy default; some specs are meaningfully ill-conditioned


def param_covariance(result, param_names, drop_cols=(), rcond=RCOND_DEFAULT, name="", direction="") -> ParamUncertainty:
    """Build a ParamUncertainty from a scipy OptimizeResult (from
    magic.pipeline.refit_second_pass), dropping any known-dead columns
    (e.g. lateral's unused param) before forming J^T J.

    KNOWN LIMITATION (not fixed today): rcond truncates in RAW parameter
    units. P-parameters span roughly 1e-3 to 1e2, so this conflates
    "genuinely non-identifiable" with "differently scaled" -- a
    correlation-style rescaling (divide by sqrt(diag(J^T J)) before
    truncating) would be more principled. Treat results near the rcond
    boundary (very large relative_std_pct) as suspect either way, not as
    a precise number.
    """
    J = np.delete(np.atleast_2d(result.jac), drop_cols, axis=1)
    p_params = np.delete(np.atleast_1d(result.x), drop_cols)
    kept_names = tuple(n for i, n in enumerate(param_names) if i not in set(drop_cols))
    # drop_cols can reference an index beyond len(param_names) -- e.g. lateral's
    # spare 21st parameter genuinely has no name (it's unused dead weight, not a
    # real Pacejka coefficient), so param_names only lists the 20 real ones.
    dropped_names = tuple(param_names[i] if i < len(param_names) else f"x[{i}] (unnamed, unused)" for i in drop_cols)

    m, n = J.shape
    dof = max(m - n, 1)
    residual_variance = float(np.sum(result.fun ** 2) / dof)

    JtJ = J.T @ J
    cov = residual_variance * np.linalg.pinv(JtJ, rcond=rcond)

    diag = np.diag(cov)
    std = np.sqrt(np.where(diag >= 0, diag, np.nan))
    relative_std_pct = std / (np.abs(p_params) + 1e-12) * 100

    cond_number = float(np.linalg.cond(JtJ))
    rank = int(np.linalg.matrix_rank(JtJ))
    well_identified = relative_std_pct < WELL_IDENTIFIED_THRESHOLD_PCT

    return ParamUncertainty(
        name=name,
        direction=direction,
        param_names=kept_names,
        p_params=p_params,
        std=std,
        cov=cov,
        relative_std_pct=relative_std_pct,
        cond_number=cond_number,
        rank=rank,
        n_params=n,
        dof=dof,
        residual_variance=residual_variance,
        well_identified=well_identified,
        dropped_param_names=dropped_names,
    )
