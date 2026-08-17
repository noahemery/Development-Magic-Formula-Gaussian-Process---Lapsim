"""Per-tire-spec configuration.

Encodes, as data instead of copy-pasted code, the six tire-spec fitting
blocks that used to be hand-duplicated in magic.py (2x for lateral, 4x for
longitudinal). Every literal value here (files, sort/bound windows and
thresholds, ET-duration filter ranges, least-squares bounds, x0 seeds) is
carried over unchanged from the original script -- this file should not
change *what* gets fit, only how many times the fitting logic itself had to
be retyped to do it.

Two things worth flagging rather than silently "fixing" while porting this:

1. `fz0_files` is NOT always the same as the files that get segmented into
   cases -- e.g. 160X75_R20_80's F_z0 is averaged from raw7/raw8/raw9, but
   only raw8 is actually segmented into fitted cases. Preserved as-is.
2. `x_scale_jac` (below) turned out to be a no-op, not a real inconsistency.
   Originally: only 160X75_R20_70 (lateral) had `x_scale="jac"` active in
   its second-pass least_squares call, every other spec had it commented
   out. That looked like a leftover from tuning one spec. Verified against
   the installed scipy source (`check_x_scale` in
   scipy/optimize/_lsq/least_squares.py): `x_scale=None` already resolves
   to `'jac'` whenever `method='lm'`, which is every spec's second-pass
   call -- so all 6 specs were always fitting with `x_scale='jac'`
   regardless of this flag. `magic/pipeline.py` now sets `x_scale='jac'`
   unconditionally to make that explicit. This field is kept on
   `TireSpecConfig` for backward compatibility only; it's no longer read
   anywhere and is safe to remove.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class SegmentFileConfig:
    path: str
    sort_load_key: str
    sort_window: int
    sort_threshold_factor: float
    bound_slip_key: str
    bound_threshold_factor: float
    et_min: float
    et_max: float
    filter_small_sa: bool = False   # longitudinal specs only: keep cases where mean(|SA|) < 1e-1


@dataclass(frozen=True)
class TireSpecConfig:
    name: str
    direction: Literal["lateral", "longitudinal"]
    force_key: str                      # "FY" or "FX" -- feeds the dynamic D-bound margin
    fz0_files: tuple
    segments: tuple                     # tuple[SegmentFileConfig, ...]
    bcde_lower: list
    bcde_upper_base: list                # index 2 (D) is a MARGIN; actual upper = base + |force|.max() per segment
    p_x0_template: list                  # x0 for the second-pass fit; one entry is a placeholder (see p_x0_c_index)
    p_x0_c_index: int                    # index into p_x0_template that gets replaced with BCDE_params[:,1].mean()
    lambda_mu: float = 1.0
    x_scale_jac: bool = False            # see module docstring point 2


CORNERING_DIR = "data/cornering_SI"
STRAIGHT_DIR = "data/straight_SI"


LATERAL_SPECS = (
    TireSpecConfig(
        name="160X75_R20_70",
        direction="lateral",
        force_key="FY",
        fz0_files=(f"{CORNERING_DIR}/B2356raw4.mat", f"{CORNERING_DIR}/B2356raw5.mat", f"{CORNERING_DIR}/B2356raw6.mat"),
        segments=(
            SegmentFileConfig(f"{CORNERING_DIR}/B2356raw4.mat", "FZ", 200, 30, "SA", 0.86, 8, 12),
            SegmentFileConfig(f"{CORNERING_DIR}/B2356raw5.mat", "FZ", 250, 60, "SA", 1.0, 8, 12),
            SegmentFileConfig(f"{CORNERING_DIR}/B2356raw6.mat", "FZ", 250, 50, "SA", 1.0, 10, 13),
        ),
        bcde_lower=[0, 1, 0, 0, -0.075, -100],
        bcde_upper_base=[50, 2, 100, 1, 0.075, 100],
        p_x0_template=[1, 0, 0, None, 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0.15, 0],
        p_x0_c_index=3,
        x_scale_jac=True,
    ),
    TireSpecConfig(
        name="160X75_R20_80",
        direction="lateral",
        force_key="FY",
        fz0_files=(f"{CORNERING_DIR}/B2356raw8.mat", f"{CORNERING_DIR}/B2356raw7.mat", f"{CORNERING_DIR}/B2356raw9.mat"),
        segments=(
            SegmentFileConfig(f"{CORNERING_DIR}/B2356raw8.mat", "FZ", 130, 90, "SA", 0.86, 6, 12),
        ),
        bcde_lower=[0, 1, 0, 0, -0.075, -75],
        bcde_upper_base=[100, 2, 75, 1, 0.075, 75],
        p_x0_template=[1, 0, 0, None, 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0.15, 0],
        p_x0_c_index=3,
        x_scale_jac=False,
    ),
)


LONGITUDINAL_SPECS = (
    TireSpecConfig(
        name="205X70_R20_70",
        direction="longitudinal",
        force_key="FX",
        fz0_files=(f"{STRAIGHT_DIR}/B2356raw51.mat", f"{STRAIGHT_DIR}/B2356raw52.mat"),
        segments=(
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw51.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw52.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
        ),
        bcde_lower=[0, 1.2, 0, 0, -0.075, -200],
        bcde_upper_base=[20, 2, 200, 1, 0.075, 200],
        p_x0_template=[1, 0, None, 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0],
        p_x0_c_index=2,
        x_scale_jac=False,
    ),
    TireSpecConfig(
        name="205X70_R20_80",
        direction="longitudinal",
        force_key="FX",
        fz0_files=(f"{STRAIGHT_DIR}/B2356raw54.mat", f"{STRAIGHT_DIR}/B2356raw55.mat"),
        segments=(
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw54.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw55.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
        ),
        bcde_lower=[0, 1.2, 0, 0, -0.075, -150],
        bcde_upper_base=[20, 2, 100, 1, 0.075, 150],
        p_x0_template=[1, 0, None, 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0],
        p_x0_c_index=2,
        x_scale_jac=False,
    ),
    TireSpecConfig(
        name="180X60_R20_60",
        direction="longitudinal",
        force_key="FX",
        fz0_files=(f"{STRAIGHT_DIR}/B2356raw69.mat", f"{STRAIGHT_DIR}/B2356raw70.mat"),
        segments=(
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw69.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw70.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
        ),
        bcde_lower=[0, 1.2, 0, 0, -0.075, -150],
        bcde_upper_base=[20, 2, 150, 1, 0.075, 150],
        p_x0_template=[1, 0, None, 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0],
        p_x0_c_index=2,
        x_scale_jac=False,
    ),
    TireSpecConfig(
        name="180X60_R20_70",
        direction="longitudinal",
        force_key="FX",
        fz0_files=(f"{STRAIGHT_DIR}/B2356raw72.mat", f"{STRAIGHT_DIR}/B2356raw73.mat"),
        segments=(
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw72.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
            SegmentFileConfig(f"{STRAIGHT_DIR}/B2356raw73.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, filter_small_sa=True),
        ),
        bcde_lower=[0, 1.2, 0, 0, -0.075, -50],
        bcde_upper_base=[20, 2, 50, 1, 0.075, 50],
        p_x0_template=[1, 0, None, 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0],
        p_x0_c_index=2,
        x_scale_jac=False,
    ),
)


ALL_SPECS = LATERAL_SPECS + LONGITUDINAL_SPECS


# ---------------------------------------------------------------------------
# Combined-slip (G_x) correction config -- see magic/combined_slip.py.
#
# Ported from William Young's magic1.py, which currently only fits this
# correction for one longitudinal spec (205X70_R20_70). Kept to that same
# scope here rather than assuming it should cover all 4 -- expanding to the
# other 3 longitudinal specs is a deliberate follow-up, not done implicitly.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CombinedSlipSegmentConfig:
    """Like SegmentFileConfig, but selects segments WITH meaningful slip
    angle present (mean(|SA|) > sa_threshold) instead of filtering them
    out -- these are the combined-slip cases the G_x correction is fit
    against. Same raw files/sort/bound params as the base spec's pure-slip
    segmentation; only the SA filter direction flips."""
    path: str
    sort_load_key: str
    sort_window: int
    sort_threshold_factor: float
    bound_slip_key: str
    bound_threshold_factor: float
    et_min: float
    et_max: float
    sa_threshold: float = 0.5   # keep cases where mean(|SA|) > this


@dataclass(frozen=True)
class CombinedSlipSpecConfig:
    """Which longitudinal tire spec to layer a G_x correction onto (reuses
    that spec's already-fitted tm_long/p_params/F_z0 as the base model),
    and which raw files/segmentation to fit the correction from."""
    base_spec_name: str          # must match a LONGITUDINAL_SPECS entry
    segments: tuple              # tuple[CombinedSlipSegmentConfig, ...]
    bces_lower: tuple = (0, 0, 0.5, -0.5)
    bces_upper: tuple = (30, 2, 1, 0.5)
    bces_x0: tuple = (10, 1.65, 0.5, 0)
    # x0 for the second-pass R-parameter fit; indices 3 (RCX1) and 6 (RHX1)
    # get seeded from the first-pass BCES fit's mean C_gx/S_hgx (see
    # magic/pipeline.py:fit_combined_slip_spec), matching how the main
    # pipeline seeds PCY1/PCX1 from the mean fitted C.
    r_x0_template: tuple = (5, 8, 0, 0, 0, 0, 0)


COMBINED_SLIP_SPECS = (
    CombinedSlipSpecConfig(
        base_spec_name="205X70_R20_70",
        segments=(
            CombinedSlipSegmentConfig(f"{STRAIGHT_DIR}/B2356raw51.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, sa_threshold=0.5),
            CombinedSlipSegmentConfig(f"{STRAIGHT_DIR}/B2356raw52.mat", "FZ", 100, 10, "SL", 1.0, 5, 15, sa_threshold=0.5),
        ),
    ),
)
