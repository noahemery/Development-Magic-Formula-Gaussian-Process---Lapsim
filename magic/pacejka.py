"""Two-pass Pacejka Magic Formula fitting for lateral (Fy) and longitudinal (Fx) force.

Pass 1 (first_pass_y / first_pass_x): fit the raw sine-based Magic Formula
(B, C, D, E, S_h, S_v) independently to each segmented test case.

Pass 2 (second_pass_y / second_pass_x): fit the standard Pacejka P-parameters,
which describe how B/C/D/E/S_h/S_v vary with normalized load and camber,
against the per-case BCDE values from pass 1, across all cases at once.

_y_model_terms / _x_model_terms / _magic_formula_curve are the single source
of truth for "given P-parameters, what are B/C/D/E/S_h/S_v and the resulting
curve". Both second_pass_y/x (fit-time residuals) and tm_lat/tm_long
(predict-time curve) call these, rather than each re-deriving the formula --
that was the root cause of two real bugs in the original script:

  - second_pass_x's very first case computed B without the +1e-8 epsilon
    guard that every other case (and the y-direction equivalent) used,
    giving segment 0 inconsistent divide-by-zero protection.
  - second_pass_y's per-case S_v term added S_vy_gamma *outside* the
    subtraction for every case after the first, instead of nesting it
    inside the model term -- fit against the wrong target for all but
    one case.

Both are structurally impossible now: there is exactly one place the S_v/B
formulas are written down.
"""

import numpy as np
from scipy.optimize import least_squares

# Named, in exact unpack order -- single source of truth for anything that
# needs to label the raw x[i] entries in second_pass_y/x's solution vector
# (e.g. parameter-uncertainty reporting). Lateral has 20 named params;
# p_x0_template carries a 21st, unused spare slot (x[20]) -- see
# _y_model_terms's docstring -- deliberately NOT included here.
LATERAL_PARAM_NAMES = (
    "PDY1", "PDY2", "PDY3", "PCY1", "PKY1", "PKY2", "PKY3", "PKY4", "PKY5",
    "PHY1", "PHY2", "PEY1", "PEY2", "PEY3", "PEY4", "PEY5", "PVY1", "PVY2", "PVY3", "PVY4",
)
LONGITUDINAL_PARAM_NAMES = (
    "PDX1", "PDX2", "PCX1", "PKX1", "PKX2", "PKX3", "PHX1", "PHX2",
    "PEX1", "PEX2", "PEX3", "PEX4", "PVX1", "PVX2",
)


def _magic_formula_curve(x_slip, terms):
    B, C, D, E, S_h, S_v = terms["B"], terms["C"], terms["D"], terms["E"], terms["S_h"], terms["S_v"]
    return D * np.sin(C * np.arctan(B * (x_slip + S_h) - E * (B * (x_slip + S_h) - np.arctan(B * (x_slip + S_h))))) + S_v


# ---------------------------------------------------------------------------
# Lateral (Fy)
# ---------------------------------------------------------------------------

def first_pass_y(data, x):
    F_y = -data["FY"]
    alpha = np.tan(data["SA"] * np.pi / 180)

    B_y, C_y, D_y, E_y, S_hy, S_vy = x

    Y = D_y * np.sin(C_y * np.arctan(B_y * (alpha + S_hy) - E_y * (B_y * (alpha + S_hy) - np.arctan(B_y * (alpha + S_hy))))) + S_vy

    return (Y - F_y).squeeze()


def _y_model_terms(F_z, F_z0, alpha, gamma, lambda_mu_y, x):
    """B/C/D/E/S_h/S_v as functions of the 20 Pacejka lateral P-parameters.

    NOTE: the original script's lateral x0_P is 21 elements long, but only
    the first 20 (PDY1..PVY4) were ever named/used -- x[20] was a spare,
    unused free parameter that least_squares optimized over for no effect
    on the residual. Preserved here (only x[:20] is consumed) rather than
    silently dropped, so this stays numerically comparable to the original
    fit -- flagged separately as a discovered inconsistency, not one of the
    originally agreed bug fixes.
    """
    (PDY1, PDY2, PDY3, PCY1, PKY1, PKY2, PKY3, PKY4, PKY5,
     PHY1, PHY2, PEY1, PEY2, PEY3, PEY4, PEY5, PVY1, PVY2, PVY3, PVY4) = x[:20]

    df_z = (F_z - F_z0) / F_z0

    mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2)
    BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0)) / (1 + PKY3 * gamma ** 2))
    S_hy = PHY1 + PHY2 * df_z
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma

    return dict(
        D=mu_y * F_z,
        C=PCY1,
        B=BCD_y / (mu_y * F_z * PCY1 + 1e-8),
        S_h=S_hy,
        E=(PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) - np.sign(alpha + S_hy)),
        S_v=(PVY1 + PVY2 * df_z) * F_z + S_vy_gamma,
    )


def second_pass_y(data, F_z0, lambda_mu_y, BCDE_params, x):
    B_y, C_y, D_y, E_y, S_hy, S_vy = BCDE_params.T

    rows = []
    for i in range(len(data)):
        F_z = -data[i]["FZ"]
        alpha = np.tan(data[i]["SA"] * np.pi / 180)
        gamma = np.sin(data[i]["IA"] * np.pi / 180)

        terms = _y_model_terms(F_z, F_z0, alpha, gamma, lambda_mu_y, x)

        residual_terms = (
            D_y[i] - terms["D"],
            C_y[i] - terms["C"],
            B_y[i] - terms["B"],
            S_hy[i] - terms["S_h"],
            E_y[i] - terms["E"],
            S_vy[i] - terms["S_v"],
        )
        rows.extend(residual_terms)

    return np.vstack(rows).squeeze()


def tm_lat(F_z, F_z0, alpha, gamma, lambda_mu_y, x):
    alpha = np.tan(alpha * np.pi / 180)
    gamma = np.sin(gamma * np.pi / 180)
    terms = _y_model_terms(F_z, F_z0, alpha, gamma, lambda_mu_y, x)
    return _magic_formula_curve(alpha, terms)


# ---------------------------------------------------------------------------
# Longitudinal (Fx)
# ---------------------------------------------------------------------------

def first_pass_x(data, x):
    F_x = data["FX"]
    s = data["SL"]

    B_x, C_x, D_x, E_x, S_hx, S_vx = x

    Y = D_x * np.sin(C_x * np.arctan(B_x * (s + S_hx) - E_x * (B_x * (s + S_hx) - np.arctan(B_x * (s + S_hx))))) + S_vx

    return (Y - F_x).squeeze()


def _x_model_terms(F_z, F_z0, s, lambda_mu_x, x):
    """B/C/D/E/S_h/S_v as functions of the 14 Pacejka longitudinal P-parameters."""
    (PDX1, PDX2, PCX1, PKX1, PKX2, PKX3, PHX1, PHX2,
     PEX1, PEX2, PEX3, PEX4, PVX1, PVX2) = x

    df_z = (F_z - F_z0) / F_z0

    mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
    BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
    S_hx = PHX1 + PHX2 * df_z

    return dict(
        D=mu_x * F_z,
        C=PCX1,
        B=BCD_x / (mu_x * F_z * PCX1 + 1e-8),
        S_h=S_hx,
        E=(PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + S_hx)),
        S_v=F_z * (PVX1 + PVX2 * df_z),
    )


def second_pass_x(data, F_z0, lambda_mu_x, BCDE_params, x):
    B_x, C_x, D_x, E_x, S_hx, S_vx = BCDE_params.T

    rows = []
    for i in range(len(data)):
        F_z = -data[i]["FZ"]
        s = data[i]["SL"]

        terms = _x_model_terms(F_z, F_z0, s, lambda_mu_x, x)

        residual_terms = (
            D_x[i] - terms["D"],
            C_x[i] - terms["C"],
            B_x[i] - terms["B"],
            S_hx[i] - terms["S_h"],
            E_x[i] - terms["E"],
            S_vx[i] - terms["S_v"],
        )
        rows.extend(residual_terms)

    return np.vstack(rows).squeeze()


def tm_long(F_z, F_z0, s, lambda_mu_x, x):
    terms = _x_model_terms(F_z, F_z0, s, lambda_mu_x, x)
    return _magic_formula_curve(s, terms)
