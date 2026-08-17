"""Combined-slip correction for longitudinal force (G_x weighting function).

The pure-longitudinal fit (magic.pacejka.tm_long / second_pass_x) assumes
zero lateral slip -- that's exactly why its case selection filters for
mean(|SA|) < 0.1. Real driving is rarely that clean (trail braking,
power-on exit mid-corner), and when slip angle is also present the tire
can't produce as much longitudinal force as the pure-slip curve predicts --
there's a shared grip limit between Fx and Fy (the friction ellipse). G_x is
a weighting function (0 < G_x <= 1) that scales the pure-slip Fx prediction
down based on how much slip angle is also present:

    Fx_combined = tm_long(...) * G_x(...)

Ported from William Young's magic1.py (first_pass_GX/second_pass_GX/GX),
rebuilt on top of THIS package's tm_long/second_pass_x -- which already has
the epsilon-guard fix his copy is still missing, see magic/pacejka.py's
module docstring -- rather than duplicating his copy of the base model.
Also fixes a real risk in his version: the G_x formula was written out
twice (once in first_pass_GX, once in GX) and could silently drift apart,
same bug class as the two documented bugs in the original script. Here
there is exactly one place it's written down (_gx_terms/_gx_curve), used
by both fit-time and predict-time code -- same pattern already used for
the pure-slip lateral/longitudinal formulas.
"""

import numpy as np

from .pacejka import tm_long

R_PARAM_NAMES = ("RBX1", "RBX2", "RBX3", "RCX1", "REX1", "REX2", "RHX1")


def _gx_terms(F_z, F_z0, gamma, s, x):
    """B/C/E/S_h for the G_x weighting curve, from the 7 R-parameters."""
    RBX1, RBX2, RBX3, RCX1, REX1, REX2, RHX1 = x
    df_z = (F_z - F_z0) / F_z0
    return dict(
        B=(RBX1 + RBX3 * gamma ** 2) * np.cos(np.arctan(RBX2 * s)),
        C=RCX1,
        E=REX1 + REX2 * df_z,
        S_h=RHX1,
    )


def _gx_curve(alpha, terms):
    B, C, E, S_h = terms["B"], terms["C"], terms["E"], terms["S_h"]
    G_x0 = np.cos(C * np.arctan(B * S_h - E * (B * S_h - np.arctan(B * S_h))))
    return np.cos(C * np.arctan(B * (alpha + S_h) - E * (B * (alpha + S_h) - np.arctan(B * (alpha + S_h))))) / G_x0


def first_pass_gx(data, x, F_z0, long_params, lambda_mu_x=1.0):
    """Per-segment fit of B_gx/C_gx/E_gx/S_hgx against the base longitudinal
    prediction (tm_long, the already-fitted pure-slip model for this spec).
    """
    F_z = -data["FZ"]
    F_x = data["FX"]
    s = data["SL"]
    alpha = np.tan(data["SA"] * np.pi / 180)

    terms = dict(B=x[0], C=x[1], E=x[2], S_h=x[3])
    G_x = _gx_curve(alpha, terms)

    Y = tm_long(F_z, F_z0, s, lambda_mu_x, long_params) * G_x
    residual = (Y - F_x).squeeze()

    # Same physical constraint William's version enforces: G_x must not
    # exceed 1 -- it's a REDUCTION in available force from combined slip,
    # never an increase. Preserved as-found (a crude but workable penalty)
    # rather than redesigned into a properly constrained optimization,
    # which is a bigger change than this pass covers.
    if np.any(G_x > 1):
        return residual ** 10
    return residual


def second_pass_gx(data, F_z0, lambda_mu_x, BCES_params, x):
    """Global R-parameter fit -- how B_gx/C_gx/E_gx/S_hgx vary with load and camber."""
    B_gx, C_gx, E_gx, S_hgx = BCES_params.T

    rows = []
    for i in range(len(data)):
        F_z = -data[i]["FZ"]
        s = data[i]["SL"]
        gamma = np.sin(data[i]["IA"] * np.pi / 180)

        terms = _gx_terms(F_z, F_z0, gamma, s, x)

        residual_terms = (
            B_gx[i] - terms["B"],
            C_gx[i] - terms["C"],
            E_gx[i] - terms["E"],
            S_hgx[i] - terms["S_h"],
        )
        rows.extend(residual_terms)

    return np.vstack(rows).squeeze()


def gx(F_z, F_z0, SL, SA, IA, x):
    """Predict-time G_x weighting factor, given the 7 fitted R-parameters.
    SA/IA in degrees (converted internally), matching tm_lat/tm_long."""
    alpha = np.tan(np.asarray(SA, dtype=float) * np.pi / 180)
    gamma = np.sin(np.asarray(IA, dtype=float) * np.pi / 180)
    terms = _gx_terms(F_z, F_z0, gamma, SL, x)
    return _gx_curve(alpha, terms)


def tm_long_combined(F_z, F_z0, SL, SA, IA, lambda_mu_x, long_params, gx_params):
    """Fx prediction WITH the combined-slip correction applied:
    tm_long(...) * gx(...). Falls back to plain tm_long behavior when
    SA == 0 (gx() -> 1 there by construction, G_x0 normalizes the curve to
    pass through 1 at alpha=0)."""
    return tm_long(F_z, F_z0, SL, lambda_mu_x, long_params) * gx(F_z, F_z0, SL, SA, IA, gx_params)
