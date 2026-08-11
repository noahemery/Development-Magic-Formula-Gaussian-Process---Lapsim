"""Combined MF + GP prediction API.

predict_fy / predict_fx return (MF + GP mean, GP std) -- a corrected
prediction plus a calibrated uncertainty band. TireModel.p_params alone
(ignore .gp entirely) still gives the plain, portable Pacejka coefficients
for anything that only wants the standard Magic Formula.
"""

import numpy as np

from .pacejka import tm_lat, tm_long


def predict_fy(model, F_z, SA, IA, return_std: bool = True):
    F_z = np.atleast_1d(np.asarray(F_z, dtype=float))
    SA = np.atleast_1d(np.asarray(SA, dtype=float))
    IA = np.atleast_1d(np.asarray(IA, dtype=float))

    mf = tm_lat(F_z, model.F_z0, SA, IA, model.lambda_mu, model.p_params)

    if model.gp is None:
        return (mf, np.zeros_like(mf)) if return_std else mf

    gp_mean, gp_std = model.gp.predict(np.column_stack([F_z, SA, IA]), return_std=True)
    combined = mf + gp_mean
    return (combined, gp_std) if return_std else combined


def predict_fx(model, F_z, SL, IA, return_std: bool = True):
    F_z = np.atleast_1d(np.asarray(F_z, dtype=float))
    SL = np.atleast_1d(np.asarray(SL, dtype=float))
    IA = np.atleast_1d(np.asarray(IA, dtype=float))

    mf = tm_long(F_z, model.F_z0, SL, model.lambda_mu, model.p_params)

    if model.gp is None:
        return (mf, np.zeros_like(mf)) if return_std else mf

    gp_mean, gp_std = model.gp.predict(np.column_stack([F_z, SL, IA]), return_std=True)
    combined = mf + gp_mean
    return (combined, gp_std) if return_std else combined
