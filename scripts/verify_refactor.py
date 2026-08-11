"""One-off verification script (not part of the shipped package): replays the
ORIGINAL, bug-preserving second-pass formula through the new config-driven
pipeline's cases/bcde_params/F_z0, and checks that reproduces the captured
baseline exactly. This isolates "did the mechanical refactor stay correct"
from "the bug fixes changed the numbers" -- the latter is expected and
already reported separately.
"""
import pickle
import numpy as np
from scipy.optimize import least_squares

from magic.persistence import load_fit_results
from magic.config import LATERAL_SPECS, LONGITUDINAL_SPECS

_LSQ_TOL_KWARGS = dict(ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16, max_nfev=int(1e8), verbose=0)


def legacy_second_pass_y(data, F_z0, lambda_mu_y, BCDE_params, x):
    """Byte-for-byte reproduction of the ORIGINAL (buggy) second_pass_y."""
    B_y, C_y, D_y, E_y, S_hy, S_vy = BCDE_params.T
    x = np.asarray(x)[:20]
    (PDY1, PDY2, PDY3, PCY1, PKY1, PKY2, PKY3, PKY4, PKY5,
     PHY1, PHY2, PEY1, PEY2, PEY3, PEY4, PEY5, PVY1, PVY2, PVY3, PVY4) = x

    F_z = -data[0]["FZ"]
    alpha = np.tan(data[0]["SA"] * np.pi / 180)
    gamma = np.sin(data[0]["IA"] * np.pi / 180)
    df_z = (F_z - F_z0) / F_z0
    mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2)
    BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0)) / (1 + PKY3 * gamma ** 2))
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
    shit = (
        D_y[0] - (mu_y * F_z), C_y[0] - PCY1, B_y[0] - BCD_y / (mu_y * F_z * PCY1 + 1e-8),
        S_hy[0] - (PHY1 + PHY2 * df_z),
        E_y[0] - ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) - np.sign(alpha + (PHY1 + PHY2 * df_z)))),
        S_vy[0] - ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma),
    )
    residuals = np.vstack(shit)
    for i in range(len(data)):
        F_z = -data[i]["FZ"]
        alpha = np.tan(data[i]["SA"] * np.pi / 180)
        gamma = np.sin(data[i]["IA"] * np.pi / 180)
        df_z = (F_z - F_z0) / F_z0
        mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2)
        BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0)) / (1 + PKY3 * gamma ** 2))
        S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
        shit = (
            residuals, D_y[i] - (mu_y * F_z), C_y[i] - PCY1, B_y[i] - BCD_y / (mu_y * F_z * PCY1 + 1e-8),
            S_hy[i] - (PHY1 + PHY2 * df_z),
            E_y[i] - ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) - np.sign(alpha + (PHY1 + PHY2 * df_z)))),
            S_vy[i] - ((PVY1 + PVY2 * df_z) * F_z) + S_vy_gamma,
        )
        residuals = np.vstack(shit)
    return residuals.squeeze()


def legacy_second_pass_x(data, F_z0, lambda_mu_x, BCDE_params, x):
    """Byte-for-byte reproduction of the ORIGINAL (buggy) second_pass_x."""
    B_x, C_x, D_x, E_x, S_hx, S_vx = BCDE_params.T
    (PDX1, PDX2, PCX1, PKX1, PKX2, PKX3, PHX1, PHX2, PEX1, PEX2, PEX3, PEX4, PVX1, PVX2) = x

    F_z = -data[0]["FZ"]
    s = data[0]["SL"]
    df_z = (F_z - F_z0) / F_z0
    mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
    BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
    shit = (
        D_x[0] - (mu_x * F_z), C_x[0] - PCX1, B_x[0] - BCD_x / (mu_x * F_z * PCX1),
        S_hx[0] - (PHX1 + PHX2 * df_z),
        E_x[0] - ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + (PHX1 + PHX2 * df_z)))),
        S_vx[0] - (F_z * (PVX1 + PVX2 * df_z)),
    )
    residuals = np.vstack(shit)
    for i in range(len(data)):
        F_z = -data[i]["FZ"]
        s = data[i]["SL"]
        df_z = (F_z - F_z0) / F_z0
        mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
        BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
        shit = (
            residuals, D_x[i] - (mu_x * F_z), C_x[i] - PCX1, B_x[i] - BCD_x / (mu_x * F_z * PCX1 + 1e-8),
            S_hx[i] - (PHX1 + PHX2 * df_z),
            E_x[i] - ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + (PHX1 + PHX2 * df_z)))),
            S_vx[i] - (F_z * (PVX1 + PVX2 * df_z)),
        )
        residuals = np.vstack(shit)
    return residuals.squeeze()


def main():
    with open('baseline_original_output.pkl', 'rb') as f:
        baseline = pickle.load(f)
    results = load_fit_results('models/mf_fits.joblib')

    spec_lookup = {s.name: s for s in LATERAL_SPECS + LONGITUDINAL_SPECS}
    name_map = {
        'lat_160X75_R20_70': ('160X75_R20_70', legacy_second_pass_y),
        'lat_160X75_R20_80': ('160X75_R20_80', legacy_second_pass_y),
        'long_205X70_R20_70': ('205X70_R20_70', legacy_second_pass_x),
        'long_205X70_R20_80': ('205X70_R20_80', legacy_second_pass_x),
        'long_180X60_R20_60': ('180X60_R20_60', legacy_second_pass_x),
        'long_180X60_R20_70': ('180X60_R20_70', legacy_second_pass_x),
    }

    print(f"{'spec':22s} {'max |diff p_params|':>20s} {'max |diff F_z0|':>14s}")
    for base_name, (spec_name, legacy_fn) in name_map.items():
        base_arr = baseline[base_name]
        r = results[spec_name]
        spec = spec_lookup[spec_name]

        p_x0 = list(spec.p_x0_template)
        p_x0[spec.p_x0_c_index] = r.bcde_params[:, 1].mean()

        fit_func = lambda x: legacy_fn(r.cases, r.F_z0, spec.lambda_mu, r.bcde_params, x)
        lsq_kwargs = dict(jac='3-point', method='lm', **_LSQ_TOL_KWARGS)
        if spec.x_scale_jac:
            lsq_kwargs['x_scale'] = 'jac'
        result = least_squares(fit_func, p_x0, **lsq_kwargs)

        legacy_arr = np.hstack((result.x, r.F_z0))
        diff = np.abs(legacy_arr - base_arr)
        print(f"{spec_name:22s} {diff[:-1].max():20.6g} {diff[-1]:14.6g}")


if __name__ == "__main__":
    main()
