"""Reproduces the plot that was commented out at the bottom of the original
magic.py, for each tire spec: raw scatter, MF-only curve, MF+GP corrected
curve, and the +/-2*std uncertainty band.

The original commented-out snippet plotted `data["FY"]` and `F_z0` directly
without the sign flips the fit itself uses (`F_y = -data["FY"]`,
`F_z = -data["FZ"]`) -- copied verbatim it would show a mirrored curve
against correctly-signed data. Fixed here.

Run: python -m scripts.plot_verification
(or via `python fit_all.py --plot` right after fitting)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from magic.pacejka import tm_lat, tm_long
from magic.predict import predict_fy, predict_fx


def _pick_representative_case(cases, slip_key):
    """Longest sweep = most points = cleanest curve to look at."""
    return max(cases, key=lambda c: len(np.atleast_1d(np.asarray(c[slip_key])).ravel()))


def _plot_spec(model, fit_result, out_dir: Path):
    direction = model.direction
    slip_key = "SA" if direction == "lateral" else "SL"
    force_key = "FY" if direction == "lateral" else "FX"
    measured_sign = -1.0 if direction == "lateral" else 1.0

    case = _pick_representative_case(fit_result.cases, slip_key)
    F_z = np.atleast_1d(np.asarray(-case["FZ"])).ravel()
    slip = np.atleast_1d(np.asarray(case[slip_key])).ravel()
    IA = np.atleast_1d(np.asarray(case["IA"])).ravel()
    measured = measured_sign * np.atleast_1d(np.asarray(case[force_key])).ravel()

    order = np.argsort(slip)
    F_z, slip, IA, measured = F_z[order], slip[order], IA[order], measured[order]

    if direction == "lateral":
        mf_only = tm_lat(F_z, model.F_z0, slip, IA, model.lambda_mu, model.p_params)
        combined, gp_std = predict_fy(model, F_z, slip, IA, return_std=True)
    else:
        mf_only = tm_long(F_z, model.F_z0, slip, model.lambda_mu, model.p_params)
        combined, gp_std = predict_fx(model, F_z, slip, IA, return_std=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(slip, measured, s=6, color="black", alpha=0.35, label="raw data")
    ax.plot(slip, mf_only, color="tab:blue", lw=2, label="MF only")
    if model.gp is not None:
        ax.plot(slip, combined, color="tab:red", lw=2, label="MF + GP")
        ax.fill_between(slip, combined - 2 * gp_std, combined + 2 * gp_std,
                         color="tab:red", alpha=0.2, label=r"$\pm 2\sigma$")

    xlabel = "Slip angle SA [deg]" if direction == "lateral" else "Slip ratio SL [-]"
    ylabel = "F$_y$ [N]" if direction == "lateral" else "F$_x$ [N]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model.name} ({direction}) -- F$_z$ approx {F_z.mean():.0f} N, representative sweep")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model.name}_{direction}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_all(models: dict, fit_results: dict, out_dir="plots"):
    out_dir = Path(out_dir)
    paths = []
    for name, model in models.items():
        path = _plot_spec(model, fit_results[name], out_dir)
        paths.append(path)
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    from magic.persistence import load_fit_results, load_tire_models

    fit_results = load_fit_results("models/mf_fits.joblib")
    models = load_tire_models("models/tire_models.joblib")
    plot_all(models, fit_results)
