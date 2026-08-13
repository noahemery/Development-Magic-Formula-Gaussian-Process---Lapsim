"""Parameter-level uncertainty on the fitted Pacejka P-parameters.

Different from cross_validate.py: that measures uncertainty on the
PREDICTED FORCE (via the GP). This measures uncertainty on the P-PARAMETERS
themselves -- how confident are we in the value of PDY1 specifically, not
just how well the curve fits.

Only re-runs the second-pass fit (cheap, ~30s/spec, reuses cached
first-pass results) -- does NOT touch the GP or re-segment any data.

Run: python -u scripts/param_uncertainty.py
"""

import numpy as np

from magic.persistence import load_fit_results, save_fit_results
from magic.pipeline import refit_second_pass
from magic.pacejka import second_pass_y, second_pass_x, LATERAL_PARAM_NAMES, LONGITUDINAL_PARAM_NAMES
from magic.param_uncertainty import param_covariance

LATERAL_DROP_COLS = (20,)   # the documented spare, unused 21st lateral parameter
LONGITUDINAL_DROP_COLS = ()


def _spec_config(direction):
    if direction == "lateral":
        return second_pass_y, LATERAL_PARAM_NAMES, LATERAL_DROP_COLS
    return second_pass_x, LONGITUDINAL_PARAM_NAMES, LONGITUDINAL_DROP_COLS


def main():
    fit_results = load_fit_results("models/mf_fits.joblib")
    param_uncertainties = {}

    for i, (name, fit_result) in enumerate(fit_results.items(), 1):
        second_pass_fn, param_names, drop_cols = _spec_config(fit_result.direction)

        print(f"\n=== [{i}/{len(fit_results)}] {name} ({fit_result.direction}) ===", flush=True)
        result = refit_second_pass(fit_result, second_pass_fn)

        max_delta = float(np.max(np.abs(result.x - fit_result.p_params)))
        print(f"  max|delta p_params| vs cached mf_fits.joblib: {max_delta:.6g}  "
              f"(near 0 confirms x_scale='jac' changed nothing; a LARGE delta on an "
              f"already-known-fragile spec is itself expected -- see DECISIONS.md)", flush=True)

        pu = param_covariance(result, param_names, drop_cols=drop_cols, name=name, direction=fit_result.direction)
        param_uncertainties[name] = pu

        if pu.dropped_param_names:
            print(f"  dropped (unused, exact-zero Jacobian column): {pu.dropped_param_names}", flush=True)
        print(f"  cond(J^T J)={pu.cond_number:.3e}  rank={pu.rank}/{pu.n_params}  dof={pu.dof}", flush=True)

        print(f"  {'param':8s} {'value':>12s} {'std':>12s} {'rel std %':>10s} {'well-id?':>9s}", flush=True)
        for pname, val, std, rel_pct, ok in zip(pu.param_names, pu.p_params, pu.std, pu.relative_std_pct, pu.well_identified):
            std_str = f"{std:12.4g}" if np.isfinite(std) else f"{'nan':>12s}"
            rel_str = f"{rel_pct:10.1f}" if np.isfinite(rel_pct) else f"{'nan':>10s}"
            print(f"  {pname:8s} {val:12.4g} {std_str} {rel_str} {str(ok):>9s}", flush=True)

    save_fit_results(param_uncertainties, "models/param_uncertainty.joblib")
    print("\nsaved models/param_uncertainty.joblib", flush=True)


if __name__ == "__main__":
    main()
