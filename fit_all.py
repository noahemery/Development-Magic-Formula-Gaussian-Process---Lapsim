"""Entry point: fit every tire spec's Magic Formula (Stage 1), then fit a
residual GP on top of each one (Stage 2), and save both.

Replaces the old top-to-bottom magic.py script. Run:

    python fit_all.py            # Stage 1 + Stage 2, saves models/*.joblib
    python fit_all.py --mf-only  # Stage 1 only
    python fit_all.py --plot     # also reproduce the verification plot
"""

import argparse
import sys
import time

import numpy as np

from magic.config import LATERAL_SPECS, LONGITUDINAL_SPECS
from magic.pipeline import fit_tire_spec_lateral, fit_tire_spec_longitudinal
from magic.persistence import TireModel, save_fit_results, save_tire_models


def fit_stage1():
    """Fit the Magic Formula for every configured tire spec. Returns
    {spec_name: TireFitResult}."""
    results = {}

    for spec in LATERAL_SPECS:
        print(f"\n=== fitting lateral spec: {spec.name} ===")
        results[spec.name] = fit_tire_spec_lateral(spec)

    for spec in LONGITUDINAL_SPECS:
        print(f"\n=== fitting longitudinal spec: {spec.name} ===")
        results[spec.name] = fit_tire_spec_longitudinal(spec)

    return results


def fit_stage2(fit_results: dict, directions=("lateral",)):
    """Fit a residual GP on top of each Stage 1 result. Returns
    {spec_name: TireModel}.

    directions: which fit_result.direction values actually get a GP fit.
    Defaults to lateral only -- held-out GroupKFold cross-validation (see
    DECISIONS.md) showed the GP gives a large, genuine improvement on the
    lateral specs (76-88% RMSE reduction) but ~0 or slightly negative
    improvement on all 4 longitudinal specs, which lines up with the
    longitudinal MF fit already being sound (not much residual left for a
    GP to correct). Specs outside `directions` get a plain MF-only
    TireModel rather than paying for a GP fit that measurably doesn't help.
    Pass directions=("lateral", "longitudinal") to fit both anyway.
    """
    from magic.gp_residual import ResidualGP, build_gp_dataset

    models = {}
    for i, (name, fit_result) in enumerate(fit_results.items(), 1):
        print(f"\n=== [{i}/{len(fit_results)}] {name} ({fit_result.direction}) ===", flush=True)

        if fit_result.direction not in directions:
            print(f"  direction not in {directions}; saving MF-only model.", flush=True)
            models[name] = TireModel.from_fit_result(fit_result)
            continue

        X, residual, segment_id = build_gp_dataset(fit_result.cases, fit_result, fit_result.direction)
        print(f"  pooled dataset: {X.shape[0]} points from {len(np.unique(segment_id))} segments", flush=True)

        if len(np.unique(segment_id)) < 3:
            print(f"  WARNING: only {len(np.unique(segment_id))} segment(s) available; "
                  f"skipping GP, saving MF-only model.", flush=True)
            models[name] = TireModel.from_fit_result(fit_result)
            continue

        t_spec = time.time()
        gp = ResidualGP().fit(X, residual)
        print(f"  done in {time.time() - t_spec:.1f}s, length_scales(F_z, slip, IA) = {gp.fitted_length_scales_}", flush=True)
        model = TireModel.from_fit_result(fit_result)
        model.gp = gp
        models[name] = model

    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mf-only", action="store_true", help="Stage 1 only, skip the GP layer")
    parser.add_argument("--plot", action="store_true", help="reproduce the verification plot after fitting")
    parser.add_argument("--gp-all-directions", action="store_true",
                         help="also fit the GP for longitudinal specs (off by default -- see fit_stage2 docstring)")
    args = parser.parse_args()

    t0 = time.time()
    fit_results = fit_stage1()
    save_fit_results(fit_results, "models/mf_fits.joblib")
    print(f"\nStage 1 done in {time.time() - t0:.1f}s, saved to models/mf_fits.joblib")

    if args.mf_only:
        # Still save a Stage-1-only TireModel set so downstream code has a consistent artifact to load.
        models = {name: TireModel.from_fit_result(r) for name, r in fit_results.items()}
        save_tire_models(models, "models/tire_models.joblib")
        return

    directions = ("lateral", "longitudinal") if args.gp_all_directions else ("lateral",)
    t1 = time.time()
    models = fit_stage2(fit_results, directions=directions)
    save_tire_models(models, "models/tire_models.joblib")
    print(f"\nStage 2 done in {time.time() - t1:.1f}s, saved to models/tire_models.joblib")

    if args.plot:
        from scripts.plot_verification import plot_all
        plot_all(models, fit_results)


if __name__ == "__main__":
    main()
