"""GroupKFold cross-validation: MF-only vs MF+GP RMSE, plus uncertainty
calibration, per tire spec.

Validation is grouped by segment id, not a random row split -- adjacent raw
points within one sweep are nearly identical, so a random split would leak
and report a misleadingly good held-out error. See magic/gp_residual.py.

Uses a CHEAPER GP config than the production fit (fewer restarts, fewer
pooled points, fewer folds) -- this is deliberate: the production fit
(n_restarts_optimizer=5, ~3000 points) took 8-22 minutes PER SPEC, so a
naive 5-fold CV refitting that same config 5x per spec across 6 specs would
be another several hours. This script's job is a directionally-reliable
held-out RMSE/calibration signal, not the best possible per-fold model, so
it trades some precision for something that actually finishes in minutes.

Run: python -u scripts/cross_validate.py
"""

import time

import numpy as np
from sklearn.model_selection import GroupKFold

from magic.persistence import load_fit_results
from magic.gp_residual import ResidualGP, ResidualGPConfig, build_gp_dataset

CV_CONFIG = ResidualGPConfig(n_restarts_optimizer=1)   # length_scale_bounds floor inherited from the (now-fixed) default
CV_MAX_TOTAL_POINTS = 1500
N_SPLITS = 3


def cross_validate_spec(name, fit_result, n_splits=N_SPLITS):
    X, residual, segment_id = build_gp_dataset(fit_result.cases, fit_result, fit_result.direction,
                                                max_total_points=CV_MAX_TOTAL_POINTS)
    n_groups = len(np.unique(segment_id))

    if n_groups < 3:
        return None  # too few segments to hold any out meaningfully

    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    mf_only_sq_err, mf_gp_sq_err, within_band = [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, residual, groups=segment_id), 1):
        t0 = time.time()
        gp = ResidualGP(CV_CONFIG).fit(X[train_idx], residual[train_idx])
        gp_mean, gp_std = gp.predict(X[val_idx], return_std=True)
        print(f"  {name} fold {fold}/{n_splits}: {time.time() - t0:.1f}s", flush=True)

        mf_only_sq_err.append(residual[val_idx] ** 2)                 # MF-only error IS the held-out residual itself
        mf_gp_sq_err.append((residual[val_idx] - gp_mean) ** 2)
        within_band.append(np.abs(residual[val_idx] - gp_mean) <= 1.96 * gp_std)

    return dict(
        mf_only_rmse=np.sqrt(np.concatenate(mf_only_sq_err).mean()),
        mf_gp_rmse=np.sqrt(np.concatenate(mf_gp_sq_err).mean()),
        calibration=np.concatenate(within_band).mean(),
        n_groups=n_groups,
        n_splits=n_splits,
    )


def main():
    fit_results = load_fit_results("models/mf_fits.joblib")

    header = f"{'spec':16s} {'n_segs':>7s} {'MF-only RMSE':>13s} {'MF+GP RMSE':>12s} {'improvement':>12s} {'95% band cov':>13s}"
    results = {}

    for i, (name, fit_result) in enumerate(fit_results.items(), 1):
        print(f"\n=== [{i}/{len(fit_results)}] cross-validating: {name} ({fit_result.direction}) ===", flush=True)
        results[name] = cross_validate_spec(name, fit_result)

    print()
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        if result is None:
            print(f"{name:16s} {'--':>7s}  skipped -- too few segments for GroupKFold")
            continue
        improvement_pct = 100 * (1 - result["mf_gp_rmse"] / result["mf_only_rmse"])
        print(
            f"{name:16s} {result['n_groups']:7d} {result['mf_only_rmse']:13.3f} "
            f"{result['mf_gp_rmse']:12.3f} {improvement_pct:11.1f}% {result['calibration'] * 100:12.1f}%"
        )


if __name__ == "__main__":
    main()
