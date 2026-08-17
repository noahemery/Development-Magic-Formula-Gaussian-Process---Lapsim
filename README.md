# Magic Formula + Gaussian Process tire model

Pacejka Magic Formula tire model fitting for lateral (Fy) and longitudinal (Fx)
force, built on Calspan TTC Round 9 data, with a hybrid residual Gaussian
Process layer on top of the lateral fits for uncertainty-aware correction.

See [DECISIONS.md](DECISIONS.md) for what changed from the original script,
why, and the actual verification results (including where the GP does and
doesn't help).

## Setup

```
pip install -r requirements.txt
```

Requires a local copy of the TTC data under `data/cornering_SI/` and
`data/straight_SI/` (not included in this repo -- see `.gitignore`).

## Usage

```
python fit_all.py                    # fit everything, GP on lateral specs only
python fit_all.py --mf-only          # Magic Formula fit only, no GP
python fit_all.py --gp-all-directions  # also fit GP for longitudinal (see DECISIONS.md for why this is off by default)
python fit_all.py --plot             # also write verification plots to plots/
```

Fitted models are saved to `models/mf_fits.joblib` (Magic Formula results,
including segmented cases) and `models/tire_models.joblib` (final MF+GP
models, used by `magic/predict.py`).

## Verification

```
python -m scripts.verify_refactor    # confirms the refactor reproduces the original script's fit
python -m scripts.cross_validate     # held-out MF-only vs MF+GP RMSE and uncertainty calibration, per tire spec
python -m scripts.plot_verification  # writes plots/ (requires models/ to already exist)
```

## `magic1.py`

**William Young's independent Magic Formula implementation, not part of this package.** Kept for reference/comparison -- see the header comment in the file itself and `DECISIONS.md` for the detailed diff against `magic.py`. `magic/combined_slip.py` is Noah's port of the combined-slip correction from this file, built on top of this repo's own longitudinal model rather than a copy of William's code.

## Layout

```
magic/
  segmentation.py   # raw time series -> clean per-condition test cases
  pacejka.py         # two-pass Magic Formula fit (per-segment BCDE, then global P-parameters)
  config.py           # per-tire-spec fitting configuration
  pipeline.py          # config-driven fitting, replaces the old hand-duplicated blocks
  gp_residual.py        # hybrid residual Gaussian Process
  persistence.py          # save/load fitted results
  predict.py                # combined MF + GP prediction API
fit_all.py                   # entry point
scripts/                      # cross-validation, plotting, refactor verification
```
