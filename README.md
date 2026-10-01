# Freight load rate prediction

Predicts the posted rate of a truck load from distance, weight, origin/destination city, equipment type and date.

* **Submission file:** `outputs/validation_predictions.csv` (`load_id,predicted_rate`, 12,000 rows, template order)
* **December chart:** `outputs/december_chart_inputs.csv` (completed template) and `outputs/scorer_results/candidate_december.png` (produced by the provided `score.py`)

## Final model (F)

`F = 0.5 * Bmi + 0.5 * C`

| Part | Model | Features |
|---|---|---|
| **Bmi** | LightGBM, L1 loss on log(rate) | distance, log distance, weight (absolute value, equipment-median fill + missing flag), equipment, day of week, pickup/delivery **coordinates** (no city names), days since 2025-01-01, **daily mean `market_index`** (+ missing flag) |
| **C** | Ridge on log(rate) + smearing correction | distance terms, equipment, day of week, weight, pickup/delivery city one-hots; no time feature |

The model, its 50/50 weights and all parameters were frozen on January-September development windows **before** October was scored.

## Setup

Python 3.12 was used.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then place the four input files in `data/` (see `data/README.md`; they are not part of this repository).

## Reproduce the submission

Run from the repository root:

```bash
python 03_final.py --step checks     # implementation checks (December path, fill rule, template)
python 03_final.py --step october    # final test: train Jan-Sep, score October
python 03_final.py --step predict    # train on all 48,000 loads, write the two output files to outputs/
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_inputs.csv \
                --output-dir outputs/scorer_results
```

`score.py` is the provided script, unchanged. It validates the two files and draws the chart; it does not compute accuracy.

Re-running `--step october` reproduces the same final test. It is not a new independent test, and the model is not changed based on it.

## Reproduce the development experiments (optional)

These produce the tables in `results/` that the report cites. Order matters where noted.

```bash
python 01_audit_and_baselines.py                      # audit.json, baselines_summary.csv, month_effects.csv
python 02_tree_experiments.py --part main             # lgbm_main.csv      (1-month folds: objectives, features)
python 02_tree_experiments.py --part cat              # catboost.csv       (CatBoost, no time; superseded by round3)
python 02_tree_experiments.py --part city             # city_holdout.csv   (first hidden-city test, 1-month folds)
python 02_tree_experiments.py --part horizon          # horizon2.csv       (2-month windows, single models)
python 02_tree_experiments.py --part blend            # blend_summary / blend_per_window / blend_bootstrap
python 02_tree_experiments.py --part round3           # round3_summary / round3_per_window
python 02_tree_experiments.py --part city2            # hidden cities on candidates (writes results/city2_parts/; optional --windows)
python 02_tree_experiments.py --part city2_summary    # city2_summary.csv            (needs city2)
python 02_tree_experiments.py --part roundF           # roundF_summary / roundF_per_window
python 02_tree_experiments.py --part cityF            # cityF_summary.csv            (needs city2)
python 02_tree_experiments.py --part pathtest         # pathtest.csv
python 03_final.py --step repro                       # frozen F == saved development predictions (needs roundF)
```

## Validation design

* **Time-based split on whole days.** Training always precedes evaluation, and no day is split between them.
* **October held out.** October prices (4,853 loads) were not used to train or select any model. Development used January-September (43,147 loads).
* **Two-month windows** (train up to month m-1, evaluate months m and m+1; m = May…August) mirror the real task, where November-December is up to two months after the October cut-off. The selection criterion was pooled MAE on these windows, on all evaluation rows.
* **Hidden-city test.** 12% of the loads to price involve 8 cities never seen in training. We hid 6 random cities (4 draws) from training and scored only loads touching them, with a control that removes the same number of other training rows.
* **Leakage rules.** Every imputation, scaling, smearing factor, outlier rule and daily index for training is computed from the training part only.

## Key results (MAE in dollars, all evaluation rows)

| | 2-month windows | Month 2 | Hidden cities | October (final test) |
|---|---|---|---|---|
| **F (selected)** | 120.17 | 126.83 | 124.23 | 106.17 |
| B+C (reference) | 125.55 | 135.00 | 129.86 | 105.26 |
| Ridge baseline (log, 1-month folds) | 135.80 | | | |

F's development gain comes mainly from one window (training ending in June, the price peak); in the other windows F and B+C are close. In October, F did not improve on the reference (MAE about 0.90 dollars higher); the choice was kept because it was made before the test. This is not evidence of statistical equivalence.

## Assumptions and limitations

* **Daily `market_index`.** The daily mean is computed from the features of the batch being priced (no prices). F therefore uses indicators attached to the November-December loads; it is not a forecast from data up to October alone.
* **December chart input.** The chart template has no `market_index` or coordinates. Coordinates come from a city-to-coordinate table (one coordinate per city in the data). The daily index is taken from the December loads in `validation.csv`, which assumes the index is shared by loads on the same day. The data supports this (date explains about 98% of its variance in January-September), but its business meaning is not proven.
* **Origin of the market_index idea.** It came from development results and is documented as such.
* **Horizon of the final test.** The October test covers a one-month horizon only.
* **Weight handling.** Absolute value versus treating negative weights as missing was compared only with the Ridge baseline (135.80 vs 135.88).
* **Hyper-parameters.** They were fixed and not tuned.
* **Outliers.** About 1.4% of loads have rates around 0.2-0.5x or 2-6x the typical level. They were kept, and a robust L1 loss was used instead; RMSE is dominated by them.
* **The chart.** It shows model predictions, not observed prices. Its y-axis does not start at zero; the total December variation is about 17 dollars (2.06% of the lowest prediction), mostly a weekly pattern.

## Reproducibility notes

* All models run single-threaded (`n_jobs=1`) with fixed seeds.
* Different library versions or thread counts may produce small numeric differences.
* Per-load files containing real prices from the provided data (`results/*_predictions.csv`, `results/city2_parts/`) are excluded from the repository by `.gitignore`.

**Verification performed.** From a clean copy of this repository in a fresh environment built from `requirements.txt` (Python 3.12.3, pandas 2.3.3), every command in this README was run.

* All commands completed with no warnings.
* `validation_predictions.csv`, the completed December file and the October predictions were **byte-identical** to the ones first produced in a pandas 3.0.2 environment.
* Every result table in `results/` was identical. `final_checks.txt` differs only because the experiment-reproduction check now has its own step (`repro_check.txt`).
* `city2_summary.csv` columns were renamed (`control_minus_seen`, `unseen_minus_control`) to describe the control condition accurately; the values are unchanged.

**Approximate run times** on one CPU core:

* Submission steps: about 1 minute in total.
* Most experiment parts: under 2 minutes each.
* `city` and `city2`: about 3-4 minutes each. `city2` can be split, e.g. `--windows 2025-05 2025-06`, then `--windows 2025-07 2025-08`.

## Changes after the model was frozen

These are implementation-only changes. Both were verified to leave every prediction identical, and no model choice changed.

1. Ridge city one-hot columns are now built in a single `pd.concat` (same columns, same order) instead of column-by-column insertion. This removed pandas `PerformanceWarning` messages under pandas 2.x.
2. The `train_end` label in the blend output is computed as `(period - 1).end_time` instead of subtracting a `Timedelta`. This removed a NumPy `DeprecationWarning`; the label values are the same.

## Repository layout

```
01_audit_and_baselines.py   data audit, whole-day expanding folds, baselines (Ridge, rate-per-mile)
02_tree_experiments.py      LightGBM/CatBoost experiments, 2-month windows, blends, hidden-city tests, model F parts
03_final.py                 frozen model F: checks, October test, final training, output files
score.py                    provided validation/chart script (unchanged)
data/                       input files go here (not included)
results/                    aggregated result tables cited in the report
outputs/                    submission file, completed December template, chart
```
