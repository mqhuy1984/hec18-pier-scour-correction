# Code and data for "Correcting HEC‑18 pier‑scour estimates with field data"

This repository reproduces every number, table and figure of the manuscript (under review) and its supplement (`BaiA/Supplement_BaiA.md`).

## Data

- **Source:** USGS Pier‑Scour Database PSDb‑2014 (Benedict & Caldwell, 2014, Data Series 845), file `ds845_psdb-2014_version1.0.xlsx`, public domain.
- **Cleaning:** `PSDb_clean/clean_psdb.py` converts the field and laboratory tables to SI units and adds eight severe and six warning/information quality flags. Nothing is deleted; flagged records are kept with their flags. Outputs: `PSDb_clean/psdb2014_field_clean.csv`, `psdb2014_lab_clean.csv`, `psdb2014_clean.xlsx` (with a data dictionary).
- **Evaluation set:** 1,253 field records from 370 bridge site groups (records without severe flags for which every baseline equation can be computed).

## Requirements

Python 3.13 and the packages in `requirements.txt`:

```
pip install -r BaiA/requirements.txt
```

## Reproducing the results

From the repository root, on Windows PowerShell:

```
powershell -ExecutionPolicy Bypass -File BaiA/run_all.ps1          # full run (about 12 h on 15 cores)
powershell -ExecutionPolicy Bypass -File BaiA/run_all.ps1 -Quick   # smoke test, minutes
```

The steps can also be run one by one (same order):

| Step | Script | Main outputs (`BaiA/outputs/`) | Manuscript |
|---|---|---|---|
| 0 | `PSDb_clean/clean_psdb.py` | cleaned and flagged tables | Section 2, Table 1 |
| 1 | `week1_baselines.py` | `week1_equations_overall.csv`, `week1_eval_set_with_predictions.csv` | Section 4.1 |
| 2 | `week2_analysis.py` | `week2_*.csv` | Section 4.1, Fig. 2 |
| 3 | `week3_xgboost.py --reps 10 --trials 60 --jobs 15` | `week3_metrics_summary.csv`, `week3_rq4_split_gap.csv`, `week3_oof_predictions.csv` | Sections 4.2–4.3, Fig. 3 |
| 4 | `week4_conformal_extrap.py --reps 10 --trials 60 --jobs 15` | `week4_conformal_cv.csv`, `week4_conformal_by_group.csv`, `week4_extrapolation.csv` | Sections 4.4–4.5, Figs. 4–5 |
| 5 | `week5_formula_sensitivity.py` | `week5_formula_coefficients.csv`, `week5_formula_conformal_factors.csv`, `week5_extrap_bootstrap.csv`, `week5_lab_to_field.csv` | Sections 4.5–4.8, Fig. 6 |
| S2 | `formula_variant_check.py` | `formula_variant_check.csv` | Supplement S2 |
| S4 | `ciria_check.py` | `ciria_equations.csv`, `ciria_by_covariate.csv`, `ciria_correction_cv.csv` | Table 2, Supplement S4 |
| S8 | `revision2_analyses.py` | `rev2_variants.csv`, `rev2_coefficients.csv`, `rev2_k_multipliers.csv`, `rev2_*.csv` | final Eq. 2, Tables 3–5, Supplement S8 |
| S8 | `revision_analyses.py` | `rev_eq2_variants.csv`, `rev_eq2_coefficients.csv`, `rev_k_multipliers.csv`, `rev_strat_timing.csv`, `rev_*.csv` | Eq. 2, Tables 4–6, Supplement S7–S8 |
| — | `make_figures_en.py` | `BaiA/figures_en/Fig1–Fig6` | Figures |
| — | `make_supplement.py` | `BaiA/Supplement_BaiA.md` | Supplement |

Notes:
- Set `OMP_NUM_THREADS=1` (done by `run_all.ps1`) when running steps 3–4 in parallel, to avoid oversubscription.
- Random seeds are fixed in the scripts (outer folds `1000 + repetition`; inner folds and Optuna samplers derived from repetition and fold), so a full run should reproduce the reported numbers; small differences can arise from library versions.
- `-Quick` writes to the same output folders and therefore overwrites the full results; it refuses to run when full results exist unless `-Overwrite` is given.
- `equations.py` contains all baseline equations (HEC‑18 5th ed., Sheppard–Melville/FDOT, Froehlich, CIRIA C742); implementation details are given in Supplement S1.

## Using the closed‑form correction

`apply_P1_P7.py` applies the closed‑form correction, its conformal multipliers and the domain‑of‑applicability checks to user‑supplied piers. Fill in a copy of `P1_P7_input_template.csv` and run:

```
python BaiA/apply_P1_P7.py my_piers.csv
```

## Licence

- **Code** (`*.py`, `run_all.ps1`): MIT licence, see `LICENSE`.
- **Derived data** (cleaned and flagged PSDb‑2014 tables, `BaiA/outputs/`, `BaiA/figures_en/`): CC BY 4.0, see `LICENSE-DATA.md`.
- **Source data** (USGS PSDb‑2014): U.S. Government work, public domain. Please cite Benedict & Caldwell (2014).
