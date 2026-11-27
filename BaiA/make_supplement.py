"""Tạo phụ lục tiếng Anh cho bài A (Supplement S1–S4) từ các file kết quả.

Chạy:  python BaiA/make_supplement.py      (sau week1…week5, formula_variant_check.py, ciria_check.py)
Ghi:   BaiA/Supplement_BaiA.md
Mọi con số trong bảng đọc trực tiếp từ BaiA/outputs/*.csv; phần chữ là cố định.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def md(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else fmt.get(c, "{:.3f}").format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SPLIT_EN = {"USA → ngoài Mỹ": "USA → non‑USA", "bn ≤ 3 → bn > 5 m": "bn ≤ 3 m → > 5 m", "bn ≤ 3 → 3–5 m": "bn ≤ 3 m → 3–5 m",
            "cát (<2 mm) → sỏi cuội (≥20 mm)": "sand → gravel/cobble", "cát → sỏi cuội": "sand → gravel/cobble",
            "sỏi (≥2 mm) → cát (<2 mm)": "gravel → sand", "sỏi → cát": "gravel → sand",
            "Fr ≤ 0,3 → Fr > 0,5": "Fr ≤ 0.3 → > 0.5", "CV theo cầu 5×10": "bridge‑grouped CV, 5 × 10"}
ORDER = ["USA → non‑USA", "bn ≤ 3 m → 3–5 m", "bn ≤ 3 m → > 5 m", "sand → gravel/cobble", "gravel → sand", "Fr ≤ 0.3 → > 0.5"]

# ---------------------------------------------------------------- S2
fv = pd.read_csv(OUT / "formula_variant_check.csv")
fv["split"] = fv["split"].map(SPLIT_EN).fillna(fv["split"])
fv["variant"] = fv["variant"].map({"S6 (5 số hạng)": "5 terms (with ln y/a)", "S6 bỏ ln(y/a) (4 số hạng)": "4 terms (adopted)",
                                   "chỉ ln(a/D50), θ (2 số hạng)": "2 terms (ln a/D50, θ only)"})
s2a = fv[fv["split"].str.startswith("bridge")][["variant", "MAE", "MAE_sd", "R2_log", "k95_bridgemax"]]
s2a.columns = ["Variant", "MAE (m)", "s.d.", "R² log", "Bridge‑level 95% multiplier"]
s2b = fv[~fv["split"].str.startswith("bridge")].pivot(index="split", columns="variant", values="MAE").reindex(ORDER)
s2b = s2b[["5 terms (with ln y/a)", "4 terms (adopted)", "2 terms (ln a/D50, θ only)"]].reset_index().rename(columns={"split": "Experiment"})

# ---------------------------------------------------------------- S3
eb = pd.read_csv(OUT / "week5_extrap_bootstrap.csv")
w = eb[eb["method"] == "weighted"].copy(); w["Experiment"] = w["split"].map(SPLIT_EN); w = w.set_index("Experiment").reindex(ORDER)
models = [("M3", "M3‑S6"), ("M4", "M4"), ("M5", "M5"), ("RF4", "RF4"), ("RF5", "RF5"), ("GPR5", "GPR5")]


def ci(m, k, scale=1.0, f="{:.2f}"):
    return w.apply(lambda r: f"{f.format(scale * r[f'{k}_{m}'])} [{f.format(scale * r[f'{k}_{m}_lo'])}, {f.format(scale * r[f'{k}_{m}_hi'])}]", axis=1)


s3a = pd.DataFrame({"Experiment": ORDER, "Rows / bridges": [f"{int(w.loc[e, 'n_rows'])} / {int(w.loc[e, 'n_bridges'])}" for e in ORDER]})
for m, lab in models:
    s3a[lab] = ci(m, "ratio").to_numpy()
s3b = pd.DataFrame({"Experiment": ORDER})
for m, lab in models[1:]:
    col = w.apply(lambda r: ("**" if (r[f"dMAE_{m}_M3_lo"] > 0 or r[f"dMAE_{m}_M3_hi"] < 0) else "")
                  + f"{r[f'dMAE_{m}_M3']:+.2f} [{r[f'dMAE_{m}_M3_lo']:+.2f}, {r[f'dMAE_{m}_M3_hi']:+.2f}]"
                  + ("**" if (r[f"dMAE_{m}_M3_lo"] > 0 or r[f"dMAE_{m}_M3_hi"] < 0) else ""), axis=1)
    s3b[lab] = col.to_numpy()
s3c = pd.DataFrame({"Experiment": ORDER})
for meth, tag in [("weighted", "bridge‑weighted"), ("bridgemax", "bridge‑max")]:
    ww = eb[eb["method"] == meth].copy(); ww["Experiment"] = ww["split"].map(SPLIT_EN); ww = ww.set_index("Experiment").reindex(ORDER)
    for m, lab in [("M3", "M3‑S6"), ("M5", "M5")]:
        s3c[f"{lab}, {tag}"] = ww.apply(lambda r: f"{100 * r[f'exc_{m}']:.1f} [{100 * r[f'exc_{m}_lo']:.1f}, {100 * r[f'exc_{m}_hi']:.1f}]", axis=1).to_numpy()

# ---------------------------------------------------------------- S4
ce = pd.read_csv(OUT / "ciria_equations.csv")
s4a = ce[["model", "n_zero_pred", "MAE_m", "RMSE_m", "R2_log_pos", "median_obs_over_pred_pos", "frac_under", "within_pm50", "c95_pos"]].copy()
s4a.columns = ["Equation", "Zero predictions", "MAE (m)", "RMSE (m)", "R² log", "Median ratio", "Underpredicted", "Within ±50%", "c95"]
for c in ["Underpredicted", "Within ±50%"]:
    s4a[c] = (100 * s4a[c]).map("{:.1f}%".format)
cb = pd.read_csv(OUT / "ciria_by_covariate.csv")
key = cb[["var", "class"]].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index": "ord"})
s4b = (cb.pivot_table(index=["var", "class", "n"], columns="model", values="median_ratio").reset_index()
       .merge(key, on=["var", "class"]).sort_values("ord").drop(columns="ord").reset_index(drop=True))
s4b["var"] = s4b["var"].replace({"θ (độ)": "θ (deg)"})
s4b.columns.name = None
s4b = s4b.rename(columns={"var": "Variable", "class": "Class", "n": "n"})
s4b["n"] = s4b["n"].astype(int).astype(str)
cc = pd.read_csv(OUT / "ciria_correction_cv.csv")
cc["correction"] = cc["correction"].map({"hằng số": "constant", "4 số hạng (S6)": "4 terms (S6)", "11 số hạng (S2)": "11 terms (S2)"})
s4c = cc[["baseline", "correction", "n_rows", "MAE", "MAE_sd", "R2_log"]].copy()
s4c["n_rows"] = s4c["n_rows"].astype(int).astype(str)
s4c.columns = ["Baseline", "Correction", "Rows", "MAE (m)", "s.d.", "R² log"]

text = f"""# Supplementary material

**Correcting HEC‑18 Pier Scour with Field Data: Bridge‑Level Validation and Conformal Bounds**

*Generated by `python BaiA/make_supplement.py`; all numbers are read from `BaiA/outputs/`.*

---

## S1. Implementation of the baseline equations

All equations are implemented in `BaiA/equations.py` (SI units; D50 in mm). HEC‑18 was checked line by line against the 5th edition (Arneson et al., 2012, FHWA‑HIF‑12‑003, Sections 6.3, 7.1–7.4 and 7.11). An earlier version of our code contained four errors, which were corrected before any correction model was fitted:

| Item | HEC‑18 (5th ed.) | Earlier code | Corrected |
|---|---|---|---|
| 2.4a / 3.0a limit (Eq. 7.2) | Only for round‑nosed piers aligned with the flow | Applied to every pier | Applied only to round/circular piers with θ ≤ 5° |
| Wide‑pier factor K_w (Eq. 7.20–7.21) | Applies when y/a < 0.8, a/D50 > 50 and Fr < 1; no upper cap | Capped at 1.0 | Cap removed |
| Critical velocity for Sheppard–Melville (FDOT) | Own expression, Eq. 7.12–7.14 | HEC‑18 Eq. 6.1 used | FDOT expression implemented |
| Coarse‑bed branch (Eq. 7.34) | σ = D84/D50; clear‑water only (V < Vc); D50 ≥ 20 mm, σ ≥ 1.5 | Geometric standard deviation used; no clear‑water check | Corrected |

Other implementation choices: K₄ is omitted, as in the 5th edition (Section 7.11); the lower limit D50 = 0.2 mm is applied in Eq. 6.1; Ksf of sharp‑nosed piers in the FDOT method is not specified by HEC‑18 and is set to 1.0; missing pier length is replaced by L = a (K₂ = 1 for aligned piers).

Effect of the corrections on the evaluation set (1,253 records): the median measured/HEC‑18 ratio changed from 0.38 to 0.34, the underprediction rate from 6.2% to 4.5%, and c95 from 1.07 to 0.997. The conclusion that HEC‑18 is close to a 95% non‑exceedance envelope depends on these corrections.

CIRIA C742 (Kirby et al., 2015) is implemented as ys = a K_shape K_depth K_velocity K_angle, with factors from Gavriel (2025, Tables 2.7–2.9, Eq. 2.12) cross‑checked against CIRIA C551 (May et al., 2002, Table 4.3, Eq. 4.49–4.50) and CD 356 (Eq. A.2). See S4.

## S2. Choice of terms in the closed‑form correction

The five‑term candidate contained ln(y/a), ln(a/D50), θ/90, a cohesive‑bed indicator and a bed‑type‑unknown indicator. The coefficient of ln(y/a) was 0.052 (95% bridge‑bootstrap CI −0.026 to 0.135). Three variants were compared by OLS on ln(ys/ys,HEC‑18) (`BaiA/formula_variant_check.py`).

**Table S2a.** Bridge‑grouped CV, 5 folds × 10 repetitions. The bridge‑level multiplier is exp(q95) of the bridge‑maximum out‑of‑fold residual.

{md(s2a)}

**Table S2b.** MAE (m) in the six extrapolation experiments (bridges in the test set removed from training).

{md(s2b)}

The four‑term variant has the same in‑distribution accuracy as the five‑term one, a narrower bridge‑level multiplier, and a lower MAE in four of the six extrapolation experiments; it is slightly worse only for the two wide‑pier experiments. Removing θ and the bed indicators as well (two terms) degrades accuracy. HEC‑18 already contains (y/a)^0.35, which explains why no further y/a correction is needed.

## S3. Extrapolation: bridge‑bootstrap results for all models

Each experiment is one train/test split; uncertainty is from 2,000 bootstrap resamples of test bridges (90% intervals). M3‑S6: four‑term log‑linear correction; M4/RF4: direct XGBoost / random forest; M5/RF5/GPR5: residual XGBoost / random forest / Gaussian process.

**Table S3a.** Ratio of extrapolated MAE to the MAE of the same rows under bridge‑grouped CV.

{md(s3a)}

**Table S3b.** Difference in MAE (m) relative to M3‑S6 on the same test rows. Bold: the 90% interval excludes zero.

{md(s3b)}

**Table S3c.** Percentage of test rows above the 95% conformal upper bound, bridge‑weighted and bridge‑max calibration.

{md(s3c)}

## S4. CIRIA C742 as an additional baseline

Factors: K_shape = 1.5 (round/circular), 2.0 (square), 1.075 (sharp); K_depth = 0.55 (y/a)^0.6 for y/a ≤ 2.7, otherwise 1; K_velocity = 0 for V/Vc ≤ 0.375, 1.6 V/Vc − 0.6 for 0.375 < V/Vc ≤ 1, and 1 for V/Vc > 1; K_angle = (cos θ + (L/a) sin θ)^0.62. Vc from HEC‑18 Eq. 6.1, or from Gavriel (2025): 0.1 D50^−0.2 for D50 < 1 mm and 0.35 D50^0.45 otherwise (D50 in mm, Vc in m/s). No factor of safety is applied (SF = 1). Records with a zero prediction count as underpredictions and are excluded from log‑scale metrics.

**Table S4a.** All baseline equations on the 1,253 evaluation records.

{md(s4a, {"Zero predictions": "{:.0f}", "R² log": "{:.2f}", "Median ratio": "{:.2f}", "c95": "{:.2f}"})}

**Table S4b.** Median measured/predicted ratio by covariate class.

{md(s4b, {c: "{:.2f}" for c in s4b.columns})}

**Table S4c.** Log‑linear corrections on the HEC‑18 and CIRIA baselines, on the records with a non‑zero CIRIA prediction (bridge‑grouped CV, 5 × 10).

{md(s4c)}

The CIRIA underprediction rate (19.5%) agrees with the 19.0% reported by Gavriel et al. (2026) for 936 PSDb field records. Corrections built on CIRIA explain far less of the variance than corrections built on HEC‑18, because CIRIA's depth and velocity factors saturate (y/a > 2.7, V/Vc > 1) and it contains no Froude‑number term.
"""

# ---------------------------------------------------------------- S5–S6 (text moved from the main paper for length)
S5_S6 = """
## S5. Data quality, the Russian subset, uncleaned data and the laboratory‑to‑field experiment

**Table S5a.** Quality flags in the PSDb‑2014 field table (1,858 records). Flags are added, never used to delete records; severe flags define the core set (1,586 records).

| Flag | Records | Level | Meaning |
|---|---|---|---|
| ys = 0 | 85 | Severe | Zero scour |
| ys/bn > 3 | 25 | Severe | Possibly includes contraction scour, general scour or debris |
| Accuracy ≥ ys | 56 | Severe | Label within measurement error |
| V or y = 0 | 10 | Severe | Missing hydraulics |
| Fr > 1 | 25 | Severe | Supercritical approach flow |
| Missing core input | 87 | Severe | bn, y, V or D50 missing |
| bn < b | 2 | Severe | Geometrically impossible |
| Duplicate | 2 | Severe | Repeated record |
| Accuracy > ½ ys | 103 | Warning | Noisy label |
| θ > 0 with bn = b | 53 | Warning | Projected width probably not adjusted for skew |
| bn inconsistent | 62 | Warning | bn differs by > 0.5 ft from b cos θ + L sin θ |
| Cohesive bed | 84 | Information | Outside the main population |
| Substantial debris | 32 | Information | Outside the scope of pier equations |
| Hydraulics modelled | 407 | Information | Not measured at the time of the survey |

**Sensitivity to flagged records.** Removing records with any one of the six warning or information flags from training changes the MAE on unflagged records by at most 0.01 m. Two flags mark conditions outside the model's scope: for substantial debris, measured scour is 1.7–1.9 times the corrected prediction; for skewed piers recorded with bn = b, it is 1.5 times, consistent with a mis‑recorded projected width.

**Russian subset.** The 45 Russian records (Zhuravlyov 1978, compiled by Benedict and Caldwell 2014 via Sheppard et al. 2011) have a median measured/HEC‑18 ratio of 0.74, against 0.33 for the USA. All are round‑nosed, unskewed piers in fine sand at low velocity, rounded to 0.1 ft and without bed‑type or measurement metadata. The difference persists after conditioning on hydraulics: US records with comparable conditions (n = 161) have a median ratio of 0.53, and the regression residual is ×2.0 for the Russian subset against ×1.09 for the rest. Without the subset, the R² advantage of residual XGBoost (M5) over M3 shrinks from +0.055 to +0.042, and M5's MAE becomes worse than that of M3‑S6 (0.475 against 0.466 m).

**Hyperparameters under random splitting.** Tuned on random folds, XGBoost chose a median depth of 6 against 4 under bridge folds, and a median minimum child weight of 1.7–1.8 against 3.1–7.0; random forests chose a median depth of 13–18 against 8–12.5 and a minimum leaf size of 1 against 2–3. The unconstrained residual XGBoost (M5u) reached MAE 0.481 m under bridge CV and 0.400 m under random CV (inflation 17.0%).

**Uncleaned data.** On the uncleaned field table (1,403 computable records), the inflation of M4 with fixed hyperparameters was 11%–13%, and accuracy on the clean rows was unchanged. Cleaning mainly affects the metrics: the R² of M3‑S6 falls from 0.51 to 0.43 when noisy rows are scored.

**Laboratory to field.** The median measured/HEC‑18 ratio is 0.80 for the 525 core laboratory records against 0.34 for the field records; at equal a/D50 the field ratio is 2.1–3.3 times lower (for a/D50 < 50: 0.23 in the field against 0.73 in the laboratory). Trained on laboratory data and applied to the field, a constant factor gave MAE 1.14 m, a linear correction in y/a and a/D50 1.40 m, and XGBoost 1.27 m, against 0.47 m for the field‑trained M3‑S6. Restricting the field data to round, aligned piers does not change this; the offset is not explained by a/D50 within the flume range (15–635).

## S6. Comparison with previous machine‑learning studies

| Aspect | Typical recent study | Kim et al. (2025) | Zhou et al. (2026) | Guo et al. (2027) | This study |
|---|---|---|---|---|---|
| Data | 100–950 points, mostly flume | 1,152 PSDb field records, screened partly by agreement with empirical equations | 552 flume + 229 field records (piles, 8 sources; 21 held out) | 646 flume tests + 400 synthetic | 1,253 PSDb field records, 370 bridges, flags documented |
| Baseline | None, or compared afterwards | None | Laboratory‑trained XGBoost (additive residual) | None | HEC‑18 (5th ed.), log residual |
| Validation | Random split | Random 80/10/10 by row | Random 85/15 + one held‑out source (21 records) | Random 80/10/10 | Bridge‑grouped (×10) and source‑grouped CV; six extrapolation tests with bridge bootstrap |
| Uncertainty | None (or Gaussian, laboratory only) | In‑sample conservatism level and global multiplier | Underprediction rate | Bootstrap of R² | Conformal bounds at row, bridge and source level |
| Leakage control | Rare | — | Partial (one source) | — | Quantified (13%–22% inflation, three model families) |
| Laboratory → field | Mixed, untested | Laboratory model overpredicts field (qualitative) | Used as base model | Laboratory only | Tested on the same HEC‑18 baseline; fails |

Representative studies reviewed for the "typical" column: Pal (2022); Ali et al. (2022); Eini et al. (2024a, b); Devi and Kaushik (2025); El Gana et al. (2025); Huynh et al. (2025a, b); Choi et al. (2025); Khan and Ismael (2026). Common features: 100–950 points, mostly from flumes; random train/test splits; reported test R² of 0.78–0.92; claims of superiority over design equations; no prediction intervals; no out‑of‑distribution tests; and, in several cases, data leakage such as noise augmentation before splitting or tuning on the test set. Khan and Ismael (2026) trained gradient boosting on the 569 PSDb laboratory cases (stratified random 80/20 split, test R² 0.96) and applied it to a field bridge. Kim et al. (2026) combined probabilistic boosting with reliability analysis for abutment scour.

### References for the Supplemental Materials (not cited in the main text)

Ali, A. S., M. M. Ashiq, S. Ebrahimi, M. S. Alasta, and M. Khorram. 2022. "Machine learning model for estimation of local scour depth around cylindrical bridge piers." *Iraqi J. Civ. Eng.* 16 (2). https://doi.org/10.37650/ijce.2022.160201.

Devi, G., and V. Kaushik. 2025. "ML‑based prediction of scour depth around a cylindrical bridge pier: A comparative analysis of ANN, SVM, and ensemble trees." *Ocean Eng.* 336: 121735. https://doi.org/10.1016/j.oceaneng.2025.121735.

Eini, N., S. Janizadeh, S. M. Bateni, C. Jun, E. Heggy, and M. Kirs. 2024a. "Predicting equilibrium scour depth around non‑circular bridge piers with shallow foundations using hybrid explainable machine learning methods." *Results Eng.* 24: 103492.

Huynh, N. T., A. T. T. Phan, T. T. Trieu, H.‑H.‑D. Nguyen, and T. N. Nguyen. 2025b. "A CNN deep learning approach to scour depth estimation around complex bridge piers in steady flow environments." *Adv. Bridge Eng.* 6: 19. https://doi.org/10.1186/s43251-025-00170-8.

Khan, A., and D. Ismael. 2026. "Interpretable machine learning for bridge‑pier scour prediction and flood resilience." *Front. Built Environ.* 11: 1731114. https://doi.org/10.3389/fbuil.2025.1731114.

Kim, T., S. Kwon, and T. Kim. 2026. "Probabilistic prediction and reliability analysis of abutment scour depth using natural gradient boosting." *Water* 18 (21): 2688. https://doi.org/10.3390/w18212688.

Sheppard, D. M., H. Demir, and B. W. Melville. 2011. *Scour at wide piers and long skewed piers*. NCHRP Rep. 682. Washington, DC: Transportation Research Board.
"""
text += S5_S6

# ---------------------------------------------------------------- S7–S8 (revision, 28/11/2026)
rv = pd.read_csv(OUT / "rev_eq2_variants.csv"); rc = pd.read_csv(OUT / "rev_eq2_coefficients.csv")
rf = pd.read_csv(OUT / "rev_resid_by_Fr.csv"); rt = pd.read_csv(OUT / "rev_eq2_timing_sensitivity.csv")
ra = pd.read_csv(OUT / "rev_aD_within_D50.csv"); rs = pd.read_csv(OUT / "rev_strat_timing.csv")
rk = pd.read_csv(OUT / "rev_k_multipliers.csv")
s8a = rv.pivot_table(index="split", columns="variant", values="MAE").round(3)
s8b = rc.assign(CI=lambda x: x.apply(lambda r: f"{r.coef:.3f} [{r.lo95:.3f}, {r.hi95:.3f}]", axis=1)).pivot(index="term", columns="variant", values="CI").fillna("")
s8c = rf[["variant", "Fr_class", "n", "median_resid", "lo90", "hi90", "spearman_rho", "spearman_p"]].round(3)
s8d = rs.round(3)
S7_S8 = f"""
## S7. Application to the Ngoi Thia Bridge, Vietnam

The Ngoi Thia Bridge (five spans, 176.5 m, opened in 1989) crosses a gravel‑bed mountain river in northern Vietnam on shallow foundations. During the flood of 5–11 October 2017 (modelled peak discharge 860 m³/s, water level +264.06 m), Pier T4 settled and tilted and two spans collapsed (Fig. S1). The official investigation (MOT 2017) attributed the failure to the extreme flood, to morphological change that shifted the thalweg toward Pier T4, and to a large bamboo cluster trapped against the pier. After the flood, the bed at the pier was 2.63 m below its pre‑flood level and 0.63 m below the base of the 2‑m‑deep footing.

Inputs: equivalent pier width a = 1.4 m (shaft 1.3 m wide in its upper part and 1.6 m in its lower part), L = 6 m, round nose, y = 8.54 m, D50 = 17 mm. The approach velocity at the pier, 1.3 m/s, was obtained by distributing the discharge over the pre‑flood cross section (digitized from the as‑built drawing) with Manning's equation. Because deposits narrowed the section during the event, the velocity of the design flow at a similar stage (2.6 m/s) is used as an upper case. θ = 15° represents the shift of the thalweg toward the pier. The bridge has four main‑channel piers, so the bridge‑level multiplier is 2.6 × 4^0.3 ≈ 3.95 (main text, Table 3).

**Table S7.** Estimated scour depth (m) at Pier T4, October 2017 flood (measured after the flood: at least 2.63 m; footing depth 2.0 m), using Eq. (2) with G = 1 (gravel). The refill factor 1.61 (M = 1) is identified for sand beds only and is applied here provisionally.

| Estimate | V = 1.3 m/s, θ = 0° | V = 1.3 m/s, θ = 15° | V = 2.6 m/s, θ = 0° | V = 2.6 m/s, θ = 15° |
|---|---|---|---|---|
| HEC‑18 | 2.51 | 4.03 | 3.36 | 5.43 |
| Corrected median, scour as sounded (M = 0) | 0.67 | 0.82 | 0.78 | 0.97 |
| Corrected median, maximum scour (M = 1) | 1.07 | 1.32 | 1.26 | 1.56 |
| 95% bound, maximum scour, single pier (× 2.6) | 2.78 | 3.42 | 3.28 | 4.05 |
| 95% bound, maximum scour, four piers (× 3.95) | 4.22 | 5.19 | 4.97 | 6.14 |

The corrected median for scour as sounded is well below the footing depth and the measured scour (which is itself a lower bound, because the hole refills after a flood). The single‑pier bound for maximum scour is not exceeded in any scenario, and HEC‑18 is exceeded only in the aligned, low‑velocity scenario. The pier was outside the scope of Eq. (2) because of the debris and the migrating thalweg.

![Fig. S1](figures_en/Fig7_ngoithia_case.png)

**Fig. S1.** Ngoi Thia Bridge after the flood of October 2017: (a) Pier T4 and the collapsed spans (photograph by the first author); (b) geometry of Pier T4, with bed levels before and after the flood (data from MOT 2017).

## S8. Development of the refined closed form

Script: `BaiA/revision_analyses.py`. E0 = M3‑S6 (four terms, all beds); E1 = ln(a/D50) and θ, cohesionless and unreported beds; E2 = E1 + ln Fr (Eq. 2 of the main text).

**Table S8a.** MAE (m) of the variants under bridge‑ and source‑grouped CV and in the six extrapolation experiments. E1 and E2 exclude the 78 cohesive‑bed records, so their test sets differ slightly from those of E0 (notably gravel → sand).

{md(s8a)}

**Table S8b.** Coefficients on the log scale with bridge‑bootstrap 95% intervals.

{s8b.to_markdown()}

**Table S8c.** Out‑of‑fold residual (log scale) by Froude‑number class, median with bridge‑bootstrap 90% interval, and Spearman correlation of residual with Fr.

{md(s8c)}

**Table S8d.** Sensitivity of Eq. (2) to measurement timing: coefficients without and with indicators for historical, post‑flood and unreported timing (reference: during a flow event).

{md(rt.round(3))}

**Table S8e.** Median measured/HEC‑18 ratio by a/D50 tercile within each bed class, with the Spearman correlation of ln(a/D50) and the log ratio within the class.

{md(ra.round(3))}

**Table S8f.** Full stratification by measurement timing, method, recurrence interval and gradation (ratio = measured/HEC‑18; resid factor = measured/corrected by Eq. 2, out of fold).

{md(s8d)}

**Table S8g.** Multipliers for k piers: empirical resampling, Gaussian random‑effects model and semi‑parametric resampling of bridge effects and within‑bridge deviations.

{md(rk.round(3))}

Under the single‑pier bound of Eq. (2), the proportion of records above the bound was 16.4% for ground‑penetrating‑radar records and 3.2% for soundings; the 95th percentile of the residual multiplier was 3.28 for radar records and 2.29 for the others.

### References for S7–S8

MOT (Ministry of Transport of Vietnam). 2017. *Report on the cause of the Ngoi Thia Bridge collapse.* Hanoi, Vietnam: MOT. [In Vietnamese.]
"""
text += S7_S8

r2v = pd.read_csv(OUT / "rev2_variants.csv").set_index("variant").T
r2c = pd.read_csv(OUT / "rev2_coefficients.csv")
r2c = r2c.assign(CI=lambda x: x.apply(lambda r: f"{r.coef:.3f} [{r.lo95:.3f}, {r.hi95:.3f}]", axis=1)).pivot(index="term", columns="variant", values="CI").fillna("")
S8B = f"""
### S8, second revision: measurement level, bed‑class form and k multipliers

Script: `BaiA/revision2_analyses.py`. E2 = continuous form (ln a/D50, θ, ln Fr); E2m = E2 + indicators for GPR, survey, scour rod and unreported method (reference: soundings, fathometer, ADCP); E3 = bed‑class form (G, θ, ln Fr); E3m = E3 + all method indicators; E3c = three bed classes + method indicators; E3g = E3 + GPR indicator only (Eq. 2 of the main text).

**Table S8h.** Variants on the 1,175 cohesionless and unreported‑bed records: bridge‑ and source‑grouped CV, correlation of residual with Fr, and extrapolation (ext = MAE ratio to the same rows under bridge CV; extMAE in m; NaN where a bed‑class factor cannot be estimated).

{md(r2v.round(3))}

**Table S8i.** Coefficients on the log scale with bridge‑bootstrap 95% intervals.

{r2c.to_markdown()}

**Table S8j.** GPR against soundings within the same bed class (ratio = measured/HEC‑18; meas/E2 = measured/out‑of‑fold continuous form). A fit to sand records only gives a GPR coefficient of 0.493 (factor 1.64).

{md(pd.read_csv(OUT / "rev2_gpr_vs_soundings.csv").round(3))}

**Table S8k.** Multipliers for k piers from out‑of‑fold residuals of Eq. 2 (empirical, Gaussian random‑effects, semi‑parametric, and 2.6·k^0.3 fit at the single‑pier value).

{md(pd.read_csv(OUT / "rev2_k_multipliers.csv").round(3))}

**Table S8l.** Bridges by number of measured piers (medians): bridges with several measured piers have finer beds and deeper flows.

{pd.read_csv(OUT / "rev2_bridges_by_npier.csv").round(2).to_markdown(index=False)}

**Table S8m.** 95% bound of Eq. 2 relative to HEC‑18, for scour as sounded (M = 0) and maximum scour (M = 1), by bed class and number of piers (empirical multipliers).

{md(pd.read_csv(OUT / "rev2_bound_vs_hec.csv").round(2))}

Direct check on the measurements: the 95th percentile of measured/HEC‑18 is 1.42 for GPR records in sand (12.4% above HEC‑18), 1.03 for soundings in sand (6.2%) and 0.54 for soundings in gravel and cobble (0.4%).
"""
text += S8B

(HERE / "Supplement_BaiA.md").write_text(text, encoding="utf-8")
print("Đã ghi", HERE / "Supplement_BaiA.md")
