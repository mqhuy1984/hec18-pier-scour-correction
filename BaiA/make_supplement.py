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

**Correcting HEC‑18 pier‑scour estimates with field data: bridge‑level validation, conformal bounds, and the limits of machine learning**

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
(HERE / "Supplement_BaiA.md").write_text(text, encoding="utf-8")
print("Đã ghi", HERE / "Supplement_BaiA.md")
