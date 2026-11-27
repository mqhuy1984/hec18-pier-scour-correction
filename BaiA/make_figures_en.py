"""English, publication-style figures for the Bai A manuscript (reads existing output CSVs only).

Run:   python BaiA/make_figures_en.py      (after week1–week5 scripts)
Writes BaiA/figures_en/Fig1_workflow.png … Fig6_formula_lab_shap.png (300 dpi)

Colours: categorical slots in fixed order (reference palette); one colour per model, never re-assigned.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; FIG = HERE / "figures_en"; FIG.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"M3_S6": SLOTS[0], "M4": SLOTS[1], "M5": SLOTS[2], "RF4": SLOTS[3], "RF5": SLOTS[4],
               "GPR5": SLOTS[5], "CQR": SLOTS[6], "M0": "#898781", "M3": "#86b6ef", "M5u": "#9ec5f4"}
MODEL_LABEL = {"M0": "HEC-18", "M2": "HEC-18 × const.", "M3": "Linear (11 terms)", "M3_S6": "Linear (4 terms)",
               "M4": "XGB direct", "M5": "XGB residual", "M5u": "XGB residual (unconstr.)", "RF4": "RF direct",
               "RF5": "RF residual", "GPR5": "GPR residual", "CQR": "CQR"}
SPLIT_EN = {"USA → ngoài Mỹ": "USA → non-USA", "bn ≤ 3 → bn > 5 m": "bn ≤ 3 → > 5 m", "bn ≤ 3 → 3–5 m": "bn ≤ 3 → 3–5 m",
            "cát (<2 mm) → sỏi cuội (≥20 mm)": "sand → gravel/cobble", "sỏi (≥2 mm) → cát (<2 mm)": "gravel → sand",
            "Fr ≤ 0,3 → Fr > 0,5": "Fr ≤ 0.3 → > 0.5"}
VAR_EN = {"y/a": "y/a", "a/D50": "a/D50", "Fr": "Froude number", "V/Vc": "V/Vc", "bn (m)": "projected width bn (m)",
          "θ (độ)": "skew θ (deg)", "D50 (mm)": "D50 (mm)"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                     "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK})


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight"); plt.close(fig); print("wrote", FIG / name)


def bars(ax, groups, series, values, lo=None, hi=None, width=0.8):
    """Grouped bars with a thin surface gap between adjacent bars."""
    k = len(series); w = width / k
    for j, s in enumerate(series):
        x = np.arange(len(groups)) + (j - (k - 1) / 2) * w
        v = np.asarray(values[s], float)
        err = None
        if lo is not None:
            err = np.vstack([v - np.asarray(lo[s], float), np.asarray(hi[s], float) - v])
        ax.bar(x, v, w * 0.92, color=MODEL_COLOR.get(s, SLOTS[j % 8]), label=MODEL_LABEL.get(s, s),
               yerr=err, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.5), zorder=2)
    ax.set_xticks(range(len(groups)))


# ------------------------------------------------------------------ Fig 1 workflow
fig, ax = plt.subplots(figsize=(7.2, 2.4)); ax.axis("off")
boxes = [("PSDb-2014\nfield table", "1,858 records\n8 severe + 6 warning\nquality flags"),
         ("Evaluation set", "1,253 rows\n370 bridges\nfull HEC-18 (5th ed.)"),
         ("Corrections", "constant · linear\nXGB · RF · GPR\nln(ys/a) or ln(ys/ys,HEC)"),
         ("Validation", "random · bridge\n· data source\n+ 6 extrapolation splits"),
         ("Conformal bounds", "pooled · weighted\n· bridge-max\nrow / bridge / source")]
for i, (t, s) in enumerate(boxes):
    x = 0.02 + i * 0.198
    ax.add_patch(plt.Rectangle((x, 0.30), 0.17, 0.62, facecolor="#eef4fc", edgecolor=SLOTS[0], lw=0.8))
    ax.text(x + 0.085, 0.80, t, ha="center", va="center", fontsize=7.2, color=INK, weight="bold")
    ax.text(x + 0.085, 0.50, s, ha="center", va="center", fontsize=6.2, color=INK2, linespacing=1.4)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + 0.198, 0.63), xytext=(x + 0.172, 0.63), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.text(0.5, 0.15, "Output: closed-form HEC-18 correction + conformal multipliers + domain of applicability",
        ha="center", fontsize=7.6, color=INK)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "Fig1_workflow.png")

# ------------------------------------------------------------------ Fig 2 HEC-18 bias
b = pd.read_csv(OUT / "week2_rq1_binned.csv"); c = pd.read_csv(OUT / "week2_rq1_categorical.csv")
ev = pd.read_csv(OUT / "week1_eval_set_with_predictions.csv", encoding="utf-8-sig", low_memory=False)
ratio = ev["ys_m"] / ev["M0_hec18_kw"]
import importlib.util
sys.path.insert(0, str(HERE)); import equations as eq  # noqa: E402
raw = {"y/a": ev["y_m"] / ev["a_m"], "a/D50": ev["a_m"] / (ev["D50_mm"] / 1000), "Fr": ev["V_ms"] / np.sqrt(9.81 * ev["y_m"]),
       "V/Vc": ev["V_ms"] / eq.critical_velocity_hec18(ev["y_m"], ev["D50_mm"]), "bn (m)": ev["bn_m"], "θ (độ)": ev["theta0"],
       "D50 (mm)": ev["D50_mm"]}
fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9)); axes = axes.ravel()
for ax, (k, x) in zip(axes, raw.items()):
    ax.scatter(x, ratio, s=2, color=MUTED, alpha=0.25, lw=0, zorder=1)
    bb = b[b["var"] == k]
    ax.errorbar(bb["x_mid"], bb["median_ratio"], yerr=[bb["median_ratio"] - bb["lo90"], bb["hi90"] - bb["median_ratio"]],
                fmt="o-", ms=3.5, lw=1.4, color=SLOTS[0], ecolor=SLOTS[0], elinewidth=0.8, capsize=1.5, zorder=3)
    ax.axhline(1, color=INK2, lw=0.7, ls="--"); ax.set_yscale("log"); ax.set_ylim(0.01, 5)
    if k not in ("θ (độ)", "Fr"): ax.set_xscale("log")
    ax.set_xlabel(VAR_EN[k], fontsize=7.5)
    if ax in (axes[0], axes[4]): ax.set_ylabel("measured / HEC-18")
ax = axes[-1]
cc = c[c["var"].isin(["Mũi trụ", "Loại đất", "Chế độ (V/Vc)"])].reset_index(drop=True)
lab_map = {"round": "round nose", "square": "square nose", "sharp": "sharp nose", "circular": "circular",
           "cohesionless": "non-cohesive bed", "cohesive": "cohesive bed", "không ghi": "bed not reported",
           "live-bed": "live-bed", "clear-water": "clear-water"}
ax.errorbar(cc["median_ratio"], np.arange(len(cc)), xerr=[cc["median_ratio"] - cc["lo90"], cc["hi90"] - cc["median_ratio"]],
            fmt="o", ms=3.5, color=SLOTS[0], elinewidth=0.8, capsize=1.5)
ax.set_yticks(range(len(cc))); ax.set_yticklabels([f"{lab_map.get(l, l)} (n={n})" for l, n in zip(cc["level"], cc["n"])], fontsize=6.5)
ax.axvline(1, color=INK2, lw=0.7, ls="--"); ax.set_xscale("log"); ax.set_xlim(0.15, 1.1)
ax.set_xticks([0.2, 0.3, 0.5, 1.0]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1"]); ax.minorticks_off()
ax.set_xlabel("median measured / HEC-18", fontsize=7.5)
fig.suptitle("HEC-18 bias on field data: binned medians with bridge-bootstrap 90% CI", fontsize=9, weight="bold", color=INK)
fig.tight_layout(); save(fig, "Fig2_hec18_bias.png")

# ------------------------------------------------------------------ Fig 3 accuracy + split inflation
oof = pd.read_csv(OUT / "week3_oof_predictions.csv"); rq4 = pd.read_csv(OUT / "week3_rq4_split_gap.csv", index_col=0)
obs = oof["ys_m"].to_numpy()
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
for ax, m in zip(axes[:3], ["M0", "M3_S6", "M5"]):
    p = oof[f"{m}_group"].to_numpy()
    ax.scatter(obs, p, s=2.5, lw=0, alpha=0.45, color=MODEL_COLOR[m])
    ax.plot([0.01, 30], [0.01, 30], color=INK2, lw=0.7, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.02, 20); ax.set_ylim(0.02, 20)
    lo, lp = np.log10(obs), np.log10(p)
    r2 = 1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2); mae = np.mean(np.abs(obs - p))
    ax.set_title(f"{MODEL_LABEL[m]}\nMAE {mae:.2f} m, R² {r2:.2f}", fontsize=7.5)
    ax.set_xlabel("measured ys (m)", fontsize=7.5)
axes[0].set_ylabel("predicted ys (m)")
ax = axes[3]; ms = [m for m in ["M3", "M3_S6", "M4", "M5", "RF4", "RF5", "GPR5"] if m in rq4.index]
x = np.arange(len(ms))
ax.bar(x - 0.2, rq4.loc[ms, "R2_group"], 0.38, color=SLOTS[0], label="bridge-grouped CV", zorder=2)
ax.bar(x + 0.2, rq4.loc[ms, "R2_random"], 0.38, color=SLOTS[1], label="random CV", zorder=2)
ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in ms], fontsize=6, rotation=40, ha="right")
ax.set_ylabel("R² (log10 ys)"); ax.set_ylim(0, 0.9); ax.legend(fontsize=6.3, loc="upper left", ncol=2, frameon=False)
ax.set_title("Inflation by random splitting", fontsize=7.5)
fig.tight_layout(); save(fig, "Fig3_accuracy_split.png")

# ------------------------------------------------------------------ Fig 4 conformal levels
cv = pd.read_csv(OUT / "week4_conformal_cv.csv"); bg = pd.read_csv(OUT / "week4_conformal_by_group.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
meths = ["pooled", "weighted", "bridgemax", "weighted+Mondrian"]; mlab = ["pooled", "bridge-weighted", "bridge-max", "Mondrian (width)"]
for ax, sp, title in [(axes[0], "cv_group", "Bridge-grouped CV"), (axes[1], "cv_source", "Source-grouped CV")]:
    s = cv[(cv["split"] == sp) & (cv["model"] == "M3_S6")].set_index("method").reindex(meths)
    x = np.arange(len(meths))
    ax.bar(x - 0.2, 100 * s["exceed95_row"], 0.38, color=SLOTS[0], label="rows above 95% bound", zorder=2)
    ax.bar(x + 0.2, 100 * s["bridges_any_exceed"], 0.38, color=SLOTS[1], label="bridges with ≥1 pier above", zorder=2)
    ax.axhline(5, color=INK2, lw=0.7, ls="--"); ax.set_xticks(x); ax.set_xticklabels(mlab, fontsize=6.3, rotation=20, ha="right")
    ax.set_ylabel("% exceeding (linear 4-term)"); ax.set_title(title, fontsize=8); ax.set_ylim(0, 22)
axes[0].legend(fontsize=6.3, loc="upper right")
ax = axes[2]
order = ["≤1 m", "1–3 m", "3–5 m", ">5 m"]
for j, (meth, lab) in enumerate([("weighted", "bridge-weighted"), ("weighted+Mondrian", "Mondrian (width)"), ("bridgemax", "bridge-max")]):
    s = bg[(bg["by"] == "cls") & (bg["model"] == "M5") & (bg["method"] == meth)].set_index("level").reindex(order)
    ax.bar(np.arange(4) + (j - 1) * 0.27, s["cov90_row"], 0.25, color=SLOTS[j], label=lab, zorder=2)
ax.axhline(0.9, color=INK2, lw=0.7, ls="--"); ax.set_ylim(0.6, 1.0); ax.set_xticks(range(4)); ax.set_xticklabels(order)
ax.set_ylabel("90% interval coverage"); ax.set_xlabel("pier width bn"); ax.set_title("XGB residual: coverage by width", fontsize=8)
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
fig.tight_layout(); save(fig, "Fig4_conformal_levels.png")

# ------------------------------------------------------------------ Fig 5 extrapolation
eb = pd.read_csv(OUT / "week5_extrap_bootstrap.csv"); ex = pd.read_csv(OUT / "week4_extrapolation.csv")
w = eb[eb["method"] == "weighted"].copy(); w["lab"] = w["split"].map(SPLIT_EN)
order = ["USA → non-USA", "bn ≤ 3 → 3–5 m", "bn ≤ 3 → > 5 m", "sand → gravel/cobble", "gravel → sand", "Fr ≤ 0.3 → > 0.5"]
w = w.set_index("lab").reindex(order)
key = {"M3_S6": "M3", "M4": "M4", "M5": "M5", "RF4": "RF4", "RF5": "RF5", "GPR5": "GPR5"}
ms = [m for m in key if f"ratio_{key[m]}" in w.columns and w[f"ratio_{key[m]}"].notna().all()]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
bars(axes[0], order, ms, {m: w[f"ratio_{key[m]}"] for m in ms}, {m: w[f"ratio_{key[m]}_lo"] for m in ms},
     {m: w[f"ratio_{key[m]}_hi"] for m in ms})
axes[0].axhline(1, color=INK2, lw=0.7); axes[0].set_xticklabels(order, rotation=25, ha="right", fontsize=6.5)
axes[0].set_ylabel("MAE extrapolated / MAE in-distribution"); axes[0].set_title("Degradation (bridge-bootstrap 90% CI)", fontsize=8)
axes[0].set_ylim(0, 5.2); axes[0].legend(fontsize=6, ncol=3, loc="upper left", frameon=False)
bars(axes[1], order, ms, {m: 100 * w[f"exc_{key[m]}"] for m in ms}, {m: 100 * w[f"exc_{key[m]}_lo"] for m in ms},
     {m: 100 * w[f"exc_{key[m]}_hi"] for m in ms})
axes[1].axhline(5, color=INK2, lw=0.7, ls="--"); axes[1].set_xticklabels(order, rotation=25, ha="right", fontsize=6.5)
axes[1].set_ylabel("% rows above 95% bound"); axes[1].set_title("Exceedance under extrapolation (bridge-weighted)", fontsize=8)
fig.tight_layout(); save(fig, "Fig5_extrapolation.png")

# ------------------------------------------------------------------ Fig 6 formula, SHAP, lab vs field
# Eq. 2 (refined closed form, cohesionless and unreported beds): coefficients from revision_analyses.py
rc = pd.read_csv(OUT / "rev2_coefficients.csv"); rc = rc[rc["variant"].str.startswith("E3g")].set_index("term")["coef"]
k1 = 2.6
m = (ev["bed"] != "cohesive").to_numpy()
aD = raw["a/D50"][m]; Fr6 = raw["Fr"][m]; th6 = ev["theta0"][m].fillna(0)
G6 = (ev["D50_mm"][m] >= 2).astype(float)
corr0 = np.exp(rc["const"] + rc["b_gravel"] * G6 + rc["x_log_Fr"] * np.log(Fr6) + rc["x_theta"] * th6 / 90)  # as sounded
corr = corr0 * np.exp(rc["m_gpr"])                                                                              # maximum scour
sand = (ev["D50_mm"][m] < 2).to_numpy()
shap = pd.read_csv(OUT / "week5_shap_importance.csv"); gb = pd.read_csv(OUT / "week5_lab_field_by_aD.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axes[0]
# colours distinct from the model colours used in panel (b) and the lab/field colours in panel (c)
for sel, lab, col in [(sand, "max. scour median, sand", SLOTS[6]), (~sand, "max. scour median, gravel/cobble", SLOTS[3])]:
    ax.scatter(aD[sel], corr[sel], s=2.5, lw=0, alpha=0.6, color=col, label=lab)
ax.scatter(aD, corr * k1, s=2.5, lw=0, alpha=0.35, color=SLOTS[7], label=f"95% bound, one pier (×{k1:.1f})")
ax.scatter(aD, corr * 2.6 * 4 ** 0.3, s=2.5, lw=0, alpha=0.35, color=MUTED, label="95% bound, four piers (×3.9)")
ax.axhline(1, color=INK2, lw=0.7, ls="--"); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("a/D50"); ax.set_ylabel("ys,corrected / ys,HEC-18")
ax.legend(fontsize=5.3, loc="lower right", markerscale=3, handletextpad=0.2, borderaxespad=0.2,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
ax.set_title("Closed-form correction, Eq. (2)", fontsize=8)
ax = axes[1]
order_f = shap.groupby("feature")["mean_abs_shap"].max().sort_values().tail(7).index
fe = {"x_log_aD": "ln a/D50", "x_log_ya": "ln y/a", "x_log_VVc": "ln V/Vc", "x_theta": "θ/90", "x_log_Fr": "ln Fr",
      "x_ptype_unknown": "pier type n/r", "x_bed_unknown": "bed type n/r", "x_cohesive": "cohesive", "x_nose_square": "square nose",
      "x_group": "pier group", "x_nose_sharp": "sharp nose", "x_nose_circular": "circular"}
for j, m in enumerate(["M4", "M5"]):
    s = shap[shap["model"] == m].set_index("feature").reindex(order_f)
    ax.barh(np.arange(len(order_f)) + (j - 0.5) * 0.4, s["mean_abs_shap"].fillna(0), 0.38, xerr=s["sd"].fillna(0),
            color=MODEL_COLOR[m], label=MODEL_LABEL[m], error_kw=dict(ecolor=INK2, elinewidth=0.6), zorder=2)
ax.set_yticks(range(len(order_f))); ax.set_yticklabels([fe.get(f, f) for f in order_f], fontsize=6.5)
ax.set_xlabel("mean |SHAP| (log units)"); ax.legend(fontsize=6, loc="lower right"); ax.set_title("SHAP, 10 bridge-CV repeats", fontsize=8)
ax = axes[2]
labs = ["<50", "50–200", "200–1k", "1k–5k", ">5k"]
for j, (nm, en) in enumerate([("thí nghiệm", "laboratory"), ("hiện trường", "field")]):
    s = gb[gb["data"] == nm]
    ax.plot(range(len(s)), s["median_ratio"], "o-", ms=3.5, lw=1.4, color=SLOTS[j], label=en)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=6.5); ax.set_yscale("log"); ax.minorticks_off()
ax.set_yticks([0.2, 0.3, 0.5, 1.0]); ax.set_yticklabels(["0.2", "0.3", "0.5", "1"])
ax.axhline(1, color=INK2, lw=0.7, ls="--"); ax.set_xlabel("a/D50"); ax.set_ylabel("median measured / HEC-18")
ax.legend(fontsize=6.5); ax.set_title("Laboratory vs field", fontsize=8)
fig.tight_layout(); save(fig, "Fig6_formula_shap_lab.png")
