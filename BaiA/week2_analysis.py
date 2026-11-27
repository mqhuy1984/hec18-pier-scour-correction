"""Tuần 2 bài A: (1) kiểm tra nhóm giả và độ nhạy cách nhóm; (2) chọn bộ biến; (3) RQ1 — sai lệch HEC-18 phụ thuộc vào gì.

Chạy:  python BaiA/week2_analysis.py     (cần chạy week1_baselines.py trước)
Đọc:   BaiA/outputs/week1_eval_set_with_predictions.csv
Ghi:   BaiA/outputs/week2_*.csv, BaiA/figures/week2_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import equations as eq  # noqa: E402

OUT = ROOT / "BaiA" / "outputs"; FIG = ROOT / "BaiA" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

d = pd.read_csv(OUT / "week1_eval_set_with_predictions.csv", encoding="utf-8-sig", low_memory=False)
d["lr"] = np.log(d["ys_m"] / d["M0_hec18_kw"])          # log tỷ số ys đo / HEC-18
print(f"Tập đánh giá: {len(d)} dòng, {d['site_group'].nunique()} nhóm cầu")

# ============================================================ 1. NHÓM GIẢ
print("\n================ 1. NHÓM GIẢ ================")
ps = d[d["site_group_is_pseudo"]]
print(f"Dòng thuộc nhóm giả: {len(ps)} ({len(ps)/len(d):.1%}), số nhóm giả: {ps['site_group'].nunique()}")
print(ps.groupby(["country", "source"]).agg(rows=("row_id", "size"), groups=("site_group", "nunique")).to_string())
sz = ps.groupby("site_group").size()
print("Kích thước nhóm giả: ", sz.describe().round(1).to_dict())
print("Nhóm giả lớn nhất:\n", ps[ps["site_group"].isin(sz.nlargest(5).index)]
      .groupby("site_group")[["country", "bn_m", "y_m", "V_ms", "D50_mm", "year"]].agg(lambda s: s.nunique()).to_string())
real = d[~d["site_group_is_pseudo"]].groupby("site_group").size()
print("Kích thước nhóm thật:", real.describe().round(1).to_dict())

# Ba cách nhóm
d["grp_main"] = d["site_group"]
d["grp_fine"] = np.where(d["site_group_is_pseudo"], 100000 + d["row_id"], d["site_group"])   # mỗi dòng giả là một nhóm
coarse_key = d["country"].astype(str) + "|" + d["source"].astype(str)
d["grp_coarse"] = np.where(d["site_group_is_pseudo"], 200000 + pd.factorize(coarse_key)[0], d["site_group"])  # gom theo quốc gia+nguồn


def group_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(pd.unique(groups))); rng.shuffle(ug)
    m = {g: i % k for i, g in enumerate(ug)}
    return np.array([m[g] for g in groups])


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def cv_m3(feats, groups, reps=10, k=5, seed0=1000):
    out = []
    X = d[feats].to_numpy(float); y = d["lr"].to_numpy(); obs = d["ys_m"].to_numpy(); hec = d["M0_hec18_kw"].to_numpy()
    for r in range(reps):
        f = group_folds(groups, k, seed0 + r)
        for i in range(k):
            tr, te = f != i, f == i
            b = ols(X[tr], y[tr]); res = y[tr] - pred(b, X[tr]); mu = pred(b, X[te])
            p = hec[te] * np.exp(mu); up = hec[te] * np.exp(mu + np.quantile(res, 0.95))
            lo, lp = np.log10(obs[te]), np.log10(p)
            out.append({"MAE": np.mean(np.abs(obs[te] - p)),
                        "R2_log": 1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2),
                        "exceed_q95": np.mean(obs[te] > up)})
    o = pd.DataFrame(out)
    return o.mean().to_dict() | {f"{c}_sd": v for c, v in o.std().to_dict().items()}


# ======================================================= 2. BIẾN ĐẦU VÀO
VVc = d["V_ms"] / eq.critical_velocity_hec18(d["y_m"], d["D50_mm"])
d["x_log_ya"] = np.log(d["y_m"] / d["a_m"])
d["x_log_Fr"] = np.log(d["V_ms"] / np.sqrt(9.81 * d["y_m"]))
d["x_log_aD"] = np.log(d["a_m"] / (d["D50_mm"] / 1000))
d["x_log_VVc"] = np.log(np.clip(VVc, 0.05, 20))
d["x_log_Vc_sqrtgy"] = np.log(eq.critical_velocity_hec18(d["y_m"], d["D50_mm"]) / np.sqrt(9.81 * d["y_m"]))
d["x_theta"] = d["theta0"] / 90
d["x_log_La"] = np.log(np.clip((d["L_m"] / d["a_m"]).fillna(1.0), 1, 12))
for k in ["square", "sharp", "circular"]:
    d[f"x_nose_{k}"] = (d["nose"] == k).astype(float)
d["x_cohesive"] = (d["bed"] == "cohesive").astype(float)
d["x_bed_unknown"] = d["bed"].isna().astype(float)
d["x_group"] = (d["pier_type"] == "Group").astype(float)
d["x_ptype_unknown"] = d["pier_type"].isna().astype(float)
cat = ["x_theta", "x_nose_square", "x_nose_sharp", "x_nose_circular", "x_cohesive", "x_bed_unknown", "x_group", "x_ptype_unknown"]

sets = {
    "S1 tuần 1 (y/a, Fr, a/D50, V/Vc)": ["x_log_ya", "x_log_Fr", "x_log_aD", "x_log_VVc"] + cat,
    "S2 bỏ Fr (y/a, a/D50, V/Vc)": ["x_log_ya", "x_log_aD", "x_log_VVc"] + cat,
    "S3 bỏ V/Vc (y/a, Fr, a/D50)": ["x_log_ya", "x_log_Fr", "x_log_aD"] + cat,
    "S4 (y/a, a/D50, V/Vc, Vc/√gy)": ["x_log_ya", "x_log_aD", "x_log_VVc", "x_log_Vc_sqrtgy"] + cat,
    "S5 S2 + L/a": ["x_log_ya", "x_log_aD", "x_log_VVc", "x_log_La"] + cat,
    "S6 tối giản (y/a, a/D50, θ, đất dính)": ["x_log_ya", "x_log_aD", "x_theta", "x_cohesive", "x_bed_unknown"],
}


def vif(cols):
    X = d[cols].to_numpy(float); out = {}
    for j, c in enumerate(cols):
        y = X[:, j]; Z = np.delete(X, j, axis=1)
        b = ols(Z, y); r2 = 1 - np.sum((y - pred(b, Z)) ** 2) / np.sum((y - y.mean()) ** 2)
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return out


print("\n================ 2. BỘ BIẾN ================")
cont = ["x_log_ya", "x_log_Fr", "x_log_aD", "x_log_VVc", "x_log_Vc_sqrtgy"]
print("Tương quan Pearson giữa các biến liên tục:\n", d[cont].corr().round(2).to_string())
rows = []
for name, cols in sets.items():
    v = vif([c for c in cols if c in cont + ["x_log_La"]])
    r = cv_m3(cols, d["grp_main"].to_numpy())
    rows.append({"set": name, "n_feat": len(cols), "max_VIF_cont": max(v.values()), **r})
    print(f"{name}: VIF liên tục = { {k: round(x, 1) for k, x in v.items()} }")
sel = pd.DataFrame(rows)
print("\nM3 với từng bộ biến (GroupKFold theo cầu, 5×10):\n", sel.round(3).to_string(index=False))

# Độ nhạy theo cách nhóm (dùng bộ S2)
base = sets["S2 bỏ Fr (y/a, a/D50, V/Vc)"]
gs = []
for g in ["grp_main", "grp_fine", "grp_coarse"]:
    gs.append({"grouping": g, "n_groups": d[g].nunique(), **cv_m3(base, d[g].to_numpy())})
gs.append({"grouping": "random (không nhóm)", "n_groups": len(d), **cv_m3(base, np.arange(len(d)))})
gs = pd.DataFrame(gs)
print("\nĐộ nhạy theo cách nhóm (M3, bộ S2):\n", gs.round(3).to_string(index=False))

# ============================================================== 3. RQ1
print("\n================ 3. RQ1 ================")
rng = np.random.default_rng(7)
groups = d["site_group"].to_numpy()
ug = np.unique(groups)
idx_by_g = {g: np.where(groups == g)[0] for g in ug}


def boot_median(mask, B=500):
    """Trung vị log tỷ số và khoảng 90% bootstrap theo cầu (lấy mẫu lại nhóm cầu)."""
    sub_g = np.unique(groups[mask]); vals = []
    lr = d["lr"].to_numpy()
    for _ in range(B):
        gg = rng.choice(sub_g, size=len(sub_g), replace=True)
        ii = np.concatenate([idx_by_g[g][mask[idx_by_g[g]]] for g in gg])
        vals.append(np.median(lr[ii]))
    return np.median(lr[mask]), np.quantile(vals, 0.05), np.quantile(vals, 0.95), len(sub_g)


cont_vars = {
    "y/a": d["y_m"] / d["a_m"], "a/D50": d["a_m"] / (d["D50_mm"] / 1000), "Fr": np.exp(d["x_log_Fr"]),
    "V/Vc": VVc, "bn (m)": d["bn_m"], "θ (độ)": d["theta0"], "D50 (mm)": d["D50_mm"],
}
binned = []
for name, x in cont_vars.items():
    if name == "θ (độ)":
        edges = [-0.1, 0.1, 10, 20, 90]
    else:
        edges = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, 7)))
    b = pd.cut(x, edges, include_lowest=True)
    for k in b.cat.categories:
        m = (b == k).to_numpy()
        if m.sum() < 15: continue
        med, lo, hi, ng = boot_median(m)
        binned.append({"var": name, "bin": str(k), "x_mid": float(np.median(x[m])), "n": int(m.sum()), "n_sites": ng,
                       "median_ratio": np.exp(med), "lo90": np.exp(lo), "hi90": np.exp(hi),
                       "frac_under": float(np.mean(d.loc[m, "ys_m"] > d.loc[m, "M0_hec18_kw"]))})
binned = pd.DataFrame(binned)
print(binned.round(3).to_string(index=False))

cats = {"Mũi trụ": d["nose"], "Loại đất": d["bed"].fillna("không ghi"), "Loại trụ": d["pier_type"].fillna("không ghi"),
        "Quốc gia": d["country"], "Cách đo": d["meas_type"].fillna("không ghi"),
        "Thủy lực": d["hydraulics_method"].fillna("không ghi"),
        "Chế độ (V/Vc)": np.where(VVc > 1, "live-bed", "clear-water")}
crows = []
for name, s in cats.items():
    s = pd.Series(s, index=d.index)
    for k in s.value_counts().index:
        m = (s == k).to_numpy()
        if m.sum() < 15: continue
        med, lo, hi, ng = boot_median(m)
        crows.append({"var": name, "level": k, "n": int(m.sum()), "n_sites": ng, "median_ratio": np.exp(med),
                      "lo90": np.exp(lo), "hi90": np.exp(hi),
                      "frac_under": float(np.mean(d.loc[m, "ys_m"] > d.loc[m, "M0_hec18_kw"]))})
crows = pd.DataFrame(crows)
print("\n", crows.round(3).to_string(index=False))

# Spearman log tỷ số với biến liên tục
sp = {k: pd.Series(np.log(np.clip(v, 1e-6, None)) if k != "θ (độ)" else v).rank().corr(d["lr"].rank()) for k, v in cont_vars.items()}
print("\nSpearman(log tỷ số, biến):", {k: round(v, 3) for k, v in sp.items()})

# Hệ số OLS bộ S2 trên toàn tập + khoảng tin cậy bootstrap theo cầu
X = d[base].to_numpy(float); y = d["lr"].to_numpy()
b0 = ols(X, y); bs = []
for _ in range(500):
    gg = rng.choice(ug, size=len(ug), replace=True); ii = np.concatenate([idx_by_g[g] for g in gg])
    bs.append(ols(X[ii], y[ii]))
bs = np.array(bs)
coef = pd.DataFrame({"coef": b0, "lo90": np.quantile(bs, 0.05, axis=0), "hi90": np.quantile(bs, 0.95, axis=0)},
                    index=["intercept"] + base)
print("\nHệ số M3 (bộ S2), khoảng 90% bootstrap theo cầu:\n", coef.round(3).to_string())

# ============================================================== HÌNH
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5)); axes = axes.ravel()
for ax, (name, x) in zip(axes, cont_vars.items()):
    ax.scatter(x, d["ys_m"] / d["M0_hec18_kw"], s=6, alpha=0.25, color="0.5")
    bb = binned[binned["var"] == name]
    ax.errorbar(bb["x_mid"], bb["median_ratio"], yerr=[bb["median_ratio"] - bb["lo90"], bb["hi90"] - bb["median_ratio"]],
                fmt="o-", color="C3", capsize=3, label="trung vị (KTC 90% theo cầu)")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.01, 5)
    if name not in ("θ (độ)", "Fr"): ax.set_xscale("log")
    ax.set_xlabel(name); ax.set_ylabel("ys đo / ys HEC-18")
axes[0].legend(fontsize=8, loc="lower left")
ax = axes[-1]
cc = crows[crows["var"].isin(["Mũi trụ", "Loại đất", "Chế độ (V/Vc)"])].reset_index(drop=True)
ax.errorbar(cc["median_ratio"], np.arange(len(cc)), xerr=[cc["median_ratio"] - cc["lo90"], cc["hi90"] - cc["median_ratio"]],
            fmt="o", color="C0", capsize=3)
ax.set_yticks(np.arange(len(cc))); ax.set_yticklabels([f"{v}: {l} (n={n})" for v, l, n in zip(cc["var"], cc["level"], cc["n"])], fontsize=7)
ax.axvline(1, color="k", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xlabel("trung vị ys đo / ys HEC-18")
ax.set_xlim(0.15, 1.1); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.2, 0.3, 0.5, 1.0]); ax.set_xticklabels(["0,2", "0,3", "0,5", "1"])
fig.suptitle("RQ1 — Sai lệch của HEC-18 (CSU bản 5, có Kw) trên số liệu hiện trường PSDb-2014, tập đánh giá chung")
fig.tight_layout(); fig.savefig(FIG / "week2_rq1_dependence.png", dpi=200)

# Ghi file
sel.to_csv(OUT / "week2_feature_sets.csv", index=False)
gs.to_csv(OUT / "week2_grouping_sensitivity.csv", index=False)
binned.to_csv(OUT / "week2_rq1_binned.csv", index=False)
crows.to_csv(OUT / "week2_rq1_categorical.csv", index=False)
coef.to_csv(OUT / "week2_M3_S2_coefficients.csv")
print("\nĐã ghi hình:", FIG / "week2_rq1_dependence.png")
