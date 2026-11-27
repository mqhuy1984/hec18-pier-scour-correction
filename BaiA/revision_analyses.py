"""Phân tích bổ sung khi sửa bài A cho JHE (28/11/2026).

1. Thảo luận thủy lực: tỷ số đo/HEC-18 và phần dư của công thức theo thời điểm đo (During Flow Event /
   Post Flood / Historical), phương pháp đo (GPR thấy hố đã bồi lấp), chu kỳ lặp lũ (không cân bằng),
   độ cấp phối đáy sỏi (armoring), và xu hướng a/D50 trong từng nhóm cỡ hạt.
2. Sửa Eq. 2: bỏ U (không ghi loại đất), loại đáy dính khỏi phạm vi, kiểm tra phần dư theo Fr, thêm Fr nếu cần.
3. Hệ số nhân cấp cầu theo số trụ k cần bao phủ đồng thời.
4. CV theo nguồn dữ liệu cho Eq. 2.

Chạy:  python BaiA/revision_analyses.py
Ghi:   BaiA/outputs/rev_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import week3_xgboost as w3  # noqa: E402
from week4_conformal_extrap import q_weighted, q_bridgemax  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
OUT = HERE / "outputs"
d = w3.load(); n = len(d)
d["Fr"] = d["V_ms"] / np.sqrt(9.81 * d["y_m"]); d["x_log_Fr"] = np.log(d["Fr"])
g_all = d["site_group"].to_numpy(); ys_all = d["ys_m"].to_numpy()
rng0 = np.random.default_rng(7)


def boot_med(v, grp, B=1000, seed=1):
    """Trung vị và KTC 90% bootstrap theo cầu."""
    v = np.asarray(v); grp = np.asarray(grp); ug = np.unique(grp)
    idx = {u: np.where(grp == u)[0] for u in ug}; rng = np.random.default_rng(seed); m = []
    for _ in range(B):
        s = rng.choice(ug, len(ug)); m.append(np.median(np.concatenate([v[idx[u]] for u in s])))
    return np.median(v), np.quantile(m, 0.05), np.quantile(m, 0.95)


# ------------------------------------------------------------------ 2. biến thể Eq. 2
VARIANTS = {
    "E0: 4 terms (current: aD, θ, C, U), all beds": (["x_log_aD", "x_theta", "x_cohesive", "x_bed_unknown"], "all"),
    "E1: aD, θ; cohesionless + unknown": (["x_log_aD", "x_theta"], "noncoh"),
    "E2: aD, θ, ln Fr; cohesionless + unknown": (["x_log_aD", "x_theta", "x_log_Fr"], "noncoh"),
}


def subset(kind):
    return np.ones(n, bool) if kind == "all" else (d["bed"] != "cohesive").to_numpy()


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return b


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def metr(ys, lp):
    p = np.exp(lp); lo, lpp = np.log10(ys), np.log10(p)
    return np.mean(np.abs(ys - p)), 1 - np.sum((lo - lpp) ** 2) / np.sum((lo - lo.mean()) ** 2)


rows, oof_store = [], {}
for name, (cols, kind) in VARIANTS.items():
    m = subset(kind); dd = d[m].reset_index(drop=True); g = dd["site_group"].to_numpy(); nn = len(dd)
    X = dd[cols].to_numpy(float); y = dd["lr"].to_numpy(); lhec = np.log(dd["M0_hec18_kw"].to_numpy())
    ys = dd["ys_m"].to_numpy(); ly = np.log(ys)
    # CV theo cầu 5 × 10
    mae, r2, kw, kb, exc_row, exc_br = [], [], [], [], [], []
    oof = np.zeros((10, nn))
    for r in range(10):
        f = w3.folds(nn, g, 5, 1000 + r, "group"); P = np.empty(nn); UPw = np.empty(nn)
        for i in range(5):
            tr, te = f != i, f == i
            # hiệu chuẩn bằng phần dư ngoài-phần của vòng trong (4 phần theo cầu) trên phần huấn luyện
            gi = g[tr]; fi = w3.folds(tr.sum(), gi, 4, 50 + 7 * r + i, "group"); Xi, yi = X[tr], y[tr]; ri = np.empty(tr.sum())
            for j in range(4):
                bj = ols(Xi[fi != j], yi[fi != j]); ri[fi == j] = yi[fi == j] - pred(bj, Xi[fi == j])
            b = ols(X[tr], y[tr]); P[te] = lhec[te] + pred(b, X[te])
            UPw[te] = P[te] + q_weighted(ri, gi, 0.05)
        oof[r] = ly - P
        a, bb = metr(ys, P); mae.append(a); r2.append(bb)
        kw.append(np.exp(q_weighted(ly - P, g, 0.05))); kb.append(np.exp(q_bridgemax(ly - P, g, 0.05)))
        e = ly > UPw; exc_row.append(e.mean()); exc_br.append(pd.Series(e).groupby(g).any().mean())
    oof_store[name] = (dd, oof.mean(0))
    rows.append({"variant": name, "split": "bridge CV 5×10", "n": nn, "bridges": len(np.unique(g)),
                 "MAE": np.mean(mae), "R2_log": np.mean(r2), "k95_row_weighted": np.mean(kw),
                 "k95_bridgemax": np.mean(kb), "exceed_row_%": 100 * np.mean(exc_row), "exceed_bridge_%": 100 * np.mean(exc_br)})
    # CV theo nguồn (5 phần, gom theo trường source)
    src = dd["source"].fillna("unknown").to_numpy(); mae, r2, er, eb, cov90 = [], [], [], [], []
    for r in range(10):
        f = w3.folds(nn, src, 5, 2000 + r, "group"); P = np.empty(nn); UP = np.empty(nn); HW = np.empty(nn)
        for i in range(5):
            tr, te = f != i, f == i; gi = g[tr]
            fi = w3.folds(tr.sum(), gi, 4, 70 + 7 * r + i, "group"); Xi, yi = X[tr], y[tr]; ri = np.empty(tr.sum())
            for j in range(4):
                bj = ols(Xi[fi != j], yi[fi != j]); ri[fi == j] = yi[fi == j] - pred(bj, Xi[fi == j])
            b = ols(X[tr], y[tr]); P[te] = lhec[te] + pred(b, X[te])
            UP[te] = P[te] + q_weighted(ri, gi, 0.05); HW[te] = q_weighted(np.abs(ri), gi, 0.10)
        a, bb = metr(ys, P); mae.append(a); r2.append(bb); e = ly > UP
        er.append(e.mean()); eb.append(pd.Series(e).groupby(g).any().mean()); cov90.append(np.mean(np.abs(ly - P) <= HW))
    rows.append({"variant": name, "split": "source CV 5×10", "n": nn, "bridges": len(np.unique(g)), "MAE": np.mean(mae),
                 "R2_log": np.mean(r2), "cov90_row": np.mean(cov90), "exceed_row_%": 100 * np.mean(er), "exceed_bridge_%": 100 * np.mean(eb)})
    # 6 phép ngoại suy
    Fr = dd["Fr"]; bn = dd["bn_m"]; D50 = dd["D50_mm"]
    ext = {"USA → other": (dd["country"] == "USA", dd["country"] != "USA"), "bn ≤ 3 → > 5 m": (bn <= 3, bn > 5),
           "bn ≤ 3 → 3–5 m": (bn <= 3, (bn > 3) & (bn <= 5)), "sand → gravel": (D50 < 2, D50 >= 20),
           "gravel → sand": (D50 >= 2, D50 < 2), "Fr ≤ 0.3 → > 0.5": (Fr <= 0.3, Fr > 0.5)}
    for en, (mtr, mte) in ext.items():
        mtr, mte = mtr.to_numpy(), mte.to_numpy(); mtr = mtr & ~np.isin(g, g[mte]); tr, te = np.where(mtr)[0], np.where(mte)[0]
        b = ols(X[tr], y[tr]); a, bb = metr(ys[te], lhec[te] + pred(b, X[te]))
        rows.append({"variant": name, "split": en, "n": len(te), "MAE": a, "R2_log": bb})
var = pd.DataFrame(rows); var.to_csv(OUT / "rev_eq2_variants.csv", index=False)
print(var.round(3).to_string(index=False))

# hệ số và KTC bootstrap theo cầu cho từng biến thể
coef_rows = []
for name, (cols, kind) in VARIANTS.items():
    m = subset(kind); dd = d[m].reset_index(drop=True); X = dd[cols].to_numpy(float); y = dd["lr"].to_numpy(); g = dd["site_group"].to_numpy()
    b = ols(X, y); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(3); B = []
    for _ in range(2000):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); B.append(ols(X[ii], y[ii]))
    B = np.array(B)
    for k, c in enumerate(["const"] + cols):
        coef_rows.append({"variant": name, "term": c, "coef": b[k], "lo95": np.quantile(B[:, k], 0.025), "hi95": np.quantile(B[:, k], 0.975)})
coef = pd.DataFrame(coef_rows); coef.to_csv(OUT / "rev_eq2_coefficients.csv", index=False); print(coef.round(3).to_string(index=False))

# phần dư theo Fr (OOF, theo cầu)
fr_rows = []
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
for name, (dd, res) in oof_store.items():
    cls = pd.cut(dd["Fr"], bins); rho, p = stats.spearmanr(dd["Fr"], res)
    for c, grp in pd.Series(res).groupby(cls.to_numpy(), observed=True):
        mm, lo, hi = boot_med(grp.to_numpy(), dd["site_group"].to_numpy()[grp.index], B=500)
        fr_rows.append({"variant": name, "Fr_class": str(c), "n": len(grp), "median_resid": mm, "lo90": lo, "hi90": hi,
                        "spearman_rho": rho, "spearman_p": p})
frr = pd.DataFrame(fr_rows); frr.to_csv(OUT / "rev_resid_by_Fr.csv", index=False); print(frr.round(3).to_string(index=False))

# ------------------------------------------------------------------ 1. thời điểm đo và cơ chế thủy lực
FINAL = "E2: aD, θ, ln Fr; cohesionless + unknown"
ddF, resF = oof_store[FINAL]
d["ratio"] = d["ys_m"] / d["M0_hec18_kw"]
d["resid_final"] = np.nan; d.loc[(d["bed"] != "cohesive").to_numpy(), "resid_final"] = resF
d["meth"] = np.where(d["meas_method"] == "Ground penetrating radar", "GPR (detects infilled holes)",
                     np.where(d["meas_method"].isin(["Soundings", "Fathometer", "ADCP"]), "Soundings/fathometer/ADCP",
                              d["meas_method"].fillna("not reported")))
d["RI_class"] = pd.cut(d["RI_yr"], [0, 2, 10, 50, 1000], labels=["≤2 yr", "2–10 yr", "10–50 yr", ">50 yr"]).astype(str).replace("nan", "not reported")
d["D50_class"] = pd.cut(d["D50_mm"], [0, 2, 20, 1000], labels=["sand <2 mm", "gravel 2–20 mm", "coarse ≥20 mm"]).astype(str)
d["grad"] = np.where(d["D50_mm"] < 2, "sand", np.where(d["sigma_g"].isna(), "gravel, σg not reported",
                     np.where(d["sigma_g"] >= 3, "gravel, σg ≥ 3 (armour-prone)", "gravel, σg < 3")))
strat = []
for col, lab in [("meas_type", "timing"), ("meth", "method"), ("RI_class", "recurrence interval"), ("grad", "gradation")]:
    for k, grp in d.groupby(d[col].fillna("not reported")):
        mr = boot_med(grp["ratio"], grp["site_group"], B=500)
        rr = grp["resid_final"].dropna()
        mres = boot_med(rr, grp.loc[rr.index, "site_group"], B=500) if len(rr) > 5 else (np.nan,) * 3
        strat.append({"factor": lab, "class": k, "n": len(grp), "bridges": grp["site_group"].nunique(),
                      "ratio_med": mr[0], "ratio_lo90": mr[1], "ratio_hi90": mr[2], "underpred_%": 100 * (grp["ratio"] > 1).mean(),
                      "resid_factor_med": np.exp(mres[0]), "resid_lo90": np.exp(mres[1]), "resid_hi90": np.exp(mres[2])})
st = pd.DataFrame(strat); st.to_csv(OUT / "rev_strat_timing.csv", index=False); print(st.round(3).to_string(index=False))

# a/D50 trong từng nhóm cỡ hạt
aD_rows = []
for dc, grp in d.groupby("D50_class"):
    rho, p = stats.spearmanr(grp["x_log_aD"], np.log(grp["ratio"]))
    for c, gg in grp.groupby(pd.qcut(grp["x_log_aD"], 3, duplicates="drop"), observed=True):
        aD_rows.append({"D50_class": dc, "aD_range": f"{np.exp(c.left):.0f}–{np.exp(c.right):.0f}", "n": len(gg),
                        "ratio_med": gg["ratio"].median(), "spearman_rho": rho, "spearman_p": p})
aDt = pd.DataFrame(aD_rows); aDt.to_csv(OUT / "rev_aD_within_D50.csv", index=False); print(aDt.round(3).to_string(index=False))

# độ nhạy: Eq. 2 thêm biến thời điểm đo — hệ số a/D50, Fr có đổi không?
dd = ddF.copy()
for k in ["Historical", "Post Flood"]:
    dd[f"t_{k}"] = (dd["meas_type"] == k).astype(float)
dd["t_na"] = dd["meas_type"].isna().astype(float)
cols = VARIANTS[FINAL][0]; b0 = ols(dd[cols].to_numpy(float), dd["lr"].to_numpy())
b1 = ols(dd[cols + ["t_Historical", "t_Post Flood", "t_na"]].to_numpy(float), dd["lr"].to_numpy())
sens = pd.DataFrame({"term": ["const"] + cols + ["Historical", "Post Flood", "timing NA"],
                     "without_timing": list(b0) + [np.nan] * 3, "with_timing": b1})
sens.to_csv(OUT / "rev_eq2_timing_sensitivity.csv", index=False); print(sens.round(3).to_string(index=False))

# ------------------------------------------------------------------ 3. hệ số nhân theo số trụ k
dd, res = ddF, resF
g = dd["site_group"].to_numpy(); pg = dd["pier_group"].to_numpy()
# một giá trị dư cho mỗi trụ: lấy ngẫu nhiên một lần đo/trụ ở mỗi lần rút
piers = pd.DataFrame({"g": g, "p": pg, "r": res})
by_bridge = {b: grp.groupby("p")["r"].apply(np.array).tolist() for b, grp in piers.groupby("g")}
npier = pd.Series({b: len(v) for b, v in by_bridge.items()})
# mô hình hiệu ứng ngẫu nhiên theo cầu: r = u_b + e
pm = piers.groupby(["g", "p"])["r"].mean().reset_index()
bm = pm.groupby("g")["r"].agg(["mean", "count"]); within = pm.merge(bm, left_on="g", right_index=True)
s2_w = np.sum((within["r"] - within["mean"]) ** 2) / max(1, (len(within) - len(bm)))
s2_b = max(0.0, bm["mean"].var() - s2_w / bm["count"].mean())
mu = pm["r"].mean()
k_rows = []
for k in [1, 2, 3, 4, 5, 6, 8, 10]:
    elig = [b for b, v in by_bridge.items() if len(v) >= k]; mx = []
    if len(elig) >= 15:
        for _ in range(200):
            for b in elig:
                v = by_bridge[b]; sel = rng0.choice(len(v), k, replace=False)
                mx.append(max(rng0.choice(v[s]) for s in sel))
        emp = np.exp(np.quantile(mx, 0.95))
    else:
        emp = np.nan
    sim = mu + rng0.normal(0, np.sqrt(s2_b), (200000, 1)) + rng0.normal(0, np.sqrt(s2_w), (200000, k))
    k_rows.append({"k_piers": k, "bridges_with_≥k_piers": len(elig), "multiplier_empirical": emp,
                   "multiplier_random_effects": np.exp(np.quantile(sim.max(1), 0.95)),
                   "indep_bridge_exceed_of_single_pier_bound_%": 100 * (1 - 0.95 ** k)})
kt = pd.DataFrame(k_rows); kt.to_csv(OUT / "rev_k_multipliers.csv", index=False)
print(f"variance components: between-bridge {s2_b:.3f}, within-bridge {s2_w:.3f}, ICC {s2_b / (s2_b + s2_w):.2f}")
print(kt.round(3).to_string(index=False))

# vượt cấp cầu của cận một trụ, theo số trụ thực có trong dữ liệu
k1 = np.exp(np.quantile(pm["r"], 0.95))
ex = piers.assign(e=piers["r"] > np.log(k1)).groupby("g")["e"].any()
cls = pd.cut(npier, [0, 1, 2, 4, 100], labels=["1", "2", "3–4", "≥5"])
obs = pd.DataFrame({"exceed": ex, "cls": cls}).groupby("cls", observed=True)["exceed"].agg(["mean", "count"])
obs["mean"] *= 100; obs.to_csv(OUT / "rev_bridge_exceed_by_k.csv"); print(obs.round(2))
print("rows/bridge distribution of piers:", npier.value_counts().sort_index().to_dict())

# ---- bổ sung: đường cong k bán tham số (u_b từ trung bình cầu, e từ độ lệch trong cầu) và vượt cận theo phương pháp đo
pmm = pm.merge(bm[["mean"]], left_on="g", right_index=True)
dev = (pmm["r"] - pmm["mean"])[pmm["g"].map(bm["count"]) >= 2].to_numpy()
dev = dev * np.sqrt(bm["count"][bm["count"] >= 2].mean() / max(1, bm["count"][bm["count"] >= 2].mean() - 1))
ub = bm["mean"].to_numpy()
semi = []
for k in [1, 2, 3, 4, 5, 6, 8, 10]:
    U = rng0.choice(ub, 100000)[:, None] * np.sqrt(max(0.0, 1 - (s2_w / bm["count"].mean()) / bm["mean"].var())) \
        + (1 - np.sqrt(max(0.0, 1 - (s2_w / bm["count"].mean()) / bm["mean"].var()))) * ub.mean()
    E = rng0.choice(dev, (100000, k))
    semi.append(np.exp(np.quantile((U + E).max(1), 0.95)))
kt["multiplier_semiparametric"] = semi; kt.to_csv(OUT / "rev_k_multipliers.csv", index=False); print(kt.round(2).to_string(index=False))
up1 = np.log(2.625)
for meth, grp in d.loc[d["resid_final"].notna()].groupby("meth"):
    print(f"{meth}: n={len(grp)}, exceed per-pier 95% bound {100*(grp['resid_final'] > up1).mean():.1f}%")
gpr = (ddF["meas_method"] == "Ground penetrating radar").to_numpy()
print("GPR sources:", ddF.loc[gpr, "source"].value_counts().to_dict())
print("q95 multiplier GPR:", round(np.exp(np.quantile(resF[gpr], 0.95)), 2), "non-GPR:", round(np.exp(np.quantile(resF[~gpr], 0.95)), 2))
print("median factor GPR:", round(np.exp(np.median(resF[gpr])), 2), "non-GPR:", round(np.exp(np.median(resF[~gpr])), 2),
      "| GPR share sand:", round((ddF.loc[gpr, "D50_mm"] < 2).mean(), 2))
q90w = q_weighted(np.abs(resF), ddF["site_group"].to_numpy(), 0.10); q90b = q_bridgemax(np.abs(resF), ddF["site_group"].to_numpy(), 0.10)
b = ols(ddF[VARIANTS[FINAL][0]].to_numpy(float), ddF["lr"].to_numpy()); fac = np.exp(pred(b, ddF[VARIANTS[FINAL][0]].to_numpy(float)))
print("E2 90% interval multipliers: per-pier", round(np.exp(q90w), 2), "bridge", round(np.exp(q90b), 2))
print("factor median", round(np.median(fac), 3), "5-95%", np.round(np.quantile(fac, [0.05, 0.95]), 3))
for kk, mk in [(1, 2.6), (2, 2.9), (4, 3.8)]:
    bnd = fac * mk; print(f"k={kk}: bound/HEC median {np.median(bnd):.2f}, 5-95% {np.quantile(bnd,0.05):.2f}-{np.quantile(bnd,0.95):.2f}, >HEC {100*(bnd>1).mean():.0f}%")
deb = (ddF["debris"] == "Substantial").to_numpy(); print("debris substantial n", deb.sum(), "median obs/pred", round(np.exp(np.median(resF[deb])), 2), "mean", round(np.exp(np.mean(resF[deb])), 2))
print("Fr range 5-95", np.round(np.quantile(ddF["Fr"], [0.05, 0.95]), 3))
