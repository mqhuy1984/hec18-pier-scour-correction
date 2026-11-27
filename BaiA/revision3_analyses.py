"""Phân tích sửa lần 3 cho JHE (28/11/2026).

Eq. 2 = dạng theo loại đáy (G, θ, ln Fr), mức trung vị như thường đo (sounding).
Eq. 3 = bước lấp hố riêng: y_max = F · y_s,c, F từ biến chỉ báo GPR (1,61; KTC 1,39–1,87), chỉ cho đáy cát.
Hệ số nhân cho xói cực đại: hiệu chuẩn Mondrian riêng trên các bản ghi GPR (phần dư so với F · Eq. 2),
kiểm tra bằng CV theo cầu trong nhóm GPR. Hệ số nhân cho k trụ (dạng Eq. 2).

Chạy:  python BaiA/revision3_analyses.py
Ghi:   BaiA/outputs/rev3_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import week3_xgboost as w3  # noqa: E402
from week4_conformal_extrap import q_weighted  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
OUT = HERE / "outputs"
d = w3.load(); d["Fr"] = d["V_ms"] / np.sqrt(9.81 * d["y_m"]); d["x_log_Fr"] = np.log(d["Fr"])
d = d[d["bed"] != "cohesive"].reset_index(drop=True); n = len(d)
d["b_gravel"] = (d["D50_mm"] >= 2).astype(float); d["m_gpr"] = (d["meas_method"] == "Ground penetrating radar").astype(float)
g = d["site_group"].to_numpy(); ys = d["ys_m"].to_numpy(); ly = np.log(ys); y = d["lr"].to_numpy()
lhec = np.log(d["M0_hec18_kw"].to_numpy()); sand = (d["D50_mm"] < 2).to_numpy(); gpr = d["m_gpr"].to_numpy() == 1
COLS = ["b_gravel", "x_theta", "x_log_Fr"]; X = d[COLS].to_numpy(float)
rng0 = np.random.default_rng(5)


def ols(A, b):
    M = np.column_stack([np.ones(len(A)), A]); c, *_ = np.linalg.lstsq(M, b, rcond=None); return c


def pred(c, A):
    return np.column_stack([np.ones(len(A)), A]) @ c


# out-of-fold dự báo log của Eq. 2 (CV theo cầu, 10 lần lặp) và hệ số lấp hố trong cùng phần huấn luyện
oof = np.zeros((10, n)); Foof = np.zeros((10, n))
XG = d[COLS + ["m_gpr"]].to_numpy(float)
for r in range(10):
    f = w3.folds(n, g, 5, 1000 + r, "group")
    for i in range(5):
        tr, te = f != i, f == i
        oof[r, te] = lhec[te] + pred(ols(X[tr], y[tr]), X[te])
        Foof[r, te] = ols(XG[tr], y[tr])[-1]                      # hệ số GPR học trên phần huấn luyện
P = oof.mean(0); lF = Foof.mean(0)
res = ly - P                                                       # so với mức như thường đo
c_all = ols(X, y); cG = ols(XG, y)
F = np.exp(cG[-1])
# KTC bootstrap theo cầu cho F
ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; Fs = []
for _ in range(2000):
    ii = np.concatenate([idx[u] for u in rng0.choice(ug, len(ug))]); Fs.append(np.exp(ols(XG[ii], y[ii])[-1]))
F_lo, F_hi = np.quantile(Fs, [0.025, 0.975])
print(f"Eq. 2 coefficients (const, G, θ/90, ln Fr): {np.round(c_all, 3)}; refill factor F = {F:.2f} [{F_lo:.2f}, {F_hi:.2f}]")

# hệ số nhân thường (mức như thường đo): phân vị 95 có trọng số theo cầu của phần dư ngoài-phần
k1_meas = np.exp(q_weighted(res, g, 0.05))
# Mondrian cho xói cực đại: phần dư của các bản ghi GPR so với F · Eq. 2
rg = (res - lF)[gpr]; gg = g[gpr]
k1_max = np.exp(q_weighted(rg, gg, 0.05))
# kiểm tra ngoài-mẫu: CV theo cầu trong nhóm GPR (hiệu chuẩn trên các cầu GPR khác)
ugg = np.unique(gg); exc = []
for r in range(50):
    perm = rng0.permutation(ugg); fold = {u: j % 5 for j, u in enumerate(perm)}; ff = np.array([fold[u] for u in gg])
    for i in range(5):
        q = q_weighted(rg[ff != i], gg[ff != i], 0.05); exc.append(np.mean(rg[ff == i] > q))
cv_exc = 100 * np.mean(exc)
print(f"single-pier multiplier: as measured {k1_meas:.2f}; maximum scour (Mondrian, GPR) {k1_max:.2f}; "
      f"out-of-bridge exceedance within GPR {cv_exc:.1f}%")

# so sánh với HEC-18 cho đáy cát, mức xói cực đại
fac_meas = np.exp(pred(c_all, X)); fac_max = fac_meas * F
rows = []
for lab, msk in [("sand, all records", sand), ("sand, GPR sites", sand & gpr)]:
    b1 = fac_max[msk] * k1_max
    rows.append({"group": lab, "n": int(msk.sum()), "median_factor_max": np.median(fac_max[msk]),
                 "bound1_over_HEC_median": np.median(b1), "p05": np.quantile(b1, 0.05), "p95": np.quantile(b1, 0.95),
                 "share_bound_above_HEC_%": 100 * np.mean(b1 > 1)})
ratio = ys / np.exp(lhec)
m = sand & gpr
rows.append({"group": "measured GPR sand: q95 of measured/HEC", "n": int(m.sum()), "bound1_over_HEC_median": np.quantile(ratio[m], 0.95),
             "share_bound_above_HEC_%": 100 * np.mean(ratio[m] > 1)})
# xói cực đại đo được (GPR, cát) vượt cận một trụ (trong mẫu)
b_rec = np.exp(P + lF) * k1_max
rows.append({"group": "GPR sand records above max-scour single-pier bound (in-sample, %)", "n": int(m.sum()),
             "share_bound_above_HEC_%": 100 * np.mean(ys[m] > b_rec[m])})
cmp = pd.DataFrame(rows); cmp.to_csv(OUT / "rev3_maxscour_bounds.csv", index=False); print(cmp.round(3).to_string(index=False))

# hệ số nhân theo k (dạng Eq. 2, mức như thường đo): thực nghiệm + bán tham số
pg = d["pier_group"].to_numpy(); piers = pd.DataFrame({"g": g, "p": pg, "r": res})
by_b = {b: grp.groupby("p")["r"].apply(np.array).tolist() for b, grp in piers.groupby("g")}
pm = piers.groupby(["g", "p"])["r"].mean().reset_index(); bm = pm.groupby("g")["r"].agg(["mean", "count"])
w_ = pm.merge(bm, left_on="g", right_index=True); s2w = np.sum((w_["r"] - w_["mean"]) ** 2) / (len(w_) - len(bm))
s2b = max(0.0, bm["mean"].var() - s2w / bm["count"].mean()); mu = pm["r"].mean()
multi = w_["count"] >= 2; dev = (w_["r"] - w_["mean"])[multi].to_numpy(); nb = w_.loc[multi, "count"].to_numpy(); dev *= np.sqrt(nb / (nb - 1))
shrink = np.sqrt(s2b / bm["mean"].var()); ub = mu + shrink * (bm["mean"].to_numpy() - mu)
kt = []
for k in [1, 2, 3, 4, 5, 6, 8, 10]:
    el = [b for b, v in by_b.items() if len(v) >= k]; mx = []
    if len(el) >= 15:
        for _ in range(200):
            for b in el:
                v = by_b[b]; sel = rng0.choice(len(v), k, replace=False); mx.append(max(rng0.choice(v[s]) for s in sel))
        emp = np.exp(np.quantile(mx, 0.95))
    else:
        emp = np.nan
    U = rng0.choice(ub, 200000)[:, None]; E = rng0.choice(dev, (200000, k))
    kt.append({"k": k, "bridges": len(el), "empirical": emp, "semiparametric": np.exp(np.quantile((U + E).max(1), 0.95)),
               "fit_2.6k^0.3": 2.6 * k ** 0.3, "max_scour_k1max_k^0.3": k1_max * k ** 0.3})
kt = pd.DataFrame(kt); kt.to_csv(OUT / "rev3_k_multipliers.csv", index=False)
print(f"ICC {s2b / (s2b + s2w):.2f}"); print(kt.round(2).to_string(index=False))
pd.DataFrame([{"F": F, "F_lo": F_lo, "F_hi": F_hi, "k1_meas": k1_meas, "k1_max": k1_max, "cv_exc_gpr_%": cv_exc,
               "const": c_all[0], "G": c_all[1], "theta90": c_all[2], "lnFr": c_all[3]}]).to_csv(OUT / "rev3_summary.csv", index=False)
