"""Phân tích sửa lần 2 cho JHE (28/11/2026).

1. Eq. 2 có biến chỉ báo phương pháp đo (tham chiếu: hồi âm/fathometer/ADCP); hệ số chặn ứng với GPR
   (thấy cả phần hố đã bồi lấp) dùng làm giá trị thiết kế cho xói cực đại.
2. So sánh GPR với hồi âm trong cùng đáy cát.
3. Dạng rời rạc theo loại đáy (cát / sỏi-cuội) thay cho (a/D50)^b.
4. Cột Eq. 2 cho bảng ngoại suy (tỷ số MAE ngoại suy / MAE cùng dòng khi CV theo cầu).
5. Hệ số nhân theo k từ phần dư ngoài-phần của dạng cuối; đặc điểm các cầu có nhiều trụ được đo.

Chạy:  python BaiA/revision2_analyses.py
Ghi:   BaiA/outputs/rev2_*.csv
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
d = w3.load()
d["Fr"] = d["V_ms"] / np.sqrt(9.81 * d["y_m"]); d["x_log_Fr"] = np.log(d["Fr"])
d = d[d["bed"] != "cohesive"].reset_index(drop=True); n = len(d)
mm = d["meas_method"]
d["m_gpr"] = (mm == "Ground penetrating radar").astype(float)
d["m_survey"] = (mm == "Survey").astype(float)
d["m_rod"] = (mm == "Scour Rod").astype(float)
d["m_na"] = mm.isna().astype(float)
d["b_gravel"] = (d["D50_mm"] >= 2).astype(float)                       # sỏi và cuội
d["b_grav2_20"] = ((d["D50_mm"] >= 2) & (d["D50_mm"] < 20)).astype(float)
d["b_cobble"] = (d["D50_mm"] >= 20).astype(float)
METH = ["m_gpr", "m_survey", "m_rod", "m_na"]
g = d["site_group"].to_numpy(); ys = d["ys_m"].to_numpy(); ly = np.log(ys)
y = d["lr"].to_numpy(); lhec = np.log(d["M0_hec18_kw"].to_numpy())
rng0 = np.random.default_rng(11)

V = {
    "E2: ln(a/D50), θ, ln Fr": ["x_log_aD", "x_theta", "x_log_Fr"],
    "E2m: E2 + method": ["x_log_aD", "x_theta", "x_log_Fr"] + METH,
    "E3: bed class (sand/gravel), θ, ln Fr": ["b_gravel", "x_theta", "x_log_Fr"],
    "E3m: E3 + method": ["b_gravel", "x_theta", "x_log_Fr"] + METH,
    "E3c: 3 bed classes, θ, ln Fr, method": ["b_grav2_20", "b_cobble", "x_theta", "x_log_Fr"] + METH,
    "E3g: E3 + GPR indicator only": ["b_gravel", "x_theta", "x_log_Fr", "m_gpr"],
}


def ols(X, yy):
    A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, yy, rcond=None); return b


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def metr(o, lp):
    p = np.exp(lp); lo, lpp = np.log10(o), np.log10(p)
    return np.mean(np.abs(o - p)), 1 - np.sum((lo - lpp) ** 2) / np.sum((lo - lo.mean()) ** 2)


def cv(cols, kind="group", seed0=1000, reps=10):
    X = d[cols].to_numpy(float); grp = g if kind == "group" else d["source"].fillna("unknown").to_numpy()
    out = {"MAE": [], "R2": [], "k95": [], "cov90": [], "exc": [], "excb": []}; oof = np.zeros((reps, n))
    for r in range(reps):
        f = w3.folds(n, grp, 5, seed0 + r, "group"); P = np.empty(n); UP = np.empty(n); HW = np.empty(n)
        for i in range(5):
            tr, te = f != i, f == i; gi = g[tr]
            fi = w3.folds(tr.sum(), gi, 4, 50 + 7 * r + i, "group"); Xi, yi = X[tr], y[tr]; ri = np.empty(tr.sum())
            for j in range(4):
                bj = ols(Xi[fi != j], yi[fi != j]); ri[fi == j] = yi[fi == j] - pred(bj, Xi[fi == j])
            b = ols(X[tr], y[tr]); P[te] = lhec[te] + pred(b, X[te])
            UP[te] = P[te] + q_weighted(ri, gi, 0.05); HW[te] = q_weighted(np.abs(ri), gi, 0.10)
        oof[r] = ly - P; a, bb = metr(ys, P)
        out["MAE"].append(a); out["R2"].append(bb); out["k95"].append(np.exp(q_weighted(ly - P, g, 0.05)))
        out["cov90"].append(np.mean(np.abs(ly - P) <= HW)); e = ly > UP
        out["exc"].append(100 * e.mean()); out["excb"].append(100 * pd.Series(e).groupby(g).any().mean())
    return {k: np.mean(v) for k, v in out.items()}, oof.mean(0)


rows, coefs, oofs = [], [], {}
Fr = d["Fr"]; bn = d["bn_m"]; D50 = d["D50_mm"]
EXT = {"USA → other": (d["country"] == "USA", d["country"] != "USA"), "bn ≤ 3 → 3–5 m": (bn <= 3, (bn > 3) & (bn <= 5)),
       "bn ≤ 3 → > 5 m": (bn <= 3, bn > 5), "sand → gravel/cobble": (D50 < 2, D50 >= 20),
       "gravel → sand": (D50 >= 2, D50 < 2), "Fr ≤ 0.3 → > 0.5": (Fr <= 0.3, Fr > 0.5)}
for name, cols in V.items():
    rb, oof = cv(cols, "group"); oofs[name] = oof
    rs, _ = cv(cols, "source", 2000)
    rho, p = stats.spearmanr(d["Fr"], oof)
    row = {"variant": name, "bridge_MAE": rb["MAE"], "bridge_R2": rb["R2"], "k95_single_pier": rb["k95"],
           "bridge_exceed_row_%": rb["exc"], "source_MAE": rs["MAE"], "source_R2": rs["R2"], "source_cov90": rs["cov90"],
           "source_exceed_row_%": rs["exc"], "source_exceed_bridge_%": rs["excb"], "rho_resid_Fr": rho, "p_resid_Fr": p}
    X = d[cols].to_numpy(float)
    for en, (mtr, mte) in EXT.items():
        mtr, mte = mtr.to_numpy(), mte.to_numpy(); mtr = mtr & ~np.isin(g, g[mte]); tr, te = np.where(mtr)[0], np.where(mte)[0]
        Xtr = X[tr]; keep = Xtr.std(0) > 0                     # biến hằng trên tập huấn luyện → không ước lượng được
        bedcols = [j for j, c in enumerate(cols) if c.startswith("b_")]
        if any((not keep[j]) and np.any(X[te][:, j] != Xtr[0, j]) for j in bedcols):
            row[f"ext_{en}"] = np.nan; continue                 # loại đáy chưa có trong tập huấn luyện
        b = ols(Xtr[:, keep], y[tr]); lp = lhec[te] + pred(b, X[te][:, keep])
        mae_ext = np.mean(np.abs(ys[te] - np.exp(lp)))
        mae_in = np.mean(np.abs(ys[te] - np.exp(ly[te] - oof[te])))
        row[f"ext_{en}"] = mae_ext / mae_in; row[f"extMAE_{en}"] = mae_ext
    rows.append(row)
    b = ols(X, y); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(3); B = []
    for _ in range(2000):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); B.append(ols(X[ii], y[ii]))
    B = np.array(B)
    for k, c in enumerate(["const"] + cols):
        coefs.append({"variant": name, "term": c, "coef": b[k], "lo95": np.quantile(B[:, k], 0.025), "hi95": np.quantile(B[:, k], 0.975)})
res = pd.DataFrame(rows); res.to_csv(OUT / "rev2_variants.csv", index=False); print(res.round(3).T.to_string())
cf = pd.DataFrame(coefs); cf.to_csv(OUT / "rev2_coefficients.csv", index=False); print(cf.round(3).to_string(index=False))

# GPR so với hồi âm trong cùng đáy cát
sand = d["D50_mm"] < 2
ratio = d["ys_m"] / d["M0_hec18_kw"]
cmp = []
for lab, msk in [("sand, GPR", sand & (d["m_gpr"] == 1)), ("sand, soundings", sand & mm.isin(["Soundings", "Fathometer", "ADCP"])),
                 ("gravel/cobble, GPR", ~sand & (d["m_gpr"] == 1)), ("gravel/cobble, soundings", ~sand & mm.isin(["Soundings", "Fathometer", "ADCP"]))]:
    cmp.append({"group": lab, "n": int(msk.sum()), "bridges": d.loc[msk, "site_group"].nunique(), "ratio_median": ratio[msk].median(),
                "meas_over_E2_median": np.exp(np.median(oofs["E2: ln(a/D50), θ, ln Fr"][msk.to_numpy()])) if msk.sum() else np.nan})
cmp = pd.DataFrame(cmp); cmp.to_csv(OUT / "rev2_gpr_vs_soundings.csv", index=False); print(cmp.round(3).to_string(index=False))
# hệ số GPR khi chỉ khớp trên đáy cát
ds = d[sand].reset_index(drop=True); cols_s = ["x_theta", "x_log_Fr"] + METH
bs = ols(ds[cols_s].to_numpy(float), ds["lr"].to_numpy()); print("sand-only fit, GPR coefficient:", round(bs[1 + cols_s.index("m_gpr")], 3))

# ---------------------------------------------------------------- dạng cuối: E3m (hoặc E2m) — hệ số nhân và so với HEC-18
FINAL = "E3g: E3 + GPR indicator only"
cols = V[FINAL]; X = d[cols].to_numpy(float); b = ols(X, y); r = oofs[FINAL]
bmap = dict(zip(["const"] + cols, b))
MC = [c for c in METH if c in cols]
Xmax = X.copy(); Xmax[:, [cols.index(c) for c in MC]] = 0; Xmax[:, cols.index("m_gpr")] = 1   # mức GPR = xói cực đại
fac_max = np.exp(pred(b, Xmax)); Xs = X.copy(); Xs[:, [cols.index(c) for c in MC]] = 0; fac_snd = np.exp(pred(b, Xs))
k1 = np.exp(q_weighted(r, g, 0.05))
pg = d["pier_group"].to_numpy(); piers = pd.DataFrame({"g": g, "p": pg, "r": r})
by_b = {bb: grp.groupby("p")["r"].apply(np.array).tolist() for bb, grp in piers.groupby("g")}
pm = piers.groupby(["g", "p"])["r"].mean().reset_index(); bm = pm.groupby("g")["r"].agg(["mean", "count"])
w_ = pm.merge(bm, left_on="g", right_index=True); s2w = np.sum((w_["r"] - w_["mean"]) ** 2) / (len(w_) - len(bm))
s2b = max(0.0, bm["mean"].var() - s2w / bm["count"].mean()); mu = pm["r"].mean()
kt = []
for k in [1, 2, 3, 4, 5, 6, 8, 10]:
    el = [bb for bb, v in by_b.items() if len(v) >= k]; mx = []
    if len(el) >= 15:
        for _ in range(200):
            for bb in el:
                v = by_b[bb]; sel = rng0.choice(len(v), k, replace=False); mx.append(max(rng0.choice(v[s]) for s in sel))
        emp = np.exp(np.quantile(mx, 0.95))
    else:
        emp = np.nan
    sim = mu + rng0.normal(0, np.sqrt(s2b), (200000, 1)) + rng0.normal(0, np.sqrt(s2w), (200000, k))
    kt.append({"k": k, "bridges": len(el), "empirical": emp, "random_effects": np.exp(np.quantile(sim.max(1), 0.95)),
               "fit_k03": k1 * k ** 0.3})
kt = pd.DataFrame(kt); kt.to_csv(OUT / "rev2_k_multipliers.csv", index=False); print(kt.round(2).to_string(index=False))
print(f"ICC {s2b / (s2b + s2w):.2f}; k1 = {k1:.2f}")
summ = []
for lab, fac in [("soundings level (descriptive)", fac_snd), ("GPR level (maximum scour, design)", fac_max)]:
    for kk in [1, 4]:
        mk = kt.set_index("k").loc[kk, "empirical"]; bnd = fac * mk
        for bed, msk in [("all", np.ones(n, bool)), ("sand", sand.to_numpy()), ("gravel/cobble", ~sand.to_numpy())]:
            summ.append({"level": lab, "k": kk, "bed": bed, "median_factor": np.median(fac[msk]), "multiplier": mk,
                         "bound_over_HEC_median": np.median(bnd[msk]), "p05": np.quantile(bnd[msk], 0.05),
                         "p95": np.quantile(bnd[msk], 0.95), "share_above_HEC_%": 100 * np.mean(bnd[msk] > 1)})
summ = pd.DataFrame(summ); summ.to_csv(OUT / "rev2_bound_vs_hec.csv", index=False); print(summ.round(2).to_string(index=False))
# kiểm tra: tỷ lệ bản ghi GPR vượt cận một trụ khi dùng mức GPR
e_gpr = (ly - (lhec + np.log(fac_max)))[d["m_gpr"] == 1] > np.log(k1)
print(f"GPR records above single-pier bound at GPR level: {100 * e_gpr.mean():.1f}% (n={e_gpr.size})")
deb = (d["debris"] == "Substantial").to_numpy(); print("debris median measured/corrected (own level):", round(np.exp(np.median(r[deb])), 2))

# đặc điểm cầu theo số trụ được đo (thiên lệch chọn mẫu khi ước lượng hệ số theo k)
npier = piers.groupby("g")["p"].nunique(); bridge = d.groupby("site_group").agg(bn=("bn_m", "median"), y=("y_m", "median"),
         D50=("D50_mm", "median"), V=("V_ms", "median"), ratio=("ys_m", lambda s: np.nan))
bridge["ratio"] = (d["ys_m"] / d["M0_hec18_kw"]).groupby(d["site_group"]).median(); bridge["npier"] = npier
bridge["cls"] = pd.cut(bridge["npier"], [0, 1, 2, 100], labels=["1 pier", "2 piers", "≥3 piers"])
sb = bridge.groupby("cls", observed=True)[["bn", "y", "D50", "V", "ratio"]].median(); sb["bridges"] = bridge.groupby("cls", observed=True).size()
sb.to_csv(OUT / "rev2_bridges_by_npier.csv"); print(sb.round(2))
print("final coefficients:", {k: round(v, 3) for k, v in bmap.items()})

# kiểm tra trực tiếp: phân vị 95 của tỷ số đo/HEC-18 theo đáy và phương pháp
for lab, msk in [("sand GPR", sand & (d["m_gpr"] == 1)), ("sand soundings", sand & mm.isin(["Soundings", "Fathometer", "ADCP"])),
                 ("gravel soundings", ~sand & mm.isin(["Soundings", "Fathometer", "ADCP"]))]:
    print(f"{lab}: n={int(msk.sum())}, c95 = q95(measured/HEC) = {ratio[msk].quantile(0.95):.2f}, exceed HEC {100*(ratio[msk] > 1).mean():.1f}%")
# bán tham số: hiệu ứng cầu lấy mẫu lại từ trung bình cầu (co về theo phương sai giữa cầu), độ lệch trong cầu lấy mẫu lại
pmm = pm.merge(bm[["mean", "count"]], left_on="g", right_index=True)
multi = pmm["count"] >= 2
dev = (pmm["r"] - pmm["mean"])[multi].to_numpy(); nb = pmm.loc[multi, "count"].to_numpy()
dev = dev * np.sqrt(nb / (nb - 1))                                       # bù bậc tự do
shrink = np.sqrt(s2b / max(bm["mean"].var(), 1e-9)); ub = mu + shrink * (bm["mean"].to_numpy() - mu)
semi = []
for k in kt["k"]:
    U = rng0.choice(ub, 200000)[:, None]; E = rng0.choice(dev, (200000, int(k)))
    semi.append(np.exp(np.quantile((U + E).max(1), 0.95)))
kt["semiparametric"] = semi; kt.to_csv(OUT / "rev2_k_multipliers.csv", index=False); print(kt.round(2).to_string(index=False))
for kk in [1, 2, 4]:
    mk = kt.set_index("k").loc[kk, "semiparametric"]
    for bed, msk in [("sand", sand.to_numpy()), ("gravel/cobble", ~sand.to_numpy())]:
        print(f"k={kk} {bed}: GPR-level bound/HEC median {np.median(fac_max[msk] * mk):.2f}; soundings-level {np.median(fac_snd[msk] * mk):.2f}")

# cột "đo/hiệu chỉnh" cho bảng phân tầng, theo dạng rời rạc không có biến phương pháp (E3)
rE3 = oofs["E3: bed class (sand/gravel), θ, ln Fr"]
cls = {"timing": d["meas_type"].fillna("not reported"),
       "method": np.where(d["m_gpr"] == 1, "GPR", np.where(mm.isin(["Soundings", "Fathometer", "ADCP"]), "soundings", mm.fillna("not reported"))),
       "RI": pd.cut(d["RI_yr"], [0, 2, 10, 50, 1000], labels=["≤2", "2–10", "10–50", ">50"]).astype(str),
       "bed": np.where(d["D50_mm"] < 2, "sand", np.where(d["sigma_g"].isna(), "gravel n/r", np.where(d["sigma_g"] >= 3, "gravel σg≥3", "gravel σg<3")))}
rowsS = []
for f, c in cls.items():
    for k_, grp in pd.Series(rE3).groupby(np.asarray(c)):
        rowsS.append({"factor": f, "class": k_, "n": len(grp), "meas_over_E3": np.exp(np.median(grp))})
pd.DataFrame(rowsS).to_csv(OUT / "rev2_strat_E3.csv", index=False); print(pd.DataFrame(rowsS).round(2).to_string(index=False))
# Ngòi Thia (đáy sỏi, G = 1) theo dạng cuối
import equations as eq
for V_ in [1.3, 2.6]:
    for th in [0, 15]:
        S = lambda v: pd.Series([v]); h = float(eq.hec18_csu(S(1.4), S(6.0), S(th), S("round"), S(8.54), S(V_), S(17.0), use_kw=True)[0])
        Fr_ = V_ / np.sqrt(9.81 * 8.54)
        base = np.exp(bmap["const"] + bmap["b_gravel"] + bmap["x_theta"] * th / 90 + bmap["x_log_Fr"] * np.log(Fr_))
        print(f"NT V={V_} th={th}: HEC {h:.2f} median(meas) {h*base:.2f} median(max) {h*base*np.exp(bmap['m_gpr']):.2f} "
              f"bound1(max) {h*base*np.exp(bmap['m_gpr'])*2.6:.2f} bound4(max) {h*base*np.exp(bmap['m_gpr'])*2.6*4**0.3:.2f}")
