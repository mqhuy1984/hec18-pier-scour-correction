"""Tuần 5 bài A.
  A. Công thức tường minh "HEC-18 hiệu chỉnh" (M3-S6) + hệ số conformal + miền áp dụng.
  B. SHAP (pred_contribs của XGBoost) cho M4/M5 và độ ổn định thứ hạng.
  C. Độ nhạy theo cờ chất lượng.
  D. RQ4 phần dữ liệu chưa làm sạch (Field_full) × chia ngẫu nhiên.
  E. Khoảng tin cậy bootstrap theo cầu cho các chỉ số ngoại suy của tuần 4.
  F. Thí nghiệm → hiện trường (G6).

Chạy:  python BaiA/week5_formula_sensitivity.py     (KHÔNG dùng -I; cần các đầu ra tuần 1, 4)
Ghi:   BaiA/outputs/week5_*.csv, BaiA/figures/week5_*.png

XGBoost ở mục B, C, D, F dùng siêu tham số cố định (gần trung vị các bộ tìm được ở tuần 3, chia theo cầu)
để tiết kiệm thời gian; M3-S6 không có siêu tham số.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equations as eq  # noqa: E402
import week3_xgboost as w3  # noqa: E402
from week4_conformal_extrap import q_weighted, q_bridgemax, cov_metrics  # noqa: E402

ROOT = HERE.parents[0]
OUT = ROOT / "BaiA" / "outputs"; FIG = ROOT / "BaiA" / "figures"
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
FIXED = dict(n_estimators=220, learning_rate=0.03, max_depth=4, min_child_weight=7, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=1e-3, gamma=1e-3)
rng = np.random.default_rng(2026)


def add_features(d):
    Vc = eq.critical_velocity_hec18(d["y_m"], d["D50_mm"])
    d["x_log_ya"] = np.log(d["y_m"] / d["a_m"])
    d["x_log_Fr"] = np.log(d["V_ms"] / np.sqrt(9.81 * d["y_m"]))
    d["x_log_aD"] = np.log(d["a_m"] / (d["D50_mm"] / 1000))
    d["x_log_VVc"] = np.log(np.clip(d["V_ms"] / Vc, 0.05, 20))
    d["x_theta"] = d["theta0"] / 90
    for k in ["square", "sharp", "circular"]:
        d[f"x_nose_{k}"] = (d["nose"] == k).astype(float)
    d["x_cohesive"] = (d["bed"] == "cohesive").astype(float)
    d["x_bed_unknown"] = d["bed"].isna().astype(float)
    d["x_group"] = (d["pier_type"] == "Group").astype(float)
    d["x_ptype_unknown"] = d["pier_type"].isna().astype(float)
    d["lr"] = np.log(d["ys_m"] / d["M0_hec18_kw"]); d["lya"] = np.log(d["ys_m"] / d["a_m"])
    return d


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return b


def olsp(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def xgb_fit(X, y, mono=True):
    m = "(1" + ",0" * (X.shape[1] - 1) + ")" if mono else None
    return w3.xgb_model(FIXED, m).fit(X, y)


def pm(ys, p):
    lo, lp = np.log10(ys), np.log10(p)
    return {"MAE": np.mean(np.abs(ys - p)), "R2_log": 1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2),
            "median_obs_over_pred": np.median(ys / p), "frac_under": np.mean(ys > p)}


def cv_oof(d, model, folds_fn, reps, train_mask=None):
    """Dự báo OOF (log ys) trung bình qua các lần lặp; train_mask: chỉ dùng các dòng này để huấn luyện."""
    n = len(d); P = np.zeros((reps, n))
    lhec = np.log(d["M0_hec18_kw"].to_numpy()); la = np.log(d["a_m"].to_numpy())
    tm = np.ones(n, bool) if train_mask is None else train_mask
    for r in range(reps):
        f = folds_fn(r)
        for i in range(5):
            tr, te = (f != i) & tm, f == i
            if model == "M3_S6":
                X = d[w3.S6].to_numpy(float); b = ols(X[tr], d["lr"].to_numpy()[tr]); P[r, te] = lhec[te] + olsp(b, X[te])
            elif model == "M4":
                X = d[w3.M4F].to_numpy(float); P[r, te] = la[te] + xgb_fit(X[tr], d["lya"].to_numpy()[tr]).predict(X[te])
            elif model == "M5":
                X = d[w3.S2].to_numpy(float); P[r, te] = lhec[te] + xgb_fit(X[tr], d["lr"].to_numpy()[tr]).predict(X[te])
    return P


d = w3.load()
ys = d["ys_m"].to_numpy(); ly = np.log(ys); g = d["site_group"].to_numpy(); n = len(d)
ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
gfold = lambda r: w3.folds(n, g, 5, 1000 + r, "group")

# ======================================================== A. CÔNG THỨC
print("================ A. CÔNG THỨC HEC-18 HIỆU CHỈNH (M3-S6) ================")
X = d[w3.S6].to_numpy(float); y = d["lr"].to_numpy()
b = ols(X, y)
B = np.array([ols(X[ii], y[ii]) for ii in (np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]) for _ in range(2000))])
TERM_VN = {"x_log_ya": "ln(y/a)", "x_log_aD": "ln(a/D50)", "x_theta": "θ/90", "x_cohesive": "đất dính",
           "x_bed_unknown": "không ghi loại đất"}
names = ["hằng số"] + [TERM_VN[c] for c in w3.S6]
coef = pd.DataFrame({"term": names, "var": ["const"] + list(w3.S6), "coef": b, "lo90": np.quantile(B, 0.05, 0), "hi90": np.quantile(B, 0.95, 0),
                     "lo95": np.quantile(B, 0.025, 0), "hi95": np.quantile(B, 0.975, 0)})
print(coef.round(4).to_string(index=False))

# Hệ số conformal từ phần dư OOF (chia theo cầu, 5 phần × 10 lần)
P = cv_oof(d, "M3_S6", gfold, 10)
conf = []
for r in range(10):
    res = ly - P[r]
    conf.append({"q90_weighted_abs": q_weighted(np.abs(res), g, 0.10), "q95_weighted_up": q_weighted(res, g, 0.05),
                 "q90_bridgemax_abs": q_bridgemax(np.abs(res), g, 0.10), "q95_bridgemax_up": q_bridgemax(res, g, 0.05),
                 "q50_up": np.quantile(res, 0.5)})
conf = pd.DataFrame(conf).agg(["mean", "std"]).T
conf["factor_exp_mean"] = np.exp(conf["mean"])
print("\nHệ số conformal (log và hệ số nhân):\n", conf.round(3).to_string())

# Công thức làm tròn và kiểm tra
br = np.round(b, 2)
lc = olsp(br, X); corr = np.exp(lc)
k95b = float(np.exp(conf.loc["q95_bridgemax_up", "mean"])); k95w = float(np.exp(conf.loc["q95_weighted_up", "mean"]))
hec = d["M0_hec18_kw"].to_numpy()
chk = pd.DataFrame([
    {"model": "HEC-18", **pm(ys, hec), "exceed": np.mean(ys > hec), "pred_over_hec_med": 1.0},
    {"model": "M3-S6 hệ số làm tròn (trung vị)", **pm(ys, hec * corr), "exceed": np.mean(ys > hec * corr),
     "pred_over_hec_med": np.median(corr)},
    {"model": "cận trên 95% theo trụ (weighted)", **pm(ys, hec * corr * k95w), "exceed": np.mean(ys > hec * corr * k95w),
     "pred_over_hec_med": np.median(corr * k95w)},
    {"model": "cận trên 95% theo cầu (bridgemax)", **pm(ys, hec * corr * k95b), "exceed": np.mean(ys > hec * corr * k95b),
     "pred_over_hec_med": np.median(corr * k95b)},
])
pb = pd.DataFrame({"g": g, "o": ys > hec * corr * k95b}).groupby("g")["o"].any().mean()
print("\nKiểm tra trên toàn tập (trong mẫu, chỉ để minh họa):\n", chk.round(3).to_string(index=False))
print(f"Tỷ lệ cầu có ≥1 trụ vượt cận trên bridgemax (trong mẫu): {pb:.3f}")
print("Phân bố hệ số hiệu chỉnh (ys hiệu chỉnh / HEC-18): ", np.quantile(corr, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
print("Phân bố cận trên bridgemax / HEC-18: ", np.quantile(corr * k95b, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3),
      f"; tỷ lệ cận trên > HEC-18: {np.mean(corr * k95b > 1):.3f}")

# Miền áp dụng
Fr = d["V_ms"] / np.sqrt(9.81 * d["y_m"])
dom = pd.DataFrame({"y/a": d["y_m"] / d["a_m"], "a/D50": d["a_m"] / (d["D50_mm"] / 1000), "θ (độ)": d["theta0"],
                    "D50 (mm)": d["D50_mm"], "Fr": Fr, "bn (m)": d["bn_m"], "a (m)": d["a_m"], "y (m)": d["y_m"],
                    "V (m/s)": d["V_ms"], "V/Vc": np.exp(d["x_log_VVc"])})
domain = dom.quantile([0, 0.05, 0.5, 0.95, 1]).T
domain.columns = ["min", "p05", "p50", "p95", "max"]
print("\nMiền dữ liệu hiệu chỉnh:\n", domain.round(3).to_string())
coef.to_csv(OUT / "week5_formula_coefficients.csv", index=False)
conf.to_csv(OUT / "week5_formula_conformal_factors.csv")
chk.to_csv(OUT / "week5_formula_check.csv", index=False)
domain.to_csv(OUT / "week5_domain.csv")

# ======================================================== B. SHAP
print("\n================ B. SHAP (M4, M5) ================")
shap_rows = []
for model, cols, tgt in [("M4", w3.M4F, "lya"), ("M5", w3.S2, "lr")]:
    Xm = d[cols].to_numpy(float); yv = d[tgt].to_numpy()
    for r in range(10):
        f = gfold(r); S = np.zeros((n, len(cols)))
        for i in range(5):
            tr, te = f != i, f == i
            m = xgb_fit(Xm[tr], yv[tr])
            import xgboost as xgb
            S[te] = m.get_booster().predict(xgb.DMatrix(Xm[te]), pred_contribs=True)[:, :-1]
        imp = np.abs(S).mean(0)
        for c, v in zip(cols, imp):
            shap_rows.append({"model": model, "rep": r, "feature": c, "mean_abs_shap": v})
shap = pd.DataFrame(shap_rows)
shap["rank"] = shap.groupby(["model", "rep"])["mean_abs_shap"].rank(ascending=False)
ss = shap.groupby(["model", "feature"]).agg(mean_abs_shap=("mean_abs_shap", "mean"), sd=("mean_abs_shap", "std"),
                                            rank_mean=("rank", "mean"), rank_min=("rank", "min"), rank_max=("rank", "max")).reset_index()
ss = ss.sort_values(["model", "rank_mean"])
print(ss.round(4).to_string(index=False))
# ổn định: tương quan Spearman giữa các lần lặp; và giữa M4/M5 trên biến chung
stab = {}
for model in ["M4", "M5"]:
    piv = shap[shap["model"] == model].pivot(index="feature", columns="rep", values="mean_abs_shap")
    c = piv.rank().corr().to_numpy(); stab[model] = c[np.triu_indices_from(c, 1)].mean()
common = [c for c in w3.S2]
a4 = ss[(ss["model"] == "M4") & ss["feature"].isin(common)].set_index("feature")["mean_abs_shap"]
a5 = ss[(ss["model"] == "M5") & ss["feature"].isin(common)].set_index("feature")["mean_abs_shap"]
rho45 = a4.rank().corr(a5.reindex(a4.index).rank())
print(f"Spearman trung bình giữa các lần lặp: M4 {stab['M4']:.3f}, M5 {stab['M5']:.3f}; giữa M4 và M5 (biến chung): {rho45:.3f}")
ss.to_csv(OUT / "week5_shap_importance.csv", index=False)

# ======================================================== C. CỜ CHẤT LƯỢNG
print("\n================ C. ĐỘ NHẠY THEO CỜ CHẤT LƯỢNG ================")
flags = ["flag_accuracy_gt_half_ys", "flag_bn_inconsistent", "flag_hydraulics_modelled", "flag_cohesive",
         "flag_debris_substantial", "flag_theta_pos_bn_eq_b"]
base = {m: cv_oof(d, m, gfold, 5).mean(0) for m in ["M3_S6", "M5"]}
frows = []
for fl in flags:
    fm = d[fl].fillna(False).astype(bool).to_numpy()
    if fm.sum() == 0: continue
    for m in ["M3_S6", "M5"]:
        # (i) hành vi của các dòng bị gắn cờ trong mô hình cơ sở
        r_f = ly[fm] - base[m][fm]; r_o = ly[~fm] - base[m][~fm]
        # (ii) loại dòng gắn cờ khỏi huấn luyện, đánh giá trên các dòng không gắn cờ
        P2 = cv_oof(d, m, gfold, 5, train_mask=~fm).mean(0)
        frows.append({"flag": fl, "n_flag": int(fm.sum()), "model": m,
                      "median_ratio_flagged": np.exp(np.median(r_f)), "median_ratio_others": np.exp(np.median(r_o)),
                      "MAE_flagged": np.mean(np.abs(ys[fm] - np.exp(base[m][fm]))),
                      "MAE_others_base": pm(ys[~fm], np.exp(base[m][~fm]))["MAE"],
                      "MAE_others_trained_without": pm(ys[~fm], np.exp(P2[~fm]))["MAE"],
                      "R2_others_base": pm(ys[~fm], np.exp(base[m][~fm]))["R2_log"],
                      "R2_others_trained_without": pm(ys[~fm], np.exp(P2[~fm]))["R2_log"]})
fl_tab = pd.DataFrame(frows); print(fl_tab.round(3).to_string(index=False))
fl_tab.to_csv(OUT / "week5_flag_sensitivity.csv", index=False)

# ======================================================== D. FIELD_FULL × CHIA NGẪU NHIÊN
print("\n================ D. RQ4 — DỮ LIỆU CHƯA LÀM SẠCH ================")
raw = pd.read_csv(ROOT / "PSDb_clean" / "psdb2014_field_clean.csv", encoding="utf-8-sig", low_memory=False)
raw["a_m"] = raw["b_m"].where(raw["b_m"].notna(), raw["bn_m"].where(raw["theta_deg"].fillna(0) == 0))
raw["theta0"] = raw["theta_deg"].fillna(0)
raw["M0_hec18_kw"] = eq.hec18_csu(raw["a_m"], raw["L_m"], raw["theta0"], raw["nose"], raw["y_m"], raw["V_ms"], raw["D50_mm"], use_kw=True)
okf = (np.isfinite(raw["M0_hec18_kw"]) & (raw["M0_hec18_kw"] > 0) & raw["a_m"].notna() & raw["nose"].notna()
       & (raw["ys_m"] > 0) & (raw["y_m"] > 0) & (raw["V_ms"] > 0) & (raw["D50_mm"] > 0))
full = add_features(raw[okf].copy().reset_index(drop=True))
print(f"Field_full tính được: {len(full)} dòng ({full['site_group'].nunique()} cầu), trong đó core {int(full['in_core'].sum())}; "
      f"tập đánh giá chính {n} dòng")
print("Dòng thêm vào so với tập chính theo cờ:",
      {f: int(full.loc[~full['row_id'].isin(d['row_id']), f].fillna(False).astype(bool).sum()) for f in
       ["flag_ys_over_b_gt3", "flag_accuracy_ge_ys", "flag_Fr_gt1", "flag_bn_lt_b", "flag_duplicate", "flag_missing_core_input"]})
drows = []
for setname, dd in [("chính (đã làm sạch)", d), ("Field_full (chưa làm sạch)", full)]:
    nn = len(dd); gg = dd["site_group"].to_numpy(); yy = dd["ys_m"].to_numpy()
    in_main = dd["row_id"].isin(d["row_id"]).to_numpy()
    for kind in ["group", "random"]:
        fn = lambda r, nn=nn, gg=gg, kind=kind: w3.folds(nn, gg, 5, 3000 + r, kind)
        for m in ["M3_S6", "M4"]:
            Pm = cv_oof(dd, m, fn, 5).mean(0)
            drows.append({"data": setname, "split": kind, "model": m, "n": nn, **pm(yy, np.exp(Pm)),
                          **{f"{k}_on_main_rows": v for k, v in pm(yy[in_main], np.exp(Pm[in_main])).items() if k in ("MAE", "R2_log")}})
dtab = pd.DataFrame(drows); print(dtab.round(3).to_string(index=False))
dtab.to_csv(OUT / "week5_rq4_uncleaned.csv", index=False)

# ======================================================== E. BOOTSTRAP NGOẠI SUY
print("\n================ E. KHOẢNG TIN CẬY NGOẠI SUY (bootstrap theo cầu) ================")
ex = pd.read_csv(OUT / "week4_extrap_rows.csv", encoding="utf-8-sig")
cvr = pd.read_csv(OUT / "week4_cv_group_rows.csv", encoding="utf-8-sig")
erows = []
for (sp, meth), s in ex[ex["method"].isin(["weighted", "bridgemax"])].groupby(["split", "method"]):
    i0 = np.sort(s["i"].unique()); gt = g[i0]; ugt = np.unique(gt); pos = {k: np.where(gt == k)[0] for k in ugt}
    EM = [m for m in ["M3_S6", "M4", "M5", "RF4", "RF5", "GPR5"] if (s["model"] == m).any()]
    key = {m: ("M3" if m == "M3_S6" else m) for m in EM}
    M = {m: s[s["model"] == m].set_index("i").loc[i0] for m in EM}
    ID = {m: cvr[(cvr["model"] == m) & (cvr["method"] == meth)].set_index("i").loc[i0] for m in EM}
    stats = {"M3_S6_MAE": [], "M5_MAE": [], "M4_MAE": []}
    for m in EM:
        stats[f"ratio_{key[m]}"] = []; stats[f"exc_{key[m]}"] = []
        if m != "M3_S6":
            stats[f"dMAE_{key[m]}_M3"] = []; stats[f"dexc_{key[m]}_M3"] = []
    for _ in range(2000):
        jj = np.concatenate([pos[k] for k in rng.choice(ugt, len(ugt))])
        o = ys[i0][jj]; L = ly[i0][jj]
        mae = {m: np.mean(np.abs(o - np.exp(M[m]["mu"].to_numpy()[jj]))) for m in EM}
        mid = {m: np.mean(np.abs(o - np.exp(ID[m]["mu"].to_numpy()[jj]))) for m in EM}
        exc = {m: np.mean(L > M[m]["up"].to_numpy()[jj]) for m in EM}
        stats["M3_S6_MAE"].append(mae["M3_S6"]); stats["M5_MAE"].append(mae["M5"]); stats["M4_MAE"].append(mae["M4"])
        for m in EM:
            stats[f"ratio_{key[m]}"].append(mae[m] / mid[m]); stats[f"exc_{key[m]}"].append(exc[m])
            if m != "M3_S6":
                stats[f"dMAE_{key[m]}_M3"].append(mae[m] - mae["M3_S6"]); stats[f"dexc_{key[m]}_M3"].append(exc[m] - exc["M3_S6"])
    row = {"split": sp, "method": meth, "n_rows": len(i0), "n_bridges": len(ugt)}
    for k, v in stats.items():
        row[k] = np.mean(v); row[k + "_lo"] = np.quantile(v, 0.05); row[k + "_hi"] = np.quantile(v, 0.95)
    erows.append(row)
etab = pd.DataFrame(erows); etab.to_csv(OUT / "week5_extrap_bootstrap.csv", index=False)
show = etab[etab["method"] == "weighted"]
for _, r in show.iterrows():
    print(f"{r['split']} ({r['n_rows']} dòng/{r['n_bridges']} cầu): MAE ngoại suy/TPP  M3-S6 {r['ratio_M3']:.2f} "
          f"[{r['ratio_M3_lo']:.2f}; {r['ratio_M3_hi']:.2f}], M4 {r['ratio_M4']:.2f} [{r['ratio_M4_lo']:.2f}; {r['ratio_M4_hi']:.2f}], "
          f"M5 {r['ratio_M5']:.2f} [{r['ratio_M5_lo']:.2f}; {r['ratio_M5_hi']:.2f}]; ΔMAE M5−M3 {r['dMAE_M5_M3']:+.3f} "
          f"[{r['dMAE_M5_M3_lo']:+.3f}; {r['dMAE_M5_M3_hi']:+.3f}]; vượt95 M3 {r['exc_M3']:.3f} [{r['exc_M3_lo']:.3f}; {r['exc_M3_hi']:.3f}] "
          f"M5 {r['exc_M5']:.3f} [{r['exc_M5_lo']:.3f}; {r['exc_M5_hi']:.3f}]")
for _, r in show.iterrows():
    extra = []
    for k in ["RF4", "RF5", "GPR5"]:
        if f"ratio_{k}" in r and pd.notna(r[f"ratio_{k}"]):
            extra.append(f"{k} {r[f'ratio_{k}']:.2f} [{r[f'ratio_{k}_lo']:.2f}; {r[f'ratio_{k}_hi']:.2f}], "
                         f"ΔMAE−M3 {r[f'dMAE_{k}_M3']:+.3f} [{r[f'dMAE_{k}_M3_lo']:+.3f}; {r[f'dMAE_{k}_M3_hi']:+.3f}], "
                         f"vượt95 {r[f'exc_{k}']:.3f}")
    print(f"  {r['split']}: " + " | ".join(extra))
for _, r in etab[etab["method"] == "bridgemax"].iterrows():
    print(f"  bridgemax {r['split']}: vượt95 M3-S6 {r['exc_M3']:.3f} [{r['exc_M3_lo']:.3f}; {r['exc_M3_hi']:.3f}], "
          f"M5 {r['exc_M5']:.3f} [{r['exc_M5_lo']:.3f}; {r['exc_M5_hi']:.3f}]")

# ======================================================== F. THÍ NGHIỆM → HIỆN TRƯỜNG
print("\n================ F. THÍ NGHIỆM → HIỆN TRƯỜNG (G6) ================")
lab = pd.read_csv(ROOT / "PSDb_clean" / "psdb2014_lab_clean.csv", encoding="utf-8-sig", low_memory=False)
lab = lab[lab["in_core"]].copy()
lab["a_m"] = lab["bn_m"]; lab["theta0"] = 0.0; lab["nose"] = "circular"; lab["L_m"] = np.nan
lab["bed"] = "cohesionless"; lab["pier_type"] = "Single"
lab["M0_hec18_kw"] = eq.hec18_csu(lab["a_m"], lab["L_m"], lab["theta0"], lab["nose"], lab["y_m"], lab["V_ms"], lab["D50_mm"], use_kw=True)
lab = lab[np.isfinite(lab["M0_hec18_kw"]) & (lab["M0_hec18_kw"] > 0) & (lab["ys_m"] > 0)].reset_index(drop=True)
lab = add_features(lab)
print(f"Thí nghiệm dùng được: {len(lab)} dòng; trung vị ys đo / HEC-18 thí nghiệm {np.exp(lab['lr'].median()):.3f} "
      f"(hiện trường {np.exp(d['lr'].median()):.3f})")
labcols = ["x_log_ya", "x_log_aD"]                       # θ, đất dính, không ghi loại đất là hằng số ở thí nghiệm
bl = ols(lab[labcols].to_numpy(float), lab["lr"].to_numpy())
p_lab_m3 = np.log(hec) + olsp(bl, d[labcols].to_numpy(float))
p_lab_m2 = np.log(hec) + lab["lr"].mean()
m5_lab = xgb_fit(lab[w3.S2].to_numpy(float), lab["lr"].to_numpy())
p_lab_m5 = np.log(hec) + m5_lab.predict(d[w3.S2].to_numpy(float))
fld_m3 = base["M3_S6"]
# Cũng hiệu chỉnh hiện trường chỉ bằng 2 biến để so sánh công bằng
P2v = np.zeros(n)
for r in range(5):
    f = gfold(r)
    for i in range(5):
        tr, te = f != i, f == i
        bb = ols(d.loc[tr, labcols].to_numpy(float), d["lr"].to_numpy()[tr]); P2v[te] += (np.log(hec[te]) + olsp(bb, d.loc[te, labcols].to_numpy(float))) / 5
sub = ((d["nose"].isin(["round", "circular"])) & (d["theta0"] == 0)).to_numpy()
lrows = []
for name, p in [("HEC-18", np.log(hec)), ("thí nghiệm → M2 (hệ số hằng)", p_lab_m2), ("thí nghiệm → M3 (y/a, a/D50)", p_lab_m3),
                ("thí nghiệm → M5 (XGBoost)", p_lab_m5), ("hiện trường CV → M3 (y/a, a/D50)", P2v), ("hiện trường CV → M3-S6", fld_m3)]:
    lrows.append({"model": name, "subset": "toàn bộ", **pm(ys, np.exp(p))})
    lrows.append({"model": name, "subset": "mũi tròn, θ=0", **pm(ys[sub], np.exp(p[sub]))})
ltab = pd.DataFrame(lrows); print(ltab.round(3).to_string(index=False))
print("Hệ số thí nghiệm (hằng, ln y/a, ln a/D50):", bl.round(3),
      "| hiện trường:", ols(d[labcols].to_numpy(float), d["lr"].to_numpy()).round(3))
# Phân bố tỷ số theo a/D50
edges = [0, 50, 200, 1000, 5000, 1e6]
lab["aD"] = np.exp(lab["x_log_aD"]); d["aD"] = np.exp(d["x_log_aD"])
gb = []
for nm, dd in [("thí nghiệm", lab), ("hiện trường", d)]:
    for k, s in dd.groupby(pd.cut(dd["aD"], edges), observed=True):
        gb.append({"data": nm, "a/D50": str(k), "n": len(s), "median_ratio": np.exp(s["lr"].median()),
                   "median_ys_over_a": np.exp(s["lya"].median())})
gb = pd.DataFrame(gb); print(gb.round(3).to_string(index=False))
ltab.to_csv(OUT / "week5_lab_to_field.csv", index=False); gb.to_csv(OUT / "week5_lab_field_by_aD.csv", index=False)
print("Phạm vi a/D50 thí nghiệm:", np.quantile(lab["aD"], [0, 0.05, 0.5, 0.95, 1]).round(0),
      "hiện trường:", np.quantile(d["aD"], [0, 0.05, 0.5, 0.95, 1]).round(0))

# ======================================================== HÌNH
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
aD = np.exp(d["x_log_aD"]); yaa = np.exp(d["x_log_ya"])
sc = ax.scatter(aD, corr, c=np.log10(yaa), s=7, cmap="viridis", alpha=0.6)
ax.axhline(1, color="k", lw=0.8, ls="--"); ax.axhline(1 / k95b, color="C3", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("a/D50"); ax.set_ylabel("hệ số hiệu chỉnh = ys hiệu chỉnh / HEC-18")
ax.set_title("Công thức M3-S6 (màu: log10 y/a; chấm đỏ: 1/k95 bridgemax)", fontsize=9); fig.colorbar(sc, ax=ax, fraction=0.04)
ax = axes[1]
order = (ss.groupby("feature")["mean_abs_shap"].max().sort_values().tail(9).index)   # thứ tự chung cho M4, M5
for k, model in enumerate(["M4", "M5"]):
    s = ss[ss["model"] == model].set_index("feature").reindex(order)
    yy = np.arange(len(order)) + (0.2 if k else -0.2)
    ax.barh(yy, s["mean_abs_shap"].fillna(0), 0.4, xerr=s["sd"].fillna(0), label=model)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([o.replace("x_", "") for o in order], fontsize=7)
ax.set_xlabel("trung bình |SHAP| (log)"); ax.legend(fontsize=8); ax.set_title("SHAP, 10 lần lặp chia theo cầu", fontsize=9)
ax = axes[2]
for nm, c in [("thí nghiệm", "C1"), ("hiện trường", "C0")]:
    s = gb[gb["data"] == nm]; ax.plot(range(len(s)), s["median_ratio"], "o-", color=c, label=nm)
ax.set_xticks(range(len(edges) - 1)); ax.set_xticklabels(["<50", "50–200", "200–1k", "1k–5k", ">5k"])
ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_xlabel("a/D50"); ax.set_ylabel("trung vị ys đo / HEC-18")
ax.legend(fontsize=8); ax.set_title("Thí nghiệm và hiện trường theo a/D50", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "week5_formula_shap_lab.png", dpi=200)
print("Đã ghi:", FIG / "week5_formula_shap_lab.png")
