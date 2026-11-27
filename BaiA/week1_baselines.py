"""Tuần 1 bài A: công thức mốc M0–M1 và hiệu chỉnh đơn giản M2–M3, đánh giá theo cầu (GroupKFold).

Chạy:  python -I BaiA/week1_baselines.py
Đọc:   PSDb_clean/psdb2014_field_clean.csv  (tập core: in_core == True)
Ghi:   BaiA/outputs/week1_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import equations as eq  # noqa: E402

OUT = ROOT / "BaiA" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

df = pd.read_csv(ROOT / "PSDb_clean" / "psdb2014_field_clean.csv", encoding="utf-8-sig", low_memory=False)
df = df[df["in_core"]].copy()

# Bề rộng a: b danh nghĩa, nếu thiếu và θ = 0 thì dùng bn
df["a_m"] = df["b_m"].where(df["b_m"].notna(), df["bn_m"].where(df["theta_deg"].fillna(0) == 0))
df["theta0"] = df["theta_deg"].fillna(0)

# ----------------------------------------------------------------- M0, M1
df["M0_hec18_kw"] = eq.hec18_csu(df["a_m"], df["L_m"], df["theta0"], df["nose"], df["y_m"], df["V_ms"], df["D50_mm"], use_kw=True)
df["M0b_hec18_nokw"] = eq.hec18_csu(df["a_m"], df["L_m"], df["theta0"], df["nose"], df["y_m"], df["V_ms"], df["D50_mm"], use_kw=False)
cb = eq.hec18_coarse_bed(df["a_m"], df["L_m"], df["theta0"], df["nose"], df["y_m"], df["V_ms"], df["D50_mm"], df["D84_mm"])
df["M0c_hec18_coarse"] = np.where(np.isfinite(cb), cb, df["M0_hec18_kw"])
df["M1a_sheppard"] = eq.sheppard_melville(df["bn_m"], df["nose"], df["theta0"], df["y_m"], df["V_ms"], df["D50_mm"])
df["M1b_froehlich"] = eq.froehlich(df["a_m"], df["bn_m"], df["nose"], df["y_m"], df["V_ms"], df["D50_mm"])
df["M1c_froehlich_design"] = eq.froehlich(df["a_m"], df["bn_m"], df["nose"], df["y_m"], df["V_ms"], df["D50_mm"], design=True)

eqs = ["M0_hec18_kw", "M0b_hec18_nokw", "M0c_hec18_coarse", "M1a_sheppard", "M1b_froehlich", "M1c_froehlich_design"]

# Tập đánh giá chung: mọi công thức đều tính được và > 0
ok = np.ones(len(df), bool)
for c in eqs:
    ok &= np.isfinite(df[c]) & (df[c] > 0)
ok &= df["a_m"].notna() & df["nose"].notna()
d = df[ok].copy().reset_index(drop=True)
print(f"Tập core: {len(df)} dòng; tập đánh giá chung: {len(d)} dòng, {d['site_group'].nunique()} nhóm cầu")
print("Số dòng áp dụng Kw:", int((d["M0_hec18_kw"] < d["M0b_hec18_nokw"] - 1e-9).sum()),
      " | áp dụng nhánh hạt thô:", int(np.isfinite(cb[ok]).sum()))


def metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    lo, lp = np.log10(obs), np.log10(pred)
    r = obs / pred
    return {
        "MAE_m": np.mean(np.abs(obs - pred)),
        "RMSE_m": np.sqrt(np.mean((obs - pred) ** 2)),
        "R2_log": 1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2),
        "median_obs_over_pred": np.median(r),
        "frac_under": np.mean(obs > pred),  # dự báo thấp hơn đo
    }


# ------------------------------------------- Đánh giá công thức trên toàn tập
rows = [{"model": c, **metrics(d["ys_m"], d[c])} for c in eqs]
eq_tab = pd.DataFrame(rows)
print("\n=== Công thức, toàn tập đánh giá chung ===\n", eq_tab.round(3).to_string(index=False))

# Theo nhóm bề rộng
d["bn_bin"] = pd.cut(d["bn_m"], [0, 1, 2, 3, 5, 30], labels=["<=1", "1-2", "2-3", "3-5", ">5"])
by = []
for c in ["M0_hec18_kw", "M0b_hec18_nokw", "M1a_sheppard", "M1b_froehlich"]:
    for k, g in d.groupby("bn_bin", observed=True):
        by.append({"model": c, "bn_bin": k, "n": len(g), "median_obs_over_pred": np.median(g["ys_m"] / g[c]),
                   "frac_under": np.mean(g["ys_m"] > g[c])})
by = pd.DataFrame(by)
print("\n=== Theo bề rộng trụ ===\n", by.pivot(index="bn_bin", columns="model", values="median_obs_over_pred").round(3).to_string())

# -------------------------------------------------- M2, M3 với GroupKFold
d["log_ratio"] = np.log(d["ys_m"] / d["M0_hec18_kw"])
VVc = (d["V_ms"] / eq.critical_velocity_hec18(d["y_m"], d["D50_mm"]))
d["f_log_ya"] = np.log(d["y_m"] / d["a_m"])
d["f_log_Fr"] = np.log(d["V_ms"] / np.sqrt(9.81 * d["y_m"]))
d["f_log_aD"] = np.log(d["a_m"] / (d["D50_mm"] / 1000))
d["f_log_VVc"] = np.log(np.clip(VVc, 0.05, 20)).fillna(0)
d["f_VVc_missing"] = VVc.isna().astype(float)
d["f_theta"] = d["theta0"] / 90
for k in ["square", "sharp", "circular"]:
    d[f"f_nose_{k}"] = (d["nose"] == k).astype(float)
d["f_cohesive"] = (d["bed"] == "cohesive").astype(float)
d["f_bed_unknown"] = d["bed"].isna().astype(float)
d["f_group"] = (d["pier_type"] == "Group").astype(float)
feats = [c for c in d.columns if c.startswith("f_")]


def group_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(pd.unique(groups)))
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups])


def random_folds(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    f = np.empty(n, int); f[idx] = np.arange(n) % k
    return f


def ols_fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def ols_pred(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def run_cv(fold_fn, label, reps=10, k=5):
    res = []
    for rep in range(reps):
        folds = fold_fn(rep)
        for f in range(k):
            tr, te = folds != f, folds == f
            obs = d.loc[te, "ys_m"].to_numpy()
            hec = d.loc[te, "M0_hec18_kw"].to_numpy()
            lr_tr = d.loc[tr, "log_ratio"].to_numpy()
            preds = {
                "M0_hec18_kw": hec,
                "M1a_sheppard": d.loc[te, "M1a_sheppard"].to_numpy(),
                "M1b_froehlich": d.loc[te, "M1b_froehlich"].to_numpy(),
                "M2_hec18_x_c": hec * np.exp(np.mean(lr_tr)),
                "M2q95_hec18_x_c95": hec * np.exp(np.quantile(lr_tr, 0.95)),
            }
            beta = ols_fit(d.loc[tr, feats].to_numpy(), lr_tr)
            resid = lr_tr - ols_pred(beta, d.loc[tr, feats].to_numpy())
            mu = ols_pred(beta, d.loc[te, feats].to_numpy())
            preds["M3_hec18_x_linear"] = hec * np.exp(mu)
            preds["M3q95_linear_upper"] = hec * np.exp(mu + np.quantile(resid, 0.95))
            for m, p in preds.items():
                res.append({"split": label, "rep": rep, "fold": f, "model": m, "n_test": int(te.sum()), **metrics(obs, p)})
    return pd.DataFrame(res)


grp = d["site_group"].to_numpy()
cv_g = run_cv(lambda r: group_folds(grp, 5, 1000 + r), "group_by_site")
cv_r = run_cv(lambda r: random_folds(len(d), 5, 2000 + r), "random")
cv = pd.concat([cv_g, cv_r])
summ = cv.groupby(["split", "model"])[["MAE_m", "RMSE_m", "R2_log", "median_obs_over_pred", "frac_under"]].agg(["mean", "std"])
print("\n=== GroupKFold theo cầu và chia ngẫu nhiên (5 phần × 10 lần) ===")
print(summ.round(3).to_string())

# Hệ số OLS trên toàn tập (để diễn giải)
beta_all = ols_fit(d[feats].to_numpy(), d["log_ratio"].to_numpy())
coef = pd.Series(beta_all, index=["intercept"] + feats)
print("\n=== Hệ số M3 (log tỷ số ys đo / HEC-18) trên toàn tập ===\n", coef.round(3).to_string())
print("Hệ số M2 toàn tập: c = exp(mean log ratio) =", round(float(np.exp(d['log_ratio'].mean())), 3),
      "; c95 =", round(float(np.exp(d['log_ratio'].quantile(0.95))), 3))

# Ghi file
eq_tab.to_csv(OUT / "week1_equations_overall.csv", index=False)
by.to_csv(OUT / "week1_equations_by_width.csv", index=False)
cv.to_csv(OUT / "week1_cv_folds.csv", index=False)
summ.to_csv(OUT / "week1_cv_summary.csv")
coef.to_csv(OUT / "week1_M3_coefficients.csv", header=["coef"])
d.to_csv(OUT / "week1_eval_set_with_predictions.csv", index=False, encoding="utf-8-sig")
