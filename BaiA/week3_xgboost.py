"""Tuần 3 bài A: XGBoost M4 (thuần dữ liệu) và M5 (học phần sai lệch so với HEC-18), CV lồng theo cầu,
so với M3/S6 (tuyến tính) và đối chứng chia ngẫu nhiên (RQ4).

Chạy:  python BaiA/week3_xgboost.py [--reps 3] [--trials 60] [--jobs 15]
        (KHÔNG dùng python -I; cần chạy week1_baselines.py trước)
Đọc:   BaiA/outputs/week1_eval_set_with_predictions.csv
Ghi:   BaiA/outputs/week3_*.csv, BaiA/figures/week3_*.png

Thiết kế
- Vòng ngoài: chia 5 phần, lặp R lần. Kiểu "group": GroupKFold theo site_group; kiểu "random": KFold theo dòng.
- Vòng trong (chỉ trên phần huấn luyện): chia 4 phần CÙNG KIỂU với vòng ngoài; Optuna TPE, N lượt thử,
  hàm mục tiêu = MSE của log ys trên phần kiểm tra trong.
- Cận trên 95%: phân vị 95% phần dư log của dự báo ngoài-phần (OOF) ở vòng trong với siêu tham số tốt nhất
  (không dùng phần dư huấn luyện vì XGBoost khớp quá mức trên tập huấn luyện).
- M4: mục tiêu log(ys/a), biến S2 + log Fr, ràng buộc đơn điệu tăng theo y/a.
- M5: mục tiêu log(ys/HEC18), biến S2, ràng buộc đơn điệu tăng theo y/a (đủ để ys dự báo tăng theo y).
- M5u: như M5 nhưng không ràng buộc.
- M3 / M3-S6: OLS trên log(ys/HEC18) (như tuần 2), chạy trên cùng các phần để so sánh cặp.
- Độ nhạy: bỏ số liệu Nga (Zhuravlyov 1978) khỏi cả huấn luyện và đánh giá (chỉ kiểu group).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import equations as eq  # noqa: E402

OUT = ROOT / "BaiA" / "outputs"; FIG = ROOT / "BaiA" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

S2 = ["x_log_ya", "x_log_aD", "x_log_VVc", "x_theta", "x_nose_square", "x_nose_sharp", "x_nose_circular",
      "x_cohesive", "x_bed_unknown", "x_group", "x_ptype_unknown"]
# Công thức tối giản. Bản 5 số hạng có thêm x_log_ya (hệ số 0,05, KTC 95% chứa 0); bỏ đi không đổi CV theo cầu
# và ngoại suy tốt hơn ở 4/6 phép thử (BaiA/formula_variant_check.py) → dùng bản 4 số hạng từ 27/11/2026.
S6 = ["x_log_aD", "x_theta", "x_cohesive", "x_bed_unknown"]
S6_5TERM = ["x_log_ya", "x_log_aD", "x_theta", "x_cohesive", "x_bed_unknown"]
M4F = S2 + ["x_log_Fr"]


def load():
    d = pd.read_csv(OUT / "week1_eval_set_with_predictions.csv", encoding="utf-8-sig", low_memory=False)
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
    d["lr"] = np.log(d["ys_m"] / d["M0_hec18_kw"])
    d["lya"] = np.log(d["ys_m"] / d["a_m"])
    return d.reset_index(drop=True)


def folds(n, groups, k, seed, kind):
    rng = np.random.default_rng(seed)
    if kind == "random":
        f = np.arange(n) % k; rng.shuffle(f); return f
    ug = np.array(sorted(pd.unique(groups))); rng.shuffle(ug)
    m = {g: i % k for i, g in enumerate(ug)}
    return np.array([m[g] for g in groups])


def ols_fit_pred(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); b, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    res = ytr - A @ b
    return np.column_stack([np.ones(len(Xte)), Xte]) @ b, np.quantile(res, 0.95)


def xgb_model(p, mono):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                            min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                            colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"],
                            gamma=p["gamma"], monotone_constraints=mono, tree_method="hist", nthread=1,
                            random_state=0)


def tune_xgb(X, y, g, kind, seed, trials, mono):
    """Vòng trong: Optuna TPE; trả về siêu tham số tốt nhất và phân vị 95% phần dư OOF trong."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fin = folds(len(y), g, 4, seed, kind)

    def oof(p):
        pr = np.empty(len(y))
        for j in range(4):
            tr, te = fin != j, fin == j
            pr[te] = xgb_model(p, mono).fit(X[tr], y[tr]).predict(X[te])
        return pr

    def obj(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 50, 800, log=True),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_depth=t.suggest_int("max_depth", 1, 6),
                 min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True),
                 reg_alpha=t.suggest_float("reg_alpha", 1e-4, 5, log=True),
                 gamma=t.suggest_float("gamma", 1e-4, 1, log=True))
        return float(np.mean((y - oof(p)) ** 2))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=trials)
    best = st.best_params
    q95 = np.quantile(y - oof(best), 0.95)
    return best, q95, st.best_value


def rf_model(p, mono_first):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                 min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                 max_samples=p["max_samples"], bootstrap=True, n_jobs=1, random_state=0,
                                 monotonic_cst=mono_first)


def tune_rf(X, y, g, kind, seed, trials, mono_first):
    """Random Forest, cùng ngân sách Optuna như XGBoost; ràng buộc đơn điệu tăng theo biến đầu (y/a)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fin = folds(len(y), g, 4, seed, kind)

    def oof(p):
        pr = np.empty(len(y))
        for j in range(4):
            tr, te = fin != j, fin == j
            pr[te] = rf_model(p, mono_first).fit(X[tr], y[tr]).predict(X[te])
        return pr

    def obj(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 100, 300, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 20),
                 min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 40, log=True),
                 max_features=t.suggest_float("max_features", 0.3, 1.0),
                 max_samples=t.suggest_float("max_samples", 0.5, 1.0))
        return float(np.mean((y - oof(p)) ** 2))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=trials)
    best = st.best_params
    return best, np.quantile(y - oof(best), 0.95), st.best_value, oof(best), fin


def gpr_fit(X, y):
    """GPR (RBF có độ dài đặc trưng riêng từng biến + nhiễu trắng); siêu tham số theo cực đại hợp lý biên."""
    import warnings
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    k = ConstantKernel(0.3, (1e-3, 10)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(0.3, (1e-3, 2))
    m = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(X, y)


def gpr_tune(X, y, g, kind, seed):
    fin = folds(len(y), g, 4, seed, kind)
    pr = np.empty(len(y))
    for j in range(4):
        tr, te = fin != j, fin == j
        pr[te] = gpr_fit(X[tr], y[tr]).predict(X[te])
    return {}, np.quantile(y - pr, 0.95), float(np.mean((y - pr) ** 2)), pr, fin


def run_task(task, d_dict, trials):
    """Một phần ngoài: trả về dự báo (median, cận trên) cho mọi mô hình trên phần kiểm tra."""
    d = pd.DataFrame(d_dict)
    kind, rep, i, f, subset = task["kind"], task["rep"], task["fold"], task["f"], task["subset"]
    tr, te = f != i, f == i
    g = d["site_group"].to_numpy()
    hec = d["M0_hec18_kw"].to_numpy(); a = d["a_m"].to_numpy()
    out = {"idx": np.where(te)[0]}
    meta = []
    # M2: hệ số không đổi
    lr = d["lr"].to_numpy()
    mu = lr[tr].mean(); out["M2"] = hec[te] * np.exp(mu); out["M2_up"] = hec[te] * np.exp(np.quantile(lr[tr], 0.95))
    out["M0"] = hec[te]; out["M0_up"] = hec[te]
    # M3, M3-S6
    for name, cols in [("M3", S2), ("M3_S6", S6)]:
        X = d[cols].to_numpy(float); p, q = ols_fit_pred(X[tr], lr[tr], X[te])
        out[name] = hec[te] * np.exp(p); out[name + "_up"] = hec[te] * np.exp(p + q)
    # XGBoost
    seed = 10000 * rep + 100 * i + (0 if kind == "group" else 7)
    specs = [("M4", M4F, "lya", "(1" + ",0" * (len(M4F) - 1) + ")", a, "xgb"),
             ("M5", S2, "lr", "(1" + ",0" * (len(S2) - 1) + ")", hec, "xgb"),
             ("M5u", S2, "lr", None, hec, "xgb"),
             ("RF4", M4F, "lya", [1] + [0] * (len(M4F) - 1), a, "rf"),
             ("RF5", S2, "lr", [1] + [0] * (len(S2) - 1), hec, "rf"),
             ("GPR5", S2, "lr", None, hec, "gpr")]
    if subset != "all":
        specs = [s for s in specs if s[0] in ("M5", "GPR5")]
    for name, cols, tgt, mono, base, fam in specs:
        X = d[cols].to_numpy(float); y = d[tgt].to_numpy()
        t0 = time.time()
        if fam == "xgb":
            best, q95, inner = tune_xgb(X[tr], y[tr], g[tr], kind, seed, trials, mono)
            m = xgb_model(best, mono).fit(X[tr], y[tr])
        elif fam == "rf":
            best, q95, inner, _, _ = tune_rf(X[tr], y[tr], g[tr], kind, seed, trials, mono)
            m = rf_model(best, mono).fit(X[tr], y[tr])
        else:
            best, q95, inner, _, _ = gpr_tune(X[tr], y[tr], g[tr], kind, seed)
            m = gpr_fit(X[tr], y[tr])
        p = m.predict(X[te])
        out[name] = base[te] * np.exp(p); out[name + "_up"] = base[te] * np.exp(p + q95)
        meta.append({"kind": kind, "subset": subset, "rep": rep, "fold": i, "model": name, "inner_mse": inner,
                     "q95_inner": q95, "sec": time.time() - t0, **best})
    return task, out, meta


def metrics(obs, pred, up=None):
    lo, lp = np.log10(obs), np.log10(pred)
    r = {"MAE": np.mean(np.abs(obs - pred)), "RMSE": np.sqrt(np.mean((obs - pred) ** 2)),
         "R2_log": 1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2),
         "median_obs_over_pred": np.median(obs / pred), "frac_under": np.mean(obs > pred)}
    if up is not None:
        r["exceed_up95"] = np.mean(obs > up); r["median_up_over_obs"] = np.median(up / obs)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=15)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from joblib import Parallel, delayed

    d = load()
    ru = (d["country"] == "Russia").to_numpy()
    print(f"Tập đánh giá: {len(d)} dòng, {d['site_group'].nunique()} nhóm cầu; Nga {ru.sum()} dòng")
    subsets = {"all": d, "no_russia": d[~ru].reset_index(drop=True)}
    tasks = []
    for subset, dd in subsets.items():
        kinds = ["group", "random"] if subset == "all" else ["group"]
        for kind in kinds:
            for rep in range(args.reps):
                f = folds(len(dd), dd["site_group"].to_numpy(), 5, 1000 + rep, kind)
                for i in range(5):
                    tasks.append({"kind": kind, "rep": rep, "fold": i, "f": f, "subset": subset})
    print(f"{len(tasks)} phần ngoài, {args.trials} lượt thử/phần, {args.jobs} tiến trình")
    t0 = time.time()
    keep = [c for c in d.columns if c.startswith("x_")] + ["lr", "lya", "M0_hec18_kw", "a_m", "site_group", "ys_m"]
    dicts = {k: v[keep].to_dict("list") for k, v in subsets.items()}
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_task)(t, dicts[t["subset"]], args.trials) for t in tasks)
    print(f"Xong sau {(time.time() - t0) / 60:.1f} phút")

    # Gom dự báo OOF theo (subset, kind, rep)
    models = ["M0", "M2", "M3", "M3_S6", "M4", "M5", "M5u", "RF4", "RF5", "GPR5"]
    oof, meta = {}, []
    for task, out, mt in res:
        key = (task["subset"], task["kind"], task["rep"]); n = len(subsets[task["subset"]])
        if key not in oof:
            oof[key] = {m: np.full(n, np.nan) for m in models} | {m + "_up": np.full(n, np.nan) for m in models}
        for m in models:
            if m in out:
                oof[key][m][out["idx"]] = out[m]; oof[key][m + "_up"][out["idx"]] = out[m + "_up"]
        meta += mt
    meta = pd.DataFrame(meta); meta.to_csv(OUT / "week3_tuning_log.csv", index=False)

    rows = []
    for (subset, kind, rep), o in oof.items():
        obs = subsets[subset]["ys_m"].to_numpy()
        for m in models:
            if np.isnan(o[m]).all(): continue
            rows.append({"subset": subset, "split": kind, "rep": rep, "model": m, **metrics(obs, o[m], o[m + "_up"])})
    per = pd.DataFrame(rows); per.to_csv(OUT / "week3_metrics_per_rep.csv", index=False)
    summ = per.groupby(["subset", "split", "model"]).agg(["mean", "std"]).drop(columns="rep")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.reset_index()
    summ.to_csv(OUT / "week3_metrics_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ[["subset", "split", "model", "MAE_mean", "MAE_std", "R2_log_mean", "R2_log_std", "frac_under_mean",
                "exceed_up95_mean", "median_up_over_obs_mean"]].round(3).to_string(index=False))

    # Bootstrap theo cầu cho hiệu MAE và R2 so với M3 (subset all, mỗi kiểu chia, gộp các lần lặp)
    rng = np.random.default_rng(11)
    dd = subsets["all"]; obs = dd["ys_m"].to_numpy(); g = dd["site_group"].to_numpy()
    ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    brows = []
    for kind in ["group", "random"]:
        keys = [k for k in oof if k[0] == "all" and k[1] == kind]
        P = {m: np.mean([oof[k][m] for k in keys], axis=0) for m in models}  # trung bình các lần lặp
        for m in ["M4", "M5", "M5u", "RF4", "RF5", "GPR5", "M3_S6"]:
            dm, dr = [], []
            for _ in range(1000):
                ii = np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])
                a1, b1 = metrics(obs[ii], P[m][ii]), metrics(obs[ii], P["M3"][ii])
                dm.append(a1["MAE"] - b1["MAE"]); dr.append(a1["R2_log"] - b1["R2_log"])
            brows.append({"split": kind, "model": m, "dMAE_vs_M3": np.mean(dm), "dMAE_lo90": np.quantile(dm, 0.05),
                          "dMAE_hi90": np.quantile(dm, 0.95), "dR2_vs_M3": np.mean(dr),
                          "dR2_lo90": np.quantile(dr, 0.05), "dR2_hi90": np.quantile(dr, 0.95),
                          "P(better MAE)": np.mean(np.array(dm) < 0)})
    boot = pd.DataFrame(brows); boot.to_csv(OUT / "week3_paired_bootstrap_vs_M3.csv", index=False)
    print("\nSo sánh cặp với M3 (bootstrap theo cầu, dự báo OOF trung bình các lần lặp):\n", boot.round(3).to_string(index=False))

    # RQ4: khoảng lạc quan = chỉ số chia ngẫu nhiên − chỉ số chia theo cầu
    a = summ[summ["subset"] == "all"].pivot(index="model", columns="split", values=["MAE_mean", "R2_log_mean", "exceed_up95_mean"])
    rq4 = pd.DataFrame({"MAE_group": a[("MAE_mean", "group")], "MAE_random": a[("MAE_mean", "random")],
                        "R2_group": a[("R2_log_mean", "group")], "R2_random": a[("R2_log_mean", "random")],
                        "exceed_group": a[("exceed_up95_mean", "group")], "exceed_random": a[("exceed_up95_mean", "random")]})
    rq4["MAE_optimism_%"] = 100 * (rq4["MAE_group"] - rq4["MAE_random"]) / rq4["MAE_group"]
    rq4["R2_optimism"] = rq4["R2_random"] - rq4["R2_group"]
    rq4.to_csv(OUT / "week3_rq4_split_gap.csv")
    print("\nRQ4 — chia ngẫu nhiên so với chia theo cầu:\n", rq4.round(3).to_string())

    # Phần dư số liệu Nga trong M3/M5 (subset all, group)
    keys = [k for k in oof if k[:2] == ("all", "group")]
    for m in ["M3", "M5"]:
        P = np.mean([oof[k][m] for k in keys], axis=0)
        print(f"{m}: trung vị ys đo/dự báo — Nga {np.median(obs[ru] / P[ru]):.2f}, còn lại {np.median(obs[~ru] / P[~ru]):.2f}")

    # Lưu OOF (subset all) để dùng cho tuần 4
    sav = dd[["site_group", "ys_m", "M0_hec18_kw"]].copy()
    sav["row_id"] = load()["row_id"]
    for kind in ["group", "random"]:
        keys = [k for k in oof if k[0] == "all" and k[1] == kind]
        for m in models:
            sav[f"{m}_{kind}"] = np.mean([oof[k][m] for k in keys], axis=0)
    sav.to_csv(OUT / "week3_oof_predictions.csv", index=False)

    # Hình: dự báo/đo cho M0, M3, M5 (group) + so sánh hai kiểu chia
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, m in zip(axes[:3], ["M0", "M3", "M5"]):
        p = sav[f"{m}_group"]; ax.scatter(obs, p, s=6, alpha=0.35)
        ax.plot([0.01, 30], [0.01, 30], "k--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.02, 20); ax.set_ylim(0.02, 20); ax.set_xlabel("ys đo (m)"); ax.set_ylabel("ys dự báo (m)")
        r = metrics(obs, p.to_numpy()); ax.set_title(f"{m} (chia theo cầu): MAE {r['MAE']:.2f} m, R² log {r['R2_log']:.2f}".replace(".", ","), fontsize=9)
    ax = axes[3]; ms = ["M3", "M3_S6", "M4", "M5", "M5u", "RF4", "RF5", "GPR5"]; x = np.arange(len(ms))
    ax.bar(x - 0.2, rq4.loc[ms, "R2_group"], 0.4, label="chia theo cầu")
    ax.bar(x + 0.2, rq4.loc[ms, "R2_random"], 0.4, label="chia ngẫu nhiên")
    ax.set_xticks(x); ax.set_xticklabels(ms); ax.set_ylabel("R² (log10)"); ax.legend(fontsize=8); ax.set_title("RQ4", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "week3_models.png", dpi=200)
    print("Đã ghi:", FIG / "week3_models.png")


if __name__ == "__main__":
    main()
