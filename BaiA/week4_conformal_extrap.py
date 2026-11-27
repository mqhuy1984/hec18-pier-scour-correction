"""Tuần 4 bài A: (1) khoảng dự báo conformal ở cấp cầu (RQ3); (2) thử nghiệm ngoại suy (H2, H3).

Chạy:  python BaiA/week4_conformal_extrap.py [--reps 3] [--trials 60] [--jobs 15]
        (KHÔNG dùng python -I; cần week1_baselines.py và week3_xgboost.py trong cùng thư mục)
Ghi:   BaiA/outputs/week4_*.csv, BaiA/figures/week4_*.png

Thiết kế
- Mô hình: M3-S6 (OLS, log tỷ số), M4 (XGBoost thuần dữ liệu), M5 (XGBoost học phần sai lệch), tìm siêu tham số
  như tuần 3 (Optuna lồng theo cầu). Thêm CQR: XGBoost phân vị (0,05; 0,95) trên log tỷ số, tham số cố định.
- Điểm hiệu chỉnh (calibration) = phần dư log ys ngoài-phần ở vòng trong (theo cầu) trên tập huấn luyện
  (kiểu CV+, không bỏ dữ liệu).
- Ba cách lấy phân vị conformal:
    pooled   : mọi dòng như nhau (bỏ qua cấu trúc cụm — cách làm thường gặp)
    weighted : trọng số 1/n_cầu cho mỗi dòng → mỗi cầu đóng góp như nhau
    bridgemax: mỗi cầu một điểm = điểm lớn nhất của cầu → phủ đồng thời mọi dòng của một cầu mới
  và Mondrian theo nhóm bề rộng bn (≤1, 1–3, 3–5, >5 m) dùng cách weighted trong từng nhóm.
- Khoảng 90% hai phía (điểm |r|, CQR: max(qlo−y, y−qhi)) và cận trên 95% một phía (điểm r, CQR: y−q95).
- Cách chia: (a) theo cầu 5 phần × R lần; (b) theo nguồn dữ liệu (GroupKFold theo `source`, 5 phần);
  (c) ngoại suy: Mỹ → ngoài Mỹ; bn ≤ 3 m → bn > 5 m; bn ≤ 3 m → 3–5 m; cát (D50 < 2 mm) → sỏi cuội (≥ 20 mm);
      sỏi cuội và sỏi (≥ 2 mm) → cát; Fr ≤ 0,3 → Fr > 0,5. Cầu có dòng trong tập kiểm tra bị loại khỏi tập huấn luyện.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import week3_xgboost as w3  # noqa: E402

ROOT = HERE.parents[0]
OUT = ROOT / "BaiA" / "outputs"; FIG = ROOT / "BaiA" / "figures"
MODELS = ["M3_S6", "M4", "M5", "RF4", "RF5", "GPR5", "CQR"]
CQR_P = dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=1e-3, gamma=1e-3)
BN_EDGES = [0, 1, 3, 5, 1e9]; BN_LAB = ["≤1 m", "1–3 m", "3–5 m", ">5 m"]


def tune_with_oof(X, y, g, seed, trials, mono):
    """Như w3.tune_xgb nhưng trả thêm dự báo OOF vòng trong (dùng làm điểm hiệu chỉnh) và chỉ số phần vòng trong."""
    best, _, _ = w3.tune_xgb(X, y, g, "group", seed, trials, mono)
    fin = w3.folds(len(y), g, 4, seed, "group")
    oof = np.empty(len(y))
    for j in range(4):
        tr, te = fin != j, fin == j
        oof[te] = w3.xgb_model(best, mono).fit(X[tr], y[tr]).predict(X[te])
    return best, oof, fin


def quant_model(alpha):
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, tree_method="hist", nthread=1,
                            random_state=0, **CQR_P)


def fit_task(task, D, trials):
    """Huấn luyện mọi mô hình trên tr, dự báo te; trả về log-dự báo trên te và điểm hiệu chỉnh trên tr."""
    d = pd.DataFrame(D)
    tr, te = np.asarray(task["tr"]), np.asarray(task["te"])
    g = d["site_group"].to_numpy(); ly = np.log(d["ys_m"].to_numpy())
    lhec = np.log(d["M0_hec18_kw"].to_numpy()); la = np.log(d["a_m"].to_numpy())
    seed = task["seed"]
    res = {"name": task["name"], "rep": task.get("rep", 0), "te": te, "tr": tr}
    fin = w3.folds(len(tr), g[tr], 4, seed, "group")
    # M3-S6 (OLS)
    X = d[w3.S6].to_numpy(float); y = d["lr"].to_numpy()
    p, _ = w3.ols_fit_pred(X[tr], y[tr], X[te]); res["M3_S6_te"] = lhec[te] + p
    oof = np.empty(len(tr))
    for j in range(4):
        a, b = fin != j, fin == j
        oof[b], _ = w3.ols_fit_pred(X[tr][a], y[tr][a], X[tr][b])
    res["M3_S6_cal"] = ly[tr] - (lhec[tr] + oof)
    # M4, M5
    meta = []
    for name, cols, tgt, base in [("M4", w3.M4F, "lya", la), ("M5", w3.S2, "lr", lhec)]:
        X = d[cols].to_numpy(float); y = d[tgt].to_numpy()
        mono = "(1" + ",0" * (len(cols) - 1) + ")"
        best, oofp, _ = tune_with_oof(X[tr], y[tr], g[tr], seed, trials, mono)
        m = w3.xgb_model(best, mono).fit(X[tr], y[tr])
        res[f"{name}_te"] = base[te] + m.predict(X[te])
        res[f"{name}_cal"] = ly[tr] - (base[tr] + oofp)
        meta.append({"task": task["name"], "rep": res["rep"], "model": name, **best})
    # RF (thuần dữ liệu và học phần sai lệch) và GPR (học phần sai lệch)
    for name, cols, tgt, base, fam in [("RF4", w3.M4F, "lya", la, "rf"), ("RF5", w3.S2, "lr", lhec, "rf"),
                                       ("GPR5", w3.S2, "lr", lhec, "gpr")]:
        X = d[cols].to_numpy(float); y = d[tgt].to_numpy()
        if fam == "rf":
            mono = [1] + [0] * (len(cols) - 1)
            best, _, _, oofp, _ = w3.tune_rf(X[tr], y[tr], g[tr], "group", seed, trials, mono)
            m = w3.rf_model(best, mono).fit(X[tr], y[tr])
        else:
            best, _, _, oofp, _ = w3.gpr_tune(X[tr], y[tr], g[tr], "group", seed)
            m = w3.gpr_fit(X[tr], y[tr])
        res[f"{name}_te"] = base[te] + m.predict(X[te])
        res[f"{name}_cal"] = ly[tr] - (base[tr] + oofp)
        meta.append({"task": task["name"], "rep": res["rep"], "model": name, **best})
    # CQR trên log tỷ số (nền HEC-18)
    X = d[w3.S2].to_numpy(float); y = d["lr"].to_numpy()
    lo_oof, hi_oof = np.empty(len(tr)), np.empty(len(tr))
    for j in range(4):
        a, b = fin != j, fin == j
        lo_oof[b] = quant_model(0.05).fit(X[tr][a], y[tr][a]).predict(X[tr][b])
        hi_oof[b] = quant_model(0.95).fit(X[tr][a], y[tr][a]).predict(X[tr][b])
    res["CQR_lo_te"] = lhec[te] + quant_model(0.05).fit(X[tr], y[tr]).predict(X[te])
    res["CQR_hi_te"] = lhec[te] + quant_model(0.95).fit(X[tr], y[tr]).predict(X[te])
    res["CQR_lo_cal"] = ly[tr] - (lhec[tr] + lo_oof)     # y − qlo (dương nếu trên cận dưới)
    res["CQR_hi_cal"] = ly[tr] - (lhec[tr] + hi_oof)     # y − qhi
    res["CQR_te"] = 0.5 * (res["CQR_lo_te"] + res["CQR_hi_te"])
    res["meta"] = meta
    return res


# ---------------------------------------------------------------- conformal
def q_pooled(s, alpha):
    s = np.sort(s); n = len(s); k = math.ceil((n + 1) * (1 - alpha))
    return s[min(k, n) - 1] if k <= n else np.inf


def q_weighted(s, g, alpha):
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]; B = len(cnt)
    o = np.argsort(s); cw = np.cumsum(w[o]) / B
    lev = min((1 - alpha) * (1 + 1 / B), 1.0)
    i = np.searchsorted(cw, lev - 1e-12)
    return s[o][min(i, len(s) - 1)] if lev < 1 else np.inf


def q_bridgemax(s, g, alpha):
    m = pd.Series(s).groupby(np.asarray(g)).max().to_numpy()
    return q_pooled(m, alpha)


def qfun(method, s, g, alpha):
    return {"pooled": lambda: q_pooled(s, alpha), "weighted": lambda: q_weighted(s, g, alpha),
            "bridgemax": lambda: q_bridgemax(s, g, alpha)}[method]()


def intervals(res, model, method, g_tr, cls_tr, cls_te, mondrian=False):
    """Trả về (lo90, hi90, up95) trên te theo log ys."""
    def per_class(fn):
        if not mondrian:
            return np.full(len(cls_te), fn(np.ones(len(cls_tr), bool)))
        out = np.empty(len(cls_te)); glob = fn(np.ones(len(cls_tr), bool))
        for c in np.unique(cls_te):
            m = cls_tr == c
            out[cls_te == c] = fn(m) if len(np.unique(g_tr[m])) >= 20 else glob   # ít hơn 20 cầu → dùng toàn cục
        return out
    if model == "CQR":
        slo, shi = res["CQR_lo_cal"], res["CQR_hi_cal"]
        s2 = np.maximum(-slo, shi)                               # max(qlo − y, y − qhi)
        q2 = per_class(lambda m: qfun(method, s2[m], g_tr[m], 0.10))
        q1 = per_class(lambda m: qfun(method, shi[m], g_tr[m], 0.05))
        return res["CQR_lo_te"] - q2, res["CQR_hi_te"] + q2, res["CQR_hi_te"] + q1
    r = res[f"{model}_cal"]; mu = res[f"{model}_te"]
    q2 = per_class(lambda m: qfun(method, np.abs(r[m]), g_tr[m], 0.10))
    q1 = per_class(lambda m: qfun(method, r[m], g_tr[m], 0.05))
    return mu - q2, mu + q2, mu + q1


def cov_metrics(ly, lo, hi, up, g):
    inside = (ly >= lo) & (ly <= hi); over = ly > up
    df = pd.DataFrame({"g": g, "in": inside, "over": over})
    pb = df.groupby("g").agg(cin=("in", "mean"), allin=("in", "all"), anyover=("over", "any"))
    return {"cov90_row": inside.mean(), "cov90_bridge_mean": pb["cin"].mean(), "bridges_all_in": pb["allin"].mean(),
            "exceed95_row": over.mean(), "bridges_any_exceed": pb["anyover"].mean(),
            "width_factor_med": float(np.median(np.exp(hi - lo))), "up_over_obs_med": float(np.median(np.exp(up - ly)))}


def point_metrics(ys, lp):
    p = np.exp(lp); lo, lpp = np.log10(ys), np.log10(p)
    return {"MAE": np.mean(np.abs(ys - p)), "R2_log": 1 - np.sum((lo - lpp) ** 2) / np.sum((lo - lo.mean()) ** 2),
            "median_obs_over_pred": np.median(ys / p), "frac_under": np.mean(ys > p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=15)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
    from joblib import Parallel, delayed

    d = w3.load()
    d["bn_cls"] = pd.cut(d["bn_m"], BN_EDGES, labels=BN_LAB, right=True).astype(str)
    d["Fr"] = d["V_ms"] / np.sqrt(9.81 * d["y_m"])
    n = len(d); g = d["site_group"].to_numpy()
    print(f"Tập đánh giá: {n} dòng, {len(np.unique(g))} cầu")

    tasks = []
    for rep in range(args.reps):                                 # (a) theo cầu
        f = w3.folds(n, g, 5, 1000 + rep, "group")
        for i in range(5):
            tasks.append({"name": "cv_group", "rep": rep, "fold": i, "tr": np.where(f != i)[0], "te": np.where(f == i)[0],
                          "seed": 20000 + 100 * rep + i})
    src = d["source"].astype(str).to_numpy()                     # (b) theo nguồn dữ liệu
    f = w3.folds(n, src, 5, 77, "group")
    for i in range(5):
        tasks.append({"name": "cv_source", "rep": 0, "fold": i, "tr": np.where(f != i)[0], "te": np.where(f == i)[0],
                      "seed": 30000 + i})
    D50 = d["D50_mm"]; bn = d["bn_m"]; Fr = d["Fr"]
    ext = {"USA → ngoài Mỹ": (d["country"] == "USA", d["country"] != "USA"),
           "bn ≤ 3 → bn > 5 m": (bn <= 3, bn > 5),
           "bn ≤ 3 → 3–5 m": (bn <= 3, (bn > 3) & (bn <= 5)),
           "cát (<2 mm) → sỏi cuội (≥20 mm)": (D50 < 2, D50 >= 20),
           "sỏi (≥2 mm) → cát (<2 mm)": (D50 >= 2, D50 < 2),
           "Fr ≤ 0,3 → Fr > 0,5": (Fr <= 0.3, Fr > 0.5)}
    for k, (ename, (mtr, mte)) in enumerate(ext.items()):
        mtr, mte = mtr.to_numpy(), mte.to_numpy()
        mtr = mtr & ~np.isin(g, g[mte])                          # loại cầu có mặt trong tập kiểm tra
        tasks.append({"name": ename, "rep": 0, "fold": 0, "tr": np.where(mtr)[0],
                      "te": np.where(mte)[0], "seed": 40000 + k})
    for t in tasks[-len(ext):]:
        print(f"  {t['name']}: huấn luyện {len(t['tr'])} dòng / {len(np.unique(g[t['tr']]))} cầu; "
              f"kiểm tra {len(t['te'])} dòng / {len(np.unique(g[t['te']]))} cầu")

    keep = [c for c in d.columns if c.startswith("x_")] + ["lr", "lya", "M0_hec18_kw", "a_m", "site_group", "ys_m"]
    D = d[keep].to_dict("list")
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=2)(delayed(fit_task)(t, D, args.trials) for t in tasks)
    print(f"Xong sau {(time.time() - t0) / 60:.1f} phút")
    pd.DataFrame([m for r in results for m in r["meta"]]).to_csv(OUT / "week4_tuning_log.csv", index=False)

    ly = np.log(d["ys_m"].to_numpy()); ys = d["ys_m"].to_numpy(); cls = d["bn_cls"].to_numpy()
    country = d["country"].to_numpy()
    methods = [("pooled", False), ("weighted", False), ("bridgemax", False), ("weighted", True)]
    mname = lambda m, mo: m + ("+Mondrian" if mo else "")

    # Dự báo/khoảng của mọi phần, lưu theo dòng để tính chỉ số theo nhóm
    rows, store, exstore = [], [], []
    for r in results:
        te, tr = r["te"], r["tr"]
        for model in MODELS:
            pm = point_metrics(ys[te], r[f"{model}_te"])
            for meth, mo in methods:
                lo, hi, up = intervals(r, model, meth, g[tr], cls[tr], cls[te], mondrian=mo)
                rows.append({"split": r["name"], "rep": r["rep"], "model": model, "method": mname(meth, mo),
                             "n_test": len(te), **pm, **cov_metrics(ly[te], lo, hi, up, g[te])})
                if r["name"] not in ("cv_group", "cv_source"):
                    exstore.append(pd.DataFrame({"split": r["name"], "model": model, "method": mname(meth, mo), "i": te,
                                                 "lo": lo, "hi": hi, "up": up, "mu": r[f"{model}_te"]}))
                else:
                    store.append(pd.DataFrame({"split": r["name"], "rep": r["rep"], "model": model,
                                               "method": mname(meth, mo), "i": te, "lo": lo, "hi": hi, "up": up,
                                               "mu": r[f"{model}_te"]}))
    per = pd.DataFrame(rows)
    st = pd.concat(store, ignore_index=True)
    # Lưu dự báo theo dòng (log ys) để tuần 5 tính khoảng tin cậy bootstrap
    rid = d["row_id"].to_numpy()
    ex_rows = pd.concat(exstore, ignore_index=True); ex_rows["row_id"] = rid[ex_rows["i"]]
    ex_rows.to_csv(OUT / "week4_extrap_rows.csv", index=False, encoding="utf-8-sig")
    cvr = st[st["split"] == "cv_group"].groupby(["model", "method", "i"])[["mu", "lo", "hi", "up"]].mean().reset_index()
    cvr["row_id"] = rid[cvr["i"]]
    cvr.to_csv(OUT / "week4_cv_group_rows.csv", index=False, encoding="utf-8-sig")

    # (a)(b) tổng hợp CV: gộp mọi phần của một lần lặp rồi tính chỉ số
    cvrows = []
    for (sp, rep, model, meth), s in st.groupby(["split", "rep", "model", "method"]):
        i = s["i"].to_numpy()
        cvrows.append({"split": sp, "rep": rep, "model": model, "method": meth, **point_metrics(ys[i], s["mu"].to_numpy()),
                       **cov_metrics(ly[i], s["lo"].to_numpy(), s["hi"].to_numpy(), s["up"].to_numpy(), g[i])})
    cv = pd.DataFrame(cvrows).drop(columns="rep").groupby(["split", "model", "method"]).mean().reset_index()
    cv.to_csv(OUT / "week4_conformal_cv.csv", index=False)
    print("\n=== Conformal, chia theo cầu và theo nguồn (trung bình các lần lặp) ===")
    print(cv[["split", "model", "method", "MAE", "R2_log", "cov90_row", "cov90_bridge_mean", "bridges_all_in",
              "exceed95_row", "bridges_any_exceed", "width_factor_med", "up_over_obs_med"]].round(3).to_string(index=False))

    # Theo nhóm bề rộng và quốc gia (chia theo cầu)
    sub = st[st["split"] == "cv_group"].copy()
    sub["cls"] = cls[sub["i"]]; sub["country"] = country[sub["i"]]
    byc = []
    for key in ["cls", "country"]:
        for (model, meth, lev), s in sub.groupby(["model", "method", key]):
            i = s["i"].to_numpy()
            if len(np.unique(i)) < 10: continue
            byc.append({"by": key, "level": lev, "model": model, "method": meth, "n_rows": len(np.unique(i)),
                        **cov_metrics(ly[i], s["lo"].to_numpy(), s["hi"].to_numpy(), s["up"].to_numpy(), g[i])})
    byc = pd.DataFrame(byc); byc.to_csv(OUT / "week4_conformal_by_group.csv", index=False)
    print("\n=== Theo nhóm bề rộng / quốc gia (chia theo cầu) — M5 và M3_S6 ===")
    print(byc[byc["model"].isin(["M5", "M3_S6", "CQR"])][["by", "level", "model", "method", "n_rows", "cov90_row",
          "exceed95_row", "width_factor_med"]].round(3).to_string(index=False))

    # (c) Ngoại suy + mốc trong phân phối trên cùng các dòng (từ CV theo cầu)
    ex = per[~per["split"].isin(["cv_group", "cv_source"])].copy()
    ref = []
    for r in results:
        if r["name"] in ("cv_group", "cv_source"): continue
        te = r["te"]
        for model in MODELS:
            s = st[(st["split"] == "cv_group") & (st["model"] == model) & (st["method"] == "weighted")]
            s = s[s["i"].isin(te)].groupby("i")[["mu", "lo", "hi", "up"]].mean()
            i = s.index.to_numpy()
            ref.append({"split": r["name"], "model": model, **{f"ID_{k}": v for k, v in point_metrics(ys[i], s["mu"].to_numpy()).items()},
                        **{f"ID_{k}": v for k, v in cov_metrics(ly[i], s["lo"].to_numpy(), s["hi"].to_numpy(), s["up"].to_numpy(), g[i]).items()}})
    ref = pd.DataFrame(ref)
    ex = ex.merge(ref, on=["split", "model"], how="left")
    ex.to_csv(OUT / "week4_extrapolation.csv", index=False)
    print("\n=== Ngoại suy (khoảng: weighted) — so với cùng các dòng khi chia theo cầu (ID_) ===")
    e = ex[ex["method"] == "weighted"]
    print(e[["split", "model", "n_test", "MAE", "ID_MAE", "R2_log", "ID_R2_log", "median_obs_over_pred",
             "frac_under", "cov90_row", "ID_cov90_row", "exceed95_row", "ID_exceed95_row", "up_over_obs_med"]]
          .round(3).to_string(index=False))

    # HEC-18 trên các tập ngoại suy (mốc)
    hrows = []
    for r in results:
        if r["name"] in ("cv_group", "cv_source"): continue
        te = r["te"]; h = d["M0_hec18_kw"].to_numpy()[te]
        hrows.append({"split": r["name"], "HEC18_MAE": np.mean(np.abs(ys[te] - h)), "HEC18_median_obs_over_pred": np.median(ys[te] / h),
                      "HEC18_frac_under": np.mean(ys[te] > h)})
    hrows = pd.DataFrame(hrows); hrows.to_csv(OUT / "week4_extrapolation_hec18.csv", index=False)
    print("\nHEC-18 trên tập ngoại suy:\n", hrows.round(3).to_string(index=False))

    # ---------------------------------------------------------- Hình
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    b = byc[(byc["by"] == "cls") & (byc["model"] == "M5")]
    for k, meth in enumerate(["pooled", "weighted", "bridgemax", "weighted+Mondrian"]):
        bb = b[b["method"] == meth].set_index("level").reindex(BN_LAB)
        ax.bar(np.arange(4) + (k - 1.5) * 0.2, bb["cov90_row"], 0.2, label=meth)
    ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.set_ylim(0.6, 1.0); ax.set_xticks(range(4)); ax.set_xticklabels(BN_LAB)
    ax.set_ylabel("độ phủ thực (dòng)"); ax.set_title("M5, khoảng 90%, chia theo cầu — theo bề rộng bn", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    e2 = e.set_index(["split", "model"])
    sp = list(ext)
    for k, model in enumerate(["M3_S6", "M4", "M5", "RF4", "RF5", "GPR5"]):
        ratio = [e2.loc[(s, model), "MAE"] / e2.loc[(s, model), "ID_MAE"] for s in sp]
        ax.bar(np.arange(len(sp)) + (k - 2.5) * 0.14, ratio, 0.14, label=model)
    ax.axhline(1, color="k", lw=0.8); ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("MAE ngoại suy / MAE trong phân phối"); ax.legend(fontsize=7); ax.set_title("Mức giảm hiệu năng khi ngoại suy", fontsize=9)
    ax = axes[2]
    for k, model in enumerate(["M3_S6", "M4", "M5", "RF4", "RF5", "GPR5", "CQR"]):
        ax.bar(np.arange(len(sp)) + (k - 3) * 0.12, [e2.loc[(s, model), "cov90_row"] for s in sp], 0.12, label=model)
    ax.axhline(0.9, color="k", ls="--", lw=0.8); ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("độ phủ thực khoảng 90%"); ax.legend(fontsize=7); ax.set_title("Độ phủ khi ngoại suy (weighted)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "week4_conformal_extrap.png", dpi=200)
    print("Đã ghi:", FIG / "week4_conformal_extrap.png")


if __name__ == "__main__":
    main()
