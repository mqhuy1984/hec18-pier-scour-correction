"""Kiểm tra có nên bỏ ln(y/a) khỏi công thức M3-S6 không: CV theo cầu 10 lần lặp + 6 phép ngoại suy (OLS, nhanh).

Chạy:  python BaiA/formula_variant_check.py
Ghi:   BaiA/outputs/formula_variant_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import week3_xgboost as w3  # noqa: E402
from week4_conformal_extrap import q_weighted, q_bridgemax  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
d = w3.load(); n = len(d); g = d["site_group"].to_numpy(); ys = d["ys_m"].to_numpy(); ly = np.log(ys)
lhec = np.log(d["M0_hec18_kw"].to_numpy()); y = d["lr"].to_numpy()
V = {"S6 (5 số hạng)": w3.S6_5TERM, "S6 bỏ ln(y/a) (4 số hạng)": [c for c in w3.S6_5TERM if c != "x_log_ya"],
     "chỉ ln(a/D50), θ (2 số hạng)": ["x_log_aD", "x_theta"]}


def fitpred(cols, tr, te):
    X = d[cols].to_numpy(float); p, _ = w3.ols_fit_pred(X[tr], y[tr], X[te]); return lhec[te] + p


def metr(i, lp):
    p = np.exp(lp); lo, lpp = np.log10(ys[i]), np.log10(p)
    return np.mean(np.abs(ys[i] - p)), 1 - np.sum((lo - lpp) ** 2) / np.sum((lo - lo.mean()) ** 2)


rows = []
for name, cols in V.items():
    mae, r2, q95b = [], [], []
    for r in range(10):
        f = w3.folds(n, g, 5, 1000 + r, "group"); P = np.empty(n)
        for i in range(5):
            P[f == i] = fitpred(cols, f != i, f == i)
        a, b = metr(np.arange(n), P); mae.append(a); r2.append(b); q95b.append(np.exp(q_bridgemax(ly - P, g, 0.05)))
    rows.append({"variant": name, "split": "CV theo cầu 5×10", "MAE": np.mean(mae), "MAE_sd": np.std(mae),
                 "R2_log": np.mean(r2), "k95_bridgemax": np.mean(q95b)})
Fr = d["V_ms"] / np.sqrt(9.81 * d["y_m"]); bn = d["bn_m"]; D50 = d["D50_mm"]
ext = {"USA → ngoài Mỹ": (d["country"] == "USA", d["country"] != "USA"), "bn ≤ 3 → bn > 5 m": (bn <= 3, bn > 5),
       "bn ≤ 3 → 3–5 m": (bn <= 3, (bn > 3) & (bn <= 5)), "cát → sỏi cuội": (D50 < 2, D50 >= 20),
       "sỏi → cát": (D50 >= 2, D50 < 2), "Fr ≤ 0,3 → Fr > 0,5": (Fr <= 0.3, Fr > 0.5)}
for en, (mtr, mte) in ext.items():
    mtr, mte = mtr.to_numpy(), mte.to_numpy(); mtr = mtr & ~np.isin(g, g[mte])
    tr, te = np.where(mtr)[0], np.where(mte)[0]
    for name, cols in V.items():
        a, b = metr(te, fitpred(cols, tr, te))
        rows.append({"variant": name, "split": en, "MAE": a, "R2_log": b})
out = pd.DataFrame(rows); print(out.round(3).to_string(index=False))
for name, cols in V.items():
    X = d[cols].to_numpy(float); A = np.column_stack([np.ones(n), X]); bb, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(name, "hệ số (hằng, " + ", ".join(cols) + "):", bb.round(3))
out.to_csv(HERE / "outputs" / "formula_variant_check.csv", index=False)
