"""Thêm CIRIA C742 vào các công thức mốc của bài A (theo gợi ý của Gavriel et al. 2026).

Chạy:  python BaiA/ciria_check.py
Cần:   BaiA/outputs/week1_eval_set_with_predictions.csv (tập đánh giá 1.253 dòng)
Ghi:   BaiA/outputs/ciria_equations.csv          (chỉ số của mọi công thức, cùng tập dòng)
       BaiA/outputs/ciria_by_covariate.csv       (trung vị ys đo / dự báo theo nhóm biến)
       BaiA/outputs/ciria_correction_cv.csv      (hiệu chỉnh hằng số và 4 số hạng trên nền CIRIA so với nền HEC-18, CV theo cầu 5×10)

Công thức CIRIA: xem equations.ciria_c742. CIRIA cho ys = 0 khi V/Vc ≤ 0,375 (không xói); các dòng này được tính là
"dự báo thấp" và loại khỏi chỉ số trên thang log (báo cáo riêng số dòng).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equations as eq  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
OUT = HERE / "outputs"
d = pd.read_csv(OUT / "week1_eval_set_with_predictions.csv", encoding="utf-8-sig", low_memory=False)
args = (d["a_m"], d["L_m"], d["theta0"], d["nose"], d["y_m"], d["V_ms"], d["D50_mm"])
d["CIRIA_vcHEC"] = eq.ciria_c742(*args, vc="hec18")
d["CIRIA_vcGavriel"] = eq.ciria_c742(*args, vc="gavriel")
ys = d["ys_m"].to_numpy(); g = d["site_group"].to_numpy()

NAMES = {"M0_hec18_kw": "HEC-18 (5th ed.)", "M0b_hec18_nokw": "HEC-18 without K_w", "M0c_hec18_coarse": "HEC-18 with coarse-bed branch",
         "M1a_sheppard": "Sheppard–Melville", "M1b_froehlich": "Froehlich best-fit",
         "M1c_froehlich_design": "Froehlich design", "CIRIA_vcHEC": "CIRIA C742 (Vc HEC-18)",
         "CIRIA_vcGavriel": "CIRIA C742 (Vc Gavriel 2025)"}


def metrics(obs, pred):
    pos = pred > 0
    lo, lp = np.log10(obs[pos]), np.log10(pred[pos])
    r = obs[pos] / pred[pos]
    return {"n_zero_pred": int((~pos).sum()), "MAE_m": np.mean(np.abs(obs - pred)),
            "RMSE_m": np.sqrt(np.mean((obs - pred) ** 2)),
            "R2_log_pos": 1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2),
            "median_obs_over_pred_pos": np.median(r), "frac_under": np.mean(obs > pred),
            "within_pm50": np.mean(np.abs(pred - obs) <= 0.5 * obs),
            "c95_pos": np.quantile(r, 0.95)}  # hệ số nhân để công thức thành cận 95% (trên dòng dự báo > 0)


rows = [{"model": NAMES[c], **metrics(ys, d[c].to_numpy())} for c in NAMES]
tab = pd.DataFrame(rows)
print("=== Các công thức trên cùng 1.253 dòng ===\n", tab.round(3).to_string(index=False))
tab.to_csv(OUT / "ciria_equations.csv", index=False)

# Sai lệch theo biến: trung vị ys đo / dự báo
Fr = d["V_ms"] / np.sqrt(9.81 * d["y_m"]); aD = d["a_m"] / (d["D50_mm"] / 1000)
bins = {"a/D50": (aD, [0, 50, 300, 3000, 1e7]), "Fr": (Fr, [0, 0.15, 0.3, 0.5, 2]),
        "θ (độ)": (d["theta0"], [-1, 0.5, 20, 90]), "bn (m)": (d["bn_m"], [0, 1, 3, 5, 50])}
by = []
for vname, (v, edges) in bins.items():
    cls = pd.cut(v, edges)
    for k, idx in d.groupby(cls, observed=True).groups.items():
        for c in ["M0_hec18_kw", "M1b_froehlich", "CIRIA_vcHEC"]:
            p = d.loc[idx, c].to_numpy(); o = d.loc[idx, "ys_m"].to_numpy(); pos = p > 0
            by.append({"var": vname, "class": str(k), "n": len(idx), "model": NAMES[c],
                       "median_ratio": np.median(o[pos] / p[pos]), "frac_under": np.mean(o > p)})
by = pd.DataFrame(by)
print("\n=== Trung vị ys đo / dự báo theo nhóm ===\n",
      by.pivot_table(index=["var", "class", "n"], columns="model", values="median_ratio").round(2).to_string())
by.to_csv(OUT / "ciria_by_covariate.csv", index=False)

# Hiệu chỉnh trên nền CIRIA so với nền HEC-18 (cùng các dòng CIRIA > 0), CV theo cầu 5 × 10
import week3_xgboost as w3  # noqa: E402

w = w3.load().reset_index(drop=True)
assert len(w) == len(d) and np.allclose(w["ys_m"].to_numpy(), ys)
pos = d["CIRIA_vcHEC"].to_numpy() > 0
XS = {"4 số hạng (S6)": w[w3.S6].to_numpy(float), "11 số hạng (S2)": w[w3.S2].to_numpy(float)}
res = []
for base, col in [("HEC-18", "M0_hec18_kw"), ("CIRIA C742", "CIRIA_vcHEC")]:
    lb = np.log(d[col].to_numpy()[pos]); y = np.log(ys[pos]) - lb; gp = g[pos]; n = pos.sum()
    for form in ["hằng số", "4 số hạng (S6)", "11 số hạng (S2)"]:
        Xp = XS.get(form, XS["4 số hạng (S6)"])[pos]
        mae, r2 = [], []
        for r in range(10):
            f = w3.folds(n, gp, 5, 1000 + r, "group"); P = np.empty(n)
            for i in range(5):
                tr, te = f != i, f == i
                if form == "hằng số":
                    P[te] = y[tr].mean()
                else:
                    P[te] = w3.ols_fit_pred(Xp[tr], y[tr], Xp[te])[0]
            pr = np.exp(lb + P); o = ys[pos]
            mae.append(np.mean(np.abs(o - pr)))
            lo, lp = np.log10(o), np.log10(pr); r2.append(1 - np.sum((lo - lp) ** 2) / np.sum((lo - lo.mean()) ** 2))
        res.append({"baseline": base, "correction": form, "n_rows": int(n), "MAE": np.mean(mae), "MAE_sd": np.std(mae),
                    "R2_log": np.mean(r2)})
res = pd.DataFrame(res)
print(f"\n=== Hiệu chỉnh trên nền CIRIA so với HEC-18 (các dòng CIRIA > 0: {pos.sum()}) ===\n", res.round(3).to_string(index=False))
res.to_csv(OUT / "ciria_correction_cv.csv", index=False)
