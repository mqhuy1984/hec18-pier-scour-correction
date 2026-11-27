"""Ví dụ áp dụng bài A cho cầu nghiên cứu: HEC-18 bản 5 → HEC-18 hiệu chỉnh (M3-S6) → khoảng conformal → cờ miền áp dụng.

Chạy:  python BaiA/apply_P1_P7.py [đường_dẫn_csv]      (mặc định BaiA/P1_P7_input.csv)
Cần:   BaiA/outputs/week5_formula_coefficients.csv, week5_formula_conformal_factors.csv, week5_domain.csv
Ghi:   BaiA/outputs/apply_P1_P7_results.csv (+ in bảng)

Cột đầu vào (xem BaiA/P1_P7_input_template.csv):
  pier        tên trụ
  a_m         bề rộng trụ danh nghĩa (m) — với trụ có vòng vây/bệ lộ ra, ghi bề rộng phần chắn dòng
  L_m         chiều dài trụ (m), để trống nếu trụ tròn
  nose        square | round | circular | sharp
  theta_deg   góc lệch dòng chảy (độ)
  y_m, V_ms   chiều sâu và vận tốc dòng tới ứng với lũ thiết kế
  D50_mm, D84_mm   cỡ hạt (D84 để trống nếu không có)
  bed         cohesionless | cohesive | (trống = không ghi)
  pier_type   Single | Group
  debris      yes | no
  ys_design_m xói cục bộ thiết kế hiện có (để so sánh, có thể trống)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import equations as eq  # noqa: E402

OUT = HERE / "outputs"
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)

src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "P1_P7_input.csv"
p = pd.read_csv(src)
need = ["a_m", "nose", "theta_deg", "y_m", "V_ms", "D50_mm"]
miss = p[need].isna().any(axis=1)
if miss.all():
    sys.exit(f"{src}: chưa có số liệu ở các cột bắt buộc {need}.")
if miss.any():
    print("Bỏ qua các trụ thiếu số liệu bắt buộc:", p.loc[miss, "pier"].tolist())
p = p[~miss].copy()
p["theta_deg"] = p["theta_deg"].fillna(0)

coef = pd.read_csv(OUT / "week5_formula_coefficients.csv").set_index("var")["coef"]
conf = pd.read_csv(OUT / "week5_formula_conformal_factors.csv", index_col=0)["mean"]
dom = pd.read_csv(OUT / "week5_domain.csv", index_col=0)

p["ys_hec18_m"] = eq.hec18_csu(p["a_m"], p["L_m"], p["theta_deg"], p["nose"], p["y_m"], p["V_ms"], p["D50_mm"], use_kw=True)
ya = p["y_m"] / p["a_m"]; aD = p["a_m"] / (p["D50_mm"] / 1000)
coh = (p["bed"] == "cohesive").astype(float); unk = p["bed"].isna().astype(float)
feat = {"const": 1.0, "x_log_ya": np.log(ya), "x_log_aD": np.log(aD), "x_theta": p["theta_deg"] / 90,
        "x_cohesive": coh, "x_bed_unknown": unk}
lc = sum(coef[k] * feat[k] for k in coef.index)
p["corr_factor"] = np.exp(lc)
p["ys_median_m"] = p["ys_hec18_m"] * p["corr_factor"]
p["ys_lo90_pier_m"] = p["ys_median_m"] / np.exp(conf["q90_weighted_abs"])
p["ys_hi90_pier_m"] = p["ys_median_m"] * np.exp(conf["q90_weighted_abs"])
p["ys_up95_pier_m"] = p["ys_median_m"] * np.exp(conf["q95_weighted_up"])
p["ys_up95_bridge_m"] = p["ys_median_m"] * np.exp(conf["q95_bridgemax_up"])

# Cờ miền áp dụng
Fr = p["V_ms"] / np.sqrt(9.81 * p["y_m"])
Vc = eq.critical_velocity_hec18(p["y_m"], p["D50_mm"])
L = p["L_m"].fillna(p["a_m"])
bn = p["a_m"] * np.cos(np.radians(p["theta_deg"])) + L * np.sin(np.radians(p["theta_deg"]))
chk = {"y/a": ya, "a/D50": aD, "θ (độ)": p["theta_deg"], "D50 (mm)": p["D50_mm"], "Fr": Fr, "bn (m)": bn,
       "a (m)": p["a_m"], "y (m)": p["y_m"], "V (m/s)": p["V_ms"], "V/Vc": p["V_ms"] / Vc}
flags = []
for i in range(len(p)):
    f = []
    for k, v in chk.items():
        x = float(np.asarray(v)[i])
        if x < dom.loc[k, "min"] or x > dom.loc[k, "max"]:
            f.append(f"{k}={x:.3g} NGOÀI DỮ LIỆU")
        elif x < dom.loc[k, "p05"] or x > dom.loc[k, "p95"]:
            f.append(f"{k}={x:.3g} ngoài 5–95%")
    if str(p["debris"].iloc[i]).lower() == "yes": f.append("vật trôi: KHÔNG áp dụng (ys đo ≈ 1,8× dự báo)")
    if p["pier_type"].iloc[i] == "Group": f.append("trụ nhóm: HEC-18 chưa tính theo nhóm cọc")
    flags.append("; ".join(f) if f else "trong miền dữ liệu")
p["domain_flags"] = flags
p["bn_m_calc"] = bn; p["Fr"] = Fr

cols = ["pier", "ys_hec18_m", "corr_factor", "ys_median_m", "ys_lo90_pier_m", "ys_hi90_pier_m", "ys_up95_pier_m",
        "ys_up95_bridge_m", "ys_design_m", "domain_flags"]
print(p[[c for c in cols if c in p.columns]].round(2).to_string(index=False))
p.to_csv(OUT / "apply_P1_P7_results.csv", index=False, encoding="utf-8-sig")
print("\nĐã ghi:", OUT / "apply_P1_P7_results.csv")
print("Lưu ý: cận trên 95% 'theo cầu' (bridgemax) là giá trị nên dùng khi cần mọi trụ của cầu cùng không vượt.")
