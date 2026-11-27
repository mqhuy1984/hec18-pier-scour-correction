"""Làm sạch cơ sở dữ liệu xói trụ USGS PSDb-2014 (Benedict & Caldwell, 2014, Data Series 845).

Chạy:  python -I PSDb_clean/clean_psdb.py ds845_psdb-2014_version1.0.xlsx PSDb_clean

Nguyên tắc: không xóa dòng nào khỏi bảng đầy đủ. Mọi vấn đề được ghi thành cột cờ (flag_*).
Tập "core" là tập con đề xuất cho mô hình hóa, lọc theo các cờ nghiêm trọng.
File gốc không bị sửa.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FT = 0.3048
G = 9.81

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def clean_col(c):
    return re.sub(r"\s+", " ", str(c)).strip()


def to_num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"--": np.nan, "": np.nan, "nan": np.nan}),
                         errors="coerce")


def to_cat(s):
    s = s.astype(str).str.strip()
    return s.where(~s.isin(["--", "nan", "", "Unknown"]), np.nan)


# ----------------------------------------------------------------- FIELD
f = pd.read_excel(SRC, sheet_name="Field Data", header=2)
f.columns = [clean_col(c) for c in f.columns]
f = f[f.iloc[:, 0].notna()].reset_index(drop=True)
names = ["state", "road", "site", "drainage_area_mi2", "slope", "meas_type", "meas_method", "date_raw",
         "time_raw", "pier_id", "side", "pier_type", "nose_raw", "b_ft", "bn_ft", "L_ft", "theta_deg",
         "RI_yr", "hydraulics_method", "V_fts", "y_ft", "debris", "bed_raw", "D16_mm", "D50_mm", "D84_mm",
         "D95_mm", "sigma_g", "ys_ft", "ys_accuracy_ft", "sheppard2011_code", "mueller2005_no", "source",
         "agency"]
assert len(f.columns) == len(names), f.columns
f.columns = names
f.insert(0, "row_id", np.arange(1, len(f) + 1))  # = dòng Excel - 3

for c in ["drainage_area_mi2", "slope", "b_ft", "bn_ft", "L_ft", "theta_deg", "RI_yr", "V_fts", "y_ft",
          "D16_mm", "D50_mm", "D84_mm", "D95_mm", "sigma_g", "ys_ft", "ys_accuracy_ft"]:
    f[c] = to_num(f[c])
for c in ["state", "road", "site", "meas_type", "meas_method", "pier_id", "side", "pier_type", "nose_raw",
          "hydraulics_method", "debris", "bed_raw", "source", "agency"]:
    f[c] = to_cat(f[c])
f["sheppard2011_code"] = to_num(f["sheppard2011_code"])

# Chuẩn hóa nhãn
nose_map = {"round": "round", "semi-circular": "round", "square": "square", "rectangular": "square",
            "sharp": "sharp", "triangular": "sharp", "cylindrical": "circular", "circular": "circular",
            "cylinder": "circular"}
f["nose"] = f["nose_raw"].str.lower().map(nose_map)
f["bed"] = f["bed_raw"].str.lower().map({"non-cohesive": "non-cohesive", "cohesive": "cohesive"})
f["country"] = np.where(f["state"].str.len() == 2, "USA", f["state"])
f["year"] = pd.to_datetime(f["date_raw"], errors="coerce").dt.year
yr = f["date_raw"].astype(str).str.extract(r"(\d{4})")[0]
f["year"] = f["year"].fillna(pd.to_numeric(yr, errors="coerce"))

# Mã nhóm cầu (để chia dữ liệu theo nhóm, tránh rò rỉ)
site_key = f["site"].fillna("")
missing_site = site_key == ""
# Dòng thiếu tên: gom theo quốc gia + nguồn + bề rộng trụ (gần đúng một công trình)
pseudo = ("NOSITE|" + f["state"].fillna("") + "|" + f["source"].fillna("") + "|b=" + f["bn_ft"].round(2).astype(str))
site_key = site_key.where(~missing_site, pseudo)
f["site_group"] = pd.factorize(site_key)[0] + 1
f["site_group_is_pseudo"] = missing_site
f["pier_group"] = pd.factorize(site_key + "|" + f["pier_id"].fillna("?"))[0] + 1

# SI
f["bn_m"] = f["bn_ft"] * FT
f["b_m"] = f["b_ft"] * FT
f["L_m"] = f["L_ft"] * FT
f["V_ms"] = f["V_fts"] * FT
f["y_m"] = f["y_ft"] * FT
f["ys_m"] = f["ys_ft"] * FT
f["ys_accuracy_m"] = f["ys_accuracy_ft"] * FT

# Vận tốc tới hạn (HEC-18, Eq. 6.1, SI): Vc = 6.19 y^(1/6) D50^(1/3), D50 (m). Chỉ hợp lệ cho đất rời.
D50m = f["D50_mm"] / 1000
f["Vc_ms_hec18"] = 6.19 * f["y_m"].where(f["y_m"] > 0) ** (1 / 6) * D50m.where(D50m > 0) ** (1 / 3)
f.loc[f["bed"] == "cohesive", "Vc_ms_hec18"] = np.nan

# Không thứ nguyên
f["ys_over_bn"] = f["ys_m"] / f["bn_m"]
f["y_over_bn"] = f["y_m"] / f["bn_m"]
f["Fr"] = f["V_ms"] / np.sqrt(G * f["y_m"].where(f["y_m"] > 0))
f["V_over_Vc"] = f["V_ms"] / f["Vc_ms_hec18"]
f["bn_over_D50"] = f["bn_m"] / D50m.where(D50m > 0)
f["L_over_b"] = f["L_m"] / f["b_m"]

# HEC-18 CSU (đơn giản: K1 theo mũi trụ, K2 theo θ và L/a, K3 = 1.1, không Kw/K4)
K1 = f["nose"].map({"square": 1.1, "round": 1.0, "circular": 1.0, "sharp": 0.9}).fillna(1.0)
th = f["theta_deg"].fillna(0)
K1 = K1.where(th <= 5, 1.0)
a = f["b_m"].fillna(f["bn_m"].where(th == 0))
LoA = np.minimum((f["L_m"] / a).fillna(1.0), 12)
K2 = (np.cos(np.radians(th)) + LoA * np.sin(np.radians(th))) ** 0.65
f["ys_hec18_csu_m"] = 2.0 * K1 * K2 * 1.1 * a ** 0.65 * f["y_m"] ** 0.35 * f["Fr"] ** 0.43
f["ratio_obs_hec18"] = f["ys_m"] / f["ys_hec18_csu_m"]

# ------------------------------------------------------------ CỜ (FLAGS)
exp_bn = f["b_ft"] * np.cos(np.radians(th)) + f["L_ft"] * np.sin(np.radians(th))
flags = {
    "flag_ys_zero": f["ys_ft"] == 0,
    "flag_ys_over_b_gt3": f["ys_over_bn"] > 3,
    "flag_accuracy_ge_ys": f["ys_accuracy_ft"] >= f["ys_ft"],
    "flag_accuracy_gt_half_ys": (f["ys_accuracy_ft"] > 0.5 * f["ys_ft"]) & ~(f["ys_accuracy_ft"] >= f["ys_ft"]),
    "flag_V_or_y_zero": (f["V_fts"] == 0) | (f["y_ft"] == 0),
    "flag_Fr_gt1": f["Fr"] > 1,
    "flag_missing_core_input": f[["bn_ft", "y_ft", "V_fts", "D50_mm"]].isna().any(axis=1),
    "flag_bn_lt_b": f["bn_ft"] < f["b_ft"] - 1e-6,
    "flag_theta_pos_bn_eq_b": (th > 0) & np.isclose(f["bn_ft"], f["b_ft"]),
    "flag_bn_inconsistent": (f["bn_ft"] - exp_bn).abs() > 0.5,
    "flag_cohesive": f["bed"] == "cohesive",
    "flag_debris_substantial": f["debris"] == "Substantial",
    "flag_hydraulics_modelled": f["hydraulics_method"].isin(["Model", "Estimate from limited historical information"]),
    "flag_duplicate": f.duplicated(["site", "pier_id", "date_raw", "time_raw", "bn_ft", "V_fts", "y_ft", "ys_ft"]),
}
for k, v in flags.items():
    f[k] = v.fillna(False).astype(bool)

severe = ["flag_ys_zero", "flag_ys_over_b_gt3", "flag_accuracy_ge_ys", "flag_V_or_y_zero", "flag_Fr_gt1",
          "flag_missing_core_input", "flag_bn_lt_b", "flag_duplicate"]
f["n_flags_severe"] = f[severe].sum(axis=1)
f["in_core"] = f["n_flags_severe"] == 0

lead = ["row_id", "in_core", "n_flags_severe", "site_group", "site_group_is_pseudo", "pier_group", "country",
        "state", "site", "pier_id", "year", "nose", "pier_type", "bed", "bn_m", "b_m", "L_m", "theta_deg",
        "y_m", "V_ms", "D50_mm", "sigma_g", "ys_m", "ys_accuracy_m", "Vc_ms_hec18", "ys_over_bn", "y_over_bn",
        "Fr", "V_over_Vc", "bn_over_D50", "L_over_b", "ys_hec18_csu_m", "ratio_obs_hec18"]
rest = [c for c in f.columns if c not in lead]
f = f[lead + rest]

# ------------------------------------------------------------------- LAB
l = pd.read_excel(SRC, sheet_name="Laboratory Data", header=2)
l = l[l.iloc[:, 0].notna()].reset_index(drop=True)
l.columns = ["source", "bn_ft", "V_fts", "Vc_fts", "y_ft", "D50_mm", "sigma_g", "duration_min", "ys_ft",
             "sheppard2011_code"]
l.insert(0, "row_id", np.arange(1, len(l) + 1))
l["source"] = l["source"].astype(str).str.strip()
for c in ["bn_ft", "V_fts", "Vc_fts", "y_ft", "D50_mm", "sigma_g", "duration_min", "ys_ft", "sheppard2011_code"]:
    l[c] = to_num(l[c])
l["bn_m"], l["V_ms"], l["Vc_ms"] = l["bn_ft"] * FT, l["V_fts"] * FT, l["Vc_fts"] * FT
l["y_m"], l["ys_m"] = l["y_ft"] * FT, l["ys_ft"] * FT
l["V_over_Vc"] = l["V_ms"] / l["Vc_ms"]
l["regime"] = np.where(l["V_over_Vc"] <= 1, "clear-water", "live-bed")
l["ys_over_bn"] = l["ys_m"] / l["bn_m"]
l["y_over_bn"] = l["y_m"] / l["bn_m"]
l["Fr"] = l["V_ms"] / np.sqrt(G * l["y_m"])
l["bn_over_D50"] = l["bn_m"] / (l["D50_mm"] / 1000)
l["flag_duration_missing"] = l["duration_min"].isna()
l["flag_duration_lt_600min"] = l["duration_min"] < 600
l["flag_ys_over_b_gt3"] = l["ys_over_bn"] > 3
l["flag_sheppard_not_used"] = l["sheppard2011_code"] == 0
l["in_core"] = ~(l["flag_duration_lt_600min"] | l["flag_ys_over_b_gt3"])

# ----------------------------------------------------------------- XUẤT
f.to_csv(OUT / "psdb2014_field_clean.csv", index=False, encoding="utf-8-sig")
l.to_csv(OUT / "psdb2014_lab_clean.csv", index=False, encoding="utf-8-sig")

dd = pd.DataFrame([
    ("row_id", "", "Số thứ tự trong sheet gốc (= dòng Excel − 3)"),
    ("in_core", "", "True nếu không có cờ nghiêm trọng nào; tập đề xuất cho mô hình hóa"),
    ("n_flags_severe", "", "Số cờ nghiêm trọng: " + ", ".join(severe)),
    ("site_group", "", "Mã nhóm cầu để chia dữ liệu theo nhóm (GroupKFold / leave-one-site-out)"),
    ("site_group_is_pseudo", "", "True nếu dòng gốc không có tên vị trí; nhóm được dựng từ quốc gia + nguồn + bn"),
    ("pier_group", "", "Mã nhóm trụ (vị trí + Pier ID)"),
    ("nose", "", "Mũi trụ đã chuẩn hóa: round / square / sharp / circular"),
    ("bed", "", "Loại đất đã chuẩn hóa: cohesive / non-cohesive"),
    ("bn_m, b_m, L_m", "m", "Bề rộng vuông góc dòng chảy, bề rộng danh nghĩa, chiều dài trụ"),
    ("y_m, V_ms", "m, m/s", "Chiều sâu và vận tốc dòng tới"),
    ("ys_m, ys_accuracy_m", "m", "Độ sâu xói đo và độ chính xác"),
    ("Vc_ms_hec18", "m/s", "Vận tốc tới hạn TỰ TÍNH: 6.19·y^(1/6)·D50^(1/3) (HEC-18), không có trong file gốc; NaN với đất dính"),
    ("ys_over_bn, y_over_bn, Fr, V_over_Vc, bn_over_D50, L_over_b", "-", "Biến không thứ nguyên"),
    ("ys_hec18_csu_m", "m", "HEC-18 CSU dạng đơn giản (K1 theo mũi, K2 theo θ, K3=1.1, không Kw/K4); chỉ để tham khảo"),
    ("ratio_obs_hec18", "-", "ys đo / ys HEC-18"),
    ("flag_*", "", "Cờ chất lượng; xem báo cáo"),
    ("cột gốc *_ft, *_fts, *_raw", "", "Giữ nguyên giá trị gốc để đối chiếu"),
], columns=["Cột", "Đơn vị", "Ý nghĩa"])

with pd.ExcelWriter(OUT / "psdb2014_clean.xlsx") as w:
    f.to_excel(w, sheet_name="Field_full", index=False)
    f[f["in_core"]].to_excel(w, sheet_name="Field_core", index=False)
    l.to_excel(w, sheet_name="Lab_full", index=False)
    dd.to_excel(w, sheet_name="Data_dictionary", index=False)

# ---------------------------------------------------------- TÓM TẮT IN RA
print("FIELD rows:", len(f), " core:", int(f["in_core"].sum()))
print("site_groups:", f["site_group"].nunique(), " pseudo groups:", f.loc[f["site_group_is_pseudo"], "site_group"].nunique())
print("core site_groups:", f.loc[f["in_core"], "site_group"].nunique())
print(f[[k for k in flags]].sum().to_string())
print("core by country:\n", f[f["in_core"]]["country"].value_counts().to_string())
print("core bed:\n", f[f["in_core"]]["bed"].value_counts(dropna=False).to_string())
print("core nose:\n", f[f["in_core"]]["nose"].value_counts(dropna=False).to_string())
c = f[f["in_core"]]
print("core describe:\n", c[["bn_m", "y_m", "V_ms", "D50_mm", "ys_m", "ys_over_bn", "y_over_bn", "Fr", "V_over_Vc"]]
      .describe(percentiles=[.05, .5, .95]).T.round(3).to_string())
print("core V/Vc>1 (live-bed):", int((c["V_over_Vc"] > 1).sum()), " <=1:", int((c["V_over_Vc"] <= 1).sum()))
r = c["ratio_obs_hec18"].replace([np.inf, -np.inf], np.nan).dropna()
print("core HEC ratio n, median, p95, frac>1:", len(r), round(r.median(), 3), round(r.quantile(.95), 3), round((r > 1).mean(), 3))
print(c.groupby(pd.cut(c["bn_m"], [0, 1, 2, 3, 5, 30]), observed=True)["ratio_obs_hec18"].agg(["median", "count"]).round(3).to_string())
print("core bn>5 m:", int((c["bn_m"] > 5).sum()), " >9 m:", int((c["bn_m"] > 9).sum()))
print("LAB rows:", len(l), " core:", int(l["in_core"].sum()))
print(l[["flag_duration_missing", "flag_duration_lt_600min", "flag_ys_over_b_gt3", "flag_sheppard_not_used"]].sum().to_string())
print(l["regime"].value_counts().to_string())
