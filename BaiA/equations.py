"""Công thức xói trụ cục bộ dùng làm mốc so sánh cho bài A (đơn vị SI: m, m/s, mm cho D50).

Mọi hàm nhận pandas Series / numpy array và trả về độ sâu xói ys (m).
Đã đối chiếu với HEC-18 bản 5 (Arneson et al., 2012, FHWA-HIF-12-003) ngày 27/11/2026:
  - CSU Eq. 7.1/7.3, K1 Bảng 7.1, K2 Eq. 7.4 (L/a ≤ 12, K1 = 1 khi θ > 5°), K3 Bảng 7.3; K4 đã bị bỏ (mục 7.11).
  - Giới hạn ys ≤ 2.4a (Fr ≤ 0.8) / 3.0a (Fr > 0.8) chỉ cho trụ mũi tròn đặt thẳng dòng (Eq. 7.2).
  - Kw Eq. 7.20–7.21; áp dụng khi y/a < 0.8, a/D50 > 50, Fr < 1; tài liệu không chặn trên Kw.
  - FDOT Eq. 7.5–7.19 với Vc riêng (Eq. 7.12–7.14); Ksf Eq. 7.16–7.17; aproj Eq. 7.18.
  - Hạt thô Eq. 7.34: chỉ xói nước trong (V < Vc), D50 ≥ 20 mm, σ = D84/D50 ≥ 1.5.
  - Vc Eq. 6.1: Ku = 6.19 (SI); giới hạn dưới D50 = 0.2 mm (mục 6.3).
Còn giả định (HEC-18 không quy định): Ksf của trụ mũi nhọn lấy = 1.0.
"""
import numpy as np
import pandas as pd

G = 9.81


def critical_velocity_hec18(y, d50_mm):
    """HEC-18 Eq. 6.1: Vc = Ku y^(1/6) D50^(1/3), Ku = 6.19 (SI), D50 (m), giới hạn dưới D50 = 0.2 mm."""
    d50 = np.maximum(np.asarray(d50_mm, dtype=float), 0.2) / 1000.0
    y = np.asarray(y, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 6.19 * np.where(y > 0, y, np.nan) ** (1 / 6) * d50 ** (1 / 3)


def critical_velocity_fdot(y, d50_mm):
    """HEC-18 Eq. 7.12–7.14 (FDOT): Vc = 5.75 u*c log10(5.53 y / D50);
    u*c = Ku (0.0377 + 0.041 D50^1.4) với 0.1 < D50 < 1 mm; u*c = Ku (0.1 D50^0.5 − 0.0213/D50) với 1 < D50 < 100 mm;
    Ku = 0.3048 (SI), D50 tính bằng mm trong u*c. Ngoài khoảng 0.1–100 mm: chặn D50 về biên."""
    d = np.clip(np.asarray(d50_mm, dtype=float), 0.1, 100.0)
    y = np.asarray(y, dtype=float)
    ustar = np.where(d < 1.0, 0.3048 * (0.0377 + 0.041 * d ** 1.4), 0.3048 * (0.1 * d ** 0.5 - 0.0213 / d))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 5.75 * ustar * np.log10(5.53 * y / (d / 1000.0))


def k1_hec18(nose, theta):
    """K1 theo mũi trụ; bằng 1.0 khi góc lệch > 5 độ."""
    k1 = pd.Series(nose).map({"square": 1.1, "round": 1.0, "circular": 1.0, "sharp": 0.9}).fillna(1.0).to_numpy()
    return np.where(np.nan_to_num(theta) > 5, 1.0, k1)


def k2_hec18(a, L, theta):
    """K2 = (cos θ + L/a sin θ)^0.65, L/a tối đa 12."""
    th = np.radians(np.nan_to_num(theta))
    LoA = np.minimum(np.nan_to_num(np.asarray(L, float) / np.asarray(a, float), nan=1.0), 12.0)
    return (np.cos(th) + LoA * np.sin(th)) ** 0.65


def kw_hec18(y, a, Fr, VVc):
    """HEC-18 Eq. 7.20–7.21. V/Vc < 1: Kw = 2.58 (y/a)^0.34 Fr^0.65 ; V/Vc ≥ 1: Kw = 1.0 (y/a)^0.13 Fr^0.25.
    Điều kiện áp dụng kiểm tra ở hàm gọi. HEC-18 không quy định chặn trên."""
    ya = np.asarray(y, float) / np.asarray(a, float)
    return np.where(np.asarray(VVc) < 1, 2.58 * ya ** 0.34 * np.asarray(Fr) ** 0.65, ya ** 0.13 * np.asarray(Fr) ** 0.25)


def hec18_csu(a, L, theta, nose, y, V, d50_mm, use_kw=True, cap=True, K3=1.1):
    """HEC-18 CSU Eq. 7.3: ys/a = 2.0 K1 K2 K3 (y/a)^0.35 Fr^0.43, nhân Kw khi thỏa điều kiện trụ rộng.
    cap (Eq. 7.2): ys ≤ 2.4a (Fr ≤ 0.8) / 3.0a (Fr > 0.8), chỉ cho trụ mũi tròn / tròn đặt thẳng dòng (θ ≤ 5°)."""
    a = np.asarray(a, float); y = np.asarray(y, float); V = np.asarray(V, float)
    Fr = V / np.sqrt(G * y)
    ys = 2.0 * k1_hec18(nose, theta) * k2_hec18(a, L, theta) * K3 * a * (y / a) ** 0.35 * Fr ** 0.43
    if use_kw:
        Vc = critical_velocity_hec18(y, d50_mm)
        aD = a / (np.asarray(d50_mm, float) / 1000.0)
        cond = (y / a < 0.8) & (aD > 50) & (Fr < 1)
        ys = np.where(cond, ys * kw_hec18(y, a, Fr, V / Vc), ys)
    if cap:
        roundish = pd.Series(nose).isin(["round", "circular"]).to_numpy() & (np.nan_to_num(theta) <= 5)
        ys = np.where(roundish, np.minimum(ys, np.where(Fr <= 0.8, 2.4, 3.0) * a), ys)
    return ys


def hec18_coarse_bed(a, L, theta, nose, y, V, d50_mm, d84_mm, Sg=2.65):
    """HEC-18 Eq. 7.34: ys/y1 = 1.1 K1 K2 (a/y1)^0.62 tanh(H^2 / (1.97 σ^1.5)),
    H = V1 / sqrt(g (Sg−1) D50), σ = D84/D50. Chỉ áp dụng khi xói nước trong (V < Vc), D50 ≥ 20 mm, σ ≥ 1.5.
    Ngoài phạm vi trả NaN."""
    a = np.asarray(a, float); y = np.asarray(y, float); V = np.asarray(V, float)
    d50_mm = np.asarray(d50_mm, float); d50 = d50_mm / 1000.0
    sig = np.asarray(d84_mm, float) / d50_mm
    H = V / np.sqrt(G * (Sg - 1) * d50)
    with np.errstate(invalid="ignore"):
        ys = 1.1 * k1_hec18(nose, theta) * k2_hec18(a, L, theta) * y * (a / y) ** 0.62 * np.tanh(H ** 2 / (1.97 * sig ** 1.5))
    ok = (d50_mm >= 20) & (sig >= 1.5) & (V < critical_velocity_hec18(y, d50_mm))
    return np.where(ok, ys, np.nan)


def sheppard_melville(bn, nose, theta, y, V, d50_mm):
    """FDOT / Sheppard–Melville, HEC-18 Eq. 7.5–7.19.
    a* = Ksf · aproj (aproj = a cosθ + L sinθ = bn); Ksf = 1.0 cho tròn / mũi tròn; mũi vuông: 0.86 + 0.97 |θπ/180 − π/4|^4.
    Mũi nhọn: HEC-18 không quy định, giả định Ksf = 1.0. Vc theo Eq. 7.12–7.14; Vlp = max(5 Vc, 0.6 sqrt(g y))."""
    bn = np.asarray(bn, float); y = np.asarray(y, float); V = np.asarray(V, float)
    d50 = np.asarray(d50_mm, float) / 1000.0
    th = np.radians(np.nan_to_num(theta))
    ks = np.where(pd.Series(nose).eq("square").to_numpy(), 0.86 + 0.97 * np.abs(th - np.pi / 4) ** 4, 1.0)
    astar = ks * bn
    Vc = critical_velocity_fdot(y, d50_mm)
    Vlp = np.maximum(5 * Vc, 0.6 * np.sqrt(G * y))
    r = V / Vc; rlp = Vlp / Vc
    f1 = np.tanh((y / astar) ** 0.4)
    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = 1 - 1.2 * np.log(r) ** 2
        ad = astar / d50
        f3 = ad / (0.4 * ad ** 1.2 + 10.6 * ad ** (-0.13))
        cw = 2.5 * f1 * f2 * f3
        lb = f1 * (2.2 * (r - 1) / (rlp - 1) + 2.5 * f3 * (rlp - r) / (rlp - 1))
        ysa = np.select([r < 0.4, r < 1, r <= rlp], [0.0, cw, lb], default=2.2 * f1)
    return np.maximum(ysa, 0) * astar


def critical_velocity_gavriel(d50_mm):
    """Vc (m/s) Gavriel (2025, luận án Bristol, Bảng 2.10) dùng cho CIRIA: 0.1 D50^-0.2 (D50 < 1 mm), 0.35 D50^0.45 (D50 ≥ 1 mm); D50 tính bằng mm."""
    d = np.asarray(d50_mm, float)
    return np.where(d < 1, 0.1 * d ** (-0.2), 0.35 * d ** 0.45)


def ciria_c742(a, L, theta, nose, y, V, d50_mm, vc="hec18"):
    """CIRIA C742 (Kirby et al., 2015; theo Breusers et al., 1977), dạng ghi trong CD 356 Eq. A.2 và luận án Gavriel (2025, mục 2.5.1):
    ys = a · K_shape · K_depth · K_velocity · K_angle  (hệ số an toàn SF = 1, tức ước lượng trung bình).
      K_shape  = 1.5 mũi tròn / tròn, 2.0 mũi vuông, 1.075 mũi nhọn (Gavriel Bảng 2.7; C551 Bảng 4.3: tròn 1.5, vuông 2.0, nhọn 1.0–1.15);
      K_depth  = 0.55 (y/a)^0.6 khi y/a ≤ 2.7, ngược lại 1.0 (May & Willoughby 1990; C551 Eq. 4.49);
      K_velocity = 0 (V/Vc ≤ 0.375), 1.6 V/Vc − 0.6 (0.375 < V/Vc ≤ 1), 1.0 (V/Vc > 1) (C551 Eq. 4.50; Gavriel Bảng 2.9);
      K_angle  = (cos θ + L/a sin θ)^0.62 (Gavriel Eq. 2.12); L thiếu → L/a = 1.
    vc: "hec18" dùng Vc theo HEC-18 Eq. 6.1 (mặc định, cùng Vc với các biến của bài A); "gavriel" dùng Vc của Gavriel (2025)."""
    a = np.asarray(a, float); y = np.asarray(y, float); V = np.asarray(V, float)
    ksh = pd.Series(nose).map({"square": 2.0, "round": 1.5, "circular": 1.5, "sharp": 1.075}).fillna(1.5).to_numpy()
    ya = y / a
    kdep = np.where(ya <= 2.7, 0.55 * ya ** 0.6, 1.0)
    Vc = critical_velocity_hec18(y, d50_mm) if vc == "hec18" else critical_velocity_gavriel(d50_mm)
    r = V / Vc
    kvel = np.where(r <= 0.375, 0.0, np.where(r <= 1, 1.6 * r - 0.6, 1.0))
    th = np.radians(np.nan_to_num(theta))
    LoA = np.nan_to_num(np.asarray(L, float) / a, nan=1.0)
    kang = (np.cos(th) + LoA * np.sin(th)) ** 0.62
    return a * ksh * kdep * kvel * kang


def froehlich(a, bn, nose, y, V, d50_mm, design=False):
    """Froehlich (1988): ys/a = 0.32 φ (a'/a)^0.62 (y/a)^0.47 Fr^0.22 (a/D50)^0.09 (+1 khi design).
    φ = 1.3 mũi vuông, 1.0 tròn, 0.7 mũi nhọn; a' = bn (bề rộng chiếu)."""
    a = np.asarray(a, float); bn = np.asarray(bn, float); y = np.asarray(y, float); V = np.asarray(V, float)
    d50 = np.asarray(d50_mm, float) / 1000.0
    phi = pd.Series(nose).map({"square": 1.3, "round": 1.0, "circular": 1.0, "sharp": 0.7}).fillna(1.0).to_numpy()
    Fr = V / np.sqrt(G * y)
    ysa = 0.32 * phi * (bn / a) ** 0.62 * (y / a) ** 0.47 * Fr ** 0.22 * (a / d50) ** 0.09
    return (ysa + (1.0 if design else 0.0)) * a
