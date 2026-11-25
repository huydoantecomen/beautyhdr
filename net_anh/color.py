"""Tăng cường màu và độ bóng da (bước cuối, sau làm nét), kiểu "ảnh đẹp" của Remini.

- color (0..1): cân bằng trắng tự động + giãn mức sáng + độ rực (vibrance: màu nhạt tăng nhiều, màu đã rực và
  màu da tăng ít) + rõ khối (clarity: tương phản cục bộ vùng tông giữa) + đường cong chữ S nhẹ.
- gloss (0..1): độ bóng da: vùng da được làm mịn tầng tần số thấp (giữ nếp nhăn/vân da ở tầng cao) và thêm
  ánh sáng loang nhẹ ở vùng sáng của da (bloom), da hơi ấm và sáng lên.
Cả hai bằng 0 thì ảnh giữ nguyên.
"""
from __future__ import annotations

import cv2
import numpy as np


def skin_mask(img: np.ndarray) -> np.ndarray:
    """Mặt nạ da 0..1 theo YCrCb (khoảng da người các tông màu), làm mềm viền."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    m = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y >= 50)).astype(np.float32)
    s = max(2.0, 0.004 * max(img.shape[:2]))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), s)


def white_balance(lab: np.ndarray, strength: float) -> np.ndarray:
    """Khử ám màu, ước lượng chỉ trên các điểm gần trung tính (xám/trắng) để không xoá màu thật của cảnh
    (trời xanh, cỏ, áo đỏ). Ít điểm trung tính quá thì bỏ qua."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    neutral = (L > 25) & (L < 95) & (np.sqrt(a * a + b * b) < 12)
    if neutral.mean() < 0.02:
        return lab
    da = float(np.clip(np.median(a[neutral]), -8, 8))
    db = float(np.clip(np.median(b[neutral]) - 2.0, -8, 8))  # giữ chút ấm cho đẹp da
    out = lab.copy()
    out[..., 1] -= strength * da
    out[..., 2] -= strength * db
    return out


def levels(L: np.ndarray, strength: float) -> np.ndarray:
    lo, hi = np.percentile(L, (0.5, 99.5))
    lo, hi = min(lo, 12.0), max(hi, 88.0)
    stretched = (L - lo) * (100.0 / max(hi - lo, 1e-3))
    return L + strength * (np.clip(stretched, 0, 100) - L)


def s_curve(L: np.ndarray, amount: float) -> np.ndarray:
    x = L / 100.0
    y = x + amount * (0.5 - 0.5 * np.cos(np.pi * x) - x)  # trộn với đường cong cos (chữ S)
    return np.clip(y, 0, 1) * 100.0


CLARITY_EPS = 12.0 ** 2  # (đơn vị L 0..100)²: cạnh tương phản mạnh hơn ~12 L được giữ trong lớp nền → không có quầng


def guided_smooth(L: np.ndarray, radius: int, eps: float = CLARITY_EPS) -> np.ndarray:
    """Lọc dẫn hướng (He và cộng sự 2010) tự dẫn: làm mờ vân và mảng nhỏ nhưng giữ cạnh mạnh (tóc trên nền sáng)."""
    L = L.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mean = cv2.boxFilter(L, -1, k)
    var = cv2.boxFilter(L * L, -1, k) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    return cv2.boxFilter(a, -1, k) * L + cv2.boxFilter(b, -1, k)


def clarity(L: np.ndarray, amount: float, size: int) -> np.ndarray:
    """Rõ khối: tăng tương phản cục bộ quanh lớp nền. Lớp nền từng là Gauss σ 1,2 % cạnh ảnh: cạnh tối (tóc, vai) trên
    nền sáng bị nền sáng thêm thành quầng rộng hàng chục px (ảnh thẻ phông xanh của huy). Nay nền lấy bằng lọc dẫn
    hướng giữ cạnh: vẫn rõ khối ở vùng chuyển nhẹ (má, nếp áo) mà không loé sáng quanh viền tóc."""
    base = guided_smooth(L, max(4, int(0.012 * size * 1.5)))
    mid = 1.0 - np.abs(L / 50.0 - 1.0) ** 2  # mạnh ở tông giữa, nhẹ ở vùng rất tối/rất sáng
    return L + amount * np.clip(mid, 0, 1) * (L - base)


def vibrance(lab: np.ndarray, amount: float, skin: np.ndarray) -> np.ndarray:
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    k = 1.0 + amount * np.clip(1.0 - chroma / 60.0, 0, 1) * (1.0 - 0.6 * skin)
    out = lab.copy()
    out[..., 1] = a * k
    out[..., 2] = b * k
    return out


def gloss_skin(lab: np.ndarray, amount: float, skin: np.ndarray, size: int) -> np.ndarray:
    L = lab[..., 0]
    s = max(3.0, 0.006 * size)
    low = cv2.bilateralFilter(L, d=0, sigmaColor=8.0, sigmaSpace=s)
    smooth_low = cv2.GaussianBlur(low, (0, 0), s)
    detail = L - cv2.GaussianBlur(L, (0, 0), s)  # vân da, nếp nhăn: giữ lại
    L2 = L + skin * amount * 0.6 * ((smooth_low + detail) - L)
    # ánh bóng: vùng sáng của da loang nhẹ (screen)
    hl = np.clip((L2 - 60.0) / 40.0, 0, 1) * skin
    bloom = cv2.GaussianBlur(hl, (0, 0), 2.5 * s)
    L2 = 100.0 - (100.0 - L2) * (1.0 - amount * 0.35 * bloom)
    L2 = L2 + skin * amount * 2.0  # da sáng lên một chút
    out = lab.copy()
    out[..., 0] = L2
    out[..., 2] = lab[..., 2] + skin * amount * 1.5  # hơi ấm
    return out


def enhance(img: np.ndarray, color: float = 0.0, gloss: float = 0.0) -> np.ndarray:
    color = float(np.clip(color, 0, 1))
    gloss = float(np.clip(gloss, 0, 1))
    if color <= 1e-3 and gloss <= 1e-3:
        return img
    size = max(img.shape[:2])
    f = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)  # L 0..100, a/b ≈ −128..127
    skin = skin_mask(img)
    if color > 1e-3:
        lab = white_balance(lab, 0.6 * color)
        L = levels(lab[..., 0], 0.7 * color)
        L = clarity(L, 0.5 * color, size)
        L = s_curve(L, 0.25 * color)
        lab[..., 0] = np.clip(L, 0, 100)
        lab = vibrance(lab, 0.45 * color, skin)
    if gloss > 1e-3:
        lab = gloss_skin(lab, gloss, skin, size)
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
