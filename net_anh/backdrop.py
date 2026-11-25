"""Đổi màu nền cho ảnh có nền đồng màu (ảnh thẻ, phông xanh, trắng, đỏ…), không cần model AI.

1. Dò nền: lấy mẫu viền ảnh (cạnh trên và hai bên phần trên, bỏ đáy vì thường là vai, áo). Màu nền = màu trung vị trong
   Lab của nhóm điểm gần nhau nhất; nền "đồng màu" khi ít nhất 60 % mẫu viền nằm trong dải sai khác nhỏ quanh màu đó
   (vân, bóng đổ nhẹ vẫn được). Không đạt thì không đổi gì và báo lý do.
2. Tách: khoảng cách màu tới nền (Lab, độ sáng nhẹ hơn) rồi chỉ lấy vùng nối liền với viền: mảng cùng màu nền nằm trong
   người (áo xanh trên phông xanh) không bị tính là nền. Quanh vùng nền, mặt nạ mờ dần (alpha 0..1) theo khoảng cách màu
   nên sợi tóc, viền vai giữ độ mềm; lọc dẫn hướng theo ảnh xám để viền bám cạnh thật.
3. Gỡ ám nền ở viền: ở điểm viền, màu ảnh = α·người + (1−α)·nền cũ, nên trừ phần nền cũ (ước lượng tại chỗ, theo
   bóng đổ của phông) để không còn quầng xanh quanh tóc khi đặt sang nền mới; chỗ α nhỏ, nơi phép trừ không tin được,
   lấy màu người từ vùng đặc gần nhất.
4. Ghép lên màu mới: ra = α·người + (1−α)·nền mới.
Không dò được nền đồng màu: trả ảnh nguyên và nói rõ lý do. Vùng nền kín không chạm viền (khe giữa tay và thân) giữ nguyên,
vì không phân biệt chắc được với áo cùng màu.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

L_WEIGHT = 0.5  # độ sáng nhẹ hơn màu khi đo khoảng cách tới nền (phông thường có bóng đổ)
UNIFORM_MIN = 0.60  # tỉ lệ mẫu viền tối thiểu thuộc màu nền để coi là đồng màu
RING_FRAC = 0.04  # bề dày dải lấy mẫu (phần cạnh ảnh)
INLIER_DE = 16.0

NAMED = {"trang": (255, 255, 255), "white": (255, 255, 255), "xanh": (219, 142, 67), "blue": (219, 142, 67),
         "xanh-nhat": (235, 206, 135), "do": (30, 30, 200), "red": (30, 30, 200), "xam": (200, 200, 200),
         "gray": (200, 200, 200)}


def parse_color(text: str) -> tuple[int, int, int]:
    """"#RRGGBB", "RRGGBB" hoặc tên (trang, xanh, xanh-nhat, do, xam) → (B, G, R)."""
    t = text.strip().lower()
    if t in NAMED:
        return NAMED[t]
    t = t.lstrip("#")
    if len(t) == 6 and all(c in "0123456789abcdef" for c in t):
        r, g, b = int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
        return b, g, r
    raise ValueError(f"không hiểu màu “{text}” (dùng #RRGGBB hoặc: {', '.join(sorted(set(NAMED)))})")


@dataclass
class Backdrop:
    color_lab: np.ndarray  # màu nền ước lượng (Lab, L 0..100)
    color_bgr: tuple[int, int, int]
    spread: float  # độ lệch điển hình quanh màu nền (ΔE)
    uniform: float  # tỉ lệ mẫu viền thuộc màu nền


def _lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _dist(lab: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = lab - c
    return np.sqrt((L_WEIGHT * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _ring_mask(h: int, w: int) -> np.ndarray:
    t = max(2, int(RING_FRAC * min(h, w)))
    m = np.zeros((h, w), bool)
    m[:t, :] = True  # cạnh trên
    m[: int(h * 0.7), :t] = True  # hai bên, phần trên (đáy thường là vai, áo)
    m[: int(h * 0.7), w - t:] = True
    return m


def detect(img_bgr: np.ndarray) -> Backdrop | None:
    """Dò nền đồng màu từ viền ảnh; None nếu viền không đồng màu."""
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, None, fx=min(1.0, 400 / max(h, w)), fy=min(1.0, 400 / max(h, w)),
                       interpolation=cv2.INTER_AREA)
    lab = _lab(small)
    ring = _ring_mask(*small.shape[:2])
    px = lab[ring]
    if len(px) < 50:
        return None
    c0 = np.median(px, axis=0)
    inl = _dist(px, c0) < INLIER_DE
    if inl.mean() < UNIFORM_MIN:
        return None
    c = np.mean(px[inl], axis=0)
    d = _dist(px, c)
    inl = d < INLIER_DE
    if inl.mean() < UNIFORM_MIN:
        return None
    bgr = cv2.cvtColor(c.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2BGR).reshape(3)
    bgr = tuple(int(np.clip(v * 255 + 0.5, 0, 255)) for v in bgr)
    return Backdrop(c.astype(np.float32), bgr, float(np.percentile(d[inl], 95)), float(inl.mean()))


def _guided(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    k = (2 * r + 1, 2 * r + 1)
    mg, ms = cv2.boxFilter(guide, -1, k), cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(guide * src, -1, k) - mg * ms
    var = cv2.boxFilter(guide * guide, -1, k) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def alpha_matte(img_bgr: np.ndarray, bd: Backdrop) -> np.ndarray:
    """Mặt nạ người 0..1 (1 = người, 0 = nền) cho ảnh có nền đồng màu `bd`."""
    h, w = img_bgr.shape[:2]
    lab = _lab(img_bgr)
    d = _dist(lab, bd.color_lab)
    t_lo = float(np.clip(3.0 * bd.spread, 10.0, 16.0))  # vân, vết bẩn trên phông lệch tới ~10 ΔE vẫn là nền
    t_hi = t_lo + 14.0
    cand = (d < t_hi).astype(np.uint8)
    seeds = _ring_mask(h, w) & (d < t_hi)
    n, lbl = cv2.connectedComponents(cand, connectivity=8)
    keep = np.unique(lbl[seeds])
    keep = keep[keep > 0]
    bgcc = np.isin(lbl, keep) if len(keep) else np.zeros((h, w), bool)
    # khe nền kín (trong lọn tóc, giữa tay và thân): mảng nhỏ đúng màu nền cũng là nền
    tight = ((d < t_lo) & ~bgcc).astype(np.uint8)
    if tight.any():
        n2, l2, st2, _ = cv2.connectedComponentsWithStats(tight, connectivity=8)
        pocket = np.where((st2[1:, cv2.CC_STAT_AREA] <= 0.004 * h * w) & (st2[1:, cv2.CC_STAT_AREA] >= 6))[0] + 1
        if len(pocket):
            grow = cv2.dilate(np.isin(l2, pocket).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            bgcc = bgcc | (grow & (d < t_hi + 10.0))
    # alpha mềm theo khoảng cách màu, chỉ trong vùng nền nối liền với viền và dải 3 px quanh nó
    a0 = np.clip((d - t_lo) / (t_hi - t_lo), 0.0, 1.0)
    r = max(2, int(0.004 * max(h, w)))
    near = cv2.dilate(bgcc.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    alpha = np.where(near, a0, 1.0).astype(np.float32)
    alpha[bgcc & (d < t_lo)] = 0.0
    # viền bám cạnh thật: lọc dẫn hướng theo ảnh xám, chỉ trong dải chuyển (0 < alpha < 1)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    band = (alpha > 0.0) & (alpha < 1.0)
    band = cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ref = np.clip(_guided(gray, alpha, max(2, r), 1e-4), 0.0, 1.0)
    alpha = np.where(band, ref, alpha)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    # tàn dư cùng sắc với nền (sáng hơn nên vượt ngưỡng ΔE) sát mép nền: coi là nền
    bab = bd.color_lab[1:3].astype(np.float32)
    if np.hypot(*bab) > 20.0:
        ab = lab[..., 1:3].astype(np.float32)
        cos = (ab @ bab) / (np.hypot(ab[..., 0], ab[..., 1]) * np.hypot(*bab) + 1e-6)
        hue_bg = (cos > 0.93) & (np.hypot(ab[..., 0], ab[..., 1]) > 0.5 * np.hypot(*bab))
        k = 2 * max(3, int(0.008 * max(h, w))) + 1
        zone = cv2.dilate((alpha < 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        alpha[hue_bg & zone] = 0.0
    # vết bẩn, bóng, bụi trên phông (mảng "không phải nền" rời khỏi người): bỏ các mảng nhỏ không dính khối người lớn nhất
    fg = (alpha > 0.02).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        small = np.where(areas < 0.05 * areas.max())[0] + 1
        if len(small):
            rm = np.isin(lbl, small).astype(np.uint8)
            rm = cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))) > 0
            alpha[rm] = 0.0
    return alpha


def _local_bg(img: np.ndarray, alpha: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Màu nền tại chỗ (có bóng đổ) dưới cả vùng người: làm mờ chuẩn hoá của các điểm nền chắc chắn."""
    h, w = alpha.shape
    m = (alpha < 0.02).astype(np.float32)
    s = max(6.0, 0.03 * max(h, w))
    num = cv2.GaussianBlur(img.astype(np.float32) * m[..., None], (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)[..., None]
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback.reshape(1, 1, 3))
    return out.astype(np.float32)


def replace(img_bgr: np.ndarray, color_bgr: tuple[int, int, int], bd: Backdrop | None = None
            ) -> tuple[np.ndarray, str]:
    """Đổi nền đồng màu của ảnh sang `color_bgr`. Trả (ảnh, ghi chú); không dò được nền thì (ảnh, lý do)."""
    bd = bd or detect(img_bgr)
    if bd is None:
        return img_bgr, ("Không đổi nền: viền ảnh không đồng màu (nền có vật thể, nhiều màu hoặc chuyển màu mạnh). "
                         "Đổi nền chỉ dùng cho ảnh thẻ, phông trơn")
    alpha = alpha_matte(img_bgr, bd)
    fg_frac = float(alpha.mean())
    if fg_frac > 0.97:  # gần như không tách được nền (viền toàn người?)
        return img_bgr, "Không đổi nền: không tách được vùng nền khỏi người"
    img = img_bgr.astype(np.float32)
    fallback = np.array(bd.color_bgr, np.float32)
    b_old = _local_bg(img_bgr, alpha, fallback)
    h, w = alpha.shape
    r = max(2, int(0.004 * max(h, w)))
    # màu người ở vùng đặc (đã xói r px vào trong: điểm sát viền còn lẫn màu nền), lan ra viền bằng làm mờ chuẩn hoá
    solid = cv2.erode((alpha > 0.97).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    s = max(3.0, 0.006 * max(h, w))
    num = cv2.GaussianBlur(img * solid[..., None], (0, 0), s)
    den = cv2.GaussianBlur(solid, (0, 0), s)[..., None]
    f_prop = np.where(den > 1e-3, num / np.maximum(den, 1e-3), img)
    # alpha theo phép chiếu ở dải viền: điểm = α·người + (1−α)·nền với người, nền đã biết. Điểm viền tóc lẫn 30 % nền
    # vẫn có khoảng cách màu lớn (alpha đo bằng khoảng cách bão hoà = 1) nhưng phép chiếu cho α ≈ 0,7: nhờ đó gỡ được
    # viền xanh sẫm quanh tóc. Người và nền gần màu nhau (mẫu số nhỏ) thì giữ alpha cũ.
    band = (alpha > 0.0) & (solid < 0.5)
    dfb = f_prop - b_old
    den2 = np.sum(dfb * dfb, axis=2)
    a_proj = np.sum((img - b_old) * dfb, axis=2) / np.maximum(den2, 1.0)
    a_proj = cv2.GaussianBlur(np.clip(a_proj, 0.0, 1.0).astype(np.float32), (0, 0), 0.8)
    use = band & (den2 > 30.0 ** 2)
    alpha = np.where(use, np.minimum(a_proj, 1.0), alpha).astype(np.float32)
    # sợi tóc rất mờ (α < 0,12) là bụi nền, không phải tóc: bỏ để không còn màng mờ quanh đầu; kéo giãn phần còn lại
    alpha = np.where(band, np.clip((alpha - 0.12) / 0.76, 0.0, 1.0), alpha).astype(np.float32)
    a = alpha[..., None]
    # màu người ở viền: trừ phần nền cũ; chỗ alpha nhỏ không tin phép chia → lấy màu người đặc gần nhất
    f_formula = np.clip((img - (1.0 - a) * b_old) / np.maximum(a, 0.05), 0, 255)
    t = np.clip((a - 0.15) / 0.35, 0.0, 1.0)  # alpha < 0,15 → màu lân cận; > 0,5 → công thức
    f = t * f_formula + (1.0 - t) * f_prop
    f = np.where((a > 0.985) & (solid[..., None] > 0.5), img, f)
    new = np.array(color_bgr, np.float32).reshape(1, 1, 3)
    out = a * f + (1.0 - a) * new
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), f"Đã đổi màu nền ({100 * (1 - fg_frac):.0f} % diện tích ảnh)"


__all__ = ["Backdrop", "NAMED", "alpha_matte", "detect", "parse_color", "replace"]
