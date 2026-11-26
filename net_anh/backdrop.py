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


def _blob_filter(alpha: np.ndarray, r: int) -> None:
    """Bụi, vết bẩn trên phông (mảng nhỏ rời khỏi người): bỏ; sợi tóc đứt rời nhưng nằm sát người thì giữ."""
    fg = (alpha > 0.5).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 2:
        return
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = int(np.argmax(areas)) + 1
    near = cv2.dilate((lbl == big).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1))) > 0
    small = np.where(areas < 0.05 * areas.max())[0] + 1
    drop = [k for k in small if not near[lbl == k].any()]
    if drop:
        rm = cv2.dilate(np.isin(lbl, drop).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        alpha[rm & ~near] = 0.0


def _matte(img_bgr: np.ndarray, bd: Backdrop) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(alpha, nền cũ tại chỗ, màu người lan ra viền, vùng người chắc chắn). Mọi bước làm ở độ phân giải của ảnh vào,
    bề rộng dải chuyển tính theo kích thước ảnh nên ảnh phóng 2×/4× vẫn có viền mềm đúng cỡ."""
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    lab = _lab(img_bgr)
    d = _dist(lab, bd.color_lab)
    t_lo = float(np.clip(3.0 * bd.spread, 10.0, 16.0))  # vân, vết bẩn trên phông lệch tới ~10 ΔE vẫn là nền
    t_hi = t_lo + 14.0
    # cạnh rõ giữa nền và vật thể: cắt vùng nền ở đó để mảng cùng màu nền nhưng có biên rõ (áo xanh trên phông xanh) không
    # bị nền "tràn" vào; nhiễu, vân phông có độ dốc thấp hơn nhiều nên không bị cắt
    sg = max(1.0, 0.0015 * m)
    gl = cv2.GaussianBlur(lab, (0, 0), sg)
    gl[..., 0] *= L_WEIGHT
    gx = cv2.Sobel(gl, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gl, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((gx * gx + gy * gy).sum(axis=2)) / 8.0
    ring = _ring_mask(h, w) & (d < t_lo)
    noise = float(np.median(grad[ring])) if ring.any() else 0.5
    edge = grad > max(0.6, 8.0 * noise)
    cut = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cand = ((d < t_hi) & ~cut).astype(np.uint8)
    seeds = _ring_mask(h, w) & (d < t_hi) & ~cut
    n, lbl = cv2.connectedComponents(cand, connectivity=8)
    keep = np.unique(lbl[seeds])
    keep = keep[keep > 0]
    bgcc = np.isin(lbl, keep) if len(keep) else np.zeros((h, w), bool)
    # mảnh nền bị cắt rời nhưng đúng màu nền (d rất nhỏ: khe giữa các lọn tóc, nền sau tay) vẫn là nền; mảnh lệch màu rõ
    # (áo xanh gần phông) thì giữ là người
    plain = ((d < t_hi)).astype(np.uint8)
    n1, l1 = cv2.connectedComponents(plain, connectivity=8)
    keep1 = np.unique(l1[_ring_mask(h, w) & (d < t_hi)])
    plain_bg = np.isin(l1, keep1[keep1 > 0]) if (keep1 > 0).any() else np.zeros((h, w), bool)
    frag = plain_bg & ~bgcc
    if frag.any():
        nf, lf = cv2.connectedComponents(frag.astype(np.uint8), connectivity=8)
        fid = lf[frag]
        dv = d[frag]
        med = np.zeros(nf, np.float32)
        order = np.argsort(fid)
        fs, ds = fid[order], dv[order]
        bounds = np.flatnonzero(np.diff(fs)) + 1
        for ids, vals in zip(np.split(fs, bounds), np.split(ds, bounds)):
            med[ids[0]] = np.median(vals)
        good = np.flatnonzero(med < 0.4 * t_lo)
        good = good[good > 0]
        if len(good):
            bgcc = bgcc | np.isin(lf, good)
    core = bgcc & (d < t_lo)
    core = core | ((cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & cut & (d < t_lo))
    # khe nền kín (trong lọn tóc, giữa tay và thân): mảng nhỏ đúng màu nền cũng là nền
    tight = ((d < t_lo) & ~bgcc).astype(np.uint8)
    if tight.any():
        n2, l2, st2, _ = cv2.connectedComponentsWithStats(tight, connectivity=8)
        pocket = np.where((st2[1:, cv2.CC_STAT_AREA] <= 0.004 * h * w) & (st2[1:, cv2.CC_STAT_AREA] >= 6))[0] + 1
        if len(pocket):
            core = core | np.isin(l2, pocket)
    # tàn dư cùng sắc với nền nhưng sáng hơn (vượt ngưỡng ΔE) sát mép nền: coi là nền
    bab = bd.color_lab[1:3].astype(np.float32)
    if np.hypot(*bab) > 20.0:
        ab = lab[..., 1:3].astype(np.float32)
        cos = (ab @ bab) / (np.hypot(ab[..., 0], ab[..., 1]) * np.hypot(*bab) + 1e-6)
        hue_bg = (cos > 0.93) & (np.hypot(ab[..., 0], ab[..., 1]) > 0.5 * np.hypot(*bab)) & (d < t_hi + 25.0)
        k = 2 * max(2, int(0.004 * m)) + 1
        zone = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        core = core | (hue_bg & zone & ~(d > t_hi + 25.0))
    R = max(3, int(round(0.005 * m)))  # nửa bề rộng dải chuyển (điểm ảnh của ảnh vào)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
    core8 = core.astype(np.uint8)
    grow_bg = cv2.dilate(core8, ker) > 0
    grow_fg = cv2.dilate(1 - core8, ker) > 0
    unk = grow_bg & grow_fg  # dải chuyển: cách mép nền không quá R, cả hai phía
    solid = ~grow_bg  # người chắc chắn (cách nền hơn R)
    # nền cũ tại chỗ (có bóng đổ) và màu người lan ra viền (chuẩn hoá theo vùng người chắc chắn, hai cỡ làm mờ)
    img = img_bgr.astype(np.float32)
    b_old = _local_bg(img_bgr, (~core).astype(np.float32), np.array(bd.color_bgr, np.float32))
    sf = float(max(2.0, 0.35 * R))
    sm = solid.astype(np.float32)
    num1 = cv2.GaussianBlur(img * sm[..., None], (0, 0), sf)
    den1 = cv2.GaussianBlur(sm, (0, 0), sf)[..., None]
    num2 = cv2.GaussianBlur(img * sm[..., None], (0, 0), 4 * sf)
    den2s = cv2.GaussianBlur(sm, (0, 0), 4 * sf)[..., None]
    f1 = num1 / np.maximum(den1, 1e-4)
    f2 = num2 / np.maximum(den2s, 1e-4)
    wgt1 = np.clip(den1 / 0.25, 0.0, 1.0)
    f_prop = np.where(den2s > 1e-3, wgt1 * f1 + (1 - wgt1) * f2, img).astype(np.float32)
    f_prop = np.where(solid[..., None], img, f_prop)
    # alpha ở dải chuyển theo phép chiếu: điểm = α·người + (1−α)·nền với người, nền đã biết (cho α liên tục, không nhị phân)
    a_bin = cv2.GaussianBlur((~core).astype(np.float32), (0, 0), max(0.8, 0.0008 * m))
    dfb = f_prop - b_old
    den2 = np.sum(dfb * dfb, axis=2)
    a_proj = np.clip(np.sum((img - b_old) * dfb, axis=2) / np.maximum(den2, 1.0), 0.0, 1.0)
    trust = np.clip((np.sqrt(den2) - 25.0) / 20.0, 0.0, 1.0)  # người và nền gần màu nhau: phép chiếu không tin được
    a_u = trust * a_proj + (1.0 - trust) * a_bin
    alpha = np.where(unk, a_u, (~core).astype(np.float32)).astype(np.float32)
    # viền bám cạnh thật: lọc dẫn hướng theo màu ảnh (xám), chỉ trong dải chuyển
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    band = cv2.dilate(unk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ref = np.clip(_guided(gray, alpha, max(2, R // 3), 2e-4), 0.0, 1.0)
    alpha = np.where(band, 0.5 * ref + 0.5 * alpha, alpha)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.6, 0.0005 * m)) * band + alpha * (~band)
    # bụi nhiễu rất mờ → 0 và đặc hơn 0,95 → 1, phần còn lại giữ nguyên (không kéo giãn mạnh: sợi tóc mờ vẫn còn)
    alpha = np.clip((alpha - 0.06) / 0.90, 0.0, 1.0).astype(np.float32)
    _blob_filter(alpha, max(2, R // 2))
    return alpha, b_old, f_prop, solid


def alpha_matte(img_bgr: np.ndarray, bd: Backdrop) -> np.ndarray:
    """Mặt nạ người 0..1 (1 = người, 0 = nền) cho ảnh có nền đồng màu `bd`."""
    return _matte(img_bgr, bd)[0]


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
    alpha, b_old, f_prop, solid = _matte(img_bgr, bd)
    fg_frac = float(alpha.mean())
    if fg_frac > 0.97:  # gần như không tách được nền (viền toàn người?)
        return img_bgr, "Không đổi nền: không tách được vùng nền khỏi người"
    img = img_bgr.astype(np.float32)
    a = alpha[..., None]
    # màu người ở viền: trừ phần nền cũ (khử ám nền); chỗ alpha nhỏ không tin phép chia → màu người đặc gần nhất
    f_formula = np.clip((img - (1.0 - a) * b_old) / np.maximum(a, 0.08), 0, 255)
    t = np.clip((a - 0.45) / 0.45, 0.0, 1.0)  # viền (α thấp, lẫn nền) lấy màu người sạch lân cận
    f = t * f_formula + (1.0 - t) * f_prop
    f = np.where(solid[..., None] & (a > 0.985), img, f)
    new = np.array(color_bgr, np.float32).reshape(1, 1, 3)
    out = a * f + (1.0 - a) * new
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), f"Đã đổi màu nền ({100 * (1 - fg_frac):.0f} % diện tích ảnh)"


__all__ = ["Backdrop", "NAMED", "alpha_matte", "detect", "parse_color", "replace"]
