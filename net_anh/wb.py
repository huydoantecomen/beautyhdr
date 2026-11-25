"""Cân bằng trắng tự động + sửa tông da (ảnh điện thoại ám vàng/đỏ).

1. Ước lượng màu nguồn sáng trên ảnh thu nhỏ, ở không gian tuyến tính (bỏ gamma 2,2):
   trung bình hình học của "shades of gray" (chuẩn Minkowski p = 6 của giá trị điểm ảnh) và "gray edge"
   (p = 6 của độ lớn đạo hàm). Gray edge ít bị kéo lệch bởi mảng màu lớn (tường vàng, da), shades of gray bổ sung khi
   ảnh ít cạnh. Bỏ điểm cháy sáng, điểm quá tối và vùng mặt dò được (da vốn ấm, để vào sẽ bị tính là ám).
   Đã thử thêm "grey pixel" (Yang 2015): trên bộ đo tools/wb_bench.py kém hơn nên không dùng.
2. Gỡ một phần ám màu: hệ số kênh g = (ước lượng trung tính / ước lượng)^wb, giới hạn 0,7–1,45, giữ độ sáng.
   Với gamma 2,2, nhân g ở không gian tuyến tính = nhân g^(1/2,2) ở giá trị điểm ảnh.
3. Sửa tông da trong vùng mặt dò được: góc màu (hue, Lab) của da ngoài dải tự nhiên 44–60° được kéo về mép dải
   (da quá đỏ: góc nhỏ; quá vàng: góc lớn), da quá đậm màu (chroma > 28) được giảm bớt.
4. Chỉnh độ bão hoà theo hãng (máy vốn rực màu thì giảm nhẹ).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

GAMMA = 2.2
# Dải góc màu (atan2(b, a), Lab) của da tự nhiên: đo trên 34 mặt người châu Á UTKFace, trung vị 50,5°,
# khoảng tứ phân vị 42–56°; chroma trung vị 21 (tứ phân vị 16–26). Chỉ sửa phần vượt ra ngoài dải.
SKIN_HUE_LO, SKIN_HUE_HI = 44.0, 60.0
# Da hồng hào (góc màu 36–44°) là sắc da đẹp người dùng muốn giữ: chỉ kéo da đỏ hẳn (dưới SKIN_RED_LO) về dải.
SKIN_RED_LO = 36.0
SKIN_CHROMA = 28.0


def estimate_illuminant(img: np.ndarray, p: float = 6.0, max_side: int = 640,
                        exclude: np.ndarray | None = None) -> np.ndarray:
    """Màu nguồn sáng ước lượng (B, G, R tuyến tính, chuẩn hoá tổng = 3). (1, 1, 1) = không ám màu.
    `exclude` (0..1, cỡ ảnh): vùng không dùng để ước lượng (vùng mặt: da vốn có màu ấm, để vào sẽ bị coi là ám)."""
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else img
    f = small.astype(np.float32) / 255.0
    mx = f.max(axis=2)
    valid = (mx < 0.95) & (mx > 0.04)  # bỏ điểm cháy sáng (đã mất màu thật) và điểm quá tối (nhiễu)
    if exclude is not None:
        ex = cv2.resize(exclude.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        if (valid & (ex < 0.3)).mean() >= 0.05:
            valid &= ex < 0.3
    if valid.mean() < 0.05:
        return np.ones(3, np.float32)
    lin = np.power(f, GAMMA)
    lin = cv2.GaussianBlur(lin, (0, 0), 1.0)
    sog = np.array([np.power(np.mean(np.power(lin[..., c][valid], p)), 1 / p) for c in range(3)], np.float64)
    edge = []
    for c in range(3):
        gx = cv2.Sobel(lin[..., c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lin[..., c], cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)[valid]
        edge.append(np.power(np.mean(np.power(mag, p)) + 1e-12, 1 / p))
    edge = np.array(edge, np.float64)
    sog /= sog.sum() / 3
    edge /= edge.sum() / 3
    est = np.sqrt(sog * edge)
    return (est / (est.sum() / 3)).astype(np.float32)


def cast_degrees(est: np.ndarray) -> float:
    """Góc lệch giữa nguồn sáng ước lượng và trắng trung tính (độ). < 2° coi như không ám."""
    v = est / np.linalg.norm(est)
    return float(math.degrees(math.acos(min(1.0, float(v.sum() / math.sqrt(3))))))


def cast_name(est: np.ndarray) -> str:
    b, g, r = (float(x) for x in est)
    parts = []
    if r > b * 1.04:
        parts.append("vàng" if g > (r + b) / 2 else "đỏ")
        if r > g * 1.06 and g > b * 1.06 and "vàng" in parts:
            parts.append("cam")
    elif b > r * 1.04:
        parts.append("xanh")
    if g > max(r, b) * 1.04:
        parts.append("xanh lá")
    elif g < min(r, b) * 0.96:
        parts.append("tím hồng")
    return "/".join(dict.fromkeys(parts)) or "nhẹ"


def gains_for(est: np.ndarray, strength: float) -> np.ndarray:
    """Hệ số nhân kênh (tuyến tính) để gỡ `strength` phần ám màu, giữ độ sáng (trọng số Rec.709)."""
    g = np.power(1.0 / np.maximum(est, 1e-6), float(np.clip(strength, 0, 1)))
    g = np.clip(g, 0.7, 1.45)
    lum = 0.0722 * g[0] + 0.7152 * g[1] + 0.2126 * g[2]
    return (g / lum).astype(np.float32)


def apply_gains(img: np.ndarray, gains: np.ndarray) -> np.ndarray:
    k = np.power(gains, 1.0 / GAMMA)
    lut = np.clip(np.arange(256, dtype=np.float32)[None, :] * k[:, None] + 0.5, 0, 255).astype(np.uint8)
    return cv2.merge([cv2.LUT(img[..., c], lut[c]) for c in range(3)])


def face_region_mask(shape: tuple[int, int], boxes: list) -> np.ndarray:
    """Vùng quanh mặt (khung nới 1,6×, gồm trán, cổ) làm mềm viền, 0..1."""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15 * (y2 - y1), (x2 - x1), (y2 - y1)
        cv2.ellipse(m, (int(cx), int(cy)), (int(0.8 * bw), int(0.95 * bh)), 0, 0, 360, 1.0, -1)
    if m.max() > 0:
        m = cv2.GaussianBlur(m, (0, 0), max(2.0, 0.05 * math.sqrt(m.sum())))
    return m


def _skin_weights(img: np.ndarray, boxes: list) -> np.ndarray | None:
    from .color import skin_mask
    skin = skin_mask(img) * face_region_mask(img.shape[:2], boxes)
    return skin if (skin > 0.5).sum() >= 200 else None


def skin_hue(img: np.ndarray, boxes: list, sel: np.ndarray | None = None) -> tuple[float, float] | None:
    """(góc màu, chroma) trung vị của da trong vùng mặt (Lab), None nếu không đủ điểm da."""
    if sel is None:
        w = _skin_weights(img, boxes) if boxes else None
        if w is None:
            return None
        sel = w > 0.5
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    ma, mb = float(np.median(lab[..., 1][sel])), float(np.median(lab[..., 2][sel]))
    return math.degrees(math.atan2(mb, ma)), math.hypot(ma, mb)


def skin_tone(img: np.ndarray, boxes: list, strength: float) -> tuple[np.ndarray, str]:
    """Kéo tông da trong vùng mặt về tự nhiên. Trả (ảnh, mô tả) — mô tả rỗng nếu da đã ổn."""
    if strength <= 1e-3 or not boxes:
        return img, ""
    skin = _skin_weights(img, boxes)
    if skin is None:
        return img, ""
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    a, b = lab[..., 1], lab[..., 2]
    sel = skin > 0.5
    ma, mb = float(np.median(a[sel])), float(np.median(b[sel]))
    hue = math.degrees(math.atan2(mb, ma))
    chroma = math.hypot(ma, mb)
    d_hue = float(np.clip(np.clip(hue, SKIN_RED_LO, SKIN_HUE_HI) - hue, -20, 20))
    if abs(d_hue) < 1 and chroma <= SKIN_CHROMA:
        return img, ""
    rot = math.radians(d_hue * strength) * skin  # xoay góc màu, mạnh ở giữa vùng da, mềm ở viền
    cos, sin = np.cos(rot), np.sin(rot)
    a2 = a * cos - b * sin
    b2 = a * sin + b * cos
    if chroma > SKIN_CHROMA:
        k = 1.0 - skin * strength * 0.6 * (chroma - SKIN_CHROMA) / chroma
        a2 *= k
        b2 *= k
    lab[..., 1], lab[..., 2] = a2, b2
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    kind = "đỏ" if d_hue > 0 else "vàng"
    note = (f"da {kind} (góc màu {hue:.0f}°, da tự nhiên {SKIN_HUE_LO:.0f}–{SKIN_HUE_HI:.0f}°)" if abs(d_hue) >= 1
            else f"da đậm màu (chroma {chroma:.0f})")
    return out, note


def saturation(img: np.ndarray, delta: float) -> np.ndarray:
    if abs(delta) < 1e-3:
        return img
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab[..., 1:] *= 1.0 + delta
    return (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1) * 255 + 0.5).astype(np.uint8)


# Da là mốc thứ hai cho cân bằng trắng: ước lượng nguồn sáng hay bị kéo lệch khi nền là mảng màu rực lớn (phông xanh, áo
# đỏ cam). Ảnh thẻ của huy bị coi là ám tím hồng ~10°, gỡ đi làm da hồng (góc màu 42°) thành vàng nhợt (67°). Luật: cân
# bằng trắng chỉ được kéo góc màu da VỀ PHÍA SKIN_CENTER (vượt qua tối đa SKIN_CROSS độ), không được đẩy ra xa; nếu mức
# đặt làm da đi sai hướng thì giảm mức. Tâm 45° (thấp hơn trung vị 50,5° của UTKFace) để giữ da hồng hào.
SKIN_CENTER, SKIN_CROSS = 45.0, 2.0


def _protect_skin(img: np.ndarray, faces: list | None, est: np.ndarray, wb: float) -> float:
    if not faces:
        return wb
    w = _skin_weights(img, faces)
    if w is None:
        return wb
    sel = w > 0.5
    h0 = skin_hue(img, faces, sel)
    if h0 is None or h0[1] < 6:  # da gần như không màu (ảnh đen trắng, quá tối): không dùng làm mốc
        return wb
    lo, hi = min(h0[0], SKIN_CENTER - SKIN_CROSS), max(h0[0], SKIN_CENTER + SKIN_CROSS)
    small = (cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.resize(
        sel.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) > 0) \
        if max(img.shape[:2]) > 1200 else (img, sel)

    def ok(k: float) -> bool:
        h1 = skin_hue(apply_gains(small[0], gains_for(est, k)), faces, small[1])
        return h1 is None or lo - 0.5 <= h1[0] <= hi + 0.5

    if ok(wb):
        return wb
    a, b = 0.0, wb
    for _ in range(6):
        mid = (a + b) / 2
        a, b = (mid, b) if ok(mid) else (a, mid)
    return a


def correct(img: np.ndarray, wb: float, skin: float = 0.0, sat: float = 0.0, faces: list | None = None,
            min_cast: float = 2.0) -> tuple[np.ndarray, list[str]]:
    """Cân bằng trắng + sửa tông da + bão hoà. Trả (ảnh, ghi chú)."""
    notes: list[str] = []
    out = img
    if wb > 1e-3:
        est = estimate_illuminant(img, exclude=face_region_mask(img.shape[:2], faces) if faces else None)
        deg = cast_degrees(est)
        if deg >= min_cast:
            wb_used = _protect_skin(img, faces, est, wb)
            if wb_used >= 0.15:
                out = apply_gains(out, gains_for(est, wb_used))
                notes.append(f"Ảnh ám {cast_name(est)} (~{deg:.0f}°): đã cân bằng trắng {wb_used * 100:.0f} %"
                             + (" (giảm để giữ màu da gốc)" if wb_used < wb - 0.01 else ""))
            else:
                notes.append(f"Ước lượng ám {cast_name(est)} (~{deg:.0f}°) nhưng da trong ảnh vẫn tự nhiên: "
                             "không cân bằng trắng (nền nhiều màu rực làm lệch ước lượng)")
    if skin > 1e-3 and faces:
        out, why = skin_tone(out, faces, skin)
        if why:
            notes.append(f"Sửa tông da: {why}")
    out = saturation(out, sat)
    return out, notes
