"""Sửa phơi sáng tự động: ảnh quá tối, ảnh quá sáng, ảnh ngược sáng (mặt tối trên nền sáng).

Máy chụp sai sáng là nhân cả ảnh tuyến tính với 2^EV; với gamma 2,2 thì đó là nhân giá trị điểm ảnh với 2^(EV/2,2).
Nên sửa bằng một hệ số nhân trên độ sáng Y, nhân R, G, B cùng tỉ lệ để giữ màu, thay vì Retinex kéo giãn cả dải
(đo trên tools/exposure_bench.py: Retinex làm ảnh tối −2 EV sáng quá tay, ΔL +40, SSIM và IDS giảm).

  - Quá tối: cả vùng sáng nhất cũng tối (L trung bình < 80 và phân vị 99 < 150, như analysis.dark). Hệ số đưa phân
    vị 99 của Y lên 0,9 nhưng trung vị không quá 0,5 (ảnh tông trầm không bị làm sáng quá tay), tối đa 3 lần; vai
    mềm để vùng sáng không cháy. Có mặt thì không đẩy mặt quá 0,7.
  - Quá sáng: vùng sáng dồn sát trắng (phân vị 95 của Y ≥ 0,97 và ≥ 2 % điểm cháy) hoặc mặt sáng quá 0,62 mà có
    điểm cháy, và không phải hình đồ hoạ. Hạ để phân vị 75 về 0,62 (không mặt: cả trung vị về 0,42), mặt sáng quá
    về 0,55, nhưng không làm mặt tối dưới 0,42; giới hạn 0,6–0,65 lần. Hạ theo lớp sáng nền (guided filter): vùng
    phẳng như mặt, tường hạ đều một hệ số nên giữ vân và khối; mảng trắng lớn (trời cháy) và chi tiết trắng có cạnh rõ
    vẫn giữ trắng (vùng cháy không có thông tin để lấy lại). Đo trên 9 ảnh × 2 mức sáng quá: SSIM 0,894, IDS 0,969 so
    với 0,877 và 0,964 khi hạ bằng một đường cong cho cả ảnh (đường cong ép vùng sáng của mặt, mặt loang lổ).
  - Ngược sáng: mặt tối (Y < 0,4) mà nền quanh đầu sáng gấp ≥ 1,8 lần. Nâng mặt về 0,4 (tối đa 2 lần) theo vùng: hệ
    số tính từ lớp sáng nền đã làm trơn giữ cạnh (guided filter), nên chỉ vùng tối được nâng, trời sau lưng không
    cháy thêm; người (đầu và thân dưới mặt) được nâng đủ, vùng tối xa người chỉ nâng 40 % (tính theo log), để áo
    tối, rèm tối ở xa không bị xám. Đo trên 3 ảnh ngược sáng giả lập: SSIM 0,870 → 0,881 (nâng cả ảnh: 0,874).
Y ở đây là độ sáng hiển thị 0..1 (0,299 R + 0,587 G + 0,114 B).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DARK_L, DARK_HI = 80.0, 150.0  # như analysis (L của OpenCV Lab, 0–255)
UNDER_TARGET = 0.90  # phân vị 99 của Y sau khi làm sáng ảnh tối
UNDER_MEDIAN = 0.50  # và trung vị không vượt mức này
MAX_GAIN = 3.0
FACE_MAX = 0.70  # không làm mặt sáng hơn mức này khi làm sáng cả ảnh
OVER_P95, OVER_CLIP = 0.97, 0.02  # quá sáng: phân vị 95 sát trắng và ≥ 2 % điểm cháy
OVER_FACE = 0.62  # hoặc mặt sáng hơn mức này (và ảnh có vùng cháy)
FACE_TARGET_OVER = 0.55
FACE_MIN_OVER = 0.42  # hạ sáng không làm mặt tối dưới mức này
OVER_P75, OVER_MEDIAN = 0.62, 0.42
BACKLIT_FACE, BACKLIT_RATIO, BACKLIT_TARGET, MAX_LIFT = 0.40, 1.8, 0.40, 2.0
BACKLIT_FAR = 0.4  # vùng tối xa người: phần hệ số nâng (theo log) so với người
SHOULDER = 0.80  # vai mềm khi làm sáng: từ đây trở lên nén dần về 1


@dataclass
class Plan:
    kind: str = ""  # "" | "tối" | "sáng" | "ngược sáng"
    gain: float = 1.0  # hệ số toàn ảnh
    lift: float = 1.0  # hệ số nâng vùng tối (ngược sáng)
    face_b: float = 0.0  # độ sáng lớp nền ở mặt (ngược sáng: vùng tối đến mức này được nâng đủ)
    note: str = ""
    boxes: list = field(default_factory=list)  # khung mặt (ngược sáng: vùng người được nâng đủ)


def luma(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    return f[..., 2] * 0.299 + f[..., 1] * 0.587 + f[..., 0] * 0.114


def _small(img: np.ndarray, side: int = 640) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    k = min(1.0, side / max(h, w))
    if k >= 1:
        return img, 1.0
    return cv2.resize(img, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA), k


def face_luma(y: np.ndarray, boxes, k: float = 1.0) -> tuple[float, float]:
    """(Y trung vị giữa mặt, Y nền quanh đầu) của các mặt; mặt lớn nhất quyết định. Nền quanh đầu: dải bao quanh
    khung mặt (từ 1,5 đến 2,4 lần khung), chỉ phần ngang và trên đầu, lấy phân vị 75 (phần sáng của nền)."""
    h, w = y.shape
    best, area = (float("nan"), float("nan")), 0.0
    for b in boxes:
        x1, y1, x2, y2 = (float(v) * k for v in b[:4])
        fw, fh = x2 - x1, y2 - y1
        if fw < 8 or fh < 8 or fw * fh <= area:
            continue
        inner = y[int(max(0, y1 + fh * 0.25)):int(min(h, y2 - fh * 0.2)), int(max(0, x1 + fw * 0.25)):int(min(w, x2 - fw * 0.25))]
        if inner.size == 0:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox0, ox1 = int(max(0, cx - 1.2 * fw)), int(min(w, cx + 1.2 * fw))
        oy0, oy1 = int(max(0, cy - 1.2 * fh)), int(min(h, cy + 0.3 * fh))
        ring = y[oy0:oy1, ox0:ox1].copy()
        m = np.ones(ring.shape, bool)
        ix0, ix1 = int(max(0, cx - 0.75 * fw)) - ox0, int(min(w, cx + 0.75 * fw)) - ox0
        iy0 = int(max(0, cy - 0.75 * fh)) - oy0
        m[max(0, iy0):, max(0, ix0):max(0, ix1)] = False
        bg = float(np.percentile(ring[m], 75)) if m.sum() > 16 else float("nan")
        best, area = (float(np.median(inner)), bg), fw * fh
    return best


def is_graphic(small: np.ndarray) -> bool:
    """Hình đồ hoạ (logo, hình vẽ, ảnh chụp màn hình): rất ít màu khác nhau, trong khi ảnh chụp thật có hàng nghìn
    (đo: logo 369 màu, ảnh chụp 3 000–12 000 trên mẫu 256×256 lượng tử 6 bit/kênh). Ảnh đen trắng luôn ít màu nên
    không tính. Giãn dải giá trị (phân vị 0,5–99,5) trước khi đếm: ảnh chụp tối chỉ dùng vài chục mức nên chưa giãn thì
    bị đếm ít màu như hình đồ hoạ (ảnh tối mẫu: 685 màu, giãn ra 12 964)."""
    s = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    if float(np.abs(s[..., 2] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 0]).mean()) < 3:
        return False
    lo, hi = np.percentile(s, (0.5, 99.5))
    q = np.clip((s - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.int32) // 4
    return len(np.unique((q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2])) < 800


def plan(img: np.ndarray, boxes=None, strength: float = 1.0) -> Plan:
    """Quyết định sửa sáng cho ảnh `img` (BGR uint8) với các khung mặt `boxes` (toạ độ ảnh)."""
    if strength <= 0:
        return Plan()
    small, k = _small(img)
    y = luma(small)
    lab_l = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    mean_l, p99_l = float(lab_l.mean()), float(np.percentile(lab_l, 99))
    p99 = float(np.percentile(y, 99))
    clip = float((y >= 0.98).mean())
    boxes = list(boxes or [])
    fy, bg = face_luma(y, boxes, k)
    has_face = fy == fy
    med = float(np.median(y))
    # quá tối: cả vùng sáng nhất cũng tối
    if mean_l < DARK_L and p99_l < DARK_HI:
        g = min(MAX_GAIN, UNDER_TARGET / max(p99, 1e-3), max(1.0, UNDER_MEDIAN / max(med, 1e-3)))
        if has_face:
            g = min(g, max(1.0, FACE_MAX / max(fy, 1e-3)))
        g = 1 + (g - 1) * strength
        if g > 1.05:
            return Plan("tối", gain=g, note=f"Ảnh thiếu sáng: làm sáng {g:.1f} lần (giữ màu, không cháy vùng sáng)")
        return Plan()
    # ngược sáng: mặt tối, nền quanh đầu sáng
    if has_face and fy < BACKLIT_FACE and bg == bg and bg >= max(0.5, BACKLIT_RATIO * fy):
        lift = min(MAX_LIFT, BACKLIT_TARGET / max(fy, 1e-3))
        lift = 1 + (lift - 1) * strength
        if lift > 1.05:
            return Plan("ngược sáng", lift=lift, face_b=fy, boxes=[np.asarray(b[:4], np.float32) for b in boxes],
                        note=f"Ảnh ngược sáng: nâng sáng người {lift:.1f} lần, giữ nền sáng")
        return Plan()
    # quá sáng: vùng sáng dồn sát trắng hoặc mặt sáng quá mà có điểm cháy; hình đồ hoạ thì bỏ qua
    p75, p95 = (float(v) for v in np.percentile(y, [75, 95]))
    face_over = has_face and fy > OVER_FACE and (clip >= 0.01 or p99 >= 0.98)
    if (face_over or (p95 >= OVER_P95 and clip >= OVER_CLIP)) and not is_graphic(small):
        g = min(1.0, OVER_P75 / max(p75, 1e-3))
        if has_face:  # trung vị không dùng: áo tối, nền tối kéo trung vị xuống dù mặt đã sáng quá
            g = max(min(g, FACE_TARGET_OVER / fy), FACE_MIN_OVER / fy, 0.6)
        else:
            g = max(min(g, OVER_MEDIAN / max(med, 1e-3)), 0.65)
        g = min(g, 1.0)
        g = 1 - (1 - g) * strength
        if g < 0.95:
            return Plan("sáng", gain=g, note=f"Ảnh quá sáng: hạ sáng còn {g * 100:.0f} %, giữ vùng trắng")
    return Plan()


def _curve(g: float) -> np.ndarray:
    """Bảng tra 0..1 → 0..1 (4096 mức) cho hệ số g trên Y: g > 1 có vai mềm; g < 1 giữ trắng là trắng."""
    x = np.linspace(0, 1, 4096, dtype=np.float32)
    if g >= 1:
        v = g * x
        k = SHOULDER
        return np.where(v <= k, v, k + (1 - k) * (1 - np.exp(-(v - k) / (1 - k)))).astype(np.float32)
    t = np.clip((x - SHOULDER) / (1 - SHOULDER), 0, 1)
    w = t * t * (3 - 2 * t)
    return (g * x * (1 - w) + x * w).astype(np.float32)


def _apply_ratio(img: np.ndarray, y: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    """Nhân R, G, B cùng tỉ lệ y_new / y (giữ màu), không để kênh nào vượt 255 (tránh lệch màu khi cháy)."""
    f = img.astype(np.float32)
    mx = f.max(axis=2) / 255.0
    r = y_new / np.maximum(y, 1e-4)
    r = np.minimum(r, 1.0 / np.maximum(mx, 1e-4))
    r = np.where(y < 1e-4, 1.0, r)
    return np.clip(f * r[..., None] + 0.5, 0, 255).astype(np.uint8)


def guided_base(y: np.ndarray, radius_frac: float = 0.03, eps: float = 0.02, side: int = 512) -> np.ndarray:
    """Lớp sáng nền của Y: guided filter (He 2010) chính Y, tính ở ảnh nhỏ rồi phóng hệ số a, b lên (fast guided
    filter), giữ cạnh rõ giữa người và nền sáng nên không bị quầng."""
    h, w = y.shape
    k = min(1.0, side / max(h, w))
    ys = cv2.resize(y, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else y
    r = max(2, round(radius_frac * max(ys.shape)))
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    m = box(ys)
    var = box(ys * ys) - m * m
    a = var / (var + eps)
    b = m - a * m
    a, b = box(a), box(b)
    if k < 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * y + b


def person_prior(shape, boxes, side: int = 256) -> np.ndarray:
    """Vùng người ước lượng từ khung mặt: đầu (elip) và thân bên dưới (hình thang loe dần tới đáy ảnh), viền rất mềm.
    Tính ở ảnh nhỏ rồi phóng lên (làm mờ lớn ở ảnh to rất chậm)."""
    h, w = shape
    k = min(1.0, side / max(h, w))
    hs, ws = max(8, round(h * k)), max(8, round(w * k))
    m = np.zeros((hs, ws), np.float32)
    sig = 2.0
    for b in boxes:
        x1, y1, x2, y2 = (float(v) * k for v in b[:4])
        fw, fh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.ellipse(m, (round(cx), round(cy - 0.1 * fh)), (max(1, round(0.9 * fw)), max(1, round(1.1 * fh))),
                    0, 0, 360, 1.0, -1)
        pts = np.array([[cx - fw, cy + 0.6 * fh], [cx + fw, cy + 0.6 * fh], [cx + 2.2 * fw, cy + 2.0 * fh],
                        [cx + 2.4 * fw, hs], [cx - 2.4 * fw, hs], [cx - 2.2 * fw, cy + 2.0 * fh]])
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0)
        sig = max(sig, 0.5 * fw)
    m = cv2.GaussianBlur(m, (0, 0), sig)
    if k < 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0, 1)


def apply(img: np.ndarray, p: Plan) -> np.ndarray:
    if not p.kind:
        return img
    y = luma(img)
    if p.kind == "tối":
        lut = _curve(p.gain)
        y_new = lut[np.clip((y * 4095 + 0.5).astype(np.int32), 0, 4095)]
        return _apply_ratio(img, y, y_new)
    if p.kind == "sáng":  # hệ số theo lớp nền: cả vùng phẳng hạ đều, giữ vân
        lut = _curve(p.gain)
        base = np.clip(guided_base(y), 1e-3, 1.0)
        f = lut[np.clip((base * 4095 + 0.5).astype(np.int32), 0, 4095)] / base
        return _apply_ratio(img, y, np.clip(y * f, 0, 1).astype(np.float32))
    # ngược sáng: hệ số theo lớp nền, đủ ở vùng tối như mặt, giảm dần về 1 ở vùng sáng; xa người nâng ít hơn
    base = guided_base(y)
    hi = max(p.face_b * 1.8, min(0.65, p.face_b + 0.25))
    t = np.clip((base - p.face_b) / max(hi - p.face_b, 1e-3), 0, 1)
    s = 1 - t * t * (3 - 2 * t)
    if p.boxes:
        s = s * (BACKLIT_FAR + (1 - BACKLIT_FAR) * person_prior(y.shape, p.boxes))
    v = y * np.power(np.float32(p.lift), s)
    k = SHOULDER
    y_new = np.where(v <= k, v, k + (1 - k) * (1 - np.exp(-(v - k) / (1 - k)))).astype(np.float32)
    y_new = np.maximum(y_new, y)  # chỉ nâng, không làm tối vùng nào
    return _apply_ratio(img, y, y_new)


def correct(img: np.ndarray, boxes=None, strength: float = 1.0) -> tuple[np.ndarray, Plan]:
    p = plan(img, boxes, strength)
    return apply(img, p), p


__all__ = ["Plan", "plan", "apply", "correct", "luma", "face_luma", "guided_base", "is_graphic", "person_prior"]
