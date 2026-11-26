"""Khử bóng loáng da (da mặt bóng dầu: vệt sáng chói ở trán, sống mũi, gò má, cằm).

Ánh bóng dầu là phản xạ gương: cộng thêm ánh sáng gần trắng lên màu da, nên vùng bóng **sáng hơn** da quanh nó và
**nhạt màu hơn** (sắc độ thấp hơn so với độ sáng). Cách làm, trên kênh Lab, chỉ trong vùng da (ParseNet: da mặt, mũi;
không đụng mắt, lông mày, môi, tóc):

1. Tách ba tầng: tầng nền rất rộng (màu, độ sáng chung của da quanh đó, tính bỏ qua chính các vệt bóng), tầng giữa
   (vệt bóng, khối mặt) và tầng mịn (lỗ chân lông, vân da, nếp nhăn).
2. Bản đồ bóng = phần tầng giữa sáng hơn tầng nền, nhân với mức mất màu (sắc độ giảm so với da quanh) và độ sáng tuyệt
   đối. Gò má sáng nhưng vẫn đủ màu (ánh sáng khuếch tán, khối mặt) bị coi là bóng rất ít, nên không làm mặt bệt.
3. Kéo tầng giữa của vùng bóng xuống gần tầng nền (giữ lại một phần để còn khối), trả lại màu da theo tầng nền; tầng mịn
   giữ gần như nguyên (chỉ giảm nhẹ chỗ bóng gắt vì ánh gương làm vân da gắt hơn thật).
4. Chỗ cháy sáng (L gần 100, mất hết vân da): thêm lại một chút vân da mượn từ vùng da xung quanh, để không thành mảng phẳng.

Không dùng model, chạy trên CPU, ~0,1–0,3 s một mặt.
"""
from __future__ import annotations

import cv2
import numpy as np

SKIN_CLASSES = (1, 2)  # da mặt, mũi (nhãn ParseNet của facexlib); tai ít khi bóng và hay bị tóc che


def skin_mask512(labels: np.ndarray) -> np.ndarray:
    """Mặt nạ da 0..1 trên mặt căn 512×512 từ nhãn ParseNet. Khác facestyle.skin_mask: không nhạt dần về mép mặt
    (trán sát chân tóc hay bóng nhất), chỉ co vào một chút để không ăn sang tóc, lông mày, môi."""
    m = np.isin(labels, SKIN_CLASSES).astype(np.uint8)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    t = 10  # viền đen do căn mặt
    m[:t], m[-t:], m[:, :t], m[:, -t:] = 0, 0, 0, 0
    return m


BIG_SIGMA = 12.0  # trên mức này làm mờ trên ảnh thu nhỏ (gblur)


def gblur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gauss như cv2.GaussianBlur nhưng nhanh với σ lớn: thu nhỏ d lần (σ còn ≥ 4 px), làm mờ, phóng lại. Nhân Gauss
    rộng σ = 100 px (mặt to trên ảnh phóng 4×) dài ~600 px mỗi chiều nên chậm; kết quả mờ mịn nên thu nhỏ không mất
    gì thấy được (đo: lệch < 0,3 mức xám trên ảnh ra)."""
    if sigma <= BIG_SIGMA:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    d = int(sigma // 4)
    sw, shh = max(2, round(w / d)), max(2, round(h / d))
    small = cv2.resize(x, (sw, shh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma * sw / w)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _nblur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Làm mờ có trọng số (chỉ lấy điểm có w > 0): Gauss(x·w) / Gauss(w)."""
    num = gblur(x * w, sigma)
    den = gblur(w, sigma)
    return num / np.maximum(den, 1e-4)


def shine_map(lab: np.ndarray, mask: np.ndarray, face_px: float) -> tuple[np.ndarray, dict]:
    """Bản đồ bóng 0..1 trên ảnh Lab (L 0..100). `face_px`: bề rộng mặt (px) để chọn thang đo."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    m = mask.astype(np.float32)
    s_mid = max(1.2, 0.012 * face_px)
    s_base = max(6.0, 0.10 * face_px)
    w = m + 1e-4
    Lm, am, bm = _nblur(L, w, s_mid), _nblur(a, w, s_mid), _nblur(b, w, s_mid)
    Cm = np.sqrt(am * am + bm * bm)
    # tầng nền lần 1, rồi tính lại bỏ qua các vệt bóng sơ bộ (để nền không bị vệt bóng kéo sáng lên)
    Lb = _nblur(Lm, w, s_base)
    s0 = np.clip((Lm - Lb - 2.0) / 6.0, 0, 1)
    w2 = m * (1.0 - s0) ** 2 + 1e-4
    Lb = _nblur(Lm, w2, s_base)
    ab, bb = _nblur(am, w2, s_base), _nblur(bm, w2, s_base)
    Cb = np.sqrt(ab * ab + bb * bb)
    excess = Lm - Lb
    # mất màu: ánh gương trắng làm sắc độ giảm trong khi độ sáng tăng
    desat = np.clip((Cb - Cm) / np.maximum(Cb, 6.0), 0, 1)
    bright = np.clip((excess - 2.0) / 10.0, 0, 1)
    spec = np.clip(0.30 + 1.6 * desat, 0, 1)
    absl = np.clip((Lm - 50.0) / 15.0, 0, 1)
    S = bright * spec * absl * m
    S = cv2.GaussianBlur(S, (0, 0), max(1.0, 0.5 * s_mid))
    return np.clip(S, 0, 1), dict(Lm=Lm, am=am, bm=bm, Lb=Lb, ab=ab, bb=bb, excess=excess, s_mid=s_mid)


def remove_shine_lab(lab: np.ndarray, mask: np.ndarray, amount: float, face_px: float, pre=None) -> np.ndarray:
    """`pre`: kết quả shine_map(lab, mask, face_px) đã tính (khỏi tính lại)."""
    amount = float(np.clip(amount, 0, 1))
    if amount <= 1e-3 or float(mask.max()) <= 1e-3:
        return lab
    S, p = pre if pre is not None else shine_map(lab, mask, face_px)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    k = amount * S
    excess = np.maximum(p["excess"], 0.0)
    fine = L - p["Lm"]  # vân da, lỗ chân lông
    # 1) tầng giữa về gần nền: giữ 25 % phần nhô để còn khối mặt
    L2 = L - 0.75 * k * excess
    # 2) ánh gương làm vân da gắt hơn thật: dịu bớt một chút
    L2 = L2 - 0.25 * k * fine
    # 3) trả màu da: tầng giữa của a, b về màu nền (màu da quanh đó), giữ biến thiên mịn của màu
    a2 = a + 0.8 * k * (p["ab"] - p["am"])
    b2 = b + 0.8 * k * (p["bb"] - p["bm"])
    # 4) đốm bóng nhỏ (chóp mũi, nhân trung, gò má) nhỏ hơn tầng giữa: điểm sáng hơn da ngay quanh nó và nhạt màu
    #    hơn. Lỗ chân lông là đốm tối, vân da sáng chỉ lệch 1–2 L nên không bị đụng.
    fpos = np.maximum(fine, 0.0)
    Cp = np.sqrt(a * a + b * b)
    Cmid = np.sqrt(p["am"] ** 2 + p["bm"] ** 2)
    desat_p = np.clip((Cmid - Cp) / np.maximum(Cmid, 6.0), 0, 1)
    # đốm to cỡ tầng giữa: so với vùng quanh rộng gấp 3
    ring = _nblur(L, mask + 1e-4, 3.0 * p["s_mid"])
    bump = np.maximum(p["Lm"] - ring, 0.0)
    spot = np.clip((fpos + bump - 3.0) / 4.0, 0, 1) * np.clip(0.5 + 2.0 * desat_p, 0, 1) * np.clip((L - 55.0) / 15.0, 0, 1)
    spot = cv2.GaussianBlur(spot * mask, (0, 0), 0.8) * amount
    L2 = L2 - 0.85 * spot * (fpos + 0.7 * bump * (1.0 - 0.75 * k))
    a2 = a2 + 0.8 * spot * (p["am"] - a)
    b2 = b2 + 0.8 * spot * (p["bm"] - b)
    # 5) chỗ cháy sáng mất vân: mượn biên độ vân da của vùng da quanh đó (nhiễu có cấu trúc, cố định theo hạt giống)
    burnt = np.clip((p["Lm"] - 92.0) / 6.0, 0, 1) * k
    if float(burnt.max()) > 0.05:
        s_mid = p["s_mid"]
        tex_amp = np.sqrt(_nblur(fine * fine, mask * (1.0 - S) + 1e-4, 6 * s_mid))
        rng = np.random.default_rng(7)
        n = rng.standard_normal(L.shape).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), max(0.6, 0.35 * s_mid))
        n /= max(float(n.std()), 1e-6)
        L2 = L2 + burnt * 0.6 * tex_amp * n
    out = lab.copy()
    out[..., 0] = np.clip(L2, 0, 100)
    out[..., 1] = a2
    out[..., 2] = b2
    return out


def remove_shine(img: np.ndarray, mask: np.ndarray, amount: float, face_px: float,
                 stats: dict | None = None) -> np.ndarray:
    """Khử bóng trên ảnh BGR uint8 trong vùng `mask` (0..1, cỡ ảnh). Chỉ xử lý khung bao quanh mặt nạ.
    `stats` (tuỳ chọn) nhận "shine": tỉ lệ da bị bóng trước khi khử."""
    if amount <= 1e-3:
        return img
    ys, xs = np.nonzero(mask > 0.01)
    if len(ys) == 0:
        return img
    pad = int(max(8, 0.25 * face_px))
    h, w = img.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    m = mask[y0:y1, x0:x1].astype(np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    pre = shine_map(lab, m, face_px)
    if stats is not None:
        stats["shine"] = float((pre[0] * m).sum() / max(float(m.sum()), 1.0))
    lab2 = remove_shine_lab(lab, m, amount, face_px, pre)
    res = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    res = crop + m[..., None] * (res - crop)  # ngoài da giữ nguyên tuyệt đối
    out = img.copy()
    out[y0:y1, x0:x1] = (np.clip(res, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return out


def shine_fraction(img: np.ndarray, mask: np.ndarray, face_px: float) -> float:
    """Tỉ lệ da bị bóng (0..1), để báo trong kết quả và để đo."""
    f = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    S, _ = shine_map(lab, mask, face_px)
    tot = float(mask.sum())
    return float((S * mask).sum() / tot) if tot > 1 else 0.0


__all__ = ["remove_shine", "remove_shine_lab", "shine_map", "skin_mask512", "shine_fraction", "SKIN_CLASSES"]
