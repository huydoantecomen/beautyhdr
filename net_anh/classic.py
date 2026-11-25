"""Xử lý cổ điển: không cần AI, chạy CPU (tài liệu thuật toán mục 2, đặc tả mục 6.1)."""
from __future__ import annotations

import cv2
import numpy as np


def clahe_l(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE trên kênh L của LAB: tăng tương phản cục bộ mà không lệch màu."""
    if clip <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def unsharp(img: np.ndarray, amount: float = 0.9, sigma: float = 1.2, threshold: int = 3) -> np.ndarray:
    """Unsharp mask có ngưỡng: chỉ làm nét vùng có cạnh, tránh halo và tăng hạt ở vùng phẳng."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    high = cv2.absdiff(img, blur)
    mask = cv2.cvtColor(high, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (0, 0), 2)[..., None].astype(np.float32) / 255.0
    sharp = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    out = img.astype(np.float32) * (1 - mask) + sharp.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def frequency_separation(img: np.ndarray, sigma: float = 9, high_boost: float = 1.4, low_smooth: int = 12) -> np.ndarray:
    """Làm mịn lớp tần số thấp, tăng lớp tần số cao: da mịn mà vẫn giữ chi tiết."""
    low = cv2.GaussianBlur(img, (0, 0), sigma)
    high = img.astype(np.float32) - low.astype(np.float32)
    low = cv2.bilateralFilter(low, low_smooth * 2 + 1, 60, 60)
    out = low.astype(np.float32) + high * high_boost
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_noise(img: np.ndarray) -> float:
    """Ước lượng độ lệch chuẩn nhiễu Gauss (phương pháp Immerkær 1996) trên ảnh xám."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = g.shape
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def blur_factor(img: np.ndarray, thresh: float = 1.2, scale: float = 0.9, crop: int = 1024) -> float:
    """Ước lượng ảnh "thật ra" chỉ có độ phân giải bằng 1/d ảnh hiện tại (đo trên tools/blur_bench.py).

    Thu nhỏ d lần rồi phóng lại bằng bicubic: ảnh mờ gần như không mất gì (sai số < `thresh` mức xám), ảnh nét mất
    nhiều. d lớn nhất còn mất ít = độ mờ; trả về 0.9·d làm tròn 0.5, trong [1.5, 4], hoặc 1.0 khi ảnh nét.
    Đo ở vùng giữa ≤ 1024 px, kích thước gốc (vệt mờ tính bằng điểm ảnh), sau khử nhiễu xám NLM h=6.
    """
    h, w = img.shape[:2]
    y0, x0 = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
    g = cv2.cvtColor(img[y0:y0 + crop, x0:x0 + crop], cv2.COLOR_BGR2GRAY)
    if min(g.shape) < 32:
        return 1.0
    g = cv2.fastNlMeansDenoising(g, None, 6, 7, 21).astype(np.float32)
    gh, gw = g.shape
    found = 1.0
    for d in (1.5, 2.0, 2.5, 3.0, 4.0):
        small = cv2.resize(g, (max(1, round(gw / d)), max(1, round(gh / d))), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (gw, gh), interpolation=cv2.INTER_CUBIC)
        if float(np.sqrt(np.mean((g - back) ** 2))) < thresh:
            found = d
    return 1.0 if found == 1.0 else float(np.clip(round(found * scale * 2) / 2, 1.5, 4.0))


def denoise(img: np.ndarray, h: float) -> np.ndarray:
    """Non-Local Means giữ cạnh. h = 0 thì bỏ qua."""
    if h <= 0.5:
        return img
    return cv2.fastNlMeansDenoisingColored(img, None, float(h), float(h), 7, 21)


def auto_denoise_h(img: np.ndarray) -> float:
    """Độ mạnh NLM theo mức nhiễu ước lượng: h = clamp(0.9·σ, 2, 12)."""
    return float(np.clip(0.9 * estimate_noise(img), 2, 12))


def brightness(img: np.ndarray, value: float) -> np.ndarray:
    """value -1..1: chỉnh gamma kênh L (dương = sáng hơn), không làm cháy vùng sáng."""
    if abs(value) < 1e-3:
        return img
    gamma = float(np.clip(1.0 - 0.6 * value, 0.4, 1.8))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def retinex_l(img: np.ndarray, sigmas=(15, 80, 250), amount: float = 0.7) -> np.ndarray:
    """Multi-Scale Retinex trên kênh L (tài liệu mục 2.5): nén ánh sáng, giữ màu.

    R = mean_i [log L − log(G_σi * L)], kéo giãn theo phân vị 1–99%, rồi trộn với L gốc theo `amount`.
    Tính ánh sáng ở ảnh thu nhỏ (cạnh dài 512) cho nhanh, σ quy đổi theo tỉ lệ.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[..., 0].astype(np.float32) + 1.0
    h, w = l.shape
    k = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(l, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    log_l = np.log(l)
    r = np.zeros_like(l)
    for s in sigmas:
        illum = cv2.GaussianBlur(small, (0, 0), max(s * k, 1.0))
        r += log_l - np.log(cv2.resize(illum, (w, h), interpolation=cv2.INTER_LINEAR))
    r /= len(sigmas)
    lo, hi = np.percentile(r, (1, 99))
    r = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    new_l = (1 - amount) * (l - 1.0) + amount * r
    lab[..., 0] = np.clip(new_l, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def quick_enhance(img: np.ndarray, upscale: int = 2, denoise_h: float | None = None, clahe_clip: float = 0.0,
                  sharpen: float = 0.9, bright: float = 0.0, low_light: bool = False) -> np.ndarray:
    """Quick Enhance bản 0.2.

    Thứ tự: khử nhiễu (ở kích thước gốc) → tăng sáng / Retinex → phóng Lanczos → CLAHE → unsharp.
    Khác bản 0.1: khử nhiễu trước khi phóng (nhanh hơn upscale² lần, đo được giữ danh tính tốt hơn),
    độ mạnh khử nhiễu tự theo mức nhiễu, CLAHE chỉ bật khi cần (clahe_clip > 0).
    """
    noise = estimate_noise(img)
    h = auto_denoise_h(img) if denoise_h is None else denoise_h
    img = denoise(img, h)
    if low_light:
        img = retinex_l(img)
    img = brightness(img, bright)
    if upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    img = clahe_l(img, clahe_clip)
    # ảnh còn nhiễu thì giảm α để không khuếch đại hạt; σ tăng theo hệ số phóng
    amount = sharpen * float(np.clip(1.0 - noise / 20.0, 0.5, 1.0))
    return unsharp(img, amount=amount, sigma=0.8 * max(upscale, 1), threshold=3)


def enhance_classic(img: np.ndarray, upscale: int = 2, denoise_h: int = 4, clahe_clip: float = 2.0,
                    amount: float = 0.9, sigma: float = 1.2) -> np.ndarray:
    """Pipeline cổ điển bản 0.1 (giữ để so sánh): upscale → NLM → CLAHE → unsharp."""
    if upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    img = denoise(img, denoise_h)
    img = clahe_l(img, clahe_clip)
    return unsharp(img, amount, sigma)
