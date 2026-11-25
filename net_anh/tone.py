"""Tông tươi sáng, trong trẻo (bước cuối, sau làm nét và tăng màu).

huy thấy ảnh sau sửa "u ám" và muốn tông "tươi sáng, trong trẻo". Đo trên 9 ảnh mẫu (tools/tone_bench.py): Tự động
giữ nguyên độ sáng ảnh gốc, không làm ảnh tươi hơn; cân bằng trắng làm ảnh tông ấm nhạt màu đi tới 30 %; ảnh quá sáng
sau khi hạ sáng mất 10–30 % độ tươi màu. Bước này làm giống các bộ lọc "trong trẻo" của app chỉnh ảnh:

1. Nâng tông giữa: Y' = Y + a·Y·(1 − Y) (đen vẫn đen, trắng vẫn trắng, không cháy thêm). Hệ số a theo tông giữa của
   ảnh: trung vị Y; có mặt thì lấy Y giữa mặt lớn nhất, nhưng mặt sáng trong cảnh tối (áo vest, nền tối) thì lấy trung
   bình của mặt và cảnh. Tông giữa càng thấp dưới 0,60 càng nâng nhiều, đã sáng thì không nâng. Ảnh tối có chủ ý
   (trung vị Y dưới 0,06: ảnh đêm, vũ trụ) không nâng, đến 0,18 nâng dần.
2. Làm trong: kéo mức đen (phân vị 0,5 của Y, tối đa 0,08) về 0, bỏ lớp "sương" xám ở vùng tối.
   R, G, B nhân cùng tỉ lệ như exposure.py nên giữ màu. Màu rực và sáng (kênh lớn nhất gần 255 dù Y chỉ ở tông giữa)
   nếu nhân đủ tỉ lệ sẽ chạm 255 và mất chi tiết, nên kênh lớn nhất đi qua một vai mềm từ KNEE lên 255, nhưng không
   thấp hơn đường cong áp cho chính kênh đó (màu trung tính không đổi), và không dùng quá 3/4 khoảng còn lại tới 255
   (giữ chuyển sắc ở màu rực sáng). Bước 3 cũng qua vai mềm này. Đo ở mức 1, ảnh mẫu tông ấm: cắt cứng ở 255 làm điểm
   cháy từ 3,9 % lên 23 %, có vai mềm còn 4,0 %; độ sáng, độ tươi màu gần như không đổi (KNEE 0,90 bắt đầu cháy
   thêm, 0,80 tối hơn mà không đỡ cháy hơn).
3. Rõ khối nhẹ (color.clarity) và độ rực (color.vibrance: màu nhạt tăng nhiều, da tăng ít).
4. Da trong vùng mặt sáng thêm một chút và bớt vàng ("da trắng hồng"); vùng trắng gần trung tính được gỡ ám.
Logo, hình đồ hoạ (exposure.is_graphic) giữ nguyên. amount 0 = tắt, 1 = mạnh nhất.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import color, exposure, wb

LIFT_MAX = 0.55  # a của đường nâng tông giữa ở mức 1 khi tông giữa thấp hơn mục tiêu từ KEY_SPAN trở lên
KEY_TARGET = 0.60  # tông giữa mong muốn (Y 0..1)
KEY_SPAN = 0.25
LOW_KEY = (0.06, 0.18)  # trung vị Y: dưới mức đầu không nâng (ảnh đêm, vũ trụ), đến mức sau nâng dần
HAZE, HAZE_MAX = 0.6, 0.08  # phần mức đen được kéo về 0, mức đen tối đa còn coi là "sương"
KNEE = 0.85  # kênh lớn nhất (0..1) vượt mức này thì vào vai mềm, không chạm 255
HEAD = 0.75  # và chỉ dùng tối đa phần này của khoảng còn lại tới 255 (đỏ rực 220–255: còn 10 mức thay vì 3)
CLARITY = 0.20
VIBRANCE = 0.30
SKIN_L = 4.0  # da trong vùng mặt sáng thêm (đơn vị L của Lab, ở mức 1)
SKIN_B = 0.15  # và bớt phần vàng b* này
WHITE_CLEAN = 0.5  # vùng trắng (L > 70, sắc độ < 20): phần ám được gỡ
WHITE_L, WHITE_C = 70.0, 20.0


def plan(img: np.ndarray, boxes=None, amount: float = 1.0) -> tuple[float, float]:
    """(a, b0): hệ số nâng tông giữa và mức đen được kéo về 0 cho ảnh `img` (BGR uint8)."""
    if amount <= 1e-3:
        return 0.0, 0.0
    small, k = exposure._small(img, 256)
    y = exposure.luma(small)
    med = float(np.median(y))
    key = med
    if boxes:
        fy, _ = exposure.face_luma(y, boxes, k)
        if fy == fy:  # mặt tối hơn cảnh: theo mặt; mặt sáng mà cảnh tối (áo, nền tối): nửa mặt nửa cảnh
            key = min(fy, 0.5 * (fy + med))
    low = float(np.clip((med - LOW_KEY[0]) / (LOW_KEY[1] - LOW_KEY[0]), 0, 1))
    a = amount * LIFT_MAX * float(np.clip((KEY_TARGET - key) / KEY_SPAN, 0, 1)) * low
    b0 = amount * HAZE * min(float(np.percentile(y, 0.5)), HAZE_MAX)
    return a, b0


def _curve(v: np.ndarray, a: float, b0: float) -> np.ndarray:
    v1 = np.clip((v - np.float32(b0)) / np.float32(1 - b0), 0, 1)
    return v1 + np.float32(a) * v1 * (1 - v1)


def _shoulder(m: np.ndarray, base: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Kênh lớn nhất mới (0..1) khi muốn đưa kênh lớn nhất từ `base` lên `m`: qua vai mềm từ KNEE tới 1 (tanh, độ dốc
    liền), không dùng quá phần HEAD của khoảng còn lại từ `base` tới 1 (màu rực sáng giữ được chuyển sắc thay vì dồn
    hết về 255), nhưng không thấp hơn min(m, floor) (floor: mức không cần nén, ví dụ đường cong áp cho chính kênh đó)."""
    k = np.float32(KNEE)
    soft = np.minimum(m, k) + np.float32(1 - KNEE) * np.tanh(np.maximum(m - k, 0) / np.float32(1 - KNEE))
    soft = np.minimum(soft, base + np.float32(HEAD) * (1 - base))
    return np.maximum(soft, np.minimum(m, floor))


def lift(img: np.ndarray, a: float, b0: float) -> np.ndarray:
    """Nâng tông giữa và kéo mức đen (bước 1–2): R, G, B nhân cùng tỉ lệ Y'/Y; kênh lớn nhất qua vai mềm."""
    f = img.astype(np.float32)
    y = exposure.luma(img)
    mx = f.max(axis=2) / np.float32(255)
    m1 = mx * _curve(y, a, b0) / np.maximum(y, np.float32(1e-4))
    m2 = _shoulder(m1, mx, _curve(mx, a, b0))
    r = np.where((y < 1e-4) | (mx < 1e-4), np.float32(1), m2 / np.maximum(mx, np.float32(1e-4)))
    return np.clip(f * r[..., None] + np.float32(0.5), 0, 255).astype(np.uint8)


def bright_clear(img: np.ndarray, amount: float, boxes=None, graphic: bool | None = None) -> tuple[np.ndarray, str]:
    """Tông tươi sáng, trong trẻo mức `amount` (0..1); `boxes`: khung mặt theo toạ độ của `img`. Trả (ảnh, ghi chú).
    `graphic`: ảnh vào là hình đồ hoạ (pipeline xét trên ảnh gốc: ảnh sau phóng to AI có thêm nhiều màu trung gian nên
    logo không còn bị nhận ra); None thì xét trên `img`."""
    amount = float(np.clip(amount, 0, 1))
    boxes = list(boxes or [])
    if graphic is None:
        graphic = exposure.is_graphic(exposure._small(img)[0])
    if amount <= 1e-3 or graphic:
        return img, ""
    a, b0 = plan(img, boxes, amount)
    out = lift(img, a, b0)
    ref = out.max(axis=2).astype(np.float32) / np.float32(255)

    lab = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    skin = color.skin_mask(img)
    lab[..., 0] = np.clip(color.clarity(lab[..., 0], amount * CLARITY, max(img.shape[:2])), 0, 100)
    lab = color.vibrance(lab, amount * VIBRANCE, skin)
    if boxes:
        sf = skin * wb.face_region_mask(img.shape[:2], boxes)
        lab[..., 0] += amount * SKIN_L * sf
        lab[..., 2] -= amount * SKIN_B * sf * np.maximum(lab[..., 2], 0)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    w = (np.clip((L - WHITE_L) / 20.0, 0, 1) * np.clip(1 - np.sqrt(A * A + B * B) / WHITE_C, 0, 1)
         * (amount * WHITE_CLEAN))
    lab[..., 1] = A * (1 - w)
    lab[..., 2] = B * (1 - w)
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # độ rực, rõ khối đẩy màu rực ra ngoài dải 0..1: nhân cả 3 kênh để kênh lớn nhất qua vai mềm (giữ sắc màu và
    # chi tiết) thay vì cắt riêng kênh đó ở 255; phần không vượt mức sau bước 2 thì giữ nguyên
    m = out.max(axis=2)
    m2 = _shoulder(m, ref, ref)
    out *= (m2 / np.maximum(m, np.float32(1e-6)))[..., None]
    out = (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    mid = a / 2  # tông giữa (Y = 0,5) sáng thêm phần này
    note = f"Tông tươi sáng: tông giữa sáng thêm {int(100 * mid + 0.5)} %, làm trong, tăng độ rực nhẹ" if mid >= 0.02 \
        else "Tông tươi sáng: làm trong, tăng độ rực nhẹ"
    return out, note


__all__ = ["bright_clear", "lift", "plan"]
