"""Phân tích ảnh đầu vào và gợi ý chế độ (đặc tả mục 4, 8)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .classic import blur_factor, estimate_noise
from .settings import PRESETS, Settings

MAX_SIDE = 4096
MIN_FACE_PX = 20
MAX_YAW_RATIO = 0.45  # độ lệch mũi so với giữa hai mắt / khoảng cách mắt (~45°)

# Ngưỡng hiệu chỉnh trên bộ ảnh thử (phương sai Laplacian ở cạnh dài 1000 px)
BLUR_HEAVY, BLUR_MILD = 100.0, 250.0
NOISE_HIGH = 6.0
DARK_L, BRIGHT_L = 80.0, 190.0
# ảnh chỉ "tối" khi cả vùng sáng nhất cũng tối (phân vị 99 của L < 150). Ảnh có áo/nền tối nhưng mặt, tường sáng
# (L trung bình thấp, vùng sáng vẫn sáng) không phải ảnh thiếu sáng: Retinex làm vải đen thành xám và nổi khối nén JPEG.
# Phân vị 99 (không phải 95): ảnh chạng vạng/đêm có đèn chỉ ~3 % điểm sáng (ảnh rocket: p95 115, p99 197) cũng không
# phải thiếu sáng; làm sáng thì trời xanh thẫm thành tím nhạt như ban ngày.
DARK_HI = 150.0
LOW_CONTRAST = 45.0
GRAY_SAT = 18.0
SMALL_FACE = 192.0  # mặt lớn nhất hẹp hơn mức này thì AI chân dung đáng dùng dù ảnh nét
TINY_IMAGE = 400  # cạnh dài dưới mức này: báo trước AI chỉ làm rõ được một phần (tools/small_bench.py: ảnh 100–300 px
# phóng 4× còn nét 0,4–0,7 lần ảnh gốc, mặt dưới 20 px không dò được)
# mờ từ mức này (mức cao nhất blur_factor đo được) là mờ rất nặng: khử mờ mạnh hơn (thu nhỏ 4,5–5,5 lần trước khi
# phóng AI) không giúp gì, đo trên ảnh mờ Gauss σ 4–5: SSIM giảm 0,01–0,02 mỗi nấc, IDS như cũ. Chỉ báo trước.
VERY_BLURRY = 3.5


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2, score (toạ độ ảnh gốc)
    landmarks: np.ndarray  # 5×2
    ok: bool = True
    reason: str = ""

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])


def check_face(bbox, lm5) -> tuple[bool, str]:
    """Luật mặt hợp lệ: rộng ≥ 20 px và quay trái/phải không quá ~45°."""
    width = float(bbox[2] - bbox[0])
    if width < MIN_FACE_PX:
        return False, f"mặt quá nhỏ ({width:.0f} px)"
    le, re_, nose = lm5[0], lm5[1], lm5[2]
    eye = float(np.linalg.norm(re_ - le))
    if eye < 1:
        return False, "không xác định được mắt"
    axis = (re_ - le) / eye
    yaw = abs(float(np.dot(nose - (le + re_) / 2, axis))) / eye
    if yaw > MAX_YAW_RATIO:
        return False, "mặt nghiêng quá 45°"
    return True, ""


def select_faces(faces: list[Face], max_faces: int) -> list[Face]:
    """Đánh dấu mặt hợp lệ; chỉ giữ `max_faces` mặt lớn nhất."""
    kept = 0
    for f in sorted(faces, key=lambda f: -f.width):
        f.ok, f.reason = check_face(f.bbox, f.landmarks)
        if f.ok:
            if kept >= max_faces:
                f.ok, f.reason = False, f"vượt giới hạn {max_faces} mặt"
            else:
                kept += 1
    return faces


@dataclass
class Analysis:
    width: int
    height: int
    blur: float
    noise: float
    brightness: float
    contrast: float
    saturation: float
    highlight: float = 255.0  # phân vị 99 của L (0..255)
    cast: float = 0.0  # độ ám màu ước lượng (độ, wb.cast_degrees)
    cast_name: str = ""
    blur_d: float = 1.0  # độ mờ ước lượng (classic.blur_factor): 1 = nét, 1.5–4 = ảnh chỉ có chi tiết của ảnh nhỏ hơn d lần
    expo: str = ""  # sửa sáng exposure.plan sẽ làm: "" | "tối" | "sáng" | "ngược sáng"
    expo_note: str = ""
    faces: list[Face] = field(default_factory=list)
    faces_checked: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def blur_label(self) -> str:
        return "mờ nặng" if self.blur < BLUR_HEAVY else "hơi mờ" if self.blur < BLUR_MILD else "nét"

    @property
    def noise_label(self) -> str:
        return "nhiều nhiễu" if self.noise >= NOISE_HIGH else "ít nhiễu"

    @property
    def light_label(self) -> str:
        if self.expo:
            return {"sáng": "quá sáng"}.get(self.expo, self.expo)
        return "tối" if self.dark else "sáng gắt" if self.brightness > BRIGHT_L else "bình thường"

    @property
    def dark(self) -> bool:
        return self.brightness < DARK_L and self.highlight < DARK_HI

    @property
    def good_faces(self) -> list[Face]:
        return [f for f in self.faces if f.ok]

    def summary(self) -> str:
        mo = f" (mờ ~{self.blur_d:g}×)" if self.blur_d > 1 else ""
        s = (f"{self.width}×{self.height} · {self.blur_label}{mo} · {self.noise_label} (σ≈{self.noise:.1f}) · "
             f"sáng {self.light_label}")
        if self.cast >= 2.0:
            s += f" · ám {self.cast_name} ~{self.cast:.0f}°"
        if self.faces_checked:
            s += f" · {len(self.faces)} mặt" + (f" ({len(self.good_faces)} xử lý được)" if self.faces else "")
        return s


Detector = Callable[[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]


def analyze(img: np.ndarray, detector: Detector | None = None, max_faces: int = 24) -> Analysis:
    h, w = img.shape[:2]
    k = min(1.0, 1000.0 / max(h, w))
    small = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    a = Analysis(width=w, height=h, blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                 noise=estimate_noise(small), brightness=float(lab[..., 0].mean()),
                 contrast=float(lab[..., 0].std()), saturation=float(hsv[..., 1].mean()),
                 highlight=float(np.percentile(lab[..., 0], 99)),
                 blur_d=blur_factor(img))
    from . import wb
    est = wb.estimate_illuminant(small)
    a.cast, a.cast_name = wb.cast_degrees(est), wb.cast_name(est)
    if max(h, w) > MAX_SIDE:
        a.warnings.append(f"Ảnh lớn hơn {MAX_SIDE} px: sẽ thu nhỏ trước khi xử lý")
    if a.blur_d >= VERY_BLURRY:
        a.warnings.append("Ảnh mờ rất nặng: AI làm rõ được một phần, chi tiết nhỏ và nét mặt có thể không giống thật "
                          "hoàn toàn")
    if max(h, w) < TINY_IMAGE:
        a.warnings.append(f"Ảnh rất nhỏ ({w}×{h} px): AI phóng 4× làm rõ được một phần, chi tiết nhỏ (chữ, mặt ở xa) "
                          "không rõ hẳn được")
    if detector is not None:
        try:
            a.faces = [Face(np.asarray(b, np.float32), np.asarray(l, np.float32)) for b, l in detector(img)]
            a.faces_checked = True
            select_faces(a.faces, max_faces)
            for f in a.faces:
                if not f.ok:
                    a.warnings.append(f"Một khuôn mặt sẽ không được làm nét bằng AI: {f.reason}")
        except Exception as e:  # thiếu model hoặc lỗi: vẫn phân tích được phần còn lại
            a.warnings.append(f"Chưa kiểm tra được khuôn mặt ({e.__class__.__name__})")
    from . import exposure
    p = exposure.plan(img, [f.bbox for f in a.faces])
    a.expo, a.expo_note = p.kind, p.note
    return a


def recommend(a: Analysis, base: Settings | None = None) -> tuple[str, Settings, list[str]]:
    """Router của preset Tự động: trả về ("auto", settings, lý do).

    Ánh sáng, tương phản, ảnh cũ quyết định trước (giữ nguyên qua mọi mức nét); mức nét 1–5 chọn theo độ mờ đo được
    (levels.auto_level): ảnh mờ ít dùng model nhẹ cho nhanh, mờ nhiều mới dùng model mặt và AI mạnh tay.

    Ảnh quá tối, quá sáng, ngược sáng: sửa phơi sáng (exposure.py, Settings.exposure) chạy đầu tiên trong pipeline, nhân
    sáng giữ màu. Không bật Retinex nữa: đo trên ảnh tối −2 EV, Retinex làm sáng quá tay (ΔL +40, SSIM 0,72 → 0,67), sửa
    phơi sáng đưa về gần ảnh gốc (ΔL ±6, SSIM 0,88); xem tools/exposure_bench.py."""
    from .levels import auto_level, level_settings
    s = base or Settings()
    why: list[str] = []
    dark = a.dark
    if a.saturation < GRAY_SAT and (a.blur < BLUR_MILD or a.noise >= NOISE_HIGH):
        s = s.with_(mode="old", contrast=max(s.contrast, 0.67))
    if a.expo and s.exposure > 0 and not s.low_light:
        why.append(a.expo_note[:1].lower() + a.expo_note[1:])
    # ảnh tối được làm sáng: thêm CLAHE làm ảnh gắt, vỡ mảng (đo: −2.6 dB); ngược sáng vốn đã tương phản cao
    if not dark and a.expo != "ngược sáng" and a.contrast < LOW_CONTRAST and s.contrast == 0 and not s.low_light:
        s = s.with_(contrast=0.67)
        why.append("tương phản thấp: bật CLAHE 2.0")
    n, reason = auto_level(a)
    why.insert(0, reason)
    s = level_settings(n, s, a)
    from .levels import auto_upscale
    if s.upscale == 0 and auto_upscale(a.width, a.height) == 4:
        why.append(f"ảnh nhỏ ({a.width}×{a.height} px): phóng 4×")
    if n == 1 and a.blur >= BLUR_MILD and a.noise < NOISE_HIGH and not s.low_light:
        s = s.with_(strength=min(s.strength, 0.6))  # ảnh đã nét: trộn 40 % ảnh gốc (đã sửa sáng), chỉ xử lý nhẹ
    return "auto", s, why


__all__ = ["Analysis", "Face", "analyze", "recommend", "check_face", "select_faces", "PRESETS"]
