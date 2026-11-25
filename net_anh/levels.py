"""Thang mức nét 1–5 theo độ nặng của model, và mức Tự động chọn theo độ mờ đo được.

Mức càng cao model càng nặng và AI vẽ lại càng nhiều:
  1 Nhẹ     không AI (OpenCV), ~0,2 s
  2 Vừa     AI nhẹ: Real-ESRGAN compact (5 MB) khử mờ cả ảnh, mặt giữ 75 % ảnh gốc, ~1 s
  3 Rõ      thêm model mặt GFPGAN (~350 MB), giữ 70 % mặt gốc, ~4–7 s mỗi ảnh trên CPU
  4 Rất rõ  mặt 50 % AI, mắt và môi 70 % AI, thêm chút màu
  5 Tối đa  kiểu Remini (preset "max"): AI vẽ lại mắt, môi, phóng ít nhất 2×, tăng màu, da bóng

Đo bằng tools/level_bench.py (6 ảnh có mặt + 6 ảnh không mặt, mờ σ 1/2/3, CPU 4 luồng, ảnh 512 px):
  - Mờ nhẹ (~1,5×): mức 2 cho IDS 0,966, SSIM 0,921 trong ~1 s; model mặt (mức 3) IDS 0,971 nhưng mất ~7 s.
  - Từ ~2× trở lên model mặt mới đáng: IDS 0,906 so với 0,857 (cổ điển) ở 2×, 0,82 so với 0,71 ở 2,5×.
  - AI mạnh tay hơn khi ảnh mờ hơn: ở 2,5× mặt 50 % AI tốt nhất; ở 3× trở lên Rõ nét tối đa nét hơn hẳn,
    IDS chỉ kém 0,01.
  - Model nền nặng RRDB (67 MB) không tốt hơn compact ở mức mờ nào (SSIM thấp hơn 0,01–0,04) mà chậm gấp 5–15 lần;
    RestoreFormer kém GFPGAN trung bình. Cả hai chỉ để chọn tay.
Ảnh không có mặt: mức 2–5 đều là compact, khác nhau ở độ làm nét, màu và hệ số phóng.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import weights
from .settings import PRESETS, Settings


@dataclass(frozen=True)
class Level:
    n: int
    name: str
    short: str
    ai: bool
    short_noface: str = ""  # mô tả khi ảnh không có mặt (chỉ làm nét nền)


LEVELS: dict[int, Level] = {lv.n: lv for lv in [
    Level(1, "Nhẹ", "Không dùng AI: khử nhiễu, làm nét nhẹ, giữ ảnh tự nhiên.", False),
    Level(2, "Vừa", "AI nhẹ (model 5 MB): khử mờ cả ảnh, mặt giữ 75 % ảnh gốc. Nhanh, không cần model mặt. "
                    "Hợp ảnh mờ nhẹ.", True,
          "AI nhẹ (model 5 MB): khử mờ và làm nét cả ảnh."),
    Level(3, "Rõ", "Thêm model mặt: AI làm rõ mặt nhưng giữ 70 % mặt gốc, làm nét nền. Giữ đúng người. "
                   "Hợp ảnh mờ vừa.", True,
          "AI khử mờ cả ảnh, làm nét vừa phải."),
    Level(4, "Rất rõ", "AI mạnh hơn: mặt 50 % AI, mắt và môi 70 % AI, làm nét mạnh, thêm chút màu.", True,
          "AI làm nét mạnh hơn, thêm chút màu."),
    Level(5, "Tối đa", "Kiểu Remini: AI vẽ lại mắt và môi, phóng ít nhất 2×, tăng màu, da bóng. "
                       "Nét nhất nhưng mặt có thể hơi khác người thật. Hợp ảnh mờ nặng.", True,
          "AI làm nét mạnh nhất, phóng ít nhất 2×, tăng màu và rõ khối."),
    Level(6, "Cao nhất", "Như mức Tối đa nhưng dùng model mới nhất, nặng nhất: RestoreFormer++ cho mặt (giữ đúng "
                         "người hơn GFPGAN ở mọi mức mờ), DAT-2 cho cả ảnh. Chậm hơn nhiều: nên có GPU NVIDIA.", True,
          "DAT-2, model nặng nhất, khử mờ và phóng to cả ảnh. Chậm: nên có GPU NVIDIA."),
]}
MAX_LEVEL = max(LEVELS)
BEST_FACE, BEST_BG = "restoreformer_pp", "dat2"  # bộ model của mức 6 "Cao nhất"

FACE_MODELS = {"gfpgan": "GFPGAN 1.4", "restoreformer": "RestoreFormer", "restoreformer_pp": "RestoreFormer++"}
BG_MODELS = {"compact": "Real-ESRGAN compact", "rrdb": "Real-ESRGAN x2plus/x4plus", "plksr": "RealPLKSR",
             "dat2": "DAT-2"}
HEAVY_BG = ("dat2",)  # model nền rất nặng: cảnh báo khi máy không có GPU (hardware.warnings)

# những thứ không phụ thuộc mức nét: được giữ nguyên khi đổi mức
_KEEP = ("upscale", "denoise", "brightness", "low_light", "exposure", "tone", "contrast", "max_faces", "tile", "color_match", "face_model",
         "protect_faces", "phone", "wb", "skin_tone", "sr_model", "deblur", "sr_denoise", "face_ai",
         "keep_lips", "subject", "face_styles", "bg_color", "repair", "fade", "auto_rotate", "damage_mask")

# Tự động: độ mờ đo được (classic.blur_factor) ≤ ngưỡng → mức; mờ hơn 2,5× → mức 5
AUTO_BLUR_LEVELS = ((1.5, 2), (2.0, 3), (2.5, 4))
SMALL_IMAGE = 1200  # ảnh không mặt cạnh dài nhỏ hơn mức này: phóng to bằng AI dù ảnh nét
SMALL_UPSCALE = 1000  # hệ số phóng Tự động: cạnh dài nhỏ hơn mức này phóng 4× (ảnh 500–1000 px ra 2000–4000 px), còn lại 2×


def auto_upscale(w: int, h: int) -> int:
    """Hệ số phóng khi chọn "Tự động" (Settings.upscale = 0): ảnh nhỏ phóng 4× để ảnh ra đủ lớn mà vẫn có chi tiết
    (đo bằng tools/small_bench.py), ảnh từ SMALL_UPSCALE px trở lên phóng 2× như trước."""
    return 4 if max(w, h) < SMALL_UPSCALE else 2


def level_settings(level: int, base: Settings, analysis=None) -> Settings:
    """Thông số của mức `level` (1–5). `base`: thông số Tự động của ảnh (đã bật Retinex/CLAHE nếu ảnh tối/nhạt, đã có
    hãng máy, model mặt, model nền, hệ số phóng); các lựa chọn đó được giữ. `analysis`: kết quả phân tích (biết có mặt
    hay không); không có thì coi như có thể có mặt."""
    n = min(MAX_LEVEL, max(1, int(level)))
    if n == 6:  # mức 5 + model nặng nhất (người dùng vẫn đổi tay được sau khi chọn mức)
        return level_settings(5, base, analysis).with_(face_model=BEST_FACE, sr_model=BEST_BG)
    s = Settings(**{k: getattr(base, k) for k in _KEEP})
    if n == 1:
        return s.with_(mode="quick", sharpness=0.4, strength=min(base.strength, 0.85))
    if n == 2:
        # ảnh cũ: khử hạt mạnh như chế độ Ảnh cũ
        return s.with_(mode="upscale", sharpness=0.5, strength=1.0,
                       sr_denoise=1.0 if base.mode == "old" and s.sr_denoise is None else s.sr_denoise)
    if base.mode == "old":
        mode = "old"
    elif analysis is not None and analysis.faces_checked and not analysis.good_faces:
        mode = "upscale"
    else:
        mode = "portrait"
    face = mode != "upscale"
    if n == 3:
        p = dict(sharpness=0.5 if face else 0.6, fidelity=0.7)
    elif n == 4:
        p = dict(sharpness=0.7, fidelity=0.5, detail_fidelity=0.3, color=0.3, gloss=0.2 if face else 0.0)
    else:
        p = dict(sharpness=0.8, fidelity=0.3, detail_fidelity=0.0, color=0.6, gloss=0.0,
                 upscale=s.upscale and max(s.upscale, 2))  # 0 (tự động) luôn từ 2× trở lên
    return s.with_(mode=mode, **p)


def auto_level(a) -> tuple[int, str]:
    """Mức nét Tự động chọn cho ảnh đã phân tích `a`, kèm lý do: ảnh mờ ít dùng model nhẹ (nhanh), mờ nhiều mới dùng
    model mặt và AI mạnh tay (số đo ở đầu file)."""
    from .analysis import BLUR_MILD, GRAY_SAT, NOISE_HIGH, SMALL_FACE
    d = a.blur_d
    faces = bool(a.good_faces)
    old = a.saturation < GRAY_SAT and (a.blur < BLUR_MILD or a.noise >= NOISE_HIGH)
    if d <= 1.0:
        sharp = a.blur >= BLUR_MILD and a.noise < NOISE_HIGH
        n = 1 if sharp or faces else 2
        cause = "ảnh nét" if sharp else "không đo thấy mờ rõ" if faces else "ảnh hơi mờ hoặc nhiễu"
    else:
        n = next((lv for th, lv in AUTO_BLUR_LEVELS if d <= th), 5)
        cause = f"ảnh mờ ~{d:g}×"
    if faces and max(f.width for f in a.good_faces) < SMALL_FACE and n < 3:
        n, cause = 3, f"mặt nhỏ (dưới {SMALL_FACE:.0f} px)"
    if not faces:
        if max(a.width, a.height) < SMALL_IMAGE and n < 2:
            n, cause = 2, "ảnh nhỏ"
        n = min(n, 3)  # không có mặt: mức 4–5 chỉ thêm làm nét và màu, đo thấy mức 3 (compact) tốt nhất
    if old:
        n, cause = min(max(n, 3), 4), "ảnh gần như đơn sắc, mờ hoặc nhiễu như ảnh cũ"
    if not a.faces_checked and n > 2:  # chưa có model dò mặt: không tự tải model mặt
        n, cause = 2, cause + ", chưa kiểm tra khuôn mặt"
    if n == 1:
        act = "xử lý nhẹ, không dùng AI"
    elif n == 2:
        act = "AI nhẹ, mặt giữ ảnh gốc, chưa cần model mặt" if faces else "AI nhẹ cho cả ảnh"
    elif not faces:
        act = "làm sạch kiểu ảnh cũ, AI khử mờ cả ảnh" if old else "AI khử mờ cả ảnh"
    else:
        act = {3: "thêm model mặt, giữ 70 % mặt gốc", 4: "model mặt mạnh tay hơn", 5: "Rõ nét tối đa"}[n]
        act = "làm sạch kiểu ảnh cũ, " + act if old else act
    return n, f"mức {n} · {LEVELS[n].name}: {cause}, {act}"


def level_for(preset: str, s: Settings) -> int:
    """Mức gần nhất với một preset/thông số (để hiện trên thang khi chọn preset)."""
    if preset == "max" or (s.mode in ("portrait", "old") and s.fidelity <= 0.35):
        return 6 if (s.face_model, s.sr_model) == (BEST_FACE, BEST_BG) else 5
    if s.mode in ("portrait", "old"):
        return 4 if s.fidelity <= 0.55 or s.detail_fidelity is not None else 3
    if s.mode == "upscale":
        return 5 if s.color >= 0.5 else 4 if s.color > 0 else 3 if s.sharpness >= 0.6 else 2
    return 1


def models_needed(s: Settings) -> list[str]:
    """Model sẽ dùng với thông số `s` (khoá trong weights.MANIFEST)."""
    who = ["retinaface", "parsenet"] + (["age_cydral", "gender_cydral"] if s.subject == "auto" else []) \
        if s.subject != "off" else []
    if s.mode in ("quick", "manual"):
        return []  # preset đối tượng ở mức 1–2 chỉ chạy khi đã có model dò mặt (không tự tải)
    if s.sr_model in weights.SPANDREL_SR:
        bg = [weights.SPANDREL_SR[s.sr_model]]
    elif s.sr_model == "compact":
        bg = ["esrgan_compact", "esrgan_compact_wdn"]
    else:
        bg = ["esrgan_x4", "esrgan_x2"] if s.upscale == 0 else ["esrgan_x4" if s.upscale == 4 else "esrgan_x2"]
    if s.mode == "upscale":
        return bg
    return bg + list(dict.fromkeys(["retinaface", "parsenet", s.face_model] + who))


def describe(level: int, s: Settings) -> str:
    """Một đoạn mô tả mức đang chọn: làm gì, dùng model nào, có phải tải model không."""
    lv = LEVELS[min(MAX_LEVEL, max(1, int(level)))]
    noface = s.mode == "upscale" and lv.n > 2
    head = f"Mức {lv.n} · {lv.name}: {lv.short_noface if noface and lv.short_noface else lv.short}"
    if s.mode in ("quick", "manual"):
        return head + " Model: không (xử lý cổ điển OpenCV), dưới 1 giây, không cần tải gì."
    bg = BG_MODELS.get(s.sr_model, s.sr_model)
    if s.mode == "upscale":
        used = f"{bg} cho toàn ảnh" + (" (không thấy mặt)" if noface else
                                       f", mặt giữ {(1 - s.face_ai) * 100:.0f} % ảnh gốc" if s.face_ai < 1 else "")
    else:
        used = f"{FACE_MODELS.get(s.face_model, s.face_model)} cho mặt (giữ {s.fidelity * 100:.0f} % mặt gốc"
        if s.detail_fidelity is not None:
            used += f", mắt môi {(1 - s.detail_fidelity) * 100:.0f} % AI"
        used += f"), {bg} cho nền"
    text = f"{head} Model: {used}."
    missing = [m for m in models_needed(s) if weights.status(m) != "ready"]
    if missing:
        mb = sum(weights.MANIFEST[m].size for m in missing) / 1e6
        text += f" Chưa có {len(missing)} model (~{mb:.0f} MB): tự tải ở lần chạy đầu."
    if s.sr_model in HEAVY_BG:
        speed = " Model nền rất nặng: có GPU thì vài chục giây, chỉ CPU thì vài phút đến hơn chục phút mỗi ảnh tuỳ cỡ ảnh."
    elif s.mode == "upscale":
        speed = " Vài giây trên CPU, nhanh hơn nữa nếu có GPU."
    else:
        speed = " Có GPU thì vài giây; chỉ CPU thì khoảng 4–7 giây mỗi mặt."
    return text + speed


__all__ = ["LEVELS", "MAX_LEVEL", "BEST_FACE", "BEST_BG", "Level", "level_settings", "auto_level", "auto_upscale", "level_for", "describe", "models_needed",
           "FACE_MODELS", "BG_MODELS", "HEAVY_BG", "PRESETS", "AUTO_BLUR_LEVELS"]
