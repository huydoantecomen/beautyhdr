"""Pipeline xử lý: Validate → Orientation → (Analysis) → Mode router → Processing → Mask/Blend → Output.

Chế độ (đặc tả mục 6):
  quick / manual  Quick Enhance cổ điển, chạy CPU
  portrait        Real-ESRGAN cho nền + GFPGAN cho từng mặt, trộn theo độ trung thực
  old             tiền xử lý cổ điển (khử nhiễu mạnh, CLAHE) rồi như portrait; không tô màu
  upscale         Real-ESRGAN toàn ảnh, không xử lý mặt riêng
Thiếu model hoặc lỗi AI → tự rơi về Quick Enhance và ghi lý do (đặc tả mục 7).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from . import classic, weights
from .analysis import MAX_SIDE, MIN_FACE_PX, Analysis, Face, analyze, select_faces
from .archs import GFPGANv1Clean, RRDBNet, SRVGGNetCompact
from . import gpu_profile
from .device import DeviceInfo, pick_device
from .settings import MODES, Settings

Progress = Callable[[str], None]
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_FILE_MB = 80


class Cancelled(Exception):
    """Người dùng bấm Huỷ."""


class ImageError(ValueError):
    """Ảnh không hỗ trợ hoặc hỏng."""


@dataclass
class Report:
    mode: str = ""
    device: str = ""
    seconds: float = 0.0
    faces_found: int = 0
    faces_restored: int = 0  # mặt AI chạy thành công
    ai_fallback: bool = False  # chế độ AI bị lỗi nên ảnh chỉ qua Cải thiện nhanh (CPU)
    faces_kept: int = 0  # mặt model lỗi, phải dùng lại mặt gốc (vẫn được ghép lại vào ảnh)
    notes: list[str] = field(default_factory=list)
    face_model: str = ""  # model mặt thật sự đã chạy ("" = không chạy)
    subjects: list[str] = field(default_factory=list)  # đối tượng từng mặt đã áp preset, vd "Nữ (~27 tuổi)"
    styled: bool = False  # đã áp preset đối tượng lên mặt
    bg_model: str = ""  # model nền thật sự đã chạy, kèm kiểu số trên GPU
    deshined: int = 0  # số mặt đã khử bóng da
    device_kind: str = ""  # thiết bị thật sự đã chạy model: "cuda" | "xpu" | "cpu"…
    model_s: float = 0.0  # thời gian chạy model AI (nền, mặt, tìm mặt, tách vùng mặt); còn lại là xử lý ảnh trên CPU

    def models(self) -> str:
        parts = []
        if self.face_model:
            parts.append(f"Mặt: {self.face_model}")
        if self.bg_model:
            parts.append(f"Nền: {self.bg_model}")
        return " · ".join(parts) or "Không dùng model AI"

    @property
    def kind(self) -> str:
        """Ảnh ra thật sự thuộc loại nào: "ai" (AI chạy đủ), "partial" (có mặt lỗi model giữ nguyên) hoặc
        "fallback" (AI lỗi, chỉ qua xử lý nhanh). Chế độ Nhanh do người dùng chọn cũng là "quick", không phải lỗi."""
        if self.ai_fallback:
            return "fallback"
        if self.mode == "quick":
            return "quick"
        return "partial" if self.faces_kept else "ai"

    def summary(self) -> str:
        parts = [("⚠ AI lỗi, chỉ xử lý nhanh" if self.ai_fallback else MODES.get(self.mode, self.mode)), self.models(),
                 f"Thiết bị: {self.device}",
                 f"Thời gian: {self.seconds:.1f} s"]
        if self.model_s > 0 and self.device_kind not in ("", "cpu"):
            parts.append(f"model AI trên GPU {self.model_s:.1f} s, xử lý màu/tông/ghép mặt trên CPU "
                         f"{max(0.0, self.seconds - self.model_s):.1f} s")
        if self.faces_found:
            parts.append(f"Khuôn mặt: làm nét {self.faces_restored}/{self.faces_found}"
                         + (f" ({self.faces_kept} mặt lỗi model, giữ nguyên)" if self.faces_kept else ""))
        if self.subjects:
            parts.append("Preset: " + ", ".join(self.subjects))
        return " · ".join(parts) + ("\n" + "\n".join(self.notes) if self.notes else "")


class BatchTally:
    """Tổng kết xử lý hàng loạt: không chỉ đếm tệp xuất thành công mà tách ảnh làm bằng AI đủ, ảnh có mặt lỗi model,
    ảnh chỉ qua xử lý nhanh do AI lỗi, ảnh do người dùng chọn chế độ Nhanh, và ảnh lỗi/bỏ qua."""

    def __init__(self, total: int = 0):
        self.total = total
        self.counts = {"ai": 0, "partial": 0, "fallback": 0, "quick": 0}
        self.failed = 0
        self.fallback_files: list[str] = []

    def add(self, rep: "Report", name: str = "") -> None:
        k = rep.kind
        self.counts[k] += 1
        if k == "fallback" and name:
            self.fallback_files.append(name)

    def fail(self) -> None:
        self.failed += 1

    @property
    def exported(self) -> int:
        return sum(self.counts.values())

    @property
    def needs_attention(self) -> bool:
        return bool(self.counts["fallback"] or self.counts["partial"] or self.failed)

    def summary(self) -> str:
        c = self.counts
        parts = [f"đã xuất {self.exported}/{self.total or self.exported + self.failed} ảnh", f"AI đủ: {c['ai']}"]
        if c["partial"]:
            parts.append(f"AI một phần (có mặt lỗi model giữ nguyên): {c['partial']}")
        if c["fallback"]:
            parts.append(f"CHỈ XỬ LÝ NHANH vì AI lỗi: {c['fallback']}")
        if c["quick"]:
            parts.append(f"chế độ Nhanh theo lựa chọn: {c['quick']}")
        if self.failed:
            parts.append(f"lỗi/bỏ qua: {self.failed}")
        s = " · ".join(parts)
        if self.fallback_files:
            more = "…" if len(self.fallback_files) > 5 else ""
            s += "\nẢnh chưa được AI xử lý: " + ", ".join(self.fallback_files[:5]) + more
        return s


# ---------- Validate + đọc / ghi ảnh ----------

def load_image(path: str | Path) -> np.ndarray:
    """Kiểm tra định dạng, dung lượng; đọc bằng Pillow và xoay theo EXIF orientation."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ImageError(f"Định dạng {path.suffix or '(không có đuôi)'} chưa được hỗ trợ (JPG, PNG, WebP, BMP, TIFF)")
    if not path.exists():
        raise ImageError("Không tìm thấy file")
    size_mb = path.stat().st_size / 1e6
    if size_mb > MAX_FILE_MB:
        raise ImageError(f"File quá lớn ({size_mb:.0f} MB, tối đa {MAX_FILE_MB} MB)")
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                # ảnh xám 16 bit / số thực (hay gặp ở ảnh scan): convert("RGB") của Pillow kẹp về 255 thành ảnh trắng
                a = np.nan_to_num(np.asarray(im, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                hi = _gray_white_level(im.mode, float(a.max()) if a.size else 0.0)
                g = np.clip(a / hi * 255.0 + 0.5, 0, 255).astype(np.uint8)
                return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            im = im.convert("RGB")
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Image.DecompressionBombError as e:
        raise ImageError("Ảnh có quá nhiều điểm ảnh (nghi file độc hại hoặc quá lớn)") from e
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise ImageError(f"Ảnh bị hỏng hoặc không đọc được ({e.__class__.__name__})") from e


def _gray_white_level(mode: str, peak: float) -> float:
    """Giá trị ứng với màu trắng của ảnh xám nhiều bit. Ảnh số thực (mode F) thường chuẩn hoá 0..1: nếu chia cho 255
    thì ảnh 8 bit chỉ còn 0 hoặc 1 (gần như đen, mất hết chi tiết sáng tối) nên phải nhận ra thang 0..1."""
    if mode.startswith("I;16"):
        return 65535.0
    if mode == "F" and peak <= 1.0:
        return 1.0
    if peak <= 255:
        return 255.0
    if peak <= 65535:
        return 65535.0
    return peak


# Cỡ xuất theo cạnh dài (px). 1K = Full HD 1920, 2K = QHD 2560, 4K = UHD 3840; 0 = giữ cỡ kết quả.
OUT_SIZES = {"goc": 0, "1k": 1920, "2k": 2560, "4k": 3840}


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    """Đổi cỡ để cạnh dài bằng `long_side` (giữ tỉ lệ). Thu nhỏ: INTER_AREA (không răng cưa); phóng: Lanczos."""
    h, w = img.shape[:2]
    if not long_side or max(h, w) == long_side:
        return img
    k = long_side / max(h, w)
    size = (max(1, round(w * k)), max(1, round(h * k)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)


def save_image(img: np.ndarray, path: str | Path, quality: int = 94) -> Path:
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 50, 100))]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(np.clip(quality, 50, 100))]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Không ghi được ảnh {path}")
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    tmp.replace(path)
    return path


def unique_output_path(src: Path, out_dir: Path | None = None, ext: str = ".jpg", tag: str = "_net") -> Path:
    """Tên file ra không bao giờ trùng ảnh gốc hay file đã có."""
    d = out_dir or src.parent
    p = d / f"{src.stem}{tag}{ext}"
    n = 2
    while p.exists() or p.resolve() == src.resolve():
        p = d / f"{src.stem}{tag}_{n}{ext}"
        n += 1
    return p


# ---------- tiện ích ----------

SMALL_FACE_ED = 32.0  # khoảng cách hai mắt (px, ảnh vào) dưới mức này là mặt nhỏ: Enhancer._presr_face


def is_small_face(lm) -> bool:
    """Mặt nhỏ trong ảnh nhóm, toàn thân. Đo (ArcFace so với ảnh vào) mặt có hai mắt cách 16–19 px: căn thẳng rồi chạy model
    mặt thì bịa ra tóc vàng, mắt xanh (IDS 0,52 ở 70 % AI); huy chấm mù lại ưa bản dựng nhiều AI hơn bản giới hạn AI, nên
    không hạ trọng số AI mà đổi đầu vào: phóng AI vùng mặt trước khi căn."""
    l = np.asarray(lm, np.float32).reshape(5, 2)
    return float(np.linalg.norm(l[0] - l[1])) < SMALL_FACE_ED


CONTOUR_AI = 0.85  # phần AI ở viền ngoài khuôn mặt (Enhancer._sharp_contour)

AI_MIN_EYE = float(os.environ.get("NETANH_AI_MIN_EYE", 10))  # px trên ảnh vào; dưới ngưỡng không chạy model mặt


def _eye_dist(lm) -> float:
    lm = np.asarray(lm, np.float32)
    return float(np.hypot(*(lm[1] - lm[0])))


def _to_tensor(img_bgr: np.ndarray, dev: DeviceInfo, half: bool) -> torch.Tensor:
    t = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float() / 255.0
    t = t.unsqueeze(0).to(dev.device)
    return t.half() if half else t


def _to_image(t: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    t = t.squeeze(0).float().clamp(lo, hi).cpu()
    t = (t - lo) / (hi - lo)
    img = (t.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def match_color(res: np.ndarray, ref: np.ndarray, region: tuple[int, int] = (128, 384)) -> np.ndarray:
    """Khớp màu trong LAB theo vùng giữa mặt: a, b khớp trung bình và độ lệch chuẩn; L chỉ khớp trung bình
    (giữ độ tương phản mới mà model tạo ra)."""
    a, b = region
    r = cv2.cvtColor(res, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        rs, gs = r[a:b, a:b, c], g[a:b, a:b, c]
        if c == 0:
            r[..., c] += gs.mean() - rs.mean()
        else:
            r[..., c] = (r[..., c] - rs.mean()) / (rs.std() + 1e-6) * gs.std() + gs.mean()
    return cv2.cvtColor(np.clip(r, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def face_union_mask(inv_mats, shape: tuple[int, int], size: int = 512, border: int = 32) -> np.ndarray:
    """Mask mềm (0..1) phủ các vùng mặt đã ghép, trong hệ toạ độ ảnh ra."""
    base = np.zeros((size, size), np.float32)
    base[border:size - border, border:size - border] = 1
    base = cv2.GaussianBlur(base, (0, 0), border / 2)
    h, w = shape
    m = np.zeros((h, w), np.float32)
    for inv in inv_mats:
        m = np.maximum(m, cv2.warpAffine(base, inv, (w, h)))
    return m


def protect_faces(out: np.ndarray, src: np.ndarray, boxes, w_ai: float) -> np.ndarray:
    """Vùng mặt (tròn, bán kính 0,75 cạnh khung, viền mờ) trộn lại với ảnh vào phóng bicubic, giữ (1 − w_ai) phần
    ảnh vào. Dùng khi phóng to AI không có model mặt: mạng nền vẽ lại mặt theo kiểu "trung bình" làm lệch nét người.
    Giống hệt protectFaces của Android (Enhancer.kt). `boxes` theo toạ độ `src`."""
    if not len(boxes) or w_ai >= 0.999:
        return out
    k = out.shape[1] / src.shape[1]
    mask = np.zeros(out.shape[:2], np.float32)
    area = 0.0
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2 * k, (b[1] + b[3]) / 2 * k
        r = 0.75 * max(b[2] - b[0], b[3] - b[1]) * k
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 1.0, -1)
        area += math.pi * r * r
    mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, 0.15 * math.sqrt(area))) * (1 - max(0.0, w_ai))
    base = cv2.resize(src, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    o = out.astype(np.float32)
    return np.clip(o + (base - o) * mask[..., None] + 0.5, 0, 255).astype(np.uint8)


def blend_strength(src: np.ndarray, out: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0, 1))
    if strength >= 0.999:
        return out
    base = cv2.resize(src, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return cv2.addWeighted(out, strength, base, 1 - strength, 0)


def _oom_classes() -> tuple[type, ...]:
    out = [torch.cuda.OutOfMemoryError]
    for mod in (torch, getattr(torch, "xpu", None)):
        c = getattr(mod, "OutOfMemoryError", None)
        if isinstance(c, type) and c not in out:
            out.append(c)
    return tuple(out)


_OOM = _oom_classes()  # hết bộ nhớ GPU: CUDA, Intel XPU


class Enhancer:
    """Giữ model trong bộ nhớ để xử lý nhiều ảnh liên tiếp. Không an toàn khi gọi song song."""

    def __init__(self, device: str = "auto", progress: Progress | None = None):
        self.dev = pick_device(device)
        self.say = progress or (lambda m: None)
        # kiểu số đã chọn cho từng model nền sau khi chạy thử: khoá → (tên kiểu, kết quả thử từng kiểu)
        self.precision: dict[str, tuple[str, list[tuple[str, gpu_profile.ProbeResult]]]] = {}
        self.vram_share = 1.0  # phần bộ nhớ GPU của engine này khi làm thư mục song song (batch.extra_engines)
        self.last_tile = 0
        if self.dev.family.tf32:  # Ampere trở lên: nhân ma trận fp32 bằng TF32 (attention của DAT-2, RestoreFormer)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.cancel_requested: Callable[[], bool] = lambda: False
        self._model_s = 0.0
        self._esrgan_run = self._timed(self._esrgan_run)  # đo phần chạy model (Report.model_s)
        self._face_run = self._timed(self._face_run)
        self._esrgan: dict = {}
        self._sr_extra: dict = {}  # model nền nạp bằng spandrel (DAT-2, RealPLKSR): model_id → (mạng, kiểu số)
        self._gfpgan: torch.nn.Module | None = None
        self._helper = None
        self._agegender = None
        self._faces: list | None = None

    def _timed(self, fn):
        """Bọc hàm chạy model: cộng thời gian vào self._model_s (chờ GPU xong để đo đúng)."""
        def run(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self._sync()
                self._model_s = getattr(self, "_model_s", 0.0) + time.perf_counter() - t
        return run

    def _sync(self) -> None:
        try:
            if self.dev.kind == "cuda":
                torch.cuda.synchronize()
            elif self.dev.kind == "xpu":
                torch.xpu.synchronize()
        except Exception:
            pass

    def _empty_cache(self) -> None:
        try:
            if self.dev.kind == "cuda":
                torch.cuda.empty_cache()
            elif self.dev.kind == "xpu":
                torch.xpu.empty_cache()
        except Exception:
            pass

    def _fallback_to_cpu(self, why: Exception, rep: "Report") -> None:
        """GPU không phải NVIDIA (Intel XPU, DirectML, Metal) lỗi lúc chạy: đưa mọi model về CPU, ghi nhớ cho các ảnh sau."""
        from .device import pick_device

        self._esrgan.clear()
        self._sr_extra.clear()
        self._gfpgan = self._helper = self._agegender = None
        for attr in ("_restoreformer", "_restoreformer_pp"):
            setattr(self, attr, None)
        self._empty_cache()
        self.dev = pick_device("cpu")
        self.gpu_failed = True
        rep.device = self.dev.name
        rep.notes.append(f"GPU lỗi ({why.__class__.__name__}: {str(why)[:120]}): đã chạy lại bằng CPU, chậm hơn. "
                         "Các ảnh sau cũng dùng CPU; mở lại app để thử GPU.")
        self.say("GPU lỗi, chạy lại bằng CPU…")

    def _check(self):
        if self.cancel_requested():
            raise Cancelled()

    _DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

    def _place(self, net: torch.nn.Module, key: str, half: bool = True, bf16: bool = False) -> tuple[torch.nn.Module, torch.dtype]:
        """Đưa model nền lên thiết bị và chọn kiểu số: thử lần lượt các kiểu dòng card cho phép (gpu_profile), mỗi kiểu
        chạy một ảnh mẫu so với fp32; kiểu đầu tiên đạt thì dùng, không đạt kiểu nào thì fp32. Kết quả nhớ theo `key`
        nên đổi mức khử nhiễu (nạp lại compact) không phải thử lại."""
        net = net.eval().to(self.dev.device)
        if self.dev.kind == "cpu":
            return net, torch.float32
        if key in self.precision:
            name = self.precision[key][0]
        else:
            fam = self.dev.family
            native_bf16 = self.dev.kind == "cuda" and (self.dev.cc or (0, 0))[0] >= 8 and not getattr(torch.version, "hip", None)
            tried: list[tuple[str, gpu_profile.ProbeResult]] = []
            name = "fp32"
            for cand in gpu_profile.candidates(fam, half, bf16, native_bf16):
                if cand == "fp32":
                    break
                r = gpu_profile.probe_dtype(net, self.dev.device, self._DT[cand])
                tried.append((cand, r))
                if r.ok:
                    name = cand
                    break
            self.precision[key] = (name, tried)
            bad = [f"{c} {r.why}" for c, r in tried if not r.ok]
            if bad:
                self.say(f"{key}: {'; '.join(bad)} trên {self.dev.name}, dùng {name}")
        dt = self._DT[name]
        net = net.to(dt)
        self._memory_format(net, gpu_profile.MODEL_OF_KEY.get(key, key), name)
        return net, dt

    def _memory_format(self, net: torch.nn.Module, model: str, dtype: str) -> None:
        """Tensor Core + fp16/bf16 + mạng tích chập: xếp trọng số kênh cuối (gpu_profile.channels_last); _forward đổi
        ảnh vào theo. Đánh dấu trên mạng để quay về fp32 (sửa tràn số) vẫn đúng."""
        cl = gpu_profile.channels_last(self.dev.kind, self.dev.cc, dtype, model)
        if cl:
            gpu_profile.to_channels_last(net)
        net._netanh_cl = cl

    def _place_face(self, net: torch.nn.Module, key: str) -> torch.nn.Module:
        """Model mặt lên thiết bị: chạy thử mặt mẫu 512 px ở các kiểu số nhanh (gpu_profile.face_candidates) so với fp32,
        đạt thì dùng. GTX 16, GTX 10, Intel, CPU: fp32 như cũ."""
        net = net.eval().to(self.dev.device)
        net._netanh_cl = False
        if self.dev.kind == "cpu":
            return net
        if key not in self.precision:
            tried: list[tuple[str, gpu_profile.ProbeResult]] = []
            name = "fp32"
            native_bf16 = self.dev.kind == "cuda" and (self.dev.cc or (0, 0))[0] >= 8 and not getattr(torch.version, "hip", None)
            for cand in gpu_profile.face_candidates(self.dev.family, key):
                if cand == "fp32" or (cand == "bf16" and not native_bf16):
                    continue
                r = gpu_profile.probe_dtype(net, self.dev.device, self._DT[cand], size=512, fn=gpu_profile.face_forward)
                tried.append((cand, r))
                if r.ok:
                    name = cand
                    break
            self.precision[key] = (name, tried)
            bad = [f"{c} {r.why}" for c, r in tried if not r.ok]
            if bad:
                self.say(f"{key}: {'; '.join(bad)} trên {self.dev.name}, dùng {name}")
        name = self.precision[key][0]
        net = net.to(self._DT[name])
        self._memory_format(net, key, name)
        return net

    def _face_run(self, net: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        """Chạy model mặt đúng kiểu số và cách xếp bộ nhớ của nó; fp16/bf16 ra NaN thì đổi model đó về fp32 và chạy lại."""
        params = getattr(net, "parameters", None)
        pdt = next(params()).dtype if params is not None else torch.float32
        x = t.to(pdt)
        if getattr(net, "_netanh_cl", False):
            x = x.contiguous(memory_format=torch.channels_last)
        o = net(x, return_rgb=False)[0]
        if pdt != torch.float32 and not torch.isfinite(o).all():
            net.float()
            net._netanh_cl = False
            gpu_profile.to_contiguous(net)
            o = net(t.float().contiguous(), return_rgb=False)[0]
        return o.float()

    # ----- nạp model khi cần -----

    def _ensure(self, model_id: str):
        return weights.ensure(model_id, self.say, cancel=self.cancel_requested)

    def _load_compact(self, dn: float) -> torch.nn.Module:
        """realesr-general-x4v3; mức khử nhiễu bằng trộn trọng số (DNI) với bản wdn, như Real-ESRGAN gốc."""
        dn = round(float(np.clip(dn, 0, 1)), 2)
        key = ("compact", dn)
        if key not in self._esrgan:
            self.say("Đang nạp Real-ESRGAN compact…")
            a = torch.load(self._ensure("esrgan_compact"), map_location="cpu", weights_only=True)
            a = a.get("params_ema", a.get("params", a))
            if dn < 1:
                b = torch.load(self._ensure("esrgan_compact_wdn"), map_location="cpu", weights_only=True)
                b = b.get("params_ema", b.get("params", b))
                a = {k: dn * v + (1 - dn) * b[k] for k, v in a.items()}
            net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
            net.load_state_dict(a, strict=True)
            net, _ = self._place(net, "compact")
            self._esrgan = {k: v for k, v in self._esrgan.items() if not isinstance(k, tuple)}  # chỉ giữ 1 mức
            self._esrgan[key] = net
        return self._esrgan[key]

    def _load_extra_sr(self, model_id: str) -> tuple[torch.nn.Module, torch.dtype]:
        """Model phóng 4× kiến trúc mới (DAT-2, RealPLKSR) nạp bằng spandrel (MIT). Chỉ giữ một model loại này trong
        bộ nhớ. Trên GPU NVIDIA: fp16 nếu model hỗ trợ, không thì bf16 (DAT-2, GPU từ RTX 30), không thì fp32."""
        if model_id not in self._sr_extra:
            import spandrel

            info = weights.MANIFEST[model_id]
            path = self._ensure(model_id)
            self.say(f"Đang nạp {info.label}…")
            desc = spandrel.ModelLoader(device="cpu").load_from_file(str(path))
            if desc.scale != 4:
                raise weights.ModelError(f"{info.file}: cần model phóng 4×, file này {desc.scale}×")
            # DAT-2 không chạy fp16 (supports_half False): RTX 30 trở lên thử bf16 thật (Turing chỉ giả lập bf16), không
            # thì fp32. torch.cuda.is_bf16_supported() trả True giả trên Turing nên xét compute capability ≥ 8.
            net, dtype = self._place(desc.model, model_id, bool(desc.supports_half), bool(desc.supports_bfloat16))
            self._sr_extra = {model_id: (net, dtype)}
        return self._sr_extra[model_id]

    def _load_esrgan(self, scale: int) -> torch.nn.Module:
        if scale not in self._esrgan:
            path = self._ensure("esrgan_x4" if scale == 4 else "esrgan_x2")
            self.say("Đang nạp Real-ESRGAN…")
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)), strict=True)
            net, _ = self._place(net, f"esrgan_x{scale}")
            self._esrgan[scale] = net
        return self._esrgan[scale]

    def _load_face_net(self, name: str = "gfpgan") -> torch.nn.Module:
        if name == "restoreformer_pp":
            if getattr(self, "_restoreformer_pp", None) is None:
                from .archs.restoreformer_arch import RestoreFormer
                path = self._ensure("restoreformer_pp")
                self.say("Đang nạp RestoreFormer++…")
                net = RestoreFormer(head_size=4, ex_multi_scale_num=1)
                ckpt = torch.load(path, map_location="cpu", weights_only=True)  # checkpoint Lightning gốc của tác giả
                sd = ckpt.get("state_dict", ckpt)
                net.load_state_dict({k.removeprefix("vqvae."): v for k, v in sd.items()}, strict=True)
                self._restoreformer_pp = self._place_face(net, "restoreformer_pp")
            return self._restoreformer_pp
        if name == "restoreformer":
            if getattr(self, "_restoreformer", None) is None:
                from .archs.restoreformer_arch import RestoreFormer
                path = self._ensure("restoreformer")
                self.say("Đang nạp RestoreFormer…")
                net = RestoreFormer()
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)), strict=True)
                self._restoreformer = self._place_face(net, "restoreformer")
            return self._restoreformer
        return self._load_gfpgan()

    def _load_gfpgan(self) -> torch.nn.Module:
        if self._gfpgan is None:
            path = self._ensure("gfpgan")
            self.say("Đang nạp GFPGAN…")
            net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                                fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1,
                                sft_half=True)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(ckpt.get("params_ema", ckpt), strict=True)
            self._gfpgan = self._place_face(net, "gfpgan")
        return self._gfpgan

    # nhãn ParseNet (CelebAMask-HQ): 2–3 lông mày, 4–5 mắt, 6 kính, 11 miệng, 12–13 môi
    # Nhãn ParseNet của facexlib (thứ tự kiểm trên ảnh thật): 0 nền, 1 da, 2 mũi, 3 kính, 4–5 mắt, 6–7 lông mày,
    # 8–9 tai, 10 miệng, 11–12 môi, 13 tóc, 14 mũ, 15 khuyên tai, 16 vòng cổ, 17 cổ, 18 áo.
    # (Bản trước ghi nhầm thứ tự CelebAMask-HQ: vùng "ngũ quan" có cả tóc và mũi, thiếu một bên lông mày và miệng.)
    DETAIL_CLASSES = (3, 4, 5, 6, 7, 10, 11, 12)

    LIP_CLASSES = (10, 11, 12)  # miệng, môi trên, môi dưới
    FACE_ONLY = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)  # mặt không gồm tóc: viền má, quai hàm giáp tóc / nền
    # vùng ghép mặt AI vào ảnh: như facexlib, mọi nhãn trừ nền 0, mũ 14, vòng cổ 16, cổ 17, áo 18 (có tóc)
    PASTE_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15)
    PASTE_GROW, PASTE_SOFT = 10, 6.0  # px trên mặt 512: nới ra ngoài đường viền, độ mềm viền

    @torch.no_grad()
    def _presr_face(self, img: np.ndarray, lm, helper, dn: float) -> np.ndarray:
        """Mặt nhỏ (hai mắt cách < SMALL_FACE_ED px): phóng 4× vùng quanh mặt bằng Real-ESRGAN compact, rồi căn về
        512×512 như thường. Căn thẳng từ ảnh nhỏ thì model mặt nhận 40 px phóng bicubic, hay bịa ra tóc vàng, mắt xanh,
        người khác hẳn; phóng AI trước thì giữ được màu da, màu tóc, nét mặt (đo trên 4 mặt 16–20 px của ảnh nhóm)."""
        lm = np.asarray(lm, np.float32).reshape(5, 2)
        ed = float(np.linalg.norm(lm[0] - lm[1]))
        cx, cy = lm.mean(0)
        r = max(24, int(ed * 3.5))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = max(0, int(cx - r)), max(0, int(cy - r)), min(w, int(cx + r)), min(h, int(cy + r))
        sr = self._esrgan_run(img[y0:y1, x0:x1], 4, 0, net=self._load_compact(dn))
        m = cv2.estimateAffinePartial2D((lm - np.array([x0, y0], np.float32)) * 4, helper.face_template, method=cv2.LMEDS)[0]
        return cv2.warpAffine(sr, m, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

    def _paste_mask(self, helper, restored: np.ndarray) -> np.ndarray:
        """Mặt nạ ghép 0..1 (512×512). facexlib làm mềm hai lần Gauss σ 11 (nhân 101): viền thật của má, cằm, quai hàm
        chỉ nhận ~50 % mặt AI, phần còn lại lấy từ nền (nhòe như ảnh gốc), lại loang ra nền thành quầng. Ở đây nới vùng
        mặt ra PASTE_GROW px rồi làm mềm σ PASTE_SOFT: cả đường viền mặt lấy từ model mặt, chuyển sang nền gọn."""
        lab = self._parse(helper, restored)
        m = np.isin(lab, self.PASTE_CLASSES).astype(np.uint8)
        k = 2 * self.PASTE_GROW + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), self.PASTE_SOFT)
        t = 10  # bỏ viền đen do căn mặt
        m[:t], m[-t:], m[:, :t], m[:, -t:] = 0, 0, 0, 0
        return m

    def _paste_faces(self, helper, bg: np.ndarray) -> np.ndarray:
        """Như FaceRestoreHelper.paste_faces_to_input_image (use_parse) nhưng mặt nạ ghép theo _paste_mask."""
        h, w = helper.input_img.shape[:2]
        up = helper.upscale_factor
        hu, wu = int(h * up), int(w * up)
        out = cv2.resize(bg, (wu, hu), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) \
            if bg.shape[:2] != (hu, wu) else bg.astype(np.float32)
        for restored, inv in zip(helper.restored_faces, helper.inverse_affine_matrices):
            if up > 1:
                inv[:, 2] += 0.5 * up  # như facexlib: bù nửa điểm ảnh khi phóng
            face = cv2.warpAffine(restored, inv, (wu, hu)).astype(np.float32)
            m = cv2.resize(self._paste_mask(helper, restored), restored.shape[1::-1])
            m = cv2.warpAffine(m, inv, (wu, hu), flags=cv2.INTER_CUBIC)
            m = np.clip(m, 0, 1)[..., None]
            out = m * face + (1 - m) * out
        return np.clip(out, 0, 255).round().astype(np.uint8)

    @torch.no_grad()
    def _parse(self, helper, face512: np.ndarray) -> np.ndarray:
        """Nhãn ParseNet (512×512) của mặt đã căn."""
        t0 = time.perf_counter()
        x = cv2.cvtColor(face512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(((x - 0.5) / 0.5).transpose(2, 0, 1)).unsqueeze(0).to(helper.device)
        lab = helper.face_parse(t)[0].argmax(dim=1).squeeze().cpu().numpy()
        self._model_s = getattr(self, "_model_s", 0.0) + time.perf_counter() - t0
        return lab

    def _subjects(self, img: np.ndarray, lms: list, s: Settings, rep: "Report") -> list[str | None]:
        """Đối tượng của từng mặt (khoá facestyle.SUBJECT_KEYS, None = không áp preset). Đoán tuổi, giới tính lỗi
        (chưa tải được model) thì dùng preset trung tính "Người lớn"."""
        from . import subject as sj
        if s.subject == "off" or not lms:
            return [None] * len(lms)
        if s.subject != "auto":
            rep.subjects += [f"{sj.SUBJECTS.get(s.subject, s.subject)} (chọn tay)"] * len(lms)
            return [s.subject] * len(lms)
        try:
            if self._agegender is None:
                self._agegender = sj.AgeGender.load(self.say)
            attrs = self._agegender(img, [np.asarray(l, np.float64) for l in lms])
        except (Cancelled, InterruptedError):
            raise
        except Exception as e:  # thiếu model, lỗi mạng, lỗi OpenCV DNN: vẫn xử lý tiếp với preset trung tính
            rep.notes.append(f"Không đoán được tuổi, giới tính ({e.__class__.__name__}): dùng preset Người lớn")
            rep.subjects += [sj.SUBJECTS["adult"]] * len(lms)
            return ["adult"] * len(lms)
        rep.subjects += [a.label for a in attrs]
        return [a.subject for a in attrs]

    def _style_face(self, helper, face: np.ndarray, st) -> np.ndarray:
        """Phần da, màu của kiểu mặt (mịn, bóng, sáng, hồng hào) trên mặt đã căn 512."""
        from . import facestyle
        if st is None or all(abs(getattr(st, k)) < 1e-3 for k in ("smooth", "glow", "bright", "warm")):
            return face
        return facestyle.apply(face, st, facestyle.skin_mask(self._parse(helper, face)))

    @torch.no_grad()
    def _style_only(self, out: np.ndarray, s: Settings, rep: "Report") -> np.ndarray:
        """Mức 1–2 (không chạy model mặt): vẫn áp preset đối tượng. Dò mặt trên ảnh ra, căn, áp kiểu rồi chỉ cộng phần
        thay đổi về ảnh (không lấy lại cả mặt đã căn: căn và ghép ngược làm mặt mềm đi)."""
        from . import facestyle
        if s.subject == "off" or not all(weights.status(m) == "ready" for m in ("retinaface", "parsenet")):
            return out
        styles = facestyle.styles_from_settings(s.face_styles)
        helper = self._load_helper(1)
        helper.read_image(out)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        faces = [Face(np.asarray(b), np.asarray(l)) for b, l in zip(helper.det_faces, helper.all_landmarks_5)]
        if not faces:
            return out
        select_faces(faces, s.max_faces)
        keep = [i for i, f in enumerate(faces) if f.ok]
        if not keep:
            return out
        helper.det_faces = [helper.det_faces[i] for i in keep]
        helper.all_landmarks_5 = [helper.all_landmarks_5[i] for i in keep]
        subj = self._subjects(out, helper.all_landmarks_5, s, rep)
        helper.align_warp_face()
        helper.get_inverse_affine(None)
        h, w = out.shape[:2]
        acc = out.astype(np.float32)
        for face, inv, sb in zip(helper.cropped_faces, helper.inverse_affine_matrices, subj):
            self._check()
            styled = self._style_face(helper, face, styles.get(sb) if sb else None)
            if styled is face:
                continue
            delta = cv2.warpAffine(styled.astype(np.float32) - face.astype(np.float32), inv, (w, h))
            m = cv2.warpAffine(self._paste_mask(helper, face), inv, (w, h), flags=cv2.INTER_CUBIC)
            acc += np.clip(m, 0, 1)[..., None] * delta
            rep.styled = True
        return np.clip(acc + 0.5, 0, 255).astype(np.uint8) if rep.styled else out

    @torch.no_grad()
    def _deshine(self, out: np.ndarray, s: Settings, rep: "Report") -> np.ndarray:
        """Khử bóng loáng da (deshine.py) trên ảnh ra: dò mặt, căn, tách vùng da bằng ParseNet, đưa mặt nạ da về ảnh rồi
        khử bóng ngay trên điểm ảnh của ảnh ra (giữ nguyên vân da ở độ phân giải thật, không qua căn và ghép ngược).
        Cường độ từng mặt = thanh Khử bóng chung + nút "Khử bóng" của preset đối tượng (−1..1)."""
        from . import deshine, facestyle
        if not all(weights.status(m) == "ready" for m in ("retinaface", "parsenet")):
            rep.notes.append("Khử bóng da cần model tìm và tách vùng mặt (chưa tải): bỏ qua")
            return out
        helper = self._load_helper(1)
        helper.read_image(out)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        faces = [Face(np.asarray(b), np.asarray(l)) for b, l in zip(helper.det_faces, helper.all_landmarks_5)]
        if not faces:
            return out
        select_faces(faces, s.max_faces)
        keep = [i for i, f in enumerate(faces) if f.ok and f.width >= MIN_FACE_PX]
        if not keep:
            return out
        helper.det_faces = [helper.det_faces[i] for i in keep]
        helper.all_landmarks_5 = [helper.all_landmarks_5[i] for i in keep]
        styles = facestyle.styles_from_settings(s.face_styles)
        if s.subject != "off" and any(abs(st.matte) > 1e-3 for st in styles.values()):
            subj = self._subjects(out, helper.all_landmarks_5, s, Report())  # nhãn đối tượng đã ghi ở bước mặt
        else:
            subj = [None] * len(keep)
        helper.align_warp_face()
        helper.get_inverse_affine(None)
        h, w = out.shape[:2]
        n_done, frac = 0, []
        for face, inv, box, sb in zip(helper.cropped_faces, helper.inverse_affine_matrices, helper.det_faces, subj):
            self._check()
            st = styles.get(sb) if sb else None
            amount = float(np.clip(s.deshine + (st.matte if st else 0.0), 0, 1))
            if amount <= 1e-3:
                continue
            m512 = deshine.skin_mask512(self._parse(helper, face))
            m = np.zeros((h, w), np.float32)
            corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], np.float32) @ inv.T
            x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, (w, h))
            if x1 <= x0 or y1 <= y0:
                continue
            sh = inv.copy()
            sh[:, 2] -= (x0, y0)  # chỉ đưa mặt nạ về khung quanh mặt (ảnh ra có thể rất lớn)
            m[y0:y1, x0:x1] = np.clip(cv2.warpAffine(m512, sh, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR), 0, 1)
            fw = float(box[2] - box[0])
            st_ = {}
            out = deshine.remove_shine(out, m, amount, fw, st_)
            frac.append(st_.get("shine", 0.0))
            n_done += 1
        if n_done:
            rep.deshined = n_done
            rep.notes.append(f"Khử bóng da {n_done} khuôn mặt (vùng bóng ~{max(frac) * 100:.0f} % da mặt)")
        return out

    def _look(self, out: np.ndarray, s: Settings, rep: "Report") -> np.ndarray:
        """Tông mẫu (look.py): mặt nạ da, môi, tóc từ ParseNet của từng mặt (gộp lại cỡ ảnh ra), rồi look.apply."""
        from . import look as lookmod
        lk = lookmod.Look.from_dict(s.look)
        if not all(weights.status(m) == "ready" for m in ("retinaface", "parsenet")):
            rep.notes.append("Tông mẫu cần model tìm và tách vùng mặt (chưa tải): bỏ qua")
            return out
        helper = self._load_helper(1)
        helper.read_image(out)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        faces = [Face(np.asarray(b), np.asarray(l)) for b, l in zip(helper.det_faces, helper.all_landmarks_5)]
        if not faces:
            rep.notes.append("Tông mẫu: không thấy khuôn mặt, giữ tông ảnh")
            return out
        select_faces(faces, s.max_faces)
        keep = [i for i, f in enumerate(faces) if f.ok and f.width >= MIN_FACE_PX]
        if not keep:
            return out
        helper.det_faces = [helper.det_faces[i] for i in keep]
        helper.all_landmarks_5 = [helper.all_landmarks_5[i] for i in keep]
        helper.align_warp_face()
        helper.get_inverse_affine(None)
        h, w = out.shape[:2]
        acc = {k: np.zeros((h, w), np.float32) for k in ("core", "skin", "lips", "hair")}
        for face, inv in zip(helper.cropped_faces, helper.inverse_affine_matrices):
            self._check()
            m512 = lookmod.masks_from_labels(self._parse(helper, face))
            m512.core[:10], m512.core[-10:], m512.core[:, :10], m512.core[:, -10:] = 0, 0, 0, 0
            corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], np.float32) @ inv.T
            x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, (w, h))
            if x1 <= x0 or y1 <= y0:
                continue
            sh = inv.copy()
            sh[:, 2] -= (x0, y0)
            for k in acc:
                part = cv2.warpAffine(getattr(m512, k), sh, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
                np.maximum(acc[k][y0:y1, x0:x1], np.clip(part, 0, 1), out=acc[k][y0:y1, x0:x1])
        acc["core"] = (acc["core"] > 0.5).astype(np.float32)
        out, info = lookmod.apply(out, lookmod.Masks(**acc), lk, s.look_amount)
        if "da_truoc" in info:
            (l0, a0, b0), (l1, a1, b1) = info["da_truoc"], info["da_sau"]
            rep.notes.append(f"Tông mẫu {lk.name}: da L {l0:.0f}→{l1:.0f}, đỏ {a0:.0f}→{a1:.0f}, vàng {b0:.0f}→{b1:.0f}"
                             + (", " + info["nen"] if "nen" in info else ""))
        return out

    def _sharp_contour(self, helper, ai: np.ndarray, blended: np.ndarray) -> np.ndarray:
        """Đường viền ngoài của mặt (má, quai hàm, cằm giáp tóc / nền / cổ) lấy CONTOUR_AI phần từ bản AI dù độ trung
        thực cao: trộn 70 % mặt gốc nhòe làm viền mặt nhòe theo ảnh gốc trong khi bên trong mặt đã nét. Viền là dải
        ~10 px quanh ranh giới vùng mặt (ParseNet), làm mềm σ 4 px."""
        lab = self._parse(helper, ai)
        m = np.isin(lab, self.FACE_ONLY).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        band = (cv2.dilate(m, k) - cv2.erode(m, k)).astype(np.float32)
        t = 12
        band[:t], band[-t:], band[:, :t], band[:, -t:] = 0, 0, 0, 0
        band = cv2.GaussianBlur(band, (0, 0), 4)[..., None] * CONTOUR_AI
        out = blended.astype(np.float32) * (1 - band) + ai.astype(np.float32) * band
        return np.clip(out, 0, 255).round().astype(np.uint8)

    def _keep_lips(self, helper, face: np.ndarray, restored: np.ndarray) -> np.ndarray:
        """Giữ dáng và màu môi của ảnh gốc: model mặt hay vẽ lại môi (mỏng hơn, đổi viền, thêm bóng). Vùng môi lấy từ
        mặt gốc (đã căn) làm nét bằng unsharp, chỉ cộng thêm vân rất mịn (tần số cao, σ 1,5 px) của bản AI cho đỡ mềm;
        viền vùng môi làm mềm để khỏi lộ chỗ ghép."""
        lab = self._parse(helper, face)
        m = np.isin(lab, self.LIP_CLASSES).astype(np.float32)
        if m.sum() < 50:
            return restored
        m = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 4)[..., None]
        f = face.astype(np.float32)
        r = restored.astype(np.float32)
        sharp = f + 0.8 * (f - cv2.GaussianBlur(f, (0, 0), 2.0))
        fine = r - cv2.GaussianBlur(r, (0, 0), 1.5)
        lips = sharp + 0.5 * fine
        return np.clip(r * (1 - m) + lips * m, 0, 255).round().astype(np.uint8)

    @torch.no_grad()
    def _detail_mask(self, helper, face512: np.ndarray) -> np.ndarray:
        """Mặt nạ 0..1 vùng ngũ quan trên mặt đã căn 512×512, làm mềm viền ~6 px."""
        lab = self._parse(helper, face512)
        m = np.isin(lab, self.DETAIL_CLASSES).astype(np.float32)
        m = cv2.dilate(m, np.ones((7, 7), np.uint8))
        return cv2.GaussianBlur(m, (0, 0), 6)

    def _load_helper(self, upscale: int):
        if self._helper is None:
            self._ensure("retinaface")
            self._ensure("parsenet")
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            self.say("Đang nạp model tìm và tách vùng khuôn mặt…")
            det_device = self.dev.device if self.dev.kind in ("cuda", "cpu") else torch.device("cpu")
            self._helper = FaceRestoreHelper(upscale, face_size=512, crop_ratio=(1, 1),
                                             det_model="retinaface_resnet50", save_ext="png", use_parse=True,
                                             device=det_device, model_rootpath=str(weights.weights_dir()))
            self._helper.get_face_landmarks_5 = self._timed(self._helper.get_face_landmarks_5)
        self._helper.clean_all()
        self._helper.set_upscale_factor(upscale)
        return self._helper

    def models_ready(self) -> bool:
        return all(weights.status(m) == "ready" for m in ("retinaface", "parsenet", "gfpgan", "esrgan_compact", "esrgan_compact_wdn"))

    # ----- phân tích -----

    @torch.no_grad()
    def detect_faces(self, img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        helper = self._load_helper(1)
        helper.read_image(img)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        return [(np.asarray(b), np.asarray(l)) for b, l in zip(helper.det_faces, helper.all_landmarks_5)]

    def _face_boxes(self, img: np.ndarray) -> list[np.ndarray]:
        """Khung mặt của ảnh đang xử lý: lấy lại kết quả dò ở bước sửa màu nếu có; chưa tải model dò mặt thì rỗng."""
        if self._faces is not None:
            return self._faces
        if any(weights.status(m) != "ready" for m in ("retinaface", "parsenet")):
            return []
        try:
            self._faces = [b for b, _ in self.detect_faces(img)]
        except Exception:
            self._faces = []
        return self._faces

    def analyze(self, img: np.ndarray, with_faces: bool = True, max_faces: int = 24) -> Analysis:
        return analyze(img, self.detect_faces if with_faces else None, max_faces)

    # ----- Real-ESRGAN chia ô -----

    @torch.no_grad()
    def _esrgan_run(self, img: np.ndarray, scale: int, tile: int, net=None, dtype: torch.dtype | None = None,
                    pad: int = 10, label: str = "Real-ESRGAN") -> np.ndarray:
        """Ảnh vào được đổi sang đúng kiểu số của model (fp32 / fp16 / bf16 do _place chọn). `dtype` giữ để tương thích."""
        net = net if net is not None else self._load_esrgan(scale)
        h, w = img.shape[:2]
        ph, pw = h % 2, w % 2  # mô hình x2 dùng pixel_unshuffle: cạnh phải chẵn
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        x = _to_tensor(img, self.dev, False).to(next(net.parameters()).dtype)
        if not tile or (x.shape[2] <= tile and x.shape[3] <= tile):
            return _to_image(self._forward(net, x))[: h * scale, : w * scale]
        return self._tiled(net, x, scale, tile, pad=pad, label=label)[: h * scale, : w * scale]

    def _forward(self, net, x: torch.Tensor) -> torch.Tensor:
        """Chạy mạng nền. fp16 có thể tràn số (ra NaN/vô cực, thành mảng đen) ở vùng sáng phẳng trên một số GPU và
        bản PyTorch: gặp thì đổi hẳn mạng đó sang fp32 và chạy lại ô này (các ô sau cũng chạy fp32)."""
        pdt = next(net.parameters()).dtype
        x = x.to(pdt)
        if getattr(net, "_netanh_cl", False):
            x = x.contiguous(memory_format=torch.channels_last)
        o = net(x)
        # ô ra toàn đen trong khi ảnh vào không tối cũng là dấu hiệu tràn số (giá trị lỗi bị kẹp về 0)
        if not torch.isfinite(o).all() or (pdt != torch.float32 and float(o.amax()) < 0.02 < float(x.amax())):
            if pdt != torch.float32:
                net.float()
                if getattr(net, "_netanh_cl", False):  # fp32 trên card cũ hơn không lợi từ kênh cuối: về như cũ
                    gpu_profile.to_contiguous(net)
                    net._netanh_cl = False
                self._fp32_fallback = True
                o = net(x.float().contiguous())
            if not torch.isfinite(o).all():  # vẫn lỗi ở fp32: thay giá trị lỗi để khỏi ra mảng đen
                self._nonfinite = True
                o = torch.nan_to_num(o.float(), nan=0.5, posinf=1.0, neginf=0.0)
        return o

    def _auto_tile(self, model: str, net) -> int:
        if self.dev.kind == "cpu":
            return gpu_profile.CPU_TILE
        dt = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(next(net.parameters()).dtype, "fp32")
        k = self.vram_share
        return gpu_profile.auto_tile(model, self.dev.free_vram_gb() * k, self.dev.vram_gb * k, dt, self.dev.family.key)

    def _tiled(self, net, x: torch.Tensor, scale: int, tile: int, pad: int, label: str = "Real-ESRGAN") -> np.ndarray:
        """Chạy chia ô, ghép vào ảnh uint8 trên RAM (không giữ tensor float cả ảnh ra: ảnh 12 MP phóng 4× là 192 MP,
        tensor float32 tới 2,3 GB). Hai ô kề nhau chồng 2·pad px và được trộn dần: model có phép toán theo cả ô (DAT-2
        chú ý theo kênh trên toàn ô) cho vân hơi khác giữa các ô, cắt thẳng sẽ lộ đường nối. Trọng số mỗi ô là tích các
        dốc ở bốn cạnh, dốc hai ô kề nhau bù nhau (tổng 1), bỏ pad/2 px sát mép ô (thiếu ngữ cảnh). Chỉ giữ một dải float
        cao một hàng ô."""
        tile, pad = tile - tile % 2, pad - pad % 2  # ảnh đã đệm về cạnh chẵn: mọi ô có cạnh chẵn
        _, c, h, w = x.shape
        out = np.empty((h * scale, w * scale, 3), np.uint8)
        ny, nx = math.ceil(h / tile), math.ceil(w / tile)

        def up(coord: np.ndarray, start: int) -> np.ndarray:
            """dốc lên của ô bắt đầu ở `start` (px ảnh vào, đã trừ pad) tại toạ độ ảnh ra `coord`"""
            if pad == 0:
                return (coord >= start * scale).astype(np.float32)
            k = coord - (start + pad // 2) * scale
            return np.clip((k + 0.5) / (pad * scale), 0, 1).astype(np.float32)

        def window(p0: int, p1: int, lo: int, hi: int, first: bool, last: bool) -> np.ndarray:
            coord = np.arange(p0 * scale, p1 * scale, dtype=np.float32)
            wgt = np.ones_like(coord) if first else up(coord, lo - pad)
            if not last:
                wgt = wgt * (1 - up(coord, hi - pad))
            return wgt

        carry, carry_r0 = None, 0
        for iy in range(ny):
            y0, y1 = iy * tile, min((iy + 1) * tile, h)
            py0, py1 = max(y0 - pad, 0), min(y1 + pad, h)
            r0, r1 = py0 * scale, py1 * scale
            acc = np.zeros((r1 - r0, w * scale, 3), np.float32)
            if carry is not None:
                acc[carry_r0 - r0:carry_r0 - r0 + len(carry)] += carry
            wy = window(py0, py1, y0, y1, iy == 0, iy == ny - 1)[:, None, None]
            for ix in range(nx):
                self._check()
                x0, x1 = ix * tile, min((ix + 1) * tile, w)
                px0, px1 = max(x0 - pad, 0), min(x1 + pad, w)
                o = self._forward(net, x[:, :, py0:py1, px0:px1])
                o = (o.squeeze(0).float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255.0)
                wx = window(px0, px1, x0, x1, ix == 0, ix == nx - 1)[None, :, None]
                acc[:, px0 * scale:px1 * scale] += o * (wy * wx)
            done = r1 if iy == ny - 1 else (y1 - pad) * scale  # các hàng ô dưới không còn góp vào
            out[r0:done] = np.clip(acc[:done - r0] + 0.5, 0, 255).astype(np.uint8)
            carry, carry_r0 = acc[done - r0:], done
            self.say(f"{label}: {100 * (iy + 1) // ny}%")
        return out

    def _background(self, img: np.ndarray, s: Settings, rep: Report) -> tuple[np.ndarray, float]:
        """Mạng nền cho toàn ảnh (Real-ESRGAN, hoặc DAT-2 / RealPLKSR). Trả về (ảnh ra, độ mờ d đã khử: 1 = không khử
        mờ). Model kiến trúc mới lỗi trên thiết bị này (không phải thiếu bộ nhớ): dùng Real-ESRGAN compact, ghi lý do."""
        extra = weights.SPANDREL_SR.get(s.sr_model)
        if extra is not None:
            try:
                return self._background_run(img, s, rep, extra)
            except (Cancelled, InterruptedError, weights.ModelError):
                raise
            except RuntimeError as e:
                self._sr_extra = {}
                label = weights.MANIFEST[extra].label
                rep.notes.insert(0, f"⚠ {label} lỗi trên {self.dev.name} ({e.__class__.__name__}: "
                                    f"{str(e).splitlines()[0][:160] if str(e) else ''}): dùng Real-ESRGAN compact thay")
                out = self._background_run(img, s.with_(sr_model="compact"), rep, None)
                rep.bg_model += f" thay cho {label} (lỗi)"
                return out
        return self._background_run(img, s, rep, None)

    def _background_run(self, img: np.ndarray, s: Settings, rep: Report, extra: str | None) -> tuple[np.ndarray, float]:
        compact = s.sr_model == "compact"
        scale = 4 if (s.upscale == 4 or compact or extra) else 2
        dtype, pad, label = None, 10, "Real-ESRGAN"
        if extra:
            net, dtype = self._load_extra_sr(extra)
            pad, label = 16, weights.MANIFEST[extra].label  # mạng chú ý theo cửa sổ: chồng ô rộng hơn cho khỏi lộ mép
        else:
            net = self._load_compact(s.sr_dn) if compact else None
        h0, w0 = img.shape[:2]
        # khử mờ: ảnh mờ d lần chỉ có chi tiết của ảnh nhỏ hơn d lần → thu nhỏ trước, để mạng 4× vẽ lại chi tiết
        d = classic.blur_factor(img) if s.deblur < 0 else max(1.0, float(s.deblur))
        if d > 1.0:
            img = cv2.resize(img, (max(8, round(w0 / d)), max(8, round(h0 / d))), interpolation=cv2.INTER_AREA)
            rep.notes.append(f"Ảnh mờ khoảng {d:g} lần: khử mờ bằng AI")
            if not compact:
                scale = 4
        tile = s.tile
        if tile < 0:  # tự động: GPU theo bộ nhớ còn trống và kiểu số, CPU 400 px như cũ
            tile = self._auto_tile(s.sr_model if s.sr_model in gpu_profile.ACT_BYTES_PER_PX else "rrdb",
                                   net if net is not None else self._load_esrgan(scale))
            if self.dev.kind not in ("cuda", "xpu") and extra:  # DirectML, Metal: không đọc được bộ nhớ, giữ trần cũ
                tile = min(tile, weights.MANIFEST[extra].tile or tile)
            # ô to thì dải cộng dồn float32 trên RAM (cao một hàng ô × rộng cả ảnh ra) cũng to: ảnh rộng 4000 px phóng 4×
            # với ô 1536 px là gần 1,2 GB. Giữ dải dưới gpu_profile.BAND_BYTES để máy 8 GB RAM không thiếu bộ nhớ.
            band = int(gpu_profile.BAND_BYTES / (12 * scale * scale * max(1, img.shape[1])))
            tile = max(gpu_profile.TILE_MIN, min(tile, band - band % 32))
        elif extra and tile:
            tile = min(tile, weights.MANIFEST[extra].tile or tile)
        self.last_tile = tile
        mkey = s.sr_model if s.sr_model in gpu_profile.MODEL_DB else "rrdb"
        lvl, why = gpu_profile.fits(self.dev.family, mkey, self.dev.vram_gb)
        if lvl != "ok" and self.dev.kind != "cpu":  # yêu cầu phần cứng khai trong du_lieu/model_profiles.json
            rep.notes.append(f"{label if extra else gpu_profile.MODEL_DB[mkey].label} trên {self.dev.name}: {why}")
        self._fp32_fallback = self._nonfinite = False
        while True:
            try:
                out = self._esrgan_run(img, scale, tile, net, dtype, pad, label)
                break
            except _OOM:
                self._empty_cache()
                if tile and tile <= 128:
                    raise
                tile = 256 if not tile else tile // 2
                rep.notes.append(f"GPU thiếu bộ nhớ, chia ô nhỏ hơn ({tile} px)")
        if compact:
            name = "Real-ESRGAN compact"
        elif extra:
            name = label
        else:
            name = f"Real-ESRGAN x{scale}plus"
        if self.dev.kind != "cpu":
            p = next((net if net is not None else self._load_esrgan(scale)).parameters()).dtype
            name += {torch.float16: " (fp16)", torch.bfloat16: " (bf16)"}.get(p, " (fp32)")
        rep.bg_model = name
        if self._fp32_fallback:
            rep.notes.append(f"{label}: fp16 tràn số trên {self.dev.name}, đã chạy lại bằng fp32 (chậm hơn)")
            if extra:
                self._sr_extra[extra] = (net, torch.float32)
        if self._nonfinite:
            rep.notes.append(f"{label}: model ra giá trị lỗi cả khi chạy fp32, vùng lỗi được thay bằng màu xám")
        tw, th = w0 * s.upscale, h0 * s.upscale
        if out.shape[:2] != (th, tw):
            interp = cv2.INTER_AREA if out.shape[1] > tw else cv2.INTER_CUBIC
            out = cv2.resize(out, (tw, th), interpolation=interp)
        return out, d

    # ----- xử lý chính -----

    @torch.no_grad()
    def process(self, img: np.ndarray, s: Settings | None = None) -> tuple[np.ndarray, Report]:
        s = s or Settings()
        rep = Report(mode=s.mode, device=self.dev.name)
        t0 = time.perf_counter()
        self._model_s = 0.0
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            img = resize_long(img, MAX_SIDE)  # chặn mỗi cạnh tối thiểu 1 px (ảnh dải rất dài và hẹp)
            rep.notes.append(f"Ảnh lớn hơn {MAX_SIDE} px nên đã thu nhỏ trước khi xử lý")
        if s.upscale <= 0:  # Tự động theo cỡ ảnh
            from .levels import auto_upscale
            s = s.with_(upscale=auto_upscale(img.shape[1], img.shape[0]))
            if s.upscale == 4:
                rep.notes.append(f"Ảnh nhỏ ({img.shape[1]}×{img.shape[0]} px): phóng 4× để ảnh ra đủ lớn")
        self._repair_mask = None
        self._scene_applied = False
        self._ai_src = None
        if s.repair > 0 or s.fade > 0 or s.auto_rotate or s.damage_mask or s.scene_vivid > 0:
            from . import restore
            try:
                img = restore.run(self, img, s.repair, s.fade, s.auto_rotate, rep, s.damage_mask, s.fade_style, s.scene_vivid,
                                  s.scene_ai if s.scene_vivid > 0 else 0.0)
            except (Cancelled, InterruptedError):
                raise
            except (weights.ModelError, OSError, RuntimeError, MemoryError) as e:
                rep.notes.append(f"Không khôi phục được ảnh cũ ({e}); bỏ qua bước này")
        self._faces = None  # vùng mặt dò được ở bước sửa màu, dùng lại khi giữ mặt lúc phóng to AI
        graphic = None  # hình đồ hoạ xét trên ảnh vào (ảnh sau phóng to AI có thêm màu trung gian)
        if s.tone > 0:
            from . import exposure
            graphic = exposure.is_graphic(exposure._small(img)[0])
        img, s = self._fix_color(img, s, rep)

        out = None
        if s.mode in ("portrait", "old", "upscale"):
            try:
                try:
                    out = self._process_ai(img, s, rep)
                except (Cancelled, InterruptedError):
                    raise
                except (RuntimeError, MemoryError) as e:
                    if self.dev.kind in ("xpu", "directml", "mps") and not isinstance(e, weights.ModelError):
                        self._fallback_to_cpu(e, rep)
                        rep.faces_found = rep.faces_restored = rep.faces_kept = 0
                        rep.face_model = rep.bg_model = ""
                        rep.subjects, rep.styled = [], False
                        out = self._process_ai(img, s, rep)
                    else:
                        raise
            except (Cancelled, InterruptedError):
                raise
            except (weights.ModelError, OSError, RuntimeError, MemoryError) as e:
                rep.notes.append(f"Không chạy được AI ({e}); đã dùng Cải thiện nhanh thay thế")
                rep.mode = "quick"
                rep.ai_fallback = True
        if out is None:
            rep.device = "CPU"
            out = self._quick(img, s)
        if not (rep.faces_restored or rep.faces_kept) and s.subject != "off":  # mức 1–2, phóng to: không có model mặt
            try:
                out = self._style_only(out, s, rep)
            except (Cancelled, InterruptedError):
                raise
            except Exception as e:
                rep.notes.append(f"Không áp được preset đối tượng ({e.__class__.__name__})")
        self._check()
        out = blend_strength(img, out, s.strength)
        if s.color > 0 or s.gloss > 0:
            from . import color
            out = color.enhance(out, s.color, s.gloss)
        if s.tone > 0:
            from . import tone
            k = out.shape[1] / img.shape[1]  # mặt dò ở ảnh trước khi phóng
            boxes = [np.asarray(b[:4], np.float32) * k for b in (self._faces or [])]
            out, note = tone.bright_clear(out, s.tone, boxes, graphic)
            if note:
                rep.notes.append(note)
        if s.deshine > 0:  # sau tông tươi sáng: đo và khử đúng độ bóng của ảnh ra
            self._check()
            self.say("Đang khử bóng loáng da…")
            try:
                out = self._deshine(out, s, rep)
            except (Cancelled, InterruptedError):
                raise
            except Exception as e:  # không để lỗi khử bóng làm mất cả kết quả
                rep.notes.append(f"Không khử được bóng da ({e.__class__.__name__})")
        if s.look and s.look_amount > 0:  # sau tông, khử bóng (đo màu da trên ảnh đã sáng), trước đổi nền
            self._check()
            self.say("Đang áp tông mẫu…")
            try:
                out = self._look(out, s, rep)
            except (Cancelled, InterruptedError):
                raise
            except Exception as e:  # không để lỗi tông mẫu làm mất cả kết quả
                rep.notes.append(f"Không áp được tông mẫu ({e.__class__.__name__})")
        if s.scene_vivid > 0 and getattr(self, "_scene_applied", False):
            from . import restore
            out = restore.scene_finish(out, s.scene_vivid)
            src = getattr(self, "_ai_src", None)
            if s.scene_ai > 0 and src is not None:
                k, chroma = restore.ai_color_weight(src)
                if k <= 0:
                    rep.notes.append(f"Bỏ qua tô màu AI: ảnh còn đủ màu (sắc độ {chroma:.1f}), tô lại sẽ loè loẹt")
            if s.scene_ai > 0 and src is not None and k > 0:
                try:
                    self.say("Đang tô màu AI (DDColor)…")
                    out = restore.scene_ai_color(self, src, out, s.scene_ai * k)
                except Exception as e:  # thiếu model/RAM: giữ bản chỉnh màu thường
                    rep.notes.append(f"Không tô màu AI được: {e}")
            self._ai_src = None
        if s.bg_color is not None:
            from . import backdrop
            try:
                out, note = backdrop.replace(out, tuple(int(v) for v in s.bg_color))
                rep.notes.append(note)
            except Exception as e:  # không để lỗi đổi nền làm mất cả kết quả
                rep.notes.append(f"Không đổi được nền ({e.__class__.__name__})")
        rep.seconds = time.perf_counter() - t0
        rep.model_s = self._model_s
        rep.device_kind = self.dev.kind
        rep.device = self.dev.name
        return out, rep

    def _fix_color(self, img: np.ndarray, s: Settings, rep: "Report") -> tuple[np.ndarray, Settings]:
        """Sửa phơi sáng (ảnh quá tối, quá sáng, ngược sáng), rồi bộ lọc theo hãng: cân bằng trắng + sửa tông da + bão
        hoà, trước mọi bước khác (AI phục hồi mặt thấy sáng và màu đúng); độ nét nhân hệ số của hãng."""
        from . import exposure, phone, wb
        prof = phone.profile(s.phone)
        w = prof.wb if s.wb < 0 else s.wb
        sk = prof.skin if s.skin_tone < 0 else s.skin_tone
        expo = 0.0 if s.low_light else s.exposure  # Retinex đã làm sáng: không làm hai lần
        ready = all(weights.status(m) == "ready" for m in ("retinaface", "parsenet"))

        def detect(x: np.ndarray) -> list:
            try:
                return [b for b, _ in self.detect_faces(x)]
            except Exception:  # dò mặt lỗi: chỉ bỏ phần cần mặt (sửa da, ngược sáng)
                return []

        faces = []  # vùng mặt: bỏ khỏi ước lượng ám màu (da vốn ấm), nơi sửa tông da, mốc sáng cho sửa phơi sáng
        if (w > 0 or sk > 0 or expo > 0) and ready:
            faces = self._faces = detect(img)
        if expo > 0:
            img, p = exposure.correct(img, faces, expo)
            if p.kind:
                rep.notes.append(p.note)
                if p.kind == "tối" and p.gain >= 1.5 and not faces and ready:  # mặt quá tối có thể chưa dò ra
                    faces = self._faces = detect(img)
        img, notes = wb.correct(img, w, sk, prof.sat if s.wb != 0 else 0.0, faces)
        rep.notes.extend(notes)
        if abs(prof.sharpen - 1.0) > 1e-3:
            s = s.with_(sharpness=float(np.clip(s.sharpness * prof.sharpen, 0, 1)))
        return img, s

    def _quick(self, img: np.ndarray, s: Settings) -> np.ndarray:
        self.say("Đang cải thiện ảnh (cổ điển)…")
        return classic.quick_enhance(img, upscale=s.upscale, denoise_h=s.nlm_h, clahe_clip=s.clahe_clip,
                                     sharpen=s.unsharp_amount, bright=s.brightness, low_light=s.low_light)

    def _process_ai(self, img: np.ndarray, s: Settings, rep: Report) -> np.ndarray:
        up = s.upscale
        if s.mode == "old":
            # ảnh cũ: khử nhiễu mạnh hơn mức ước lượng + tương phản cục bộ ở kích thước gốc, chưa làm nét
            self.say("Đang làm sạch ảnh cũ…")
            h = s.nlm_h if s.nlm_h is not None else classic.auto_denoise_h(img) + 2
            img = classic.quick_enhance(img, upscale=1, denoise_h=h, clahe_clip=max(s.clahe_clip, 1.5), sharpen=0,
                                        bright=s.brightness, low_light=s.low_light)
        else:
            if s.low_light or abs(s.brightness) > 1e-3:
                img = classic.brightness(classic.retinex_l(img) if s.low_light else img, s.brightness)
            # Tương phản (CLAHE): trước bản 0.15.2 chỉ chạy ở mức 1 và Ảnh cũ, nên thanh Tương phản (và CLAHE mà Tự động bật
            # cho ảnh nhạt) không đổi gì ở mức 2–6 (audit 0.15.2). Chạy ở cỡ gốc, trước mạng nền, như chế độ Ảnh cũ.
            if s.clahe_clip > 1e-3:
                img = classic.clahe_l(img, s.clahe_clip)
            # Khử nhiễu chỉnh tay: compact nhận mức này qua sr_dn (trộn bản khử nhiễu của model); RRDB, RealPLKSR, DAT-2 không
            # có nút khử nhiễu nên trước 0.15.2 thanh Khử nhiễu không đổi gì với các model này. Khử NLM ở cỡ gốc như mức 1.
            if s.nlm_h is not None and s.nlm_h > 0 and s.sr_model != "compact" and s.sr_denoise is None:
                img = classic.denoise(img, s.nlm_h)
        self._check()

        bg_name = weights.MANIFEST[weights.SPANDREL_SR[s.sr_model]].label if s.sr_model in weights.SPANDREL_SR else "Real-ESRGAN"
        self.say(f"Đang làm nét toàn ảnh bằng {bg_name} trên {self.dev.name}…")
        bg, d = self._background(img, s, rep)
        self._check()
        post_amount = 0.4 * s.unsharp_amount  # model đã làm nét: giảm ~60% so với cổ điển

        def post(x: np.ndarray, face_mask: np.ndarray | None = None) -> np.ndarray:
            sharp = classic.unsharp(x, amount=post_amount, sigma=1.0)
            if face_mask is None:
                return sharp
            m = face_mask[..., None]
            return (x.astype(np.float32) * m + sharp.astype(np.float32) * (1 - m)).round().astype(np.uint8)

        if s.mode == "upscale":
            out = post(bg)
            if d > 1.0 and s.face_ai < 0.999:  # khử mờ AI trên ảnh có người: giữ nét mặt gốc
                boxes = [b for b in self._face_boxes(img) if b[2] - b[0] >= MIN_FACE_PX]
                if boxes:
                    rep.notes.append(f"Giữ nét mặt gốc cho {len(boxes)} khuôn mặt ({s.face_ai * 100:.0f} % AI)")
                    out = protect_faces(out, img, boxes, s.face_ai)
            return out

        # ① tìm mặt trên ảnh gốc (đo được giữ danh tính tốt hơn tìm trên nền đã qua Real-ESRGAN)
        self.say("Đang tìm khuôn mặt…")
        helper = self._load_helper(up)
        helper.read_image(img)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        faces = [Face(np.asarray(b), np.asarray(l)) for b, l in zip(helper.det_faces, helper.all_landmarks_5)]
        rep.faces_found = len(faces)
        if not faces:
            rep.notes.append("Không thấy khuôn mặt: chỉ làm nét toàn ảnh")
            return post(bg)
        select_faces(faces, s.max_faces)
        from collections import Counter
        for reason, cnt in Counter(f.reason for f in faces if not f.ok).items():
            rep.notes.append(f"Bỏ qua {cnt} mặt: {reason} (chỉ làm nét cùng nền)")
        keep = [i for i, f in enumerate(faces) if f.ok]
        small = [i for i in keep if _eye_dist(helper.all_landmarks_5[i]) < AI_MIN_EYE]
        if small:  # mặt rất nhỏ (ảnh toàn thân, ảnh nhóm): model mặt không còn gì để dựng, để model nền làm nét
            keep = [i for i in keep if i not in small]
            rep.notes.append(f"{len(small)} mặt quá nhỏ (hai mắt cách dưới {AI_MIN_EYE:g} px): không dùng model mặt")
        # mặt không qua model mặt (quá nhỏ, vượt giới hạn): mạng nền vẽ chúng thành mặt nạ có vạch ngang, nên trộn lại
        # với ảnh vào theo face_ai (như chế độ Phóng to)
        left = [np.asarray(helper.det_faces[i][:4], np.float32) for i in range(len(faces))
                if i not in keep and faces[i].width >= MIN_FACE_PX]
        if left and s.face_ai < 0.999:
            for bx in left:
                bg = protect_faces(bg, img, [bx], s.face_ai)
            rep.notes.append(f"Giữ nét gốc cho {len(left)} mặt nhỏ ({s.face_ai * 100:.0f} % AI)")
        helper.det_faces = [helper.det_faces[i] for i in keep]
        helper.all_landmarks_5 = [helper.all_landmarks_5[i] for i in keep]
        if not keep:
            return post(bg)

        # ② căn mặt về 512×512 chuẩn FFHQ
        helper.align_warp_face()
        # ③ GFPGAN + khớp màu + trộn theo độ trung thực
        net = self._load_face_net(s.face_model)
        w_ai0 = float(np.clip(s.ai_weight, 0, 1))
        w_det0 = w_ai0 if s.detail_fidelity is None else float(np.clip(1 - s.detail_fidelity, 0, 1))
        from . import facestyle
        styles = facestyle.styles_from_settings(s.face_styles)
        subj = self._subjects(img, helper.all_landmarks_5, s, rep)
        ok_faces = 0
        for n, face in enumerate(helper.cropped_faces, 1):
            st = styles.get(subj[n - 1]) if subj[n - 1] else None
            w_ai, w_det = st.weights(w_ai0, w_det0) if st else (w_ai0, w_det0)
            small = is_small_face(helper.all_landmarks_5[n - 1])
            self._check()
            self.say(f"Đang làm nét khuôn mặt {n}/{len(helper.cropped_faces)}…")
            if small and s.small_face_sr:  # mặt nhỏ: phóng AI trước rồi mới căn, model mặt nhận ảnh thật thay vì ảnh bicubic mờ
                try:
                    face = self._presr_face(img, helper.all_landmarks_5[n - 1], helper, s.sr_dn)
                except (RuntimeError, MemoryError, weights.ModelError, OSError):
                    pass
            t = (_to_tensor(face, self.dev, False) - 0.5) / 0.5
            try:
                restored = _to_image(self._face_run(net, t), -1.0, 1.0)
                ok_faces += 1
            except RuntimeError as e:  # lỗi model với mặt này: giữ mặt gốc (vẫn ghép lại để đủ danh sách mặt)
                rep.notes.append(f"Mặt {n} lỗi khi xử lý ({e.__class__.__name__}), giữ nguyên")
                restored = face
            if s.color_match and restored is not face:
                restored = match_color(restored, face)
            ai = restored
            dmg = None
            if getattr(self, "_repair_mask", None) is not None:
                from . import restore
                dmg = restore.face_damage_weight(self, helper, n - 1)
            if abs(w_det - w_ai) > 1e-3 and helper.use_parse:
                # trộn theo vùng: mắt/lông mày/môi lấy w_det phần AI, còn lại (da, mũi, tai) w_ai
                wmap = w_ai + (w_det - w_ai) * self._detail_mask(helper, restored)
                restored = (restored * wmap[..., None] + face * (1 - wmap[..., None])).round().clip(0, 255).astype(np.uint8)
            elif w_ai < 0.999:
                restored = cv2.addWeighted(restored, w_ai, face, 1 - w_ai, 0)
            if dmg is not None and ai is not face:
                # mặt nhỏ có chỗ hỏng: phần trộn AI của cả mặt cũng giảm, vì mặt đã mất ngũ quan thì model mặt bịa ra
                # kính, mắt lệch, tuổi khác (đo trên mặt trẻ em ~40 px); ảnh gốc ở đây là ảnh đã vá
                fw0 = float(helper.det_faces[n - 1][2] - helper.det_faces[n - 1][0])
                if fw0 < 90:
                    cap = 0.2 + 0.5 * float(np.clip((fw0 - 30) / 60, 0, 1))
                    if w_ai > cap:
                        restored = cv2.addWeighted(ai, cap, face, 1 - cap, 0)
                # vùng hỏng nặng (lá vàng, vết ố che mặt): phần đã vá chỉ là da mịn, nên lấy 100 % dựng lại của model
                # mặt mạnh nhất có sẵn (RestoreFormer++, giữ đúng người hơn) thay cho trộn với ảnh gốc
                heavy = ai
                if s.face_model != "restoreformer_pp" and weights.status("restoreformer_pp") == "ready":
                    try:
                        net2 = self._load_face_net("restoreformer_pp")
                        heavy = _to_image(self._face_run(net2, t), -1.0, 1.0)
                        if s.color_match:
                            heavy = match_color(heavy, face)
                    except (RuntimeError, OSError, weights.ModelError):
                        heavy = ai
                # mặt nhỏ (< ~90 px) mà mất ngũ quan: model mặt bịa ra người khác (đo: mặt trẻ em 40 px thành người lớn đeo
                # kính), nên mặt càng nhỏ càng ít lấy phần dựng lại; dưới 30 px chỉ giữ chỗ đã vá
                fw = float(helper.det_faces[n - 1][2] - helper.det_faces[n - 1][0])
                dmg = dmg * float(np.clip((fw - 30) / 60, 0, 1)) * 0.8
                restored = (restored * (1 - dmg[..., None]) + heavy * dmg[..., None]).round().clip(0, 255).astype(np.uint8)
                rep.notes.append(f"Mặt {n} ({fw:.0f} px): vùng hỏng nặng, dựng lại tối đa {dmg.max() * 100:.0f} % bằng AI"
                                 + ("" if fw >= 60 else "; mặt nhỏ nên giữ phần đã vá, không bịa thêm"))
            if w_ai < CONTOUR_AI and helper.use_parse and ai is not face:
                restored = self._sharp_contour(helper, ai, restored)
            if s.keep_lips and helper.use_parse and ai is not face and dmg is None:  # mặt hỏng: môi gốc cũng hỏng
                restored = self._keep_lips(helper, face, restored)
            if st is not None and helper.use_parse:
                styled = self._style_face(helper, restored, st)
                rep.styled = rep.styled or styled is not restored
                restored = styled
            helper.add_restored_face(restored)
        # chỉ đếm mặt AI chạy được; mặt lỗi model dùng lại ảnh gốc vẫn nằm trong helper.restored_faces để ghép
        rep.faces_restored = ok_faces
        rep.faces_kept = len(helper.restored_faces) - ok_faces
        from .levels import FACE_MODELS
        rep.face_model = FACE_MODELS.get(s.face_model, s.face_model)

        # ④ tách vùng (ParseNet) + ⑤ ghép ngược có làm mềm viền
        self.say("Đang ghép khuôn mặt vào ảnh…")
        helper.get_inverse_affine(None)
        out = self._paste_faces(helper, bg)
        # ⑥ hậu xử lý nhẹ, không làm nét lại vùng mặt
        mask = face_union_mask(helper.inverse_affine_matrices, out.shape[:2]) if s.protect_faces else None
        return post(out, mask)
