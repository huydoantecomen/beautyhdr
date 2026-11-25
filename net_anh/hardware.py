"""Cấu hình máy (CPU, RAM, GPU, bộ nhớ GPU) và ước tính thời gian từng model, để cảnh báo trước khi chạy model nặng.

Chỉ cảnh báo, không chặn: model nặng vẫn chạy được trên máy yếu (thiếu bộ nhớ GPU thì chia ô nhỏ hơn), chỉ chậm hơn.

Tốc độ CPU đo bằng tools/speed_bench.py trên CPU 4 luồng (máy cloud, Xeon), máy khác nhân theo số luồng. Tốc độ GPU
chưa đo trên máy thật: lấy thời gian CPU chia GPU_SPEEDUP, ước lượng thô cho GPU NVIDIA tầm trung (RTX 3060). Thời gian
thật của mỗi ảnh hiện trong nhật ký và lịch sử.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_gb: float  # 0 = không rõ
    usable: bool  # PyTorch bản đang cài chạy được trên GPU này


@dataclass(frozen=True)
class Machine:
    cpu: str
    threads: int
    ram_gb: float
    ram_free_gb: float
    gpus: tuple[Gpu, ...]
    torch_cuda: bool  # PyTorch có CUDA (bản cài GPU NVIDIA)

    @property
    def nvidia_unused(self) -> Gpu | None:
        """GPU NVIDIA có trên máy nhưng bản đang cài chỉ chạy CPU."""
        return next((g for g in self.gpus if not g.usable and "nvidia" in g.name.lower()), None)

    def summary(self) -> str:
        parts = [f"CPU {self.cpu} ({self.threads} luồng)"]
        if self.ram_gb:
            parts.append(f"RAM {self.ram_gb:.0f} GB (còn trống {self.ram_free_gb:.0f} GB)")
        for g in self.gpus:
            v = f" {g.vram_gb:.0f} GB" if g.vram_gb else ""
            parts.append(f"GPU {g.name}{v}" + ("" if g.usable else " (bản đang cài chưa dùng được)"))
        if not self.gpus:
            parts.append("không thấy GPU dùng được")
        return " · ".join(parts)


@dataclass(frozen=True)
class Cost:
    """Chi phí một model. Model nền: giây cho 1 megapixel ảnh vào mạng (trước khi phóng 4×). Model mặt: giây mỗi mặt."""
    label: str
    cpu_s: float  # CPU 4 luồng
    vram_gb: float  # bộ nhớ GPU nên có để chạy ô 400 px không phải chia nhỏ
    ram_gb: float  # RAM máy nên có
    heavy: bool  # model nặng: hiện cảnh báo khi máy không có GPU


# tools/speed_bench.py --device cpu --threads 4, 2026-11-24: ô 256 px, lần nhanh nhất trong 2 lần sau 1 lần chạy nóng
# (compact 7,8; x4plus 127; x2plus 28,5 ≈ x4plus / 4; RealPLKSR 66,9; DAT-2 446 giây mỗi MP; mặt 1,21 / 2,58 / 2,64 giây).
# Bộ nhớ: đỉnh RAM khi chạy một ô trên CPU (≈ bộ nhớ GPU fp32 cho cùng ô), cộng dung lượng model, làm tròn lên.
COST: dict[str, Cost] = {
    "compact": Cost("Real-ESRGAN compact", 7.8, 1.0, 4, False),
    "rrdb": Cost("Real-ESRGAN x4plus", 127.0, 2.0, 6, False),
    "plksr": Cost("RealPLKSR", 67.0, 2.0, 6, False),
    "dat2": Cost("DAT-2", 446.0, 4.0, 8, True),
    "gfpgan": Cost("GFPGAN 1.4", 1.2, 1.5, 4, False),
    "restoreformer": Cost("RestoreFormer", 2.6, 1.5, 4, False),
    "restoreformer_pp": Cost("RestoreFormer++", 2.6, 1.5, 4, False),
}
FACE_COSTS = ("gfpgan", "restoreformer", "restoreformer_pp")
# Phần ngoài mạng AI, đo từng bước ở mức 5 trên ảnh 288×360, 800×1000, 1281×1600 (CPU 4 luồng): ước tính lệch thật dưới 12 %
FACE_EXTRA_S = 1.8  # mỗi mặt, ngoài mạng mặt: tách vùng mắt, môi (ParseNet), chỉnh màu, độ bóng
DETECT_S = 0.35  # tìm mặt, một lần mỗi ảnh
BASE_S = 0.5  # sửa màu, sáng trước khi vào mạng
POST_S_PER_MP = 1.1  # ghép mặt, tông, màu trên ảnh ra: giây cho mỗi MP ảnh ra
GPU_SPEEDUP = {"cuda": 25.0, "directml": 8.0, "mps": 8.0}  # ước lượng thô, xem đầu file
SLOW_SECONDS = 60  # ước tính chạy lâu hơn mức này thì cảnh báo


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return " ".join(str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).split())
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=3).stdout.strip()
            if out:
                return out
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return " ".join(line.split(":", 1)[1].split())
    except Exception:
        pass
    import platform

    return platform.processor() or platform.machine() or "không rõ"


def _ram() -> tuple[float, float]:
    """(tổng, còn trống) GB; (0, 0) nếu không đọc được."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / 1024**3, vm.available / 1024**3
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / 1024**3, st.ullAvailPhys / 1024**3
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {line.split(":")[0]: float(line.split()[1]) for line in f if ":" in line}
        return info["MemTotal"] / 1024**2, info.get("MemAvailable", info.get("MemFree", 0)) / 1024**2
    except Exception:
        return 0.0, 0.0


def _nvidia_smi() -> list[Gpu]:
    """GPU NVIDIA theo driver (có cả khi PyTorch bản CPU). nvidia-smi đi kèm driver NVIDIA."""
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if exe is None:
        return []
    try:
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: không bật cửa sổ đen
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=flags).stdout
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        m = re.match(r"\s*(.+?)\s*,\s*([\d.]+)\s*$", line)
        if m:
            gpus.append(Gpu(m[1], float(m[2]) / 1024, False))
    return gpus


@functools.lru_cache(maxsize=1)
def machine() -> Machine:
    gpus: list[Gpu] = []
    cuda = torch.cuda.is_available()
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(Gpu(p.name, p.total_memory / 1024**3, True))
    else:
        gpus += _nvidia_smi()
    try:
        import torch_directml  # type: ignore

        gpus += [Gpu(torch_directml.device_name(i), 0.0, True) for i in range(torch_directml.device_count())]
    except Exception:
        pass
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        gpus.append(Gpu("Apple (Metal)", 0.0, True))
    total, free = _ram()
    return Machine(_cpu_name(), os.cpu_count() or 1, total, free, tuple(gpus), cuda)


def _speedup(kind: str, threads: int) -> float:
    """Nhanh hơn CPU 4 luồng (nơi đo COST) bao nhiêu lần."""
    if kind in GPU_SPEEDUP:
        return GPU_SPEEDUP[kind]
    return max(0.5, min(threads, 16) / 4)


def estimate(sr_model: str, face_model: str | None, w: int, h: int, faces: int = 0, blur_d: float = 1.0,
             kind: str = "cpu", threads: int = 4, upscale: int = 2, overhead: bool = True) -> float:
    """Giây ước tính để làm nét một ảnh w×h (chưa gồm thời gian nạp model). `blur_d` > 1: ảnh mờ được thu nhỏ d lần
    trước khi vào mạng nền (pipeline._background); `faces`: số mặt chạy model mặt (None = không). `overhead=False`: chỉ
    tính riêng mạng AI (để so tốc độ các model), bỏ các bước khác như tìm mặt, ghép mặt, chỉnh màu, tông."""
    d = max(1.0, blur_d)
    mp = w * h / d / d / 1e6
    t = COST[sr_model].cpu_s * mp if sr_model in COST else 0.0
    if sr_model == "rrdb" and upscale < 4 and d <= 1:
        t /= 4  # RealESRGAN_x2plus chạy thân mạng ở nửa cạnh ảnh
    if face_model and faces:
        t += (COST[face_model].cpu_s if face_model in COST else 0.0) * faces
        if overhead:
            t += FACE_EXTRA_S * faces + DETECT_S  # chạy trên cùng thiết bị với model
    t /= _speedup(kind, threads)
    if overhead:  # bước xử lý ảnh thường (OpenCV, numpy) luôn chạy trên CPU
        t += (BASE_S + POST_S_PER_MP * w * h * upscale * upscale / 1e6) / _speedup("cpu", threads)
    return t


def fmt_seconds(t: float) -> str:
    if t < 10:
        return f"{max(t, 1):.0f} giây" if t >= 1 else "dưới 1 giây"
    if t < 90:
        return f"{t:.0f} giây"
    return f"{t / 60:.0f} phút" if t < 3600 else f"{t / 3600:.1f} giờ"


def warnings(sr_model: str, face_model: str | None, w: int, h: int, faces: int, blur_d: float, kind: str,
             threads: int, m: Machine | None = None, upscale: int = 2) -> list[str]:
    """Cảnh báo cho lựa chọn model hiện tại trên máy này (rỗng = ổn). Không chặn chạy."""
    m = m or machine()
    out: list[str] = []
    models = [sr_model] + ([face_model] if face_model and faces else [])
    heavy = [COST[x].label for x in models if x in COST and COST[x].heavy]
    t = estimate(sr_model, face_model, w, h, faces, blur_d, kind, threads, upscale)
    if kind == "cpu" and t >= SLOW_SECONDS:
        tg = estimate(sr_model, face_model, w, h, faces, blur_d, "cuda", threads, upscale)
        lighter = sr_model != "compact" or (face_model is not None and faces > 0 and face_model != "gfpgan")
        out.append(f"Chạy trên CPU: ước tính khoảng {fmt_seconds(t)} cho ảnh này"
                   + (f" ({', '.join(heavy)} là model nặng)" if heavy else "")
                   + f". Có GPU NVIDIA thì khoảng {fmt_seconds(tg)}. Vẫn chạy được, chỉ chậm"
                   + ("; muốn nhanh hãy chọn model nhẹ hơn." if lighter else "."))
    elif t >= 5 * SLOW_SECONDS:
        out.append(f"Ước tính khoảng {fmt_seconds(t)} cho ảnh này.")
    g = m.nvidia_unused
    if g is not None and kind == "cpu":
        out.append(f"Máy có {g.name} nhưng bản Beauty HDR đang cài chỉ chạy CPU. Mở lại BeautyHDR.exe "
                   "và chọn bản GPU để dùng card (giữ nguyên model).")
    if kind == "cuda":
        vram = max((x.vram_gb for x in m.gpus if x.usable), default=0)
        need = max((COST[x].vram_gb for x in models if x in COST), default=0)
        if vram and vram < need:
            out.append(f"GPU chỉ có {vram:.0f} GB bộ nhớ, {', '.join(COST[x].label for x in models if x in COST and COST[x].vram_gb > vram)}"
                       f" nên có {need:.0f} GB: app tự chia ô nhỏ hơn, chậm hơn một chút.")
    need_ram = max((COST[x].ram_gb for x in models if x in COST), default=0)
    if m.ram_gb and m.ram_gb + 0.5 < need_ram:
        out.append(f"Máy có {m.ram_gb:.0f} GB RAM, model đang chọn nên có {need_ram:.0f} GB: ảnh lớn có thể chậm hoặc "
                   "thiếu bộ nhớ (app tự thu nhỏ ảnh quá lớn).")
    return out


__all__ = ["Gpu", "Machine", "Cost", "COST", "FACE_COSTS", "machine", "estimate", "warnings", "fmt_seconds"]
