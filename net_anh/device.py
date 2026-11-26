"""Chọn thiết bị chạy model: GPU của máy nếu có, nếu không thì CPU."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import gpu_profile
from .gpu_profile import Family


@dataclass
class DeviceInfo:
    device: torch.device | object
    name: str
    kind: str  # "cuda" | "xpu" | "directml" | "mps" | "cpu"
    half: bool  # dòng card cho thử fp16 (vẫn qua bước kiểm từng model, xem gpu_profile.probe_dtype)
    family: Family = field(default_factory=lambda: gpu_profile.FAMILIES["cpu"])
    vram_gb: float = 0.0
    cc: tuple[int, int] | None = None  # compute capability (NVIDIA, AMD ROCm)

    def __str__(self) -> str:
        return self.name

    def free_vram_gb(self) -> float:
        """Bộ nhớ GPU còn trống lúc này (đã trừ app khác đang dùng); 0 = không đọc được."""
        try:
            if self.kind == "cuda":
                return torch.cuda.mem_get_info(self.device)[0] / 1024**3
            if self.kind == "xpu" and hasattr(torch.xpu, "mem_get_info"):
                return torch.xpu.mem_get_info(self.device)[0] / 1024**3
        except Exception:
            pass
        return 0.0


def fp16_ok(gpu_name: str) -> bool:
    """GTX 16xx (Turing TU116/TU117, không có Tensor Core) hay ra NaN, ảnh đen khi chạy fp16: dùng fp32.
    Giữ để tương thích; quyết định thật theo gpu_profile (dòng card) + chạy thử từng model."""
    return "fp16" in gpu_profile.nvidia_family(gpu_name).precisions


def _hip() -> bool:
    return bool(getattr(torch.version, "hip", None))


UNSUPPORTED: list[tuple[str, str]] = []  # (tên card, lý do) card có nhưng bản PyTorch này không chạy được


def _cpu() -> DeviceInfo:
    import os

    threads = torch.get_num_threads()
    return DeviceInfo(torch.device("cpu"), f"CPU ({os.cpu_count()} luồng, dùng {threads})", "cpu", False)


def list_devices() -> list[DeviceInfo]:
    """Liệt kê các thiết bị dùng được, GPU đứng trước CPU."""
    found: list[DeviceInfo] = []
    UNSUPPORTED.clear()
    if torch.cuda.is_available():
        hip = _hip()
        try:
            arch = [] if hip else torch.cuda.get_arch_list()
        except Exception:
            arch = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024**3
            cc = (props.major, props.minor)
            fam = gpu_profile.family_of("cuda", props.name, cc, hip)
            if not hip and not gpu_profile.arch_supported(cc, arch):
                # vd. GTX 1060 (sm_61) với PyTorch CUDA 12.8+: chạy là lỗi "no kernel image", nên không liệt kê
                UNSUPPORTED.append((props.name, f"bản PyTorch đang cài (CUDA {torch.version.cuda}) không còn hỗ trợ "
                                                f"{fam.label}; cần bản GPU cho card đời cũ"))
                continue
            found.append(DeviceInfo(torch.device(f"cuda:{i}"), f"GPU {props.name} ({vram:.1f} GB)", "cuda",
                                    "fp16" in fam.precisions, fam, vram, cc))
    xpu = getattr(torch, "xpu", None)
    try:  # Intel Arc, Iris Xe qua bản PyTorch XPU. fp32: chưa thử fp16 trên card Intel thật nên chọn đường chắc chắn
        if xpu is not None and xpu.is_available():
            for i in range(xpu.device_count()):
                props = xpu.get_device_properties(i)
                vram = getattr(props, "total_memory", 0) / 1024**3
                found.append(DeviceInfo(torch.device(f"xpu:{i}"), f"GPU {props.name} ({vram:.1f} GB, Intel XPU)", "xpu",
                                        False, gpu_profile.family_of("xpu", props.name), vram))
    except Exception:
        pass
    try:  # GPU AMD / Intel trên Windows qua DirectML (tuỳ chọn: pip install torch-directml)
        import torch_directml  # type: ignore

        for i in range(torch_directml.device_count()):
            found.append(DeviceInfo(torch_directml.device(i), f"GPU {torch_directml.device_name(i)} (DirectML)", "directml",
                                    False, gpu_profile.FAMILIES["amd_dml"]))
    except Exception:
        pass
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        found.append(DeviceInfo(torch.device("mps"), "GPU Apple (Metal)", "mps", False, gpu_profile.FAMILIES["apple"]))
    found.append(_cpu())
    return found


def pick_device(preference: str = "auto") -> DeviceInfo:
    """preference: "auto" (GPU nếu có), "gpu", "cpu"."""
    devices = list_devices()
    if preference == "cpu":
        return devices[-1]
    gpus = [d for d in devices if d.kind != "cpu"]
    if gpus:
        return gpus[0]
    return devices[-1]
