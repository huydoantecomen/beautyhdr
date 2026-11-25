"""Chọn thiết bị chạy model: GPU của máy nếu có, nếu không thì CPU."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device | object
    name: str
    kind: str  # "cuda" | "directml" | "mps" | "cpu"
    half: bool  # dùng fp16 hay không

    def __str__(self) -> str:
        return self.name


def fp16_ok(gpu_name: str) -> bool:
    """GTX 16xx (Turing TU116/TU117, không có Tensor Core) hay ra NaN, ảnh đen khi chạy fp16: dùng fp32."""
    return "GTX 16" not in gpu_name.upper()


def _cpu() -> DeviceInfo:
    import os

    threads = torch.get_num_threads()
    return DeviceInfo(torch.device("cpu"), f"CPU ({os.cpu_count()} luồng, dùng {threads})", "cpu", False)


def list_devices() -> list[DeviceInfo]:
    """Liệt kê các thiết bị dùng được, GPU đứng trước CPU."""
    found: list[DeviceInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024**3
            found.append(DeviceInfo(torch.device(f"cuda:{i}"), f"GPU {props.name} ({vram:.1f} GB)", "cuda",
                                    fp16_ok(props.name)))
    try:  # GPU AMD / Intel trên Windows qua DirectML (tuỳ chọn: pip install torch-directml)
        import torch_directml  # type: ignore

        for i in range(torch_directml.device_count()):
            found.append(DeviceInfo(torch_directml.device(i), f"GPU {torch_directml.device_name(i)} (DirectML)", "directml", False))
    except Exception:
        pass
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        found.append(DeviceInfo(torch.device("mps"), "GPU Apple (Metal)", "mps", False))
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
