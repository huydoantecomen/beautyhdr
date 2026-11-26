"""Tô màu AI cho ảnh cảnh cũ (DDColor, Kang và cộng sự, ICCV 2023, Apache-2.0).

Model đoán màu (kênh a, b của Lab) từ độ sáng của ảnh, nên biết áo trắng là trắng, cây là xanh, jeans là xanh dương,
điều mà phép chỉnh màu toàn ảnh không làm được. Ta chỉ lấy a, b của nó; độ sáng giữ của ảnh gốc."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

INPUT = 512


def _load(enh):
    net = getattr(enh, "_ddcolor", None)
    if net is None:
        from .archs.ddcolor_arch.model import DDColor
        path = enh._ensure("ddcolor")
        net = DDColor(encoder_name="convnext-l", decoder_name="MultiScaleColorDecoder",
                      input_size=[INPUT, INPUT], num_output_channels=2, last_norm="Spectral", do_normalize=False,
                      num_queries=100, num_scales=3, dec_layers=9)
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        net.load_state_dict(sd.get("params", sd) if isinstance(sd, dict) else sd, strict=False)
        dev = enh.dev.device if enh.dev.kind == "cuda" else torch.device("cpu")
        enh._ddcolor = net.to(dev).eval()  # luôn fp32 (GTX 1660 Ti ra ảnh đen ở fp16)
        enh._ddcolor_dev = dev
    return net


@torch.no_grad()
def predict_ab(enh, img: np.ndarray) -> np.ndarray:
    """img BGR uint8 → a, b của Lab (float32, H×W×2, thang OpenCV float: a, b ∈ ≈[-110, 110]) ở đúng cỡ ảnh."""
    net = _load(enh)
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255
    small = cv2.cvtColor(cv2.resize(f, (INPUT, INPUT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2Lab)[..., :1]
    grey = cv2.cvtColor(np.concatenate([small, np.zeros((INPUT, INPUT, 2), np.float32)], -1), cv2.COLOR_LAB2RGB)
    t = torch.from_numpy(grey.transpose(2, 0, 1)).float().unsqueeze(0).to(enh._ddcolor_dev)
    ab = net(t).float().cpu()
    ab = F.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)[0].numpy().transpose(1, 2, 0)
    return ab.astype(np.float32)
