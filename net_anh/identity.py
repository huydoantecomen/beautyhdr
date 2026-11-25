"""Đo độ giữ danh tính (IDS) bằng ArcFace IR-SE50: cos giữa hai embedding khuôn mặt."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from . import weights


# Crop 512 theo chuẩn FFHQ rộng hơn crop 112 của InsightFace (mắt cách nhau 0.246 so với 0.314 chiều rộng).
# Cắt vùng [56:456] rồi thu về 112 để khoảng cách mắt và vị trí mắt khớp chuẩn ArcFace.
_ARC_BOX = (56, 456)


class Identity:
    def __init__(self, device: torch.device):
        from facexlib.recognition.arcface_arch import Backbone

        path = weights.ensure("arcface")
        net = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.device = device if isinstance(device, torch.device) and device.type in ("cuda", "cpu") else torch.device("cpu")
        self.net = net.eval().to(self.device)

    @torch.no_grad()
    def embed(self, faces512: list[np.ndarray]) -> torch.Tensor:
        a, b = _ARC_BOX
        batch = []
        for f in faces512:
            crop = cv2.resize(f[a:b, a:b], (112, 112), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(torch.from_numpy(((rgb - 0.5) / 0.5).transpose(2, 0, 1)))
        return self.net(torch.stack(batch).to(self.device))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        e = self.embed([a, b])
        return float((e[0] * e[1]).sum())
