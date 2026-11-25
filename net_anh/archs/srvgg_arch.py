# Trích từ Real-ESRGAN (realesrgan/archs/srvgg_arch.py), BSD-3-Clause, (c) Xintao Wang.
# Mạng "compact": 16–32 lớp conv ở độ phân giải thấp rồi pixel_shuffle, ~1,2 triệu tham số (5 MB).
from torch import nn as nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))

        def act():
            if act_type == "relu":
                return nn.ReLU(inplace=True)
            if act_type == "prelu":
                return nn.PReLU(num_parameters=num_feat)
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.body.append(act())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(act())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # học phần dư so với phóng nearest
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
