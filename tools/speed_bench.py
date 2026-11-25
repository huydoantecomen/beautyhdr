"""Đo tốc độ từng model trên máy đang chạy (CPU hoặc GPU), để cập nhật hardware.COST.

Model nền: giây cho 1 megapixel ảnh vào mạng (đo ô 256×256, lấy lần nhanh nhất trong 2 lần sau 1 lần chạy nóng máy).
Model mặt: giây mỗi mặt 512×512. Model chưa tải thì bỏ qua (tải trước ở Quản lý model).

    python tools/speed_bench.py [--device cpu|auto] [--threads 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import weights  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402


def best(fn, n=2):
    fn()  # chạy nóng máy
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    eng = Enhancer(a.device)
    print(f"Thiết bị: {eng.dev.name}, {torch.get_num_threads()} luồng")
    rng = np.random.default_rng(0)
    img = (rng.random((a.size, a.size, 3)) * 255).astype(np.uint8)
    mp = a.size * a.size / 1e6
    bg = {"compact": (("esrgan_compact", "esrgan_compact_wdn"), lambda: eng._esrgan_run(img, 4, 0, eng._load_compact(0.5))),
          "rrdb": (("esrgan_x4",), lambda: eng._esrgan_run(img, 4, 0, eng._load_esrgan(4))),
          "rrdb x2": (("esrgan_x2",), lambda: eng._esrgan_run(img, 2, 0, eng._load_esrgan(2)))}
    for key, mid in weights.SPANDREL_SR.items():
        bg[key] = ((mid,), lambda mid=mid: eng._esrgan_run(img, 4, 0, *eng._load_extra_sr(mid)))
    for name, (need, fn) in bg.items():
        if any(weights.status(m) != "ready" for m in need):
            print(f"{name:18s} chưa tải, bỏ qua")
            continue
        t = best(fn)
        print(f"{name:18s} {t / mp:8.1f} giây / MP ảnh vào")
    face = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    x = (torch.from_numpy(face.transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1).to(eng.dev.device)
    for name in ("gfpgan", "restoreformer", "restoreformer_pp"):
        if weights.status(name) != "ready":
            print(f"{name:18s} chưa tải, bỏ qua")
            continue
        net = eng._load_face_net(name)
        t = best(lambda: net(x, return_rgb=False), 3)
        print(f"{name:18s} {t:8.2f} giây / mặt")


if __name__ == "__main__":
    main()
