"""Đo trên ảnh KHÔNG có mặt (phong cảnh, vật, chữ, kết cấu): bổ sung cho tools/evaluate.py vốn chỉ chấm mặt.

Ảnh gốc là bộ ảnh mẫu đi kèm scikit-image (không cần tải). Làm hỏng giống evaluate.py, rồi so với gốc bằng:
  PSNR, SSIM          (cao = tốt; ưu tiên ảnh hơi mờ nhưng không sai)
  GMSD                (thấp = tốt; độ lệch độ lớn gradient, nhạy với cạnh/chi tiết hơn PSNR)
  nét/GT              (độ nét so với ảnh gốc; ~1 là đúng, >1 là làm nét quá)

    python tools/evaluate_general.py --csv ket_qua_chung.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import classic  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from evaluate import LEVELS, degrade, psnr, sharp  # noqa: E402

IMAGES = ["coffee", "chelsea", "rocket", "astronaut", "hubble_deep_field", "immunohistochemistry",
          "brick", "grass", "gravel", "text", "page", "camera"]

VARIANTS = {
    "lanczos": None,
    "quick": Settings(mode="quick"),
    "upscale-rrdb": Settings(mode="upscale", sr_model="rrdb"),
    "compact-dn0.2": Settings(mode="upscale", sr_model="compact", sr_denoise=0.2),
    "compact-dn0.5": Settings(mode="upscale", sr_model="compact", sr_denoise=0.5),
    "compact-dn1": Settings(mode="upscale", sr_model="compact", sr_denoise=1.0),
}


def gmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Gradient Magnitude Similarity Deviation (Xue et al. 2014), trên ảnh xám thu nửa như bài gốc."""
    def gm(x):
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g = cv2.resize(cv2.blur(g, (2, 2)), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float64) / 3
        return np.sqrt(cv2.filter2D(g, -1, k) ** 2 + cv2.filter2D(g, -1, k.T) ** 2)
    ga, gb = gm(a), gm(b)
    c = 170.0
    gms = (2 * ga * gb + c) / (ga ** 2 + gb ** 2 + c)
    return float(gms.std())


def load(name: str, max_side: int) -> np.ndarray:
    im = getattr(skdata, name)()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2BGR)
    h, w = im.shape[:2]
    k = min(1.0, max_side / max(h, w))
    im = cv2.resize(im, (round(w * k) // 2 * 2, round(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--csv")
    a = p.parse_args()
    eng = Enhancer("auto")
    rows = []
    for ni, name in enumerate(IMAGES):
        gt = load(name, a.max_side)
        for li, level in enumerate(a.levels):
            lq = degrade(gt, level, seed=li * 1000 + ni)
            for v in a.variants:
                t0 = time.perf_counter()
                if VARIANTS[v] is None:
                    out = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                else:
                    out, _ = eng.process(lq, VARIANTS[v])
                    out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA) \
                        if out.shape[:2] != gt.shape[:2] else out
                dt = time.perf_counter() - t0
                r = dict(image=name, level=level, variant=v, psnr=psnr(out, gt),
                         ssim=structural_similarity(out, gt, channel_axis=2, data_range=255),
                         gmsd=gmsd(out, gt), sharp=sharp(out) / max(sharp(gt), 1e-6), sec=dt)
                rows.append(r)
                print(f"{name:22s} {level:4s} {v:14s} PSNR {r['psnr']:5.2f}  SSIM {r['ssim']:.3f}  "
                      f"GMSD {r['gmsd']:.3f}  nét {r['sharp']:.2f}  {dt:4.1f}s", flush=True)
    print("\nTrung bình:")
    print(f"{'mức':5s}{'biến thể':15s}{'PSNR':>7s}{'SSIM':>7s}{'GMSD':>7s}{'nét/GT':>8s}{'giây':>6s}")
    for level in a.levels:
        for v in a.variants:
            rs = [r for r in rows if r["level"] == level and r["variant"] == v]
            m = {k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "gmsd", "sharp", "sec")}
            print(f"{level:5s}{v:15s}{m['psnr']:7.2f}{m['ssim']:7.3f}{m['gmsd']:7.3f}{m['sharp']:8.2f}{m['sec']:6.1f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
