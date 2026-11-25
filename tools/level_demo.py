"""Lưới ảnh so 5 mức nét trên cùng một ảnh mờ, kèm thời gian từng mức và mức mà Tự động chọn (để xem bằng mắt).

Mỗi hàng: ảnh vào (đã làm mờ σ, cùng cách với tools/level_bench.py) | mức 1 … mức 5. Ô của mức Tự động chọn có viền
xanh. Cắt quanh mặt lớn nhất (không có mặt thì giữa ảnh) và phóng các ô về cùng cỡ để so chi tiết.

    python tools/level_demo.py --images obama two_people --sigma 1 3 --out /tmp/level_demo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from net_anh.analysis import recommend  # noqa: E402
from net_anh.levels import LEVELS, auto_level, level_settings  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from level_bench import degrade  # noqa: E402
from phone_cast_demo import label  # noqa: E402

TILE = 360


def crop_frac(an, h: int, w: int) -> tuple[float, float, float, float]:
    """Vùng cắt (tỉ lệ 0..1 theo y0, x0, y1, x1) quanh mặt lớn nhất, rộng gấp 2,2 lần mặt; không có mặt: giữa ảnh."""
    faces = an.good_faces or an.faces
    if faces:
        f = max(faces, key=lambda f: f.width)
        x1, y1, x2, y2 = (float(v) for v in f.bbox[:4])
        side = max(x2 - x1, y2 - y1) * 2.2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    else:
        side = min(h, w) * 0.6
        cx, cy = w / 2, h / 2
    side = min(side, h, w)
    x0 = float(np.clip(cx - side / 2, 0, w - side))
    y0 = float(np.clip(cy - side / 2, 0, h - side))
    return y0 / h, x0 / w, (y0 + side) / h, (x0 + side) / w


def tile(img: np.ndarray, fr) -> np.ndarray:
    h, w = img.shape[:2]
    y0, x0, y1, x1 = round(fr[0] * h), round(fr[1] * w), round(fr[2] * h), round(fr[3] * w)
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (TILE, TILE), interpolation=cv2.INTER_AREA if c.shape[0] > TILE else cv2.INTER_CUBIC)


def load(name: str, max_side: int) -> np.ndarray:
    p = Path(name)
    if not p.exists():
        p = Path("/tmp/claude-0/gt") / f"{name}.jpg"
    im = cv2.imread(str(p))
    if im is None:
        raise SystemExit(f"không đọc được {name}")
    h, w = im.shape[:2]
    k = min(1.0, max_side / max(h, w))
    return cv2.resize(im, (round(w * k) // 2 * 2, round(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["obama"])
    ap.add_argument("--sigma", nargs="+", type=float, default=[1.0, 3.0])
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--out", default="/tmp/claude-0/level_demo")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = Enhancer("auto")
    eng.process(np.full((64, 64, 3), 128, np.uint8), level_settings(5, recommend(eng.analyze(
        np.full((64, 64, 3), 128, np.uint8)))[1]))  # nạp sẵn model
    print(f"{'ảnh':12s}{'σ':>4s}{'blur_d':>7s}{'tự động':>8s}" + "".join(f"{'mức ' + str(n):>8s}" for n in LEVELS))
    for name in a.images:
        gt = load(name, a.max_side)
        rows = []
        for si, sigma in enumerate(a.sigma):
            lq = degrade(gt, sigma, 7 + si) if sigma > 0 else gt
            an = eng.analyze(lq, with_faces=True)
            _, base, _ = recommend(an)
            auto, _ = auto_level(an)
            fr = crop_frac(an, *lq.shape[:2])
            tiles = [label(tile(lq, fr), f"Ảnh vào · mờ σ{sigma:g} (~{an.blur_d:g}×)")]
            secs = []
            for n, lv in LEVELS.items():
                s = level_settings(n, base, an)
                t0 = time.perf_counter()
                res = eng.process(lq, s)[0]
                dt = time.perf_counter() - t0
                secs.append(dt)
                t = tile(res, fr)
                if n == auto:
                    cv2.rectangle(t, (0, 0), (TILE - 1, TILE - 1), (60, 170, 60), 6)
                tiles.append(label(t, f"{n} · {lv.name} · {dt:.1f} s" + ("  ← Tự động" if n == auto else "")))
            rows.append(np.hstack(tiles))
            print(f"{name:12s}{sigma:4g}{an.blur_d:7.1f}{auto:8d}" + "".join(f"{x:8.1f}" for x in secs), flush=True)
        grid = np.vstack(rows)
        cv2.imwrite(str(out_dir / f"{Path(name).stem}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
