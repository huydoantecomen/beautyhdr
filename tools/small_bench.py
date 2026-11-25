"""Đo Tự động với ảnh quá nhỏ và ảnh mờ do bị phóng to (giả lập từ ảnh tốt).

Hai kiểu ảnh hỏng, đều nén JPEG như ảnh gửi qua Zalo, Messenger:
  nhỏ /2, nhỏ /4   ảnh gốc thu nhỏ 2 hoặc 4 lần (cạnh dài 100–1100 px), JPEG 80
  phóng ×2, ×3, ×4 ảnh gốc thu nhỏ d lần rồi phóng lại bằng bicubic về cỡ gốc (ảnh bị phóng to nên mờ), JPEG 85
Mỗi ảnh hỏng chạy phân tích + Tự động (recommend) như người dùng bấm Làm nét, với hệ số phóng 2× (mặc định cũ) và
4× (ảnh nhỏ) hoặc 1× (ảnh bị phóng to), rồi đưa ảnh ra về cỡ ảnh gốc để so: SSIM, GMSD (thấp = tốt), độ nét so với
gốc (1 = bằng gốc), IDS mặt, cạnh dài ảnh ra, thời gian. "bicubic": phóng thường, làm mốc.

    python tools/small_bench.py --csv ket_qua_anh_nho.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from net_anh.analysis import recommend  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from evaluate import sharp  # noqa: E402
from evaluate_general import gmsd, load as sk_load  # noqa: E402
from exposure_bench import FACES, GENERAL, load_face  # noqa: E402

KINDS = {"nhỏ /2": ("small", 2), "nhỏ /4": ("small", 4), "phóng ×2": ("zoom", 2), "phóng ×3": ("zoom", 3),
         "phóng ×4": ("zoom", 4)}


def jpeg(img, q):
    _, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def degrade(gt, kind, d):
    h, w = gt.shape[:2]
    small = cv2.resize(gt, (max(8, round(w / d)), max(8, round(h / d))), interpolation=cv2.INTER_AREA)
    if kind == "small":
        return jpeg(small, 80)
    return jpeg(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 85)


def to_size(img, w, h):
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=1200)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--csv")
    a = ap.parse_args()
    eng = Enhancer("auto")
    from net_anh.identity import Identity
    ident = Identity(eng.dev.device)
    eng.process(np.full((64, 64, 3), 128, np.uint8), Settings(mode="upscale"))  # nạp sẵn model
    rows = []
    for name in FACES + GENERAL:
        if a.only and name not in a.only:
            continue
        has_face = name in FACES
        gt = load_face(name, a.max_side) if has_face else sk_load(name, a.max_side)
        h, w = gt.shape[:2]
        mats, gts = [], []
        if has_face:
            helper = eng._load_helper(1)
            helper.read_image(gt)
            helper.get_face_landmarks_5(resize=640, eye_dist_threshold=20)
            helper.align_warp_face()
            mats = [m.copy() for m in helper.affine_matrices]
            gts = [c.copy() for c in helper.cropped_faces]
        for kname in a.kinds:
            kind, d = KINDS[kname]
            lq = degrade(gt, kind, d)
            an = eng.analyze(lq, with_faces=True)
            ups = (2, 4) if kind == "small" else (2, 1)
            methods = [("bicubic", None)] + [(f"Tự động {u}×", u) for u in ups]
            print(f"{name:12s} {kname:9s} vào {lq.shape[1]}×{lq.shape[0]}  blur_d {an.blur_d:.1f}  "
                  f"mặt {len(an.good_faces)}", flush=True)
            for mname, up in methods:
                t0 = time.perf_counter()
                if up is None:
                    out, why = cv2.resize(lq, (w, h), interpolation=cv2.INTER_CUBIC), ""
                else:
                    _, s, why = recommend(an, Settings(upscale=up))
                    out = eng.process(lq, s)[0]
                dt = time.perf_counter() - t0
                side = max(out.shape[:2])
                o = to_size(out, w, h)
                ids = float(np.mean([ident.similarity(g, cv2.warpAffine(o, mt, (512, 512), borderValue=(135, 133, 132)))
                                     for mt, g in zip(mats, gts)])) if mats else float("nan")
                r = dict(image=name, face=has_face, kind=kname, method=mname, w_in=lq.shape[1], h_in=lq.shape[0],
                         blur_d=an.blur_d, side_out=side, ssim=structural_similarity(o, gt, channel_axis=2, data_range=255),
                         gmsd=gmsd(o, gt), sharp=sharp(o) / max(sharp(gt), 1e-6), ids=ids, sec=dt,
                         why="; ".join(why[:1]) if why else "")
                rows.append(r)
                print(f"   {mname:12s} ra {side:5d} px  SSIM {r['ssim']:.3f} GMSD {r['gmsd']:.3f} nét {r['sharp']:.2f} "
                      f"IDS {ids:.3f} {dt:5.1f}s  {r['why']}", flush=True)
    print("\nTrung bình (ảnh có mặt | ảnh không mặt):")
    print(f"{'kiểu':10s}{'cách':14s}{'SSIM':>7s}{'GMSD':>7s}{'nét':>6s}{'IDS':>7s}{'cạnh ra':>8s}{'giây':>6s}")
    for face in (True, False):
        for kname in a.kinds:
            for m in dict.fromkeys(r["method"] for r in rows if r["kind"] == kname):
                rs = [r for r in rows if r["method"] == m and r["kind"] == kname and r["face"] == face]
                if not rs:
                    continue
                mm = {k: float(np.nanmean([r[k] for r in rs])) for k in ("ssim", "gmsd", "sharp", "ids", "side_out", "sec")}
                print(f"{kname:10s}{m:14s}{mm['ssim']:7.3f}{mm['gmsd']:7.3f}{mm['sharp']:6.2f}{mm['ids']:7.3f}"
                      f"{mm['side_out']:8.0f}{mm['sec']:6.1f}" + ("   (có mặt)" if face else ""))
        print()
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
