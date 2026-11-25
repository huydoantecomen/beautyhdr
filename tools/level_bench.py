"""Đo từng model theo mức mờ: ảnh mờ ít thì model nhẹ có đủ không, ảnh mờ nhiều thì model nặng hơn được bao nhiêu,
và mỗi cách mất bao lâu. Dùng để gán model cho thang mức nét (levels.py) và để Tự động chọn mức theo độ mờ.

Ảnh gốc nét → làm mờ Gauss 3 mức (σ 1, 2, 3) + nhiễu nhẹ + JPEG 85 → xử lý ở cùng kích thước → so với ảnh gốc:
SSIM (cao = tốt), GMSD (thấp = tốt), độ nét so với gốc (1 = bằng gốc), IDS với ảnh có mặt, thời gian trên máy này.
In thêm số đo độ mờ của bộ phân tích (phương sai Laplacian, blur_d) ở từng mức để đặt ngưỡng cho Tự động.

    python tools/level_bench.py --csv ket_qua_muc.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_anh.analysis import analyze  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from evaluate import sharp  # noqa: E402
from evaluate_general import gmsd, load  # noqa: E402

FACES = ["obama", "biden", "lena", "two_people", "bald_guys", "2009_004587"]
GENERAL = ["coffee", "chelsea", "rocket", "astronaut", "brick", "text"]
SIGMAS = {"nhẹ": 1.0, "vừa": 2.0, "nặng": 3.0}

OFF = dict(upscale=1, phone="off")  # cùng kích thước, tắt sửa màu để chỉ so độ nét
METHODS = {  # tên: (thông số, chỉ ảnh có mặt)
    "1 cổ điển nhẹ": (Settings(mode="quick", sharpness=0.4, strength=0.85, **OFF), False),
    "2 cổ điển mạnh": (Settings(mode="quick", sharpness=0.8, **OFF), False),
    "AI nền compact": (Settings(mode="upscale", sharpness=0.6, **OFF), False),
    "AI nền RRDB": (Settings(mode="upscale", sharpness=0.6, sr_model="rrdb", **OFF), False),
    "compact + GFPGAN 0,7": (Settings(mode="portrait", sharpness=0.5, fidelity=0.7, **OFF), True),
    "compact + GFPGAN 0,5": (Settings(mode="portrait", sharpness=0.7, fidelity=0.5, detail_fidelity=0.3, **OFF), True),
    "compact + GFPGAN 0,3": (Settings(mode="portrait", sharpness=0.8, fidelity=0.3, detail_fidelity=0.0, **OFF), True),
    "RRDB + GFPGAN 0,3": (Settings(mode="portrait", sharpness=0.8, fidelity=0.3, detail_fidelity=0.0, sr_model="rrdb",
                                   **OFF), True),
    "compact + RestoreFormer 0,3": (Settings(mode="portrait", sharpness=0.8, fidelity=0.3, detail_fidelity=0.0,
                                             face_model="restoreformer", **OFF), True),
}


def degrade(gt: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = cv2.GaussianBlur(gt, (0, 0), sigma).astype(np.float32) + rng.normal(0, 2.0, gt.shape)
    _, b = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def images(max_side: int):
    for n in FACES:
        im = cv2.imread(f"/tmp/claude-0/gt/{n}.jpg")
        h, w = im.shape[:2]
        k = min(1.0, max_side / max(h, w))
        yield n, True, cv2.resize(im, (round(w * k) // 2 * 2, round(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)
    for n in GENERAL:
        yield n, False, load(n, max_side)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--methods", nargs="+")
    p.add_argument("--csv")
    p.add_argument("--only", nargs="+", help="chỉ các ảnh này")
    a = p.parse_args()
    eng = Enhancer("auto")
    from net_anh.identity import Identity
    ident = Identity(eng.dev.device)
    names = a.methods or list(METHODS)
    for m in names:  # nạp sẵn model để thời gian không tính lần nạp đầu
        if METHODS[m][0].mode != "quick":
            eng.process(np.full((64, 64, 3), 128, np.uint8), METHODS[m][0])
    rows = []
    for ii, (name, has_face, gt) in enumerate(images(a.max_side)):
        if a.only and name not in a.only:
            continue
        mats, gts = [], []
        if has_face:
            helper = eng._load_helper(1)
            helper.read_image(gt)
            helper.get_face_landmarks_5(resize=640, eye_dist_threshold=20)
            helper.align_warp_face()
            mats = [m.copy() for m in helper.affine_matrices]
            gts = [c.copy() for c in helper.cropped_faces]
        for bi, (lv, sigma) in enumerate(SIGMAS.items()):
            lq = degrade(gt, sigma, ii * 10 + bi)
            an = analyze(lq)
            print(f"{name:12s} mờ {lv:5s} Laplacian {an.blur:7.1f}  blur_d {an.blur_d:.1f}", flush=True)
            for m in names:
                s, face_only = METHODS[m]
                if face_only and not has_face:
                    continue
                t0 = time.perf_counter()
                out = eng.process(lq, s)[0]
                dt = time.perf_counter() - t0
                if out.shape[:2] != gt.shape[:2]:
                    out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                ids = float(np.mean([ident.similarity(g, cv2.warpAffine(out, mt, (512, 512), borderValue=(135, 133, 132)))
                                     for mt, g in zip(mats, gts)])) if mats else float("nan")
                r = dict(image=name, face=has_face, blur=lv, method=m, lap=an.blur, blur_d=an.blur_d,
                         ssim=structural_similarity(out, gt, channel_axis=2, data_range=255), gmsd=gmsd(out, gt),
                         sharp=sharp(out) / max(sharp(gt), 1e-6), ids=ids, sec=dt)
                rows.append(r)
                print(f"   {m:28s} SSIM {r['ssim']:.3f} GMSD {r['gmsd']:.3f} nét {r['sharp']:.2f} IDS {ids:.3f} {dt:5.1f}s",
                      flush=True)
    print("\nTrung bình (ảnh có mặt | ảnh không mặt):")
    print(f"{'mờ':6s}{'cách':30s}{'SSIM':>7s}{'GMSD':>7s}{'nét':>6s}{'IDS':>7s}{'giây':>6s}")
    for face in (True, False):
        for lv in SIGMAS:
            for m in names:
                rs = [r for r in rows if r["method"] == m and r["blur"] == lv and r["face"] == face]
                if not rs:
                    continue
                mm = {k: float(np.mean([r[k] for r in rs])) for k in ("ssim", "gmsd", "sharp", "ids", "sec")}
                print(f"{lv:6s}{m:30s}{mm['ssim']:7.3f}{mm['gmsd']:7.3f}{mm['sharp']:6.2f}{mm['ids']:7.3f}{mm['sec']:6.1f}"
                      + ("   (có mặt)" if face else ""))
        print()
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
