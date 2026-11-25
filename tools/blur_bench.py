"""Bộ đo ảnh MỜ THẬT: ảnh giữ nguyên kích thước, bị mờ nét (đĩa), rung tay (vệt) hoặc Gauss, thêm nhiễu nhẹ + JPEG.

Khác evaluate.py (thu nhỏ 2× là chính), ở đây ảnh vào đã đủ điểm ảnh nhưng mất chi tiết: đúng kiểu "ảnh chụp bị mờ".
So các cách khử mờ ở cùng kích thước gốc: SSIM, GMSD (thấp = tốt), độ nét so với ảnh gốc, PSNR, thời gian.

    python tools/blur_bench.py --methods input quick sr-c-d2 sr-c-d3 --csv ket_qua_mo.csv
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
from net_anh import classic  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from evaluate import psnr, sharp  # noqa: E402
from evaluate_general import gmsd, load  # noqa: E402

FACE_DIR = Path("/tmp/claude-0/gt")
GENERAL = ["coffee", "chelsea", "rocket", "astronaut", "brick", "text", "page", "camera"]


def disk_kernel(r: float) -> np.ndarray:
    n = int(np.ceil(r)) * 2 + 1
    y, x = np.mgrid[:n, :n] - n // 2
    k = (x * x + y * y <= r * r).astype(np.float32)
    return k / k.sum()


def motion_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, m, (length, length))
    return k / k.sum()


BLURS = {
    "gauss": lambda im: cv2.GaussianBlur(im, (0, 0), 2.5),
    "defocus": lambda im: cv2.filter2D(im, -1, disk_kernel(4)),
    "motion": lambda im: cv2.filter2D(im, -1, motion_kernel(11, 30)),
}


def degrade(gt: np.ndarray, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = BLURS[kind](gt).astype(np.float32) + rng.normal(0, 2.0, gt.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    _, b = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def sr_down_up(eng: Enhancer, img: np.ndarray, d: float, model: str, dn: float = 0.5) -> np.ndarray:
    """Thu nhỏ d lần (vệt mờ co lại cỡ 1 điểm ảnh), để mạng 4× vẽ lại chi tiết, rồi đưa về kích thước gốc."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, round(w / d)), max(8, round(h / d))), interpolation=cv2.INTER_AREA)
    net = eng._load_compact(dn) if model == "compact" else None
    up = eng._esrgan_run(small, 4, 400, net)
    interp = cv2.INTER_AREA if 4 / d > 1 else cv2.INTER_CUBIC
    return cv2.resize(up, (w, h), interpolation=interp)


def rl_gauss(img: np.ndarray, sigma: float, iters: int = 15) -> np.ndarray:
    """Richardson–Lucy với nhân Gauss trên kênh L (khử chập cổ điển, cần biết độ mờ)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[..., 0].astype(np.float32) / 255.0 + 1e-3
    est = l.copy()
    for _ in range(iters):
        conv = cv2.GaussianBlur(est, (0, 0), sigma) + 1e-6
        est *= cv2.GaussianBlur(l / conv, (0, 0), sigma)
        est = np.clip(est, 0, 1.5)
    lab[..., 0] = np.clip(est * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def face_mask_for(eng: Enhancer, img: np.ndarray) -> np.ndarray | None:
    """Mặt nạ mềm 0..1 quanh các mặt (tròn bán kính 0,75 cạnh khung, làm mềm Gauss), như Enhancer.protectFaces của Android."""
    mask = np.zeros(img.shape[:2], np.float32)
    for b, _ in eng.detect_faces(img):
        x1, y1, x2, y2 = b[:4]
        cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, 0.75 * max(x2 - x1, y2 - y1)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 1.0, -1)
    if mask.max() == 0:
        return None
    return cv2.GaussianBlur(mask, (0, 0), max(3.0, 0.15 * np.sqrt(mask.sum())))


def face_protect(eng: Enhancer, img: np.ndarray, w_ai: float) -> np.ndarray:
    """Cách cho Android (chưa có GFPGAN): khử mờ AI toàn ảnh, nhưng vùng mặt trộn lại với ảnh gốc theo w_ai."""
    out = eng.process(img, Settings(mode="upscale", upscale=1))[0]
    mask = face_mask_for(eng, img)
    if mask is None:
        return out
    mask = mask[..., None] * (1 - w_ai)
    return (out * (1 - mask) + img * mask).round().clip(0, 255).astype(np.uint8)


def methods(eng: Enhancer):
    return {
        "input": lambda x: x,
        "quick": lambda x: eng.process(x, Settings(mode="quick", upscale=1))[0],
        "quick-sharp": lambda x: eng.process(x, Settings(mode="quick", upscale=1, sharpness=1.0))[0],
        "rl-s2": lambda x: rl_gauss(x, 2.0),
        "rl-s2.5": lambda x: rl_gauss(x, 2.5),
        "sr-c-d2": lambda x: sr_down_up(eng, x, 2, "compact"),
        "sr-c-d3": lambda x: sr_down_up(eng, x, 3, "compact"),
        "sr-c-d4": lambda x: sr_down_up(eng, x, 4, "compact"),
        "sr-c1-d3": lambda x: sr_down_up(eng, x, 3, "compact", 1.0),
        "sr-r-d2": lambda x: sr_down_up(eng, x, 2, "rrdb"),
        "sr-r-d3": lambda x: sr_down_up(eng, x, 3, "rrdb"),
        "sr-r-d4": lambda x: sr_down_up(eng, x, 4, "rrdb"),
        "portrait-nodeblur": lambda x: eng.process(x, Settings(mode="portrait", upscale=1, deblur=1))[0],
        "portrait": lambda x: eng.process(x, Settings(mode="portrait", upscale=1))[0],
        "upscale": lambda x: eng.process(x, Settings(mode="upscale", upscale=1))[0],
        "upscale-face0": lambda x: face_protect(eng, x, 0.0),
        "upscale-face0.5": lambda x: face_protect(eng, x, 0.5),
        "upscale-nodeblur": lambda x: eng.process(x, Settings(mode="upscale", upscale=1, deblur=1))[0],
    }


def gt_images(max_side: int):
    for f in sorted(FACE_DIR.iterdir()):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        k = min(1.0, max_side / max(h, w))
        yield "face:" + f.stem, cv2.resize(im, (round(w * k) // 2 * 2, round(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)
    for n in GENERAL:
        yield n, load(n, max_side)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+")
    p.add_argument("--blurs", nargs="+", default=list(BLURS))
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--save", help="thư mục lưu ảnh kết quả để xem")
    p.add_argument("--csv")
    p.add_argument("--faces-only", action="store_true", help="chỉ ảnh có mặt, chấm thêm IDS (ArcFace)")
    a = p.parse_args()
    eng = Enhancer("auto")
    ident = None
    if a.faces_only:
        from net_anh.identity import Identity
        ident = Identity(eng.dev.device)
    M = methods(eng)
    names = a.methods or list(M)
    rows = []
    for ii, (name, gt) in enumerate(gt_images(a.max_side)):
        mats = gts = []
        if ident is not None:
            if not name.startswith("face:"):
                continue
            helper = eng._load_helper(1)
            helper.read_image(gt)
            helper.get_face_landmarks_5(resize=640, eye_dist_threshold=30)
            if not helper.all_landmarks_5:
                continue
            helper.align_warp_face()
            mats = [m.copy() for m in helper.affine_matrices]
            gts = [c.copy() for c in helper.cropped_faces]
        for bi, kind in enumerate(a.blurs):
            lq = degrade(gt, kind, ii * 10 + bi)
            for m in names:
                t0 = time.perf_counter()
                out = M[m](lq)
                dt = time.perf_counter() - t0
                if out.shape[:2] != gt.shape[:2]:
                    out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                r = dict(image=name, blur=kind, method=m, psnr=psnr(out, gt),
                         ssim=structural_similarity(out, gt, channel_axis=2, data_range=255),
                         gmsd=gmsd(out, gt), sharp=sharp(out) / max(sharp(gt), 1e-6), sec=dt,
                         ids=float(np.mean([ident.similarity(g, cv2.warpAffine(out, m, (512, 512), borderValue=(135, 133, 132)))
                                            for m, g in zip(mats, gts)])) if mats else float("nan"))
                rows.append(r)
                print(f"{name:20s} {kind:8s} {m:18s} PSNR {r['psnr']:5.2f} SSIM {r['ssim']:.3f} "
                      f"GMSD {r['gmsd']:.3f} nét {r['sharp']:.2f} IDS {r['ids']:.3f} {dt:5.1f}s", flush=True)
                if a.save:
                    Path(a.save).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(f"{a.save}/{name.replace(':', '_')}_{kind}_{m}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("\nTrung bình:")
    print(f"{'mờ':9s}{'cách':18s}{'PSNR':>7s}{'SSIM':>7s}{'GMSD':>7s}{'nét/GT':>8s}{'IDS':>7s}{'giây':>6s}")
    for kind in a.blurs + ["ALL"]:
        for m in names:
            rs = [r for r in rows if r["method"] == m and (kind == "ALL" or r["blur"] == kind)]
            mm = {k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "gmsd", "sharp", "sec", "ids")}
            print(f"{kind:9s}{m:18s}{mm['psnr']:7.2f}{mm['ssim']:7.3f}{mm['gmsd']:7.3f}{mm['sharp']:8.2f}{mm['ids']:7.3f}{mm['sec']:6.1f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
