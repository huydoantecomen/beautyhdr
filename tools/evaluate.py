"""Đo chất lượng trên ảnh làm mờ giả lập.

Ảnh gốc nét (GT) → làm hỏng (blur, thu nhỏ 2×, nhiễu, nén JPEG) → làm nét bằng phần mềm → so với GT.
Chỉ số trên từng khuôn mặt (crop 512 căn theo landmark của GT):
  IDS   cos ArcFace giữa mặt GT và mặt kết quả (càng cao càng giữ đúng người)
  PSNR  sai số điểm ảnh trên crop mặt (dB)
  Sharp độ nét = phương sai Laplacian, tính theo tỉ lệ so với GT (1.0 = nét bằng ảnh gốc)
Và PSNR toàn ảnh.

Ví dụ:
    python tools/evaluate.py thu_muc_anh_net --variants lanczos quick portrait --csv ket_qua.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net_anh.cli import EXTS  # noqa: E402
from net_anh.identity import Identity  # noqa: E402
from net_anh.pipeline import Enhancer, load_image  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from net_anh import classic  # noqa: E402

LEVELS = {
    #        blur σ, nhiễu σ, JPEG q
    "nhe": (1.2, 3.0, 75),
    "nang": (2.5, 8.0, 40),
}

VARIANTS = {
    "lanczos": None,
    "quick01": "classic01",  # Quick Enhance bản 0.1: phóng → NLM h=4 → CLAHE 2.0 → unsharp 0.9
    "quick": Settings(mode="quick"),
    "upscale": Settings(mode="upscale", sr_model="rrdb"),
    "upscale-compact": Settings(mode="upscale"),
    "upscale-compact-dn1": Settings(mode="upscale", sr_model="compact", sr_denoise=1.0),
    "upscale-compact-dn0.2": Settings(mode="upscale", sr_model="compact", sr_denoise=0.2),
    "portrait-rrdb": Settings(mode="portrait", sr_model="rrdb"),
    "portrait-ai100": Settings(mode="portrait", fidelity=0.0, color_match=False, protect_faces=False),  # ~ bản 0.1
    "portrait": Settings(mode="portrait"),  # mặc định 0.2: fidelity 0.5, khớp màu, không làm nét lại mặt
    "portrait-f0.3": Settings(mode="portrait", fidelity=0.3),
    "portrait-f0.7": Settings(mode="portrait", fidelity=0.7),
}


def degrade(gt: np.ndarray, level: str, seed: int) -> np.ndarray:
    blur, noise, q = LEVELS[level]
    rng = np.random.default_rng(seed)
    x = cv2.GaussianBlur(gt, (0, 0), blur)
    x = cv2.resize(x, (gt.shape[1] // 2, gt.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    x = np.clip(x.astype(np.float32) + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def sharp(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def prepare(path: Path, max_side: int) -> np.ndarray:
    img = load_image(path)
    k = min(1.0, max_side / max(img.shape[:2]))
    h, w = int(img.shape[0] * k) // 2 * 2, int(img.shape[1] * k) // 2 * 2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gt_dir")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--min-eye", type=float, default=30, help="chỉ chấm mặt GT có khoảng cách mắt ≥ giá trị này (px)")
    p.add_argument("--device", default="auto")
    p.add_argument("--csv")
    a = p.parse_args()

    eng = Enhancer(a.device)
    ident = Identity(eng.dev.device)
    files = sorted(f for f in Path(a.gt_dir).iterdir() if f.suffix.lower() in EXTS)
    rows = []
    for f in files:
        gt = prepare(f, a.max_side)
        helper = eng._load_helper(1)
        helper.read_image(gt)
        helper.get_face_landmarks_5(resize=640, eye_dist_threshold=a.min_eye)
        if not helper.all_landmarks_5:
            print(f"{f.name}: không có mặt đủ lớn, bỏ qua")
            continue
        helper.align_warp_face()
        mats = [m.copy() for m in helper.affine_matrices]
        gt_faces = [c.copy() for c in helper.cropped_faces]
        for li, level in enumerate(a.levels):
            lq = degrade(gt, level, seed=li * 1000 + len(rows))
            for v in a.variants:
                t0 = time.perf_counter()
                if VARIANTS[v] is None:
                    out = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                elif VARIANTS[v] == "classic01":
                    out = classic.enhance_classic(lq, 2)
                else:
                    out, _ = eng.process(lq, VARIANTS[v])
                    out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA) \
                        if out.shape[:2] != gt.shape[:2] else out
                sec = time.perf_counter() - t0
                for fi, (m, g) in enumerate(zip(mats, gt_faces)):
                    o = cv2.warpAffine(out, m, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
                    rows.append(dict(image=f.name, face=fi, level=level, variant=v, ids=ident.similarity(g, o),
                                     psnr_face=psnr(g, o), sharp_ratio=sharp(o) / max(sharp(g), 1e-6),
                                     psnr_full=psnr(gt, out), seconds=sec))
                print(f"{f.name} {level:5s} {v:8s} {sec:6.1f}s  " + "  ".join(
                    f"ids={r['ids']:.3f}" for r in rows if r["image"] == f.name and r["level"] == level and r["variant"] == v))

    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print("\nTrung bình theo mức hỏng và biến thể:")
    print(f"{'mức':6s} {'biến thể':9s} {'n':>3s} {'IDS':>6s} {'PSNR mặt':>9s} {'nét/GT':>7s} {'PSNR ảnh':>9s} {'giây':>6s}")
    for level in a.levels:
        for v in a.variants:
            rs = [r for r in rows if r["level"] == level and r["variant"] == v]
            if rs:
                m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
                print(f"{level:6s} {v:9s} {len(rs):3d} {m('ids'):6.3f} {m('psnr_face'):9.2f} {m('sharp_ratio'):7.2f} "
                      f"{m('psnr_full'):9.2f} {m('seconds'):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
