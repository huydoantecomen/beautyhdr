"""Đo chất lượng retouch mặt theo NHÓM TUỔI (trẻ em → người già) trên ảnh UTKFace có ghi tuổi.

UTKFace (chỉ dùng nghiên cứu phi thương mại) là mặt đã cắt 200×200, tên file `tuổi_giới_chủng_thời-gian.jpg.chip.jpg`.
Mỗi ảnh được viền thêm (replicate) để bộ dò mặt chạy ổn, làm hỏng theo hai kiểu rồi phục hồi:
  - mo:  cùng kích thước, mờ Gauss σ2,2 + nhiễu + JPEG 75 (ảnh chụp mờ)
  - nho: thu nhỏ 4× + mờ nhẹ + JPEG 75, phục hồi phóng 4× (ảnh cũ, mặt nhỏ trong ảnh đông người)
Chấm trên vùng mặt gốc 200×200:
  - IDS: cos ArcFace giữa mặt gốc và mặt sau xử lý (cao = giữ đúng người)
  - SSIM
  - lệch tuổi: tuổi ước lượng (InsightFace genderage) của ảnh ra − của ảnh gốc (âm = trông trẻ đi)
  - vân da: năng lượng Laplace vùng giữa mặt so với ảnh gốc (<1 = mất nếp nhăn/vân da, >1 = thêm chi tiết giả)

    python tools/age_bench.py --dir /path/UTKFace_chọn --genderage genderage.onnx --csv ket_qua_tuoi.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from blur_bench import face_mask_for  # noqa: E402

GROUPS = [(0, 9, "0–9"), (10, 19, "10–19"), (20, 39, "20–39"), (40, 59, "40–59"), (60, 79, "60–79"), (80, 200, "80+")]
PAD = 56


def group_of(age: int) -> str:
    return next(g for lo, hi, g in GROUPS if lo <= age <= hi)


class AgeNet:
    """InsightFace genderage.onnx (buffalo_l): ảnh 96×96 cắt quanh khung mặt ×1,5, ra [nữ, nam, tuổi/100]."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, img: np.ndarray, box) -> float:
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        k = 96 / (max(x2 - x1, y2 - y1) * 1.5)
        m = np.array([[k, 0, 48 - k * cx], [0, k, 48 - k * cy]], np.float32)
        a = cv2.warpAffine(img, m, (96, 96), borderValue=0)
        blob = cv2.dnn.blobFromImage(a, 1.0, (96, 96), (0, 0, 0), swapRB=True)
        return float(self.s.run(None, {self.inp: blob})[0][0][2] * 100)


def degrade(gt: np.ndarray, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = gt.shape[:2]
    if kind == "mo":
        x = cv2.GaussianBlur(gt, (0, 0), 2.2).astype(np.float32) + rng.normal(0, 3.0, gt.shape)
        q = 75
    else:
        x = cv2.resize(cv2.GaussianBlur(gt, (0, 0), 1.0), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) + rng.normal(0, 1.5, x.shape)
        q = 75
    x = np.clip(x, 0, 255).astype(np.uint8)
    _, b = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def android(eng: Enhancer, lq: np.ndarray, up: int, w_ai: float = 0.5) -> np.ndarray:
    """Đường Android 0.2: Phóng to AI (khử mờ) toàn ảnh, vùng mặt trộn lại 50 % ảnh vào (phóng bicubic)."""
    out = eng.process(lq, Settings(mode="upscale", upscale=up))[0]
    base = cv2.resize(lq, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC) if up > 1 else lq
    mask = face_mask_for(eng, base)
    if mask is None:
        return out
    mask = mask[..., None] * (1 - w_ai)
    return (out * (1 - mask) + base * mask).round().clip(0, 255).astype(np.uint8)


def methods(eng: Enhancer, up: int):
    return {
        "dau-vao": lambda x: cv2.resize(x, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else x,
        "nhanh": lambda x: eng.process(x, Settings(mode="quick", upscale=up))[0],
        "pc-chan-dung": lambda x: eng.process(x, Settings(mode="portrait", upscale=up))[0],
        "pc-chan-dung-f0.7": lambda x: eng.process(x, Settings(mode="portrait", upscale=up, fidelity=0.7))[0],
        "android": lambda x: android(eng, x, up),
    }


def texture(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = g[50:170, 45:155]  # má, trán, quanh mắt của mặt cắt UTKFace
    return float(np.mean(np.abs(cv2.Laplacian(c, cv2.CV_32F))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--genderage", required=True)
    p.add_argument("--kinds", nargs="+", default=["mo", "nho"])
    p.add_argument("--methods", nargs="+")
    p.add_argument("--save")
    p.add_argument("--csv")
    a = p.parse_args()
    eng = Enhancer("auto")
    from net_anh.identity import Identity
    ident = Identity(eng.dev.device)
    agenet = AgeNet(a.genderage)
    helper = eng._load_helper(1)
    rows = []
    files = sorted(Path(a.dir).glob("*.jpg"))
    for ii, f in enumerate(files):
        age = int(re.match(r"(\d+)_", f.name)[1])
        chip = cv2.imread(str(f))
        gt = cv2.copyMakeBorder(chip, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        helper.clean_all()
        helper.read_image(gt)
        helper.get_face_landmarks_5(resize=640, eye_dist_threshold=15)
        if not helper.all_landmarks_5:
            print(f"bỏ {f.name}: không dò được mặt", flush=True)
            continue
        helper.align_warp_face()
        mat = helper.affine_matrices[0].copy()
        box = helper.det_faces[0][:4].copy()
        gt512 = helper.cropped_faces[0].copy()
        age_gt = agenet(gt, box)
        tex_gt = max(texture(chip), 1e-6)
        for ki, kind in enumerate(a.kinds):
            up = 4 if kind == "nho" else 1
            lq = degrade(gt, kind, ii * 10 + ki)
            M = methods(eng, up)
            for m in a.methods or list(M):
                t0 = time.perf_counter()
                out = M[m](lq)
                dt = time.perf_counter() - t0
                if out.shape[:2] != gt.shape[:2]:
                    out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                oc = out[PAD:-PAD, PAD:-PAD]
                r = dict(file=f.name, age=age, group=group_of(age), kind=kind, method=m,
                         ids=ident.similarity(gt512, cv2.warpAffine(out, mat, (512, 512), borderValue=(135, 133, 132))),
                         ssim=structural_similarity(oc, chip, channel_axis=2, data_range=255),
                         age_gt=age_gt, age_shift=agenet(out, box) - age_gt,
                         texture=texture(oc) / tex_gt, sec=dt)
                rows.append(r)
                print(f"{f.name[:28]:28s} {r['group']:6s} {kind:4s} {m:18s} IDS {r['ids']:.3f} SSIM {r['ssim']:.3f} "
                      f"tuổi {r['age_shift']:+5.1f} vân {r['texture']:.2f} {dt:4.1f}s", flush=True)
                if a.save:
                    Path(a.save).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(f"{a.save}/{f.stem}_{kind}_{m}.png", oc)
            if a.save:
                lqc = lq if up == 1 else cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{a.save}/{f.stem}_{kind}_lq.png", lqc[PAD:-PAD, PAD:-PAD])
        if a.save:
            cv2.imwrite(f"{a.save}/{f.stem}_gt.png", chip)
    print("\nTrung bình theo nhóm tuổi:")
    ms = list(dict.fromkeys(r["method"] for r in rows))
    for kind in a.kinds:
        print(f"\n[{kind}] {'tuổi':6s}{'cách':20s}{'IDS':>7s}{'SSIM':>7s}{'lệch tuổi':>10s}{'vân da':>8s}{'n':>4s}")
        for _, _, g in GROUPS + [(0, 0, "tất cả")]:
            for m in ms:
                rs = [r for r in rows if r["kind"] == kind and r["method"] == m and (g == "tất cả" or r["group"] == g)]
                if not rs:
                    continue
                mm = {k: float(np.mean([r[k] for r in rs])) for k in ("ids", "ssim", "age_shift", "texture")}
                print(f"      {g:6s}{m:20s}{mm['ids']:7.3f}{mm['ssim']:7.3f}{mm['age_shift']:+10.1f}{mm['texture']:8.2f}{len(rs):4d}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
