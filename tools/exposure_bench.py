"""Đo Tự động với ảnh quá tối, quá sáng, ngược sáng và rất mờ (giả lập từ ảnh tốt), và với ảnh tối/sáng tự nhiên.

Làm hỏng ở không gian tuyến tính (gamma 2,2) giống máy ảnh chụp sai phơi sáng:
  tối −2 EV, rất tối −3 EV (thêm nhiễu như ảnh thiếu sáng), sáng +1 EV, rất sáng +1,6 EV (cháy vùng sáng),
  ngược sáng: người (đầu, cổ, vai) −1,7 EV, nền giữ nguyên; rất mờ: Gauss σ 4 và 5.
Mỗi ảnh hỏng chạy qua phân tích + Tự động (recommend) như người dùng bấm Làm nét, rồi so với ảnh gốc:
SSIM, lệch độ sáng ΔL (trung bình kênh L, 0–255), lệch màu ΔE (CIE76), IDS mặt, phần điểm ảnh cháy trắng, thời gian.
Ảnh tối/sáng tự nhiên (vũ trụ, đáy mắt, logo nền trắng, trang chữ…) chỉ đo độ thay đổi so với chính nó: Tự động
không được làm sáng ảnh vũ trụ hay làm xám nền trắng.

    python tools/exposure_bench.py --csv ket_qua_anh_sang.csv
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
from net_anh import wb  # noqa: E402
from net_anh.analysis import recommend  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from evaluate_general import load as sk_load  # noqa: E402

FACES = ["obama", "biden", "lena", "two_people", "2009_004587"]
GENERAL = ["coffee", "chelsea", "rocket", "astronaut"]
NATURAL = ["hubble_deep_field", "retina", "moon", "logo", "page", "immunohistochemistry"]  # tối/sáng sẵn có


def to_lin(img):
    return np.power(img.astype(np.float32) / 255.0, wb.GAMMA)


def to_srgb(lin):
    return (np.power(np.clip(lin, 0, 1), 1 / wb.GAMMA) * 255 + 0.5).astype(np.uint8)


def finish(img, noise, seed):
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) + rng.normal(0, noise, img.shape)
    _, b = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def person_mask(shape, boxes):
    """Dáng người giả định: đầu (elip lớn hơn khung mặt một chút), cổ và vai bên dưới, viền mềm."""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    blur = 3.0
    for b in boxes:
        fw, fh = float(b[2] - b[0]), float(b[3] - b[1])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        cv2.ellipse(m, (round(cx), round(cy - 0.05 * fh)), (round(0.8 * fw), round(1.0 * fh)), 0, 0, 360, 1.0, -1)
        cv2.rectangle(m, (round(cx - 0.4 * fw), round(cy)), (round(cx + 0.4 * fw), round(cy + 1.3 * fh)), 1.0, -1)
        cv2.ellipse(m, (round(cx), round(cy + 2.3 * fh)), (round(2.0 * fw), round(1.3 * fh)), 0, 0, 360, 1.0, -1)
        blur = max(blur, 0.12 * fw)
    if not len(boxes):  # không có mặt: vật ở giữa ảnh
        cv2.ellipse(m, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 1.0, -1)
        blur = max(blur, 0.03 * max(h, w))
    return cv2.GaussianBlur(m, (0, 0), blur)


def degrade(gt, kind, seed, boxes):
    lin = to_lin(gt)
    if kind == "tối":
        return finish(to_srgb(lin * 0.25), 3.0, seed)
    if kind == "rất tối":
        return finish(to_srgb(lin * 0.125), 4.0, seed)
    if kind == "sáng":
        return finish(to_srgb(lin * 2.0), 1.5, seed)
    if kind == "rất sáng":
        return finish(to_srgb(lin * 3.0), 1.5, seed)
    if kind == "ngược sáng":
        m = person_mask(gt.shape[:2], boxes)[..., None]
        return finish(to_srgb(lin * (m * 2 ** -1.7 + (1 - m))), 2.5, seed)
    if kind.startswith("mờ σ"):
        s = float(kind[4:])
        return finish(cv2.GaussianBlur(gt, (0, 0), s), 2.0, seed)
    raise ValueError(kind)


KINDS = ["tối", "rất tối", "sáng", "rất sáng", "ngược sáng", "mờ σ4", "mờ σ5"]


def load_face(name, max_side):
    im = cv2.imread(f"/tmp/claude-0/gt/{name}.jpg")
    h, w = im.shape[:2]
    k = min(1.0, max_side / max(h, w))
    return cv2.resize(im, (round(w * k) // 2 * 2, round(h * k) // 2 * 2), interpolation=cv2.INTER_AREA)


def lab_stats(a, b):
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    d_l = float(la[..., 0].mean() - lb[..., 0].mean()) * 2.55  # thang 0–255 như analysis
    d_e = float(np.sqrt(((la - lb) ** 2).sum(-1)).mean())
    return d_l, d_e


def clip_frac(img):
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    return float((l >= 250).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    eng = Enhancer("auto")
    from net_anh.identity import Identity
    ident = Identity(eng.dev.device)
    eng.process(np.full((64, 64, 3), 128, np.uint8), Settings(mode="upscale", upscale=1))
    rows = []

    def run(name, kind, lq, gt, mats, gts):
        an = eng.analyze(lq, with_faces=True)
        preset, s, why = recommend(an, Settings(upscale=1))
        t0 = time.perf_counter()
        out = eng.process(lq, s)[0]
        dt = time.perf_counter() - t0
        if out.shape[:2] != gt.shape[:2]:
            out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        ids = (lambda x: float(np.mean([ident.similarity(g, cv2.warpAffine(x, mt, (512, 512),
                                                                           borderValue=(135, 133, 132)))
                                        for mt, g in zip(mats, gts)])) if mats else float("nan"))
        dl_in, de_in = lab_stats(lq, gt)
        dl, de = lab_stats(out, gt)
        r = dict(image=name, kind=kind, light=an.light_label, blur=an.blur_label, blur_d=an.blur_d,
                 L=an.brightness, p99=an.highlight, preset=preset, mode=s.mode, low_light=s.low_light,
                 brightness=s.brightness, contrast=s.contrast,
                 ssim_in=structural_similarity(lq, gt, channel_axis=2, data_range=255),
                 ssim=structural_similarity(out, gt, channel_axis=2, data_range=255),
                 dl_in=dl_in, dl=dl, de_in=de_in, de=de, ids_in=ids(lq), ids=ids(out),
                 clip_gt=clip_frac(gt), clip=clip_frac(out), sec=dt, why="; ".join(why))
        rows.append(r)
        print(f"{name:20s}{kind:11s} {an.light_label:10s} L{an.brightness:5.0f} → {preset:9s} {s.mode:8s} "
              f"Retinex {'có' if s.low_light else '–'} sáng {s.brightness:+.2f} | SSIM {r['ssim_in']:.3f}→{r['ssim']:.3f} "
              f"ΔL {dl_in:+6.1f}→{dl:+6.1f} ΔE {de_in:5.1f}→{de:5.1f} IDS {r['ids_in']:.3f}→{r['ids']:.3f} {dt:5.1f}s",
              flush=True)

    for ii, name in enumerate(FACES + GENERAL):
        if a.only and name not in a.only:
            continue
        gt = load_face(name, a.max_side) if name in FACES else sk_load(name, a.max_side)
        boxes, mats, gts = [], [], []
        if name in FACES:
            helper = eng._load_helper(1)
            helper.read_image(gt)
            helper.get_face_landmarks_5(resize=640, eye_dist_threshold=20)
            boxes = [np.asarray(b) for b in helper.det_faces]
            helper.align_warp_face()
            mats = [m.copy() for m in helper.affine_matrices]
            gts = [c.copy() for c in helper.cropped_faces]
        for ki, kind in enumerate(a.kinds):
            run(name, kind, degrade(gt, kind, ii * 100 + ki, boxes), gt, mats, gts)
    for name in NATURAL:
        if a.only and name not in a.only:
            continue
        img = sk_load(name, a.max_side)
        run(name, "tự nhiên", img, img, [], [])

    print("\nTrung bình theo kiểu hỏng (vào → ra):")
    print(f"{'kiểu':11s}{'SSIM':>14s}{'|ΔL|':>14s}{'ΔE':>12s}{'IDS mặt':>14s}{'cháy %':>13s}{'giây':>6s}")
    for kind in [*a.kinds, "tự nhiên"]:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        m = lambda k: float(np.nanmean([r[k] for r in rs]))
        ma = lambda k: float(np.mean([abs(r[k]) for r in rs]))
        print(f"{kind:11s}{m('ssim_in'):7.3f}→{m('ssim'):.3f}{ma('dl_in'):7.1f}→{ma('dl'):5.1f}"
              f"{m('de_in'):6.1f}→{m('de'):4.1f}{m('ids_in'):7.3f}→{m('ids'):.3f}"
              f"{100 * m('clip_gt'):6.1f}→{100 * m('clip'):4.1f}{float(np.median([r['sec'] for r in rs])):6.1f}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
