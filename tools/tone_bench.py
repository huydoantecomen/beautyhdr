"""Đo tông ảnh sau Tự động, có và không có bước tông tươi sáng (net_anh/tone.py).

Ảnh: 5 ảnh có mặt + 4 ảnh không mặt (như tools/exposure_bench.py), mỗi ảnh ở dạng gốc và ba kiểu hỏng (tối −2 EV,
rất sáng +1,6 EV, ngược sáng), cộng 6 ảnh vốn tối hoặc sáng (vũ trụ, đáy mắt, mặt trăng, logo, trang chữ, hiển vi).
Mỗi ảnh chạy Tự động với tông 0 (giữ tông ảnh gốc), rồi áp tone.bright_clear ở các mức (bước cuối nên tương đương
chạy lại cả quy trình). Số đo:
  L      độ sáng trung bình (L của Lab, 0–100)
  L mặt  trung vị L giữa các mặt
  M      độ tươi màu (colorfulness, Hasler & Süsstrunk 2003)
  đen    phân vị 2 của L (cao là đen bị xám, "đục")
  cháy   % điểm ảnh có kênh ≥ 254
  b* da  độ vàng của da trong vùng mặt

    python tools/tone_bench.py --grid luoi_tong.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from net_anh import color, exposure, tone, wb  # noqa: E402
from net_anh.analysis import recommend  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from evaluate_general import load as sk_load  # noqa: E402
from exposure_bench import FACES, GENERAL, NATURAL, degrade, load_face  # noqa: E402

KINDS = ["gốc", "tối", "rất sáng", "ngược sáng"]
SEED = {"tối": 0, "rất sáng": 3, "ngược sáng": 4}


def colorfulness(img: np.ndarray) -> float:
    b, g, r = (c.astype(np.float32) for c in cv2.split(img))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def stats(img: np.ndarray, boxes) -> dict:
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    face = np.zeros(L.shape, bool)
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b[:4])
        fw, fh = x1 - x0, y1 - y0
        face[max(y0 + fh // 4, 0):max(y1 - fh // 4, 0), max(x0 + fw // 4, 0):max(x1 - fw // 4, 0)] = True
    skin = (color.skin_mask(img) * wb.face_region_mask(img.shape[:2], boxes)) > 0.5 if boxes else np.zeros_like(face)
    return dict(L=float(L.mean()), faceL=float(np.median(L[face])) if face.any() else float("nan"),
                M=colorfulness(img), black=float(np.percentile(L, 2)),
                clip=100.0 * float((img.max(axis=2) >= 254).mean()),
                skin_b=float(np.median(lab[..., 2][skin])) if skin.sum() > 100 else float("nan"))


def label(img: np.ndarray, text: str) -> np.ndarray:
    from phone_cast_demo import label as lb
    return lb(img, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", type=float, default=[0.6, 1.0])
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--grid", help="ghi lưới trước/sau (ảnh có mặt) ra file này")
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    eng = Enhancer("auto")
    rows, grid = [], []
    jobs = []
    for ii, name in enumerate(FACES + GENERAL):
        for kind in KINDS:
            jobs.append((ii, name, kind))
    jobs += [(None, name, "tự nhiên") for name in NATURAL]
    for ii, name, kind in jobs:
        if a.only and name not in a.only:
            continue
        gt = load_face(name, a.max_side) if name in FACES else sk_load(name, a.max_side)
        if kind in ("gốc", "tự nhiên"):
            lq = gt
        else:
            boxes0 = [b for b, _ in eng.detect_faces(gt)] if name in FACES else []
            if kind == "ngược sáng" and not boxes0:
                continue
            lq = degrade(gt, kind, ii * 100 + SEED[kind], boxes0)
        an = eng.analyze(lq, with_faces=True)
        _, s, _ = recommend(an, Settings(upscale=1))
        out0 = eng.process(lq, s.with_(tone=0.0))[0]
        if out0.shape[:2] != gt.shape[:2]:
            out0 = cv2.resize(out0, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        k = out0.shape[1] / lq.shape[1]
        boxes = [np.asarray(b[:4], np.float32) * k for b in (eng._faces or [])]
        outs = {0.0: out0}
        graphic = exposure.is_graphic(exposure._small(lq)[0])  # như pipeline: xét trên ảnh vào
        for t in a.levels:
            outs[t] = tone.bright_clear(out0, t, boxes, graphic)[0]
        r = {"image": name, "kind": kind, "gt": stats(gt, boxes), **{t: stats(o, boxes) for t, o in outs.items()}}
        rows.append(r)
        line = " | ".join(f"tông {t:g}: L {r[t]['L']:4.1f} mặt {r[t]['faceL']:4.1f} M {r[t]['M']:4.1f} "
                          f"đen {r[t]['black']:4.1f} cháy {r[t]['clip']:4.1f}" for t in outs)
        print(f"{name:20s} {kind:10s} gốc L {r['gt']['L']:4.1f} M {r['gt']['M']:4.1f} || {line}", flush=True)
        if a.grid and name in FACES and kind in ("gốc", "tối", "rất sáng"):
            w = 300
            fit = lambda x: cv2.resize(x, (w, round(x.shape[0] * w / x.shape[1])), interpolation=cv2.INTER_AREA)
            grid.append(np.hstack([label(fit(lq), "Ảnh vào" if kind == "gốc" else f"Ảnh {kind}")]
                                  + [label(fit(o), "Bản 0.6" if t == 0 else f"Tươi sáng {t:g}")
                                     for t, o in outs.items()]))

    print("\nTrung bình (ảnh chụp: gốc + 3 kiểu hỏng; tự nhiên riêng):")
    for group, sel in (("ảnh chụp", lambda r: r["kind"] != "tự nhiên"), ("tự nhiên", lambda r: r["kind"] == "tự nhiên")):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        for t in [0.0, *a.levels]:
            m = lambda key: float(np.nanmean([r[t][key] for r in rs]))
            print(f"{group:9s} tông {t:<4g} L {m('L'):5.1f}  L mặt {m('faceL'):5.1f}  M {m('M'):5.1f}  đen {m('black'):4.1f}"
                  f"  cháy {m('clip'):4.1f} %  b* da {m('skin_b'):5.1f}")
        mg = lambda key: float(np.nanmean([r["gt"][key] for r in rs]))
        print(f"{group:9s} ảnh gốc   L {mg('L'):5.1f}  L mặt {mg('faceL'):5.1f}  M {mg('M'):5.1f}  đen {mg('black'):4.1f}"
              f"  cháy {mg('clip'):4.1f} %  b* da {mg('skin_b'):5.1f}")
    if a.grid and grid:
        wmax = max(g.shape[1] for g in grid)
        grid = [np.pad(g, ((0, 0), (0, wmax - g.shape[1]), (0, 0)), constant_values=255) for g in grid]
        cv2.imwrite(a.grid, np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
