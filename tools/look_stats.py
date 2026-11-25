"""Đo thống kê màu theo vùng (da, môi, tóc, nền, áo) của ảnh chân dung, để chọn thông số cho tông mẫu (net_anh/look.py).

    python tools/look_stats.py anh1.png anh2.jpg ...

Vùng lấy từ ParseNet (căn mặt 512, đưa nhãn về ảnh), nền từ backdrop.alpha_matte. In ra Lab trung bình, sắc độ C,
góc màu h, phân vị độ sáng và độ "vân" (độ lệch tầng mịn của L) cho từng vùng.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net_anh import backdrop  # noqa: E402
from net_anh.pipeline import Enhancer  # noqa: E402

REGIONS = {"da": (1, 2), "moi": (11, 12), "toc": (13,), "co": (17,), "ao": (18,)}


def region_labels(en: Enhancer, img: np.ndarray) -> np.ndarray:
    """Ảnh nhãn ParseNet cỡ ảnh (mặt lớn nhất); −1 = ngoài mặt căn."""
    helper = en._load_helper(1)
    helper.read_image(img)
    helper.get_face_landmarks_5(only_center_face=True, resize=640, eye_dist_threshold=5)
    helper.align_warp_face()
    helper.get_inverse_affine(None)
    lab = en._parse(helper, helper.cropped_faces[0]).astype(np.float32)
    inv = helper.inverse_affine_matrices[0]
    h, w = img.shape[:2]
    return cv2.warpAffine(lab, inv, (w, h), flags=cv2.INTER_NEAREST, borderValue=-1).astype(np.int16)


def stats(lab: np.ndarray, m: np.ndarray) -> dict:
    L, a, b = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    C = np.sqrt(a * a + b * b)
    return dict(n=int(m.sum()), L=float(L.mean()), L10=float(np.percentile(L, 10)), L90=float(np.percentile(L, 90)),
                a=float(a.mean()), b=float(b.mean()), C=float(C.mean()),
                h=float(np.degrees(np.arctan2(b.mean(), a.mean()))))


def measure(en: Enhancer, path: str) -> dict:
    img = cv2.imread(path)
    labels = region_labels(en, img)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    out = {}
    for name, cls in REGIONS.items():
        m = np.isin(labels, cls)
        if name == "da":
            m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if m.sum() > 50:
            out[name] = stats(lab, m)
    bd = backdrop.detect(img)
    if bd is not None:
        al = backdrop.alpha_matte(img, bd)
        out["nen"] = stats(lab, al > 0.95)
    # vân da: độ lệch tầng mịn (L − Gauss σ 1 % bề rộng mặt) trong vùng da
    skin = np.isin(labels, REGIONS["da"])
    fw = float(np.sqrt(skin.sum()))
    L = lab[..., 0]
    fine = L - cv2.GaussianBlur(L, (0, 0), max(1.0, 0.01 * fw))
    out["van_da"] = float(fine[skin].std())
    out["tuong_phan"] = float(np.percentile(L, 95) - np.percentile(L, 5))
    return out


def main(paths: list[str]) -> None:
    en = Enhancer("cpu")
    for p in paths:
        r = measure(en, p)
        print(f"== {Path(p).name}: vân da {r['van_da']:.2f}  tương phản p5–p95 {r['tuong_phan']:.1f}")
        for k in ("da", "moi", "toc", "co", "nen", "ao"):
            if k in r:
                s = r[k]
                print(f"  {k:4s} L {s['L']:5.1f} ({s['L10']:5.1f}–{s['L90']:5.1f})  a {s['a']:5.1f}  b {s['b']:5.1f}"
                      f"  C {s['C']:5.1f}  h {s['h']:5.1f}°")


if __name__ == "__main__":
    main(sys.argv[1:])
