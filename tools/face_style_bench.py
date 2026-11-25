"""So các kiểu phục hồi mặt: giữ đúng người (fidelity cao) ↔ rõ nét tối đa kiểu Remini (AI 100 %, RestoreFormer).

Ảnh mặt thật (FACE_DIR của blur_bench) làm hỏng hai kiểu: mo (Gauss σ2,5, giữ kích thước, phóng 2×) và
nho (thu 4×, phóng 4×). Chấm IDS (ArcFace, giữ đúng người) và độ nét vùng mặt so với ảnh gốc
(phương sai Laplace; >1 = nét hơn ảnh gốc, kiểu Remini thường >1). Lưu lưới ảnh mặt để xem bằng mắt.

    python tools/face_style_bench.py --save /tmp/kieu_mat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from blur_bench import FACE_DIR  # noqa: E402

VARIANTS = {
    "Hiện tại (GFPGAN, AI 50 %)": dict(fidelity=0.5),
    "Giữ người (AI 30 %)": dict(fidelity=0.7),
    "GFPGAN AI 100 %": dict(fidelity=0.0),
    "RestoreFormer AI 100 %": dict(fidelity=0.0, face_model="restoreformer"),
    "RestoreFormer AI 70 %": dict(fidelity=0.3, face_model="restoreformer"),
    "Theo vùng: da 70 %, ngũ quan 100 %": dict(fidelity=0.3, detail_fidelity=0.0),
    "Theo vùng: da 50 %, ngũ quan 100 %": dict(fidelity=0.5, detail_fidelity=0.0),
}


def degrade(gt, kind, seed):
    rng = np.random.default_rng(seed)
    h, w = gt.shape[:2]
    if kind == "mo":
        x = cv2.GaussianBlur(gt, (0, 0), 2.5).astype(np.float32) + rng.normal(0, 2.5, gt.shape)
    else:
        x = cv2.resize(cv2.GaussianBlur(gt, (0, 0), 1.0), (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        x += rng.normal(0, 1.5, x.shape)
    _, b = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def lapvar(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--save")
    p.add_argument("--only", nargs="+", help="chỉ chạy các cách có tên chứa chuỗi này")
    a = p.parse_args()
    eng = Enhancer("auto")
    from net_anh.identity import Identity
    ident = Identity(eng.dev.device)
    helper = eng._load_helper(1)
    res = {}
    for ii, f in enumerate(sorted(FACE_DIR.iterdir())):
        im = cv2.imread(str(f))
        k = min(1.0, a.max_side / max(im.shape[:2]))
        gt = cv2.resize(im, (round(im.shape[1] * k) // 4 * 4, round(im.shape[0] * k) // 4 * 4), interpolation=cv2.INTER_AREA)
        helper.clean_all(); helper.read_image(gt)
        helper.get_face_landmarks_5(resize=640, eye_dist_threshold=20)
        if not helper.all_landmarks_5:
            continue
        helper.align_warp_face()
        mats = [m.copy() for m in helper.affine_matrices]
        gts = [c.copy() for c in helper.cropped_faces]
        for ki, (kind, up) in enumerate([("mo", 2), ("nho", 4)]):
            lq = degrade(gt, kind, ii * 10 + ki)
            outs = {"Ảnh hỏng": cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)}
            for name, kw in VARIANTS.items():
                if a.only and not any(o in name for o in a.only):
                    continue
                out = eng.process(lq, Settings(mode="portrait", upscale=up, **kw))[0]
                outs[name] = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
            tiles = [cv2.resize(gts[0], (256, 256), interpolation=cv2.INTER_AREA)]
            for name, out in outs.items():
                faces = [cv2.warpAffine(out, m, (512, 512), borderValue=(135, 133, 132)) for m in mats]
                ids = float(np.mean([ident.similarity(g, o) for g, o in zip(gts, faces)]))
                sh = float(np.mean([lapvar(o[96:416, 96:416]) / max(lapvar(g[96:416, 96:416]), 1e-6) for g, o in zip(gts, faces)]))
                res.setdefault((kind, name), []).append((ids, sh))
                tiles.append(cv2.resize(faces[0], (256, 256), interpolation=cv2.INTER_AREA))
            if a.save:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{a.save}/{f.stem}_{kind}.png", np.hstack(tiles))
            print(f.stem, kind, "xong", flush=True)
    print(f"\n{'kiểu':5s}{'cách':30s}{'IDS':>7s}{'nét/gốc':>9s}")
    for (kind, name), v in res.items():
        v = np.array(v)
        print(f"{kind:5s}{name:30s}{v[:, 0].mean():7.3f}{v[:, 1].mean():9.2f}")


if __name__ == "__main__":
    main()
