"""Quét thông số để chọn bộ tối ưu cho mặt người, theo nhóm tuổi (dùng bộ UTKFace như age_bench.py).

Quét ba họ:
  - nhanh:   Khử nhiễu (tự động, 0, 2, 4, 6) × Độ nét (0,2; 0,4; 0,6; 0,8; 1,0)
  - pc:      Chân dung GFPGAN, fidelity 0,3 … 1,0
  - android: Phóng to AI + trộn mặt gốc, phần AI trên mặt 0 … 1
GFPGAN và nền Real-ESRGAN được nhớ đệm theo ảnh vào, nên mỗi biến thể chỉ tốn phần trộn/ghép.

Ảnh chia đôi theo thứ tự trong từng nhóm tuổi: nửa "chon" để chọn thông số, nửa "kiem" để kiểm lại.
Điểm tổng = IDS + SSIM − 0,005·|lệch tuổi| − 0,05·|ln(vân da)|
(1 tuổi lệch ≈ 0,005; vân da mất hay thêm gấp đôi ≈ 0,035).

    python tools/param_sweep.py --dir UTK --genderage genderage.onnx --csv quet.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_anh.pipeline import Enhancer  # noqa: E402
from net_anh.settings import Settings  # noqa: E402
from age_bench import GROUPS, PAD, AgeNet, android, degrade, group_of, texture  # noqa: E402

DENOISE = [-1.0, 0.0, 2.0, 4.0, 6.0]
SHARP = [0.2, 0.4, 0.6, 0.8, 1.0]
FIDELITY = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
W_AI = [0.0, 0.25, 0.5, 0.75, 1.0]


def score(r: dict) -> float:
    return r["ids"] + r["ssim"] - 0.005 * abs(r["age_shift"]) - 0.05 * abs(math.log(max(r["texture"], 1e-3)))


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()


def add_caches(eng: Enhancer) -> None:
    bg_orig = eng._background
    bg_cache: dict = {}

    def bg(img, s, rep):
        k = _key(img, s.upscale, s.sr_dn, s.deblur, s.sr_model)
        if k not in bg_cache:
            bg_cache.clear()
            bg_cache[k] = bg_orig(img, s, rep)
        return bg_cache[k].copy()

    eng._background = bg
    gf_orig = eng._load_gfpgan
    gf_cache: dict = {}

    class Cached:
        def __init__(self, net):
            self.net = net

        def __call__(self, t, return_rgb=False):
            k = _key(t.detach().cpu().numpy())
            if k not in gf_cache:
                if len(gf_cache) > 16:
                    gf_cache.clear()
                gf_cache[k] = self.net(t, return_rgb=return_rgb)
            return gf_cache[k]

    holder = {}

    def load():
        if "n" not in holder:
            holder["n"] = Cached(gf_orig())
        return holder["n"]

    eng._load_gfpgan = load


def all_variants(eng: Enhancer, up: int):
    out = []
    for d in DENOISE:
        for sh in SHARP:
            name = f"khử nhiễu {'tự động' if d < 0 else f'{d:g}'}, nét {sh:g}"
            out.append(("nhanh", name, lambda x, d=d, sh=sh: eng.process(
                x, Settings(mode="quick", upscale=up, denoise=d, sharpness=sh))[0]))
    for f in FIDELITY:
        out.append(("pc", f"fidelity {f:g}", lambda x, f=f: eng.process(
            x, Settings(mode="portrait", upscale=up, fidelity=f))[0]))
    for w in W_AI:
        out.append(("android", f"AI trên mặt {w:g}", lambda x, w=w: android(eng, x, up, w)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--genderage", required=True)
    p.add_argument("--kinds", nargs="+", default=["mo", "nho"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--csv")
    a = p.parse_args()
    eng = Enhancer("auto")
    add_caches(eng)
    from net_anh.identity import Identity
    ident = Identity(eng.dev.device)
    agenet = AgeNet(a.genderage)
    helper = eng._load_helper(1)

    files = sorted(Path(a.dir).glob("*.jpg"), key=lambda f: (int(f.name.split("_")[0]), f.name))
    seen = defaultdict(int)
    rows = []
    done: set[str] = set()
    if a.csv and Path(a.csv).exists():  # chạy tiếp từ chỗ dừng
        with open(a.csv, newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("ids", "ssim", "age_shift", "texture", "score"):
                    r[k] = float(r[k])
                r["age"] = int(r["age"])
                rows.append(r)
                done.add(r["file"])
    for ii, f in enumerate(files[: a.limit or None]):
        age = int(re.match(r"(\d+)_", f.name)[1])
        g = group_of(age)
        chip = cv2.imread(str(f))
        gt = cv2.copyMakeBorder(chip, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        helper.clean_all()
        helper.read_image(gt)
        helper.get_face_landmarks_5(resize=640, eye_dist_threshold=15)
        if not helper.all_landmarks_5:
            continue
        split = "chon" if seen[g] % 2 == 0 else "kiem"
        seen[g] += 1
        if f.name in done:
            continue
        new = []
        helper.align_warp_face()
        mat = helper.affine_matrices[0].copy()
        box = helper.det_faces[0][:4].copy()
        gt512 = helper.cropped_faces[0].copy()
        age_gt = agenet(gt, box)
        tex_gt = max(texture(chip), 1e-6)
        t0 = time.perf_counter()
        for ki, kind in enumerate(a.kinds):
            up = 4 if kind == "nho" else 1
            lq = degrade(gt, kind, ii * 10 + ki)
            for fam, name, fn in all_variants(eng, up):
                out = fn(lq)
                if out.shape[:2] != gt.shape[:2]:
                    out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                oc = out[PAD:-PAD, PAD:-PAD]
                r = dict(file=f.name, age=age, group=g, split=split, kind=kind, family=fam, params=name,
                         ids=ident.similarity(gt512, cv2.warpAffine(out, mat, (512, 512), borderValue=(135, 133, 132))),
                         ssim=structural_similarity(oc, chip, channel_axis=2, data_range=255),
                         age_shift=agenet(out, box) - age_gt, texture=texture(oc) / tex_gt)
                r["score"] = score(r)
                new.append(r)
        rows += new
        if a.csv:
            first = not Path(a.csv).exists()
            with open(a.csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(new[0]))
                if first:
                    w.writeheader()
                w.writerows(new)
        print(f"{ii + 1}/{len(files)} {f.name} {g} {split} {time.perf_counter() - t0:.0f}s", flush=True)

    report(rows, a.kinds)


def mean(rs, k):
    return float(np.mean([r[k] for r in rs]))


def report(rows, kinds):
    groups = [g for _, _, g in GROUPS]
    for kind in kinds:
        for fam in ("nhanh", "pc", "android"):
            R = [r for r in rows if r["kind"] == kind and r["family"] == fam]
            params = list(dict.fromkeys(r["params"] for r in R))
            tr = {pp: [r for r in R if r["params"] == pp and r["split"] == "chon"] for pp in params}
            best = max(params, key=lambda pp: mean(tr[pp], "score"))
            print(f"\n[{kind} / {fam}] 5 bộ tốt nhất trên nửa chọn:")
            for pp in sorted(params, key=lambda pp: -mean(tr[pp], "score"))[:5]:
                print(f"   {pp:32s} điểm {mean(tr[pp], 'score'):.4f} IDS {mean(tr[pp], 'ids'):.3f} "
                      f"SSIM {mean(tr[pp], 'ssim'):.3f} tuổi {mean(tr[pp], 'age_shift'):+.1f} vân {mean(tr[pp], 'texture'):.2f}")
            te = [r for r in R if r["params"] == best and r["split"] == "kiem"]
            print(f"   → kiểm lại '{best}' trên nửa kiểm: điểm {mean(te, 'score'):.4f} IDS {mean(te, 'ids'):.3f} "
                  f"SSIM {mean(te, 'ssim'):.3f} tuổi {mean(te, 'age_shift'):+.1f} vân {mean(te, 'texture'):.2f}")
            print("   tốt nhất theo nhóm tuổi (cả hai nửa):")
            for g in groups:
                gr = {pp: [r for r in R if r["params"] == pp and r["group"] == g] for pp in params}
                if not gr[best]:
                    continue
                bg = max(params, key=lambda pp: mean(gr[pp], "score"))
                print(f"     {g:6s} {bg:32s} điểm {mean(gr[bg], 'score'):.4f} (bộ chung {mean(gr[best], 'score'):.4f}) "
                      f"tuổi {mean(gr[bg], 'age_shift'):+.1f} vân {mean(gr[bg], 'texture'):.2f}")


if __name__ == "__main__":
    main()
