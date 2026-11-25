"""So model mới với model đang dùng, trên ảnh làm hỏng giả lập (có ảnh gốc để so).

Hai phần:
  --nen  model cả ảnh (phóng to, khử mờ): compact, RRDB và các model 4× nạp bằng spandrel (DAT, PLKSR…).
         Ảnh hỏng: "nhỏ /4" (thu 4 lần, JPEG 80), "phóng ×3" (thu 3 lần rồi phóng bicubic lại, JPEG 85),
         "mờ nhiễu" (mờ σ 1,2, thu 2 lần, nhiễu σ 4, JPEG 60). Chạy như app: đo độ mờ d (classic.blur_factor),
         thu nhỏ d lần, phóng 4× rồi đưa về cỡ ảnh gốc.
  --mat  model mặt trên mặt căn 512×512 cắt từ ảnh gốc: GFPGAN 1.4, RestoreFormer, RestoreFormer++.
         Ảnh hỏng kiểu GFPGAN: mờ σ, thu r lần, nhiễu, JPEG, phóng lại 512: nhẹ (1,5; 2; 3; 80), vừa (3; 4; 5; 60),
         nặng (5; 8; 8; 40). Đo đầu ra thuần của model (100 % AI).
Chỉ số: SSIM, GMSD (thấp = tốt), MUSIQ (0–100, chấm cảm nhận không cần ảnh gốc, cao = đẹp), độ nét so với gốc
(1 = bằng gốc), IDS mặt (ArcFace, cao = giữ đúng người), thời gian.

    python tools/model_bench.py --nen --model dat2=4xRealWebPhoto_v4_dat2.safetensors --musiq musiq.pth --csv nen.csv
    python tools/model_bench.py --mat --rfpp RestoreFormer++.ckpt --musiq musiq.pth --csv mat.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from net_anh import classic  # noqa: E402
from net_anh.identity import Identity  # noqa: E402
from net_anh.pipeline import Enhancer, _to_image, _to_tensor  # noqa: E402
from evaluate import sharp  # noqa: E402
from evaluate_general import gmsd, load as sk_load  # noqa: E402
from exposure_bench import FACES, GENERAL, load_face  # noqa: E402

BG_KINDS = ("nhỏ /4", "phóng ×3", "mờ nhiễu")
FACE_LEVELS = {"nhẹ": (1.5, 2, 3, 80), "vừa": (3.0, 4, 5, 60), "nặng": (5.0, 8, 8, 40)}
MORE_FACES = ["t1", "bald_guys", "2008_002470"]


def jpeg(img, q):
    _, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def degrade_bg(gt, kind, seed):
    h, w = gt.shape[:2]
    if kind == "nhỏ /4":
        return jpeg(cv2.resize(gt, (w // 4, h // 4), interpolation=cv2.INTER_AREA), 80)
    if kind == "phóng ×3":
        small = cv2.resize(gt, (round(w / 3), round(h / 3)), interpolation=cv2.INTER_AREA)
        return jpeg(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 85)
    rng = np.random.default_rng(seed)
    x = cv2.resize(cv2.GaussianBlur(gt, (0, 0), 1.2), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    x = np.clip(x.astype(np.float32) + rng.normal(0, 4, x.shape), 0, 255).astype(np.uint8)
    return jpeg(x, 60)


def degrade_face(face, level, seed):
    sigma, r, noise, q = FACE_LEVELS[level]
    rng = np.random.default_rng(seed)
    x = cv2.GaussianBlur(face, (0, 0), sigma)
    x = cv2.resize(x, (512 // r, 512 // r), interpolation=cv2.INTER_LINEAR)
    x = np.clip(x.astype(np.float32) + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)
    return cv2.resize(jpeg(x, q), (512, 512), interpolation=cv2.INTER_LINEAR)


class Musiq:
    def __init__(self, path):
        self.net = None
        if path:
            from pyiqa.archs.musiq_arch import MUSIQ
            self.net = MUSIQ(pretrained_model_path=path).eval()

    @torch.no_grad()
    def __call__(self, img):
        if self.net is None:
            return float("nan")
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float()[None] / 255
        return float(self.net(x))


def spandrel_net(path):
    import spandrel
    m = spandrel.ModelLoader().load_from_file(path)
    assert m.scale == 4, f"{path}: cần model 4×, model này {m.scale}×"
    return m.model.eval()


def run4x(eng, net, img, tile=256):
    """Phóng 4× có chia ô (như app) cho mọi mạng 4×."""
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = cv2.copyMakeBorder(img, 0, h % 2, 0, w % 2, cv2.BORDER_REFLECT)
    x = _to_tensor(img, eng.dev, False)
    if x.shape[2] <= tile and x.shape[3] <= tile:
        out = _to_image(net(x))
    else:
        out = eng._tiled(net, x, 4, tile, pad=16)
    return out[: h * 4, : w * 4]


def align(img, lm, template):
    m = cv2.estimateAffinePartial2D(np.asarray(lm, np.float32), template, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, m, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))


def ssim(a, b):
    return structural_similarity(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                                 data_range=255)


def bench_bg(a, eng, ids, musiq, template, writer):
    models = {"compact": lambda: eng._load_compact(0.5), "rrdb": lambda: eng._load_esrgan(4)}
    for spec in a.model:
        name, path = spec.split("=", 1)
        models[name] = (lambda p=path: spandrel_net(p))
    names = a.only or list(models)
    images = [(n, load_face(n, a.max_side)) for n in FACES] + [(n, sk_load(n, a.max_side)) for n in GENERAL]
    for mi, name in enumerate(names):
        net = models[name]()
        for ii, (iname, gt) in enumerate(images):  # ii theo danh sách đầy đủ: cùng ảnh hỏng (seed) dù lọc --anh
            if a.anh and iname not in a.anh:
                continue
            h, w = gt.shape[:2]
            faces = [lm for _, lm in eng.detect_faces(gt)] if iname in FACES else []
            faces = [lm for lm in faces if np.linalg.norm(lm[0] - lm[1]) >= 25]
            gt_faces = [align(gt, lm, template) for lm in faces]
            for ki, kind in enumerate(BG_KINDS):
                inp = degrade_bg(gt, kind, 1000 * ii + ki)
                d = classic.blur_factor(inp)
                x = inp if d <= 1 else cv2.resize(inp, (round(inp.shape[1] / d), round(inp.shape[0] / d)),
                                                  interpolation=cv2.INTER_AREA)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y = run4x(eng, net, x)
                dt = time.perf_counter() - t0
                y = cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA if y.shape[1] > w else cv2.INTER_CUBIC)
                row = dict(phan="nen", model=name, anh=iname, kieu=kind, d=d, ssim=ssim(gt, y), gmsd=gmsd(gt, y),
                           musiq=musiq(y), net=sharp(y) / max(sharp(gt), 1e-6), giay=dt,
                           ids=float(np.mean([ids.similarity(g, align(y, lm, template))
                                              for g, lm in zip(gt_faces, faces)])) if faces else float("nan"))
                writer(row)
                if a.save:
                    out = Path(a.save) / f"nen_{iname}_{ki}_{name}.jpg"
                    cv2.imencode(".jpg", y, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(out))
                    if mi == 0:
                        cv2.imencode(".jpg", inp, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                            str(Path(a.save) / f"nen_{iname}_{ki}_vao.jpg"))
        del net


def bench_face(a, eng, ids, musiq, template, writer):
    from net_anh.archs.restoreformer_arch import RestoreFormer
    models = {"gfpgan": lambda: eng._load_face_net("gfpgan"), "restoreformer": lambda: eng._load_face_net("restoreformer")}
    if a.rfpp:
        def rfpp():
            ck = torch.load(a.rfpp, map_location="cpu", weights_only=True)
            sd = {k.removeprefix("vqvae."): v for k, v in ck.get("state_dict", ck).items()}
            net = RestoreFormer(head_size=4, ex_multi_scale_num=1)
            net.load_state_dict(sd, strict=True)
            return net.eval()
        models["restoreformer++"] = rfpp
    crops = []
    for n in FACES + MORE_FACES:
        gt = load_face(n, 4096)
        for k, (_, lm) in enumerate(eng.detect_faces(gt)):
            if np.linalg.norm(lm[0] - lm[1]) >= a.min_eye:  # mặt gốc đủ lớn để làm mốc
                crops.append((f"{n}#{k}", align(gt, lm, template)))
    print(f"{len(crops)} mặt làm mốc", flush=True)
    for name in a.only or list(models):
        net = models[name]()
        for ci, (cname, gt) in enumerate(crops):
            for li, level in enumerate(FACE_LEVELS):
                lq = degrade_face(gt, level, 100 * ci + li)
                t = (_to_tensor(lq, eng.dev, False) - 0.5) / 0.5
                t0 = time.perf_counter()
                with torch.no_grad():
                    y = _to_image(net(t, return_rgb=False)[0], -1.0, 1.0)
                dt = time.perf_counter() - t0
                row = dict(phan="mat", model=name, anh=cname, kieu=level, d=0, ssim=ssim(gt, y), gmsd=gmsd(gt, y),
                           musiq=musiq(y), net=sharp(y) / max(sharp(gt), 1e-6), giay=dt, ids=ids.similarity(gt, y))
                writer(row)
                if a.save:
                    stem = cname.replace("#", "_")
                    cv2.imencode(".jpg", y, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                        str(Path(a.save) / f"mat_{stem}_{li}_{name}.jpg"))
                    if name == (a.only or list(models))[0]:
                        cv2.imencode(".jpg", lq, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                            str(Path(a.save) / f"mat_{stem}_{li}_vao.jpg"))
                        cv2.imencode(".jpg", gt, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                            str(Path(a.save) / f"mat_{stem}_goc.jpg"))
        del net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nen", action="store_true")
    ap.add_argument("--mat", action="store_true")
    ap.add_argument("--model", action="append", default=[], help="tên=đường_dẫn model 4× (spandrel)")
    ap.add_argument("--rfpp", help="RestoreFormer++.ckpt")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--anh", nargs="+", help="chỉ đo các ảnh này (phần --nen)")
    ap.add_argument("--append", action="store_true", help="ghi tiếp vào file csv đã có")
    ap.add_argument("--musiq")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--min-eye", type=float, default=60)
    ap.add_argument("--save")
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    eng = Enhancer("cpu")
    ids = Identity(eng.dev.device)
    musiq = Musiq(a.musiq)
    template = eng._load_helper(1).face_template.astype(np.float32)
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
    keys = ["phan", "model", "anh", "kieu", "d", "ssim", "gmsd", "musiq", "net", "ids", "giay"]
    with open(a.csv, "a" if a.append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        if not a.append:
            w.writeheader()

        def writer(row):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
            f.flush()
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

        if a.nen:
            bench_bg(a, eng, ids, musiq, template, writer)
        if a.mat:
            bench_face(a, eng, ids, musiq, template, writer)


if __name__ == "__main__":
    main()
