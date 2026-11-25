"""Chạy thử 30 ca qua dòng lệnh thật (python -m net_anh.cli, mỗi ca một tiến trình riêng) và kiểm tra lỗi.

Mỗi ca ghi: mã thoát, thời gian, bộ nhớ đỉnh (RSS), kích thước ảnh ra, độ lệch màu so với ảnh vào
(trung bình L/a/b), tỉ lệ điểm cháy đen/trắng, và có tìm thấy mặt hay không. Ca "ảnh hỏng" phải báo lỗi gọn
(mã thoát 1, không traceback). Lưu lưới ảnh trước/sau để xem bằng mắt.

    python tools/stress_test.py --work /tmp/stress
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FACES = Path("/tmp/claude-0/gt")
UTK = Path("/tmp/claude-0/agedata/utk")


def make_inputs(d: Path) -> dict[str, Path]:
    import skimage.data as sd
    d.mkdir(parents=True, exist_ok=True)
    p: dict[str, Path] = {}
    for f in sorted(FACES.iterdir()):
        p[f.stem] = f
    for n in ("coffee", "chelsea", "astronaut", "rocket", "camera", "page", "brick", "text"):
        im = getattr(sd, n)()
        if im.ndim == 2:
            im = np.dstack([im] * 3)
        q = d / f"{n}.png"
        Image.fromarray(im[..., :3]).save(q)
        p[n] = q
    coffee = cv2.imread(str(p["coffee"]))
    cv2.imwrite(str(d / "tiny.png"), cv2.resize(coffee, (40, 30), interpolation=cv2.INTER_AREA)); p["tiny"] = d / "tiny.png"
    big = cv2.resize(cv2.imread(str(p["two_people"])), None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(d / "huge.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 90]); p["huge"] = d / "huge.jpg"
    rgba = Image.open(p["chelsea"]).convert("RGBA"); a = np.array(rgba); a[..., 3] = 128
    Image.fromarray(a).save(d / "rgba.png"); p["rgba"] = d / "rgba.png"
    dark = (cv2.imread(str(p["obama"])).astype(np.float32) * 0.18).astype(np.uint8)
    cv2.imwrite(str(d / "dark.jpg"), dark); p["dark"] = d / "dark.jpg"
    im = Image.open(p["biden"]).rotate(90, expand=True)  # lưu nằm ngang + EXIF bảo xoay lại
    exif = im.getexif(); exif[0x0112] = 6
    im.save(d / "exif_rot.jpg", exif=exif); p["exif_rot"] = d / "exif_rot.jpg"
    Image.open(p["lena"]).save(d / "lena.webp", quality=80); p["webp"] = d / "lena.webp"
    (d / "hong.jpg").write_bytes(b"\xff\xd8\xff\xe0" + os.urandom(3000)); p["hong"] = d / "hong.jpg"
    utk = sorted(UTK.glob("8*_*_2_*.jpg"))[0] if UTK.exists() else p["lena"]
    p["utk_old"] = utk
    return p


CASES = [  # (tên ảnh, tham số dòng lệnh, ghi chú)
    *[(n, [], "Tự động") for n in ("obama", "biden", "lena", "t1", "two_people", "bald_guys", "2008_002470", "2009_004587")],
    *[(n, ["--preset", "max"], "Rõ nét tối đa") for n in ("obama", "t1", "two_people", "utk_old")],
    ("lena", ["--mode", "portrait", "--upscale", "4"], "Chân dung 4×"),
    ("bald_guys", ["--mode", "portrait", "--face-model", "restoreformer"], "Chân dung RestoreFormer"),
    ("biden", ["--mode", "quick", "--color", "1", "--gloss", "1"], "Nhanh + màu/bóng tối đa"),
    *[(n, [], "Tự động") for n in ("coffee", "chelsea", "astronaut", "rocket", "brick")],
    ("rocket", ["--mode", "upscale", "--upscale", "4"], "Phóng to AI 4×"),
    ("page", ["--mode", "old"], "Ảnh cũ (chữ)"),
    ("tiny", [], "Ảnh 40×30"),
    ("huge", [], "Ảnh lớn 1.9k px, nhiều mặt"),
    ("camera", ["--preset", "max"], "Ảnh xám + Rõ nét tối đa"),
    ("rgba", [], "PNG trong suốt"),
    ("dark", [], "Ảnh rất tối có mặt"),
    ("exif_rot", ["--preset", "max"], "Ảnh xoay EXIF"),
    ("webp", [], "WebP"),
    ("hong", [], "File hỏng: phải báo lỗi gọn"),
]


def lab_mean(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/claude-0/stress")
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    work = Path(a.work)
    inputs = make_inputs(work / "in")
    outd = work / "out"; outd.mkdir(parents=True, exist_ok=True)
    rows = []
    sys.path.insert(0, str(ROOT))
    from net_anh.pipeline import load_image
    for i, (name, args, note) in enumerate(CASES, 1):
        if a.only and i not in a.only:
            continue
        src = inputs[name]
        out = outd / f"{i:02d}_{name}.jpg"
        out.unlink(missing_ok=True)
        argv = ["net_anh", str(src), "-o", str(out), "--device", "cpu", *args]
        code = ("import resource, runpy, sys\nsys.argv = %r\ntry:\n    runpy.run_module('net_anh.cli', run_name='__main__')\n"
                "finally:\n    print('RSS_KB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, file=sys.stderr)\n") % argv
        cmd = [sys.executable, "-c", code]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        dt = time.perf_counter() - t0
        rss = next((int(ln.split()[1]) for ln in r.stderr.splitlines() if ln.startswith("RSS_KB")), 0) / 1024
        row = dict(i=i, ca=note, anh=name, code=r.returncode, giay=round(dt, 1), rss_mb=round(rss), loi="")
        tb = "Traceback" in r.stderr
        if name == "hong":
            row["dat"] = r.returncode == 1 and not tb and not out.exists()
            row["loi"] = next((ln for ln in r.stderr.splitlines() if "hong.jpg" in ln), "")[:160]
        elif r.returncode != 0 or not out.exists():
            row["dat"] = False
            row["loi"] = (r.stderr.strip().splitlines() or ["?"])[-2:][0][:200]
        else:
            inp = load_image(src)
            res = cv2.imread(str(out))
            ok = res is not None and res.ndim == 3 and np.isfinite(res).all()
            ref = cv2.resize(inp, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_AREA)
            dl = lab_mean(res) - lab_mean(ref)
            clip = float(((res.max(2) <= 2) | (res.min(2) >= 253)).mean() - ((ref.max(2) <= 2) | (ref.min(2) >= 253)).mean())
            ratio = res.shape[1] / inp.shape[1]
            row.update(vao=f"{inp.shape[1]}×{inp.shape[0]}", ra=f"{res.shape[1]}×{res.shape[0]}", he_so=round(ratio, 2),
                       lech_L=round(float(dl[0]), 1), lech_a=round(float(dl[1]), 1), lech_b=round(float(dl[2]), 1),
                       chay_them=round(clip * 100, 2),
                       mat=next((ln.strip() for ln in r.stderr.splitlines() if "Tự động:" in ln), "")[:160],
                       tom_tat=r.stdout.strip()[-200:])
            ok = ok and abs(dl[0]) < 20 and abs(dl[1]) < 8 and abs(dl[2]) < 10 and clip < 0.03 and not tb
            row["dat"] = bool(ok)
            if not ok:
                row["loi"] = "lệch màu/cháy sáng vượt ngưỡng" if not tb else "có traceback"
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (work / "ket_qua.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(r["dat"] for r in rows)
    print(f"\nĐẠT {n_ok}/{len(rows)}")


if __name__ == "__main__":
    main()
