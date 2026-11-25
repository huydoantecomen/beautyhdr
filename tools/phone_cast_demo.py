"""Lưới ảnh trước/sau cho ám màu kiểu điện thoại (giả lập), để xem bằng mắt.

Mỗi hàng: ảnh gốc | ảnh ám màu giả lập | sau sửa bằng bộ "Khác" | sau sửa bằng bộ của hãng. Các kiểu ám:
  đèn vàng   đèn sợi đốt / đèn vàng trong nhà (nhân kênh tuyến tính B 0,70, R 1,25)
  da đỏ      chế độ làm đẹp kiểu Oppo/Vivo: da kéo về đỏ hồng (xoay góc màu da −15°, chroma ×1,3) + ám ấm nhẹ
  ám cam     ảnh tối trong nhà, máy ưu tiên giữ "không khí ấm" kiểu iPhone (B 0,80, G 1,02, R 1,15)

    python tools/phone_cast_demo.py --out /tmp/phone_cast
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from net_anh import color, wb  # noqa: E402
from net_anh.pipeline import Enhancer, Report  # noqa: E402
from net_anh.settings import Settings  # noqa: E402


def linear_cast(img, g):
    lin = np.power(img.astype(np.float32) / 255.0, wb.GAMMA) * np.array(g, np.float32)
    return (np.clip(np.power(np.clip(lin, 0, 1), 1 / wb.GAMMA), 0, 1) * 255 + 0.5).astype(np.uint8)


def red_skin(img, boxes, deg=-15.0, chroma=1.3):
    """Giả lập da bị kéo đỏ hồng trong vùng mặt (kiểu làm đẹp của máy)."""
    m = color.skin_mask(img) * wb.face_region_mask(img.shape[:2], boxes)
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    a, b = lab[..., 1].copy(), lab[..., 2].copy()
    r = math.radians(deg) * m
    k = 1 + (chroma - 1) * m
    lab[..., 1] = (a * np.cos(r) - b * np.sin(r)) * k
    lab[..., 2] = (a * np.sin(r) + b * np.cos(r)) * k
    out = (np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1) * 255 + 0.5).astype(np.uint8)
    return linear_cast(out, (0.92, 1.0, 1.08))


CASTS = [  # (tên, hàm tạo ám, hãng)
    ("đèn vàng", lambda x, f: linear_cast(x, (0.70, 1.0, 1.25)), "apple"),
    ("da đỏ hồng", red_skin, "oppo"),
    ("ám cam", lambda x, f: linear_cast(x, (0.80, 1.02, 1.15)), "samsung"),
]


def label(img, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pad = 34
    canvas = Image.new("RGB", (im.width, im.height + pad), (255, 255, 255))
    canvas.paste(im, (0, pad))
    d = ImageDraw.Draw(canvas)
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(f, 17)
            break
        except OSError:
            pass
    d.text((8, 7), text, fill=(20, 20, 20), font=font or ImageFont.load_default())
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def skin_hue(img, boxes):
    m = (color.skin_mask(img) * wb.face_region_mask(img.shape[:2], boxes)) > 0.5
    if m.sum() < 200:
        return float("nan")
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return math.degrees(math.atan2(np.median(lab[..., 2][m]), np.median(lab[..., 1][m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/claude-0/phone_cast")
    ap.add_argument("--images", nargs="*", default=["obama", "t1", "lena", "two_people"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from net_anh.pipeline import load_image
    eng = Enhancer(device="cpu")
    from net_anh.phone import PROFILES
    for n in a.images:
        x = load_image(Path("/tmp/claude-0/gt") / f"{n}.jpg")
        k = min(1.0, 420 / max(x.shape[:2]))
        x = cv2.resize(x, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        boxes = [b for b, _ in eng.detect_faces(x)]
        rows = []
        for cname, fn, brand in CASTS:
            cx = fn(x, boxes)
            cells = [label(x, f"Ảnh gốc · da {skin_hue(x, boxes):.0f}°"),
                     label(cx, f"Giả lập {cname} · da {skin_hue(cx, boxes):.0f}°")]
            for key in ("generic", brand):
                rep = Report()
                y, _ = eng._fix_color(cx, Settings(phone=key), rep)
                note = "; ".join(rep.notes) or "không sửa"
                print(n, cname, key, "|", note, flush=True)
                cells.append(label(y, f"Bộ {PROFILES[key].label.split(' /')[0]} · da {skin_hue(y, boxes):.0f}°"))
            rows.append(np.hstack(cells))
        cv2.imwrite(str(out / f"{n}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
