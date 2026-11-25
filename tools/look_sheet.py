"""Ghép ảnh so sánh: các ảnh cùng chiều cao, nhãn trên mỗi ảnh.  python tools/look_sheet.py ra.jpg H nhãn=ảnh ..."""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def main(out, H, items):
    H = int(H)
    tiles = []
    for it in items:
        lab, p = it.split("=", 1)
        im = cv2.imread(p)
        im = cv2.resize(im, (round(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = font(max(14, H // 32))
        bb = d.textbbox((0, 0), lab, font=f)
        d.rectangle((0, 0, bb[2] + 16, bb[3] + 12), fill=(0, 0, 0))
        d.text((8, 4), lab, font=f, fill=(255, 255, 255))
        tiles.append(np.asarray(pil))
    gap = np.full((H, 6, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    Image.fromarray(row).save(out, quality=90)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
