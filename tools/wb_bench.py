"""Đo cân bằng trắng trên ảnh ám màu giả lập kiểu điện thoại.

Mỗi ảnh x được phủ một ám màu c (nhân kênh ở không gian tuyến tính): đèn vàng trong nhà, ám vàng, ám đỏ hồng.
Thước đo "nhất quán màu" (color constancy tương đối): Δab (lệch màu, bỏ độ sáng) trung bình giữa WB(c(x)) và WB(x). Ám màu được gỡ đúng
thì hai ảnh như nhau dù ảnh vào khác màu. So với mốc ΔE(c(x), x) khi không sửa. Thêm ca không phủ ám (c = 1) để xem
WB có tự làm đổi màu ảnh vốn đã ổn bao nhiêu: ΔE(WB(x), x).

    python tools/wb_bench.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import wb  # noqa: E402

CASTS = {  # hệ số tuyến tính (B, G, R)
    "đèn vàng": (0.70, 1.0, 1.25),
    "ám vàng": (0.75, 1.06, 1.12),
    "ám đỏ hồng": (0.95, 0.95, 1.20),
    "ám xanh": (1.25, 1.0, 0.85),
}


def apply_cast(img, g):
    lin = np.power(img.astype(np.float32) / 255.0, wb.GAMMA) * np.array(g, np.float32)
    return (np.clip(np.power(np.clip(lin, 0, 1), 1 / wb.GAMMA), 0, 1) * 255 + 0.5).astype(np.uint8)


def de76(a, b):
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    # chỉ phần màu (a, b): WB giữ độ sáng theo ảnh vào nên độ sáng hai ảnh có thể lệch một hằng số, không tính
    return float(np.sqrt(((la[..., 1:] - lb[..., 1:]) ** 2).sum(-1)).mean())


def images():
    import skimage.data as sd
    for f in sorted(Path("/tmp/claude-0/gt").glob("*.jpg")):
        im = cv2.imread(str(f))
        k = min(1.0, 800 / max(im.shape[:2]))
        yield f.stem, cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    for n in ("coffee", "chelsea", "astronaut", "rocket", "brick", "camera"):
        im = getattr(sd, n)()
        im = np.dstack([im] * 3) if im.ndim == 2 else im[..., :3]
        yield n, cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def main():
    res = {}
    for name, x in images():
        for strength in (0.6, 1.0):
            wx, _ = wb.correct(x, strength)
            res.setdefault(("không ám", strength), []).append((0.0, de76(wx, x)))
            for cn, g in CASTS.items():
                cx = apply_cast(x, g)
                wcx, _ = wb.correct(cx, strength)
                res.setdefault((cn, strength), []).append((de76(cx, x), de76(wcx, wx)))
        print(name, "xong", flush=True)
    print(f"\n{'ám màu':12s}{'mức':>5s}{'ΔE không sửa':>14s}{'ΔE sau WB':>11s}{'giảm':>7s}")
    for (cn, st), v in res.items():
        v = np.array(v)
        a, b = v[:, 0].mean(), v[:, 1].mean()
        print(f"{cn:12s}{st:5.1f}{a:14.2f}{b:11.2f}" + (f"{100 * (1 - b / a):6.0f}%" if a > 0 else "   (đổi màu ảnh vốn ổn)"))


if __name__ == "__main__":
    main()
