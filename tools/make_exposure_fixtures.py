"""Tạo ảnh mẫu để kiểm tra bản Android (Exposure.kt) cho ra kết quả giống net_anh/exposure.py.

Từ faces.png (2 mặt, khung trong faces_ref.json), input_dark.png và sharp.png của thư mục parity, ghi:
  expo_dark.png     faces.png tối −2 EV (như máy chụp thiếu sáng)
  expo_bright.png   faces.png sáng +1,6 EV (cháy vùng sáng)
  expo_backlit.png  người (đầu, cổ, vai) tối −1,7 EV, nền sáng thêm +0,3 EV (ngược sáng)
  expo_logo.png     hình đồ hoạ nền trắng (không được coi là quá sáng)
kết quả sửa expected_expo_*.png và kế hoạch sửa (loại, hệ số, độ sáng mặt…) trong expo_ref.json.

    python tools/make_exposure_fixtures.py --parity ../net-anh-android/core/src/test/resources/parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import exposure, wb  # noqa: E402


def ev(img, stops, mask=None):
    lin = np.power(img.astype(np.float32) / 255.0, wb.GAMMA)
    gain = 2.0 ** stops if mask is None else mask[..., None]
    return (np.power(np.clip(lin * gain, 0, 1), 1 / wb.GAMMA) * 255 + 0.5).astype(np.uint8)


def person_mask(shape, boxes):
    """Như tools/exposure_bench.py: đầu, cổ, vai bên dưới mỗi mặt, viền mềm."""
    m = np.zeros(shape, np.float32)
    blur = 3.0
    for b in boxes:
        fw, fh = float(b[2] - b[0]), float(b[3] - b[1])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        cv2.ellipse(m, (round(cx), round(cy - 0.05 * fh)), (round(0.8 * fw), round(1.0 * fh)), 0, 0, 360, 1.0, -1)
        cv2.rectangle(m, (round(cx - 0.4 * fw), round(cy)), (round(cx + 0.4 * fw), round(cy + 1.3 * fh)), 1.0, -1)
        cv2.ellipse(m, (round(cx), round(cy + 2.3 * fh)), (round(2.0 * fw), round(1.3 * fh)), 0, 0, 360, 1.0, -1)
        blur = max(blur, 0.12 * fw)
    return cv2.GaussianBlur(m, (0, 0), blur)


def logo():
    x = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(x, (150, 140), 80, (30, 60, 200), -1)
    cv2.rectangle(x, (60, 240), (240, 270), (40, 40, 40), -1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", default=str(Path(__file__).resolve().parents[2] / "net-anh-android/core/src/test/resources/parity"))
    d = Path(ap.parse_args().parity)
    faces = cv2.imread(str(d / "faces.png"))
    boxes = [np.asarray(r[:4], np.float32) for r in json.loads((d / "faces_ref.json").read_text())]
    m = person_mask(faces.shape[:2], boxes)
    backlit_gain = m * 2.0 ** -1.7 + (1 - m) * 2.0 ** 0.3
    cases = {
        "dark": (ev(faces, -2.0), boxes),
        "bright": (ev(faces, 1.6), boxes),
        "backlit": (ev(faces, 0, backlit_gain), boxes),
        "dark_noface": (cv2.imread(str(d / "input_dark.png")), []),
        "sharp": (cv2.imread(str(d / "sharp.png")), []),
        "logo": (logo(), []),
    }
    ref = {}
    for name, (img, bx) in cases.items():
        if name in ("dark", "bright", "backlit", "logo"):
            cv2.imwrite(str(d / f"expo_{name}.png"), img)
        out, p = exposure.correct(img, bx)
        if p.kind:
            cv2.imwrite(str(d / f"expected_expo_{name}.png"), out)
        small, k = exposure._small(img)
        y = exposure.luma(small)
        fy, bg = exposure.face_luma(y, bx, k)
        ref[name] = {"kind": p.kind, "gain": p.gain, "lift": p.lift, "face_b": p.face_b, "note": p.note,
                     "face_y": None if fy != fy else fy, "bg_y": None if bg != bg else bg,
                     "p99": float(np.percentile(y, 99)), "median": float(np.median(y)),
                     "graphic": bool(exposure.is_graphic(small))}
    (d / "expo_ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ref, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
