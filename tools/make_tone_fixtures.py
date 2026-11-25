"""Tạo ảnh mẫu để kiểm tra bản Android (Tone.kt) cho ra kết quả giống net_anh/tone.py.

Từ faces.png (2 mặt, khung trong faces_ref.json), input_dark.png, sharp.png và expo_logo.png của thư mục parity, ghi
expected_tone_*.png (tông tươi sáng) và tone_ref.json (hệ số nâng tông giữa a, mức đen b0 của từng ca).

    python tools/make_tone_fixtures.py --parity ../net-anh-android/core/src/test/resources/parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import tone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", default=str(Path(__file__).resolve().parents[2] / "net-anh-android/core/src/test/resources/parity"))
    d = Path(ap.parse_args().parity)
    faces = cv2.imread(str(d / "faces.png"))
    boxes = [np.asarray(r[:4], np.float32) for r in json.loads((d / "faces_ref.json").read_text())]
    cases = {
        "faces": (faces, boxes, 0.6),
        "faces_full": (faces, boxes, 1.0),
        "faces_nobox": (faces, [], 0.6),
        "dark_noface": (cv2.imread(str(d / "input_dark.png")), [], 0.6),
        "sharp": (cv2.imread(str(d / "sharp.png")), [], 1.0),
        "logo": (cv2.imread(str(d / "expo_logo.png")), [], 1.0),
    }
    ref = {}
    for name, (img, bx, amount) in cases.items():
        out, note = tone.bright_clear(img, amount, bx)
        a, b0 = tone.plan(img, bx, amount)
        if out is not img:
            cv2.imwrite(str(d / f"expected_tone_{name}.png"), out)
        ref[name] = {"amount": amount, "a": a, "b0": b0, "changed": out is not img, "note": note}
    (d / "tone_ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ref, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
