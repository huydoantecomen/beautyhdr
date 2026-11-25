"""Tạo ảnh mẫu để kiểm tra bản Android (Wb.kt) cho ra kết quả giống net_anh/wb.py.

Ghi vào thư mục parity của dự án Android: ảnh faces.png phủ ám đèn vàng (warm_faces.png) và ám đỏ hồng (pink_faces.png),
kết quả sửa màu có/không có mặt, chỉ sửa da, và các số trung gian (ước lượng nguồn sáng, hệ số kênh, phân tích độ sáng)
trong wb_ref.json.

    python tools/make_wb_fixtures.py --parity ../net-anh-android/core/src/test/resources/parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import wb  # noqa: E402
from net_anh.analysis import analyze  # noqa: E402

WARM = (0.75, 1.0, 1.18)
PINK = (1.0, 0.9, 1.15)


def cast(img, g):
    lin = np.power(img.astype(np.float32) / 255.0, wb.GAMMA) * np.array(g, np.float32)
    return (np.clip(np.power(np.clip(lin, 0, 1), 1 / wb.GAMMA), 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", default=str(Path(__file__).resolve().parents[2] / "net-anh-android/core/src/test/resources/parity"))
    d = Path(ap.parse_args().parity)
    faces = cv2.imread(str(d / "faces.png"))
    boxes = [r[:4] for r in json.loads((d / "faces_ref.json").read_text())]
    warm, pink = cast(faces, WARM), cast(faces, PINK)
    cv2.imwrite(str(d / "warm_faces.png"), warm)
    cv2.imwrite(str(d / "pink_faces.png"), pink)
    out, notes = wb.correct(pink, 0.7, 0.8, -0.05, boxes)  # bộ Oppo: cân bằng trắng + sửa da + giảm bão hoà
    cv2.imwrite(str(d / "expected_wb.png"), out)
    skin, skin_note = wb.skin_tone(pink, boxes, 0.8)
    cv2.imwrite(str(d / "expected_skin.png"), skin)
    out2, notes2 = wb.correct(warm, 0.6)
    cv2.imwrite(str(d / "expected_wb_noface.png"), out2)
    est = wb.estimate_illuminant(warm)
    est_ex = wb.estimate_illuminant(warm, exclude=wb.face_region_mask(warm.shape[:2], boxes))
    dark = cv2.imread(str(d / "input_dark.png"))
    ref = {
        "est": est.tolist(), "est_ex": est_ex.tolist(), "deg": wb.cast_degrees(est), "name": wb.cast_name(est),
        "gains": wb.gains_for(est, 0.7).tolist(), "notes": notes, "notes_noface": notes2, "skin_note": skin_note,
        "highlight_faces": analyze(faces).highlight, "highlight_dark": analyze(dark).highlight,
        "cast_warm": analyze(warm).cast,
    }
    (d / "wb_ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ref, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
