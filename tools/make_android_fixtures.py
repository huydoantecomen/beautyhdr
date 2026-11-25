"""Ảnh mẫu để kiểm tra bản Android 0.8 (FaceStyle.kt, Subject.kt, Backdrop.kt, Restore.kt, Wb.kt) cho ra kết quả giống
bản PC: kiểu mặt, đoán tuổi/giới tính, đổi nền, khôi phục ảnh cũ (đốm bụi, màu phai), cân bằng trắng lấy da làm mốc.

    python tools/make_android_fixtures.py --parity ../net-anh-android/core/src/test/resources/parity \
        --age age-v1.onnx --gender gender-v1.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch  # noqa: F401
except ImportError:  # các hàm dùng ở đây không cần torch (chỉ để nạp được restore.py trên máy không cài torch)
    import types
    sys.modules["torch"] = types.SimpleNamespace(no_grad=lambda: (lambda f: f))
from net_anh import backdrop, facestyle, restore, wb  # noqa: E402


def synth_portrait(seed=3, w=360, h=440):
    """Ảnh thẻ giả: phông xanh có bóng đổ nhẹ, người (đầu, vai, áo tối), tóc có sợi lòa xòa, nhiễu."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.array([214, 140, 70], np.float32)  # BGR xanh
    shade = 1.0 - 0.10 * ((xx - w * 0.5) / w) ** 2 - 0.06 * (yy / h)
    img = bg[None, None] * shade[..., None]
    # áo + vai
    body = ((xx - w / 2) / (w * 0.48)) ** 2 + ((yy - h * 1.02) / (h * 0.42)) ** 2 < 1
    img[body] = (40, 40, 60)
    # cổ, mặt
    neck = (np.abs(xx - w / 2) < w * 0.08) & (yy > h * 0.48) & (yy < h * 0.68)
    face = ((xx - w / 2) / (w * 0.17)) ** 2 + ((yy - h * 0.40) / (h * 0.17)) ** 2 < 1
    skin = np.array([120, 150, 205], np.float32)
    img[neck | face] = skin * 0.92
    # tóc: mũ tóc + sợi
    hair = (((xx - w / 2) / (w * 0.21)) ** 2 + ((yy - h * 0.34) / (h * 0.19)) ** 2 < 1) & ~(face & (yy > h * 0.30))
    img[hair] = (25, 28, 35)
    for _ in range(40):
        x0 = rng.uniform(w * 0.3, w * 0.7)
        y0 = h * 0.18 + rng.uniform(0, h * 0.05)
        pts = np.array([[x0 + rng.uniform(-25, 25) * t, y0 - 18 * t + 6 * t * t] for t in np.linspace(0, 1, 8)], np.int32)
        cv2.polylines(img, [pts], False, (30, 32, 40), 1, cv2.LINE_AA)
    # mắt, miệng cho có chi tiết
    for ex in (w / 2 - w * 0.07, w / 2 + w * 0.07):
        cv2.ellipse(img, (int(ex), int(h * 0.38)), (9, 4), 0, 0, 360, (40, 40, 50), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(w / 2), int(h * 0.47)), (18, 5), 0, 0, 360, (90, 90, 170), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def old_photo(faces, seed=5):
    """Ảnh in cũ giả: phai màu (tương phản thấp, ám vàng), đốm bụi sáng và đốm vàng nhỏ."""
    rng = np.random.default_rng(seed)
    f = faces.astype(np.float32)
    f = 70 + f * 0.55
    f[..., 0] *= 0.85  # bớt xanh dương → ám vàng
    out = np.clip(f, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    for _ in range(25):
        c = (int(rng.uniform(10, w - 10)), int(rng.uniform(10, h - 10)))
        r = int(rng.uniform(1, 3))
        col = (235, 245, 250) if rng.random() < 0.6 else (90, 190, 230)
        cv2.circle(out, c, r, col, -1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", required=True)
    ap.add_argument("--age", required=True)
    ap.add_argument("--gender", required=True)
    a = ap.parse_args()
    d = Path(a.parity)
    faces = cv2.imread(str(d / "faces.png"))
    rows = json.loads((d / "faces_ref.json").read_text())
    ref: dict = {}

    # 1. kiểu mặt trên mặt đã căn 512 (mặt nạ da cho sẵn để hai bản dùng cùng mặt nạ)
    from net_anh.subject import FFHQ_512, _similarity
    lm = np.array(rows[0][5:15], np.float64).reshape(5, 2) if len(rows[0]) >= 15 else None
    assert lm is not None, "faces_ref.json cần 5 điểm mốc"
    m = _similarity(lm, FFHQ_512)
    face512 = cv2.warpAffine(faces, m, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    cv2.imwrite(str(d / "style_face512.png"), face512)
    skin = facestyle.face_window(512) * (cv2.GaussianBlur(
        ((np.abs(np.mgrid[0:512, 0:512][1] - 256) < 150) & (np.mgrid[0:512, 0:512][0] > 150)).astype(np.float32), (0, 0), 8))
    cv2.imwrite(str(d / "style_skin16.png"), np.round(skin * 65535).astype(np.uint16))
    skin = np.round(skin * 65535).astype(np.uint16).astype(np.float32) / 65535  # đúng mặt nạ Kotlin đọc lại
    st = facestyle.DEFAULT_SET.styles["woman"]
    cv2.imwrite(str(d / "expected_style_woman.png"), facestyle.apply(face512, st, skin))
    st2 = facestyle.FaceStyle(smooth=1.0, glow=1.0, bright=1.0, warm=-1.0)
    cv2.imwrite(str(d / "expected_style_max.png"), facestyle.apply(face512, st2, skin))

    # 2. file chia sẻ: bản PC ghi, Android đọc; file lỗi → cùng cảnh báo
    (d / "preset_pc.netanh.json").write_text(facestyle.dumps([facestyle.DEFAULT_SET.copy("Bộ của PC")], "Nét Ảnh PC"),
                                             encoding="utf-8")
    bad = {"format": facestyle.FORMAT, "version": 2, "sets": [
        {"name": "Thử‮ lỗi", "id": "co dau cach", "created": "hôm nay",
         "styles": {"woman": {"smooth": 1.5, "warm": -3, "keep": "x", "la": 1}, "meo": {}, "man": 5}},
        {"name": "Rỗng", "styles": {}}]}
    (d / "preset_bad.netanh.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    got, warn = facestyle.loads((d / "preset_bad.netanh.json").read_bytes())
    ref["bad_warn"] = warn
    ref["bad_name"] = got[0].name
    ref["bad_woman"] = got[0].styles["woman"].to_dict()

    # 3. tuổi, giới tính (OpenCV DNN như bản PC)
    from net_anh.subject import AgeGender
    ag = AgeGender(a.age, a.gender)
    lms = [np.array(r[5:15], np.float64).reshape(5, 2) for r in rows]
    pa, pm = ag.probs(faces, lms)
    ref["age_probs"] = pa.tolist()
    ref["p_male"] = pm.tolist()
    ref["subjects"] = [x.subject for x in ag(faces, lms)]

    # 4. đổi nền
    por = synth_portrait()
    cv2.imwrite(str(d / "bg_portrait.png"), por)
    bd = backdrop.detect(por)
    out, note = backdrop.replace(por, (255, 255, 255), bd)
    cv2.imwrite(str(d / "expected_bg_white.png"), out)
    ref["bg"] = {"lab": bd.color_lab.tolist(), "bgr": list(bd.color_bgr), "spread": bd.spread, "uniform": bd.uniform,
                 "note": note, "faces_detect": backdrop.detect(faces) is not None}
    cv2.imwrite(str(d / "expected_bg_alpha.png"), np.round(backdrop.alpha_matte(por, bd) * 255).astype(np.uint8))

    # 5. khôi phục ảnh cũ (bản nhẹ của điện thoại: đốm bụi + màu phai)
    old = old_photo(faces)
    cv2.imwrite(str(d / "old_photo.png"), old)
    sp = restore.speck_mask(old, 1.0)
    cv2.imwrite(str(d / "expected_speck.png"), sp.astype(np.uint8) * 255)
    fd, note = restore.fade_restore(old, 1.0)
    cv2.imwrite(str(d / "expected_fade.png"), fd)
    gray = cv2.cvtColor(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(d / "old_gray.png"), gray)
    fg, note_g = restore.fade_restore(gray, 1.0)
    cv2.imwrite(str(d / "expected_fade_gray.png"), fg)
    ref["restore"] = {"speck_frac": float(sp.mean()), "fade_note": note, "fade_gray_note": note_g,
                      "mono_old": restore.is_mono(old), "mono_gray": restore.is_mono(gray)}

    # 6. cân bằng trắng lấy da làm mốc: phông màu rực trên, áo đỏ dưới kéo lệch ước lượng nguồn sáng
    boxes = [r[:4] for r in rows]
    h = faces.shape[0]
    ref["skinref"] = {}
    for name, top in (("blue", (200, 120, 20)), ("purple", (180, 60, 160))):
        x = faces.copy()
        x[: int(h * .35)] = np.clip(x[: int(h * .35)] * 0.3 + np.array(top) * 0.7, 0, 255).astype(np.uint8)
        x[int(h * .75):] = np.clip(x[int(h * .75):] * 0.3 + np.array((60, 60, 220)) * 0.7, 0, 255).astype(np.uint8)
        cv2.imwrite(str(d / f"{name}_faces.png"), x)
        out, notes = wb.correct(x, 0.8, 0.0, 0.0, boxes)
        cv2.imwrite(str(d / f"expected_wb_{name}.png"), out)
        est = wb.estimate_illuminant(x, exclude=wb.face_region_mask(x.shape[:2], boxes))
        ref["skinref"][name] = {"notes": notes, "wb_used": wb._protect_skin(x, boxes, est, 0.8)}

    (d / "android08_ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ref, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
