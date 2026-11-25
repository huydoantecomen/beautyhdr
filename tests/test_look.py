"""PC 0.14.0: tông mẫu (look.py) – bộ preset Trong sáng theo ảnh mẫu của huy."""
import json

import cv2
import numpy as np
import pytest

from net_anh import facestyle as fs
from net_anh import look
from net_anh.settings import PRESETS, Settings, apply_preset


def _portrait(skin=(150, 175, 215), bg=(200, 160, 60)):
    """Ảnh chân dung giả: nền đồng màu có vết bẩn, mặt elip màu da ấm vàng, môi, tóc; nhãn ParseNet tương ứng."""
    h, w = 400, 300
    img = np.full((h, w, 3), bg, np.uint8)
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    for _ in range(30):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, 120))), 3, (120, 90, 40), -1)
    labels = np.full((h, w), -1, np.int16)
    labels[60:320, 40:260] = 0
    cv2.ellipse(img, (150, 150), (80, 105), 0, 0, 360, (30, 40, 70), -1)  # tóc nâu đỏ
    cv2.ellipse(labels, (150, 150), (80, 105), 0, 0, 360, 13, -1)
    cv2.ellipse(img, (150, 175), (60, 80), 0, 0, 360, skin, -1)
    cv2.ellipse(labels, (150, 175), (60, 80), 0, 0, 360, 1, -1)
    noise = rng.normal(0, 4, (h, w, 1))
    face = labels == 1
    img[face] = np.clip(img[face].astype(np.float32) + noise[face], 0, 255).astype(np.uint8)
    cv2.ellipse(img, (150, 215), (20, 7), 0, 0, 360, (70, 90, 170), -1)
    cv2.ellipse(labels, (150, 215), (20, 7), 0, 0, 360, 12, -1)
    return img, labels


def _lab_mean(img, m):
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab[m].mean(axis=0)


def test_look_moves_skin_toward_target_keeps_texture():
    img, labels = _portrait()
    mk = look.masks_from_labels(labels)
    out, info = look.apply(img, mk, look.Look())
    core = mk.core > 0.5
    L0, a0, b0 = _lab_mean(img, core)
    L1, a1, b1 = _lab_mean(out, core)
    lk = look.Look()
    assert abs(b1 - lk.skin_b) < abs(b0 - lk.skin_b) * 0.4  # bớt vàng
    assert L1 >= L0 - 0.5  # không làm tối da
    # vân da (tầng mịn) gần như giữ nguyên
    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f0 = (g0 - cv2.GaussianBlur(g0, (0, 0), 2))[core].std()
    f1 = (g1 - cv2.GaussianBlur(g1, (0, 0), 2))[core].std()
    assert 0.7 < f1 / f0 < 1.4
    # tóc bớt sắc độ, môi bớt vàng
    hair = cv2.erode((labels == 13).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    _, ha0, hb0 = _lab_mean(img, hair)
    _, ha1, hb1 = _lab_mean(out, hair)
    assert np.hypot(ha1, hb1) < np.hypot(ha0, hb0)
    assert "da_truoc" in info and "da_sau" in info


def test_look_amount_zero_and_no_face_is_noop():
    img, labels = _portrait()
    out, _ = look.apply(img, look.masks_from_labels(labels), look.Look(), 0.0)
    assert np.array_equal(out, img)
    empty = look.masks_from_labels(np.full(labels.shape, -1, np.int16))
    out, info = look.apply(img, empty, look.Look())
    assert np.array_equal(out, img) and not info


def test_clean_backdrop_flattens_but_keeps_protected():
    img, labels = _portrait()
    protect = (labels >= 1).astype(np.float32)
    out, note = look.clean_backdrop(img, 1.0, 0.0, protect)
    assert note
    corner = np.zeros(labels.shape, bool)
    corner[5:50, 5:60] = True
    assert out[corner].astype(np.float32).std(axis=0).mean() < img[corner].astype(np.float32).std(axis=0).mean() * 0.6
    face = cv2.erode((labels == 1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.array_equal(out[face], img[face])  # người không bị đụng


def test_look_dict_roundtrip_and_clamp():
    warn = []
    lk = look.Look.from_dict({"skin_L": 999, "lips": "x", "la": 1, "name": "Của tôi"}, warn)
    assert lk.skin_L == look.RANGES["skin_L"][1] and lk.name == "Của tôi" and len(warn) == 2
    assert look.Look.from_dict(look.Look().to_dict()) == look.Look()


def test_preset_file_v3_carries_look_and_old_reader_paths():
    st = fs.TRONG_SANG_SET
    assert st.builtin and st.look and st.look["name"] == "Trong sáng"
    text = fs.dumps([st])
    d = json.loads(text)
    assert d["version"] == 3 and d["sets"][0]["look"]["skin_b"] == pytest.approx(look.Look().skin_b)
    back, warn = fs.loads(text)
    assert back[0].look == st.look and not warn
    v2 = json.dumps({"format": fs.FORMAT, "version": 2, "sets": [{"name": "cũ", "styles": {"man": {"matte": 0.2}}}]})
    assert fs.loads(v2)[0][0].look is None
    bad = json.dumps({"format": fs.FORMAT, "version": 3, "sets": [{"name": "x", "styles": {"man": {}}, "look": 5}]})
    sets, warn = fs.loads(bad)
    assert sets[0].look is None and warn


def test_library_lists_builtin_look_and_imported(tmp_path):
    lib = fs.Library(tmp_path / "lib")
    assert [s.id for s in lib.looks()] == ["trong-sang", "da-dep-the"]
    with pytest.raises(fs.PresetError):
        lib.delete("trong-sang")
    lib.export(["trong-sang"], tmp_path / "ts.netanh.json")
    other = fs.Library(tmp_path / "other")
    got, _ = other.import_file(tmp_path / "ts.netanh.json")
    assert got[0].look == fs.TRONG_SANG_SET.look and len(other.looks()) == 3


def test_defaults_unchanged():
    s = Settings()
    assert s.look is None and s.tone == 0.6 and s.deshine == 0.0
    assert all(apply_preset(Settings(), k).look is None for k in PRESETS)


def test_cli_look_flags():
    from net_anh import cli

    class A:
        upscale, max_faces, tile = 2, 24, 400
        face_model = sr_model = None
        preset, mode, level = "custom", "quick", None
        low_light = False
        look, look_amount = "trong_sang", 0.5
    for k in ("strength", "sharpness", "denoise", "contrast", "brightness", "fidelity", "color", "gloss", "wb",
              "skin_tone", "exposure", "tone", "deshine"):
        setattr(A, k, None)
    s, _ = cli.build_settings(A(), None, None)
    assert s.look == look.Look().to_dict() and s.look_amount == 0.5
    A.look = "off"
    assert cli.build_settings(A(), None, None)[0].look is None


def test_gui_look_combo_shared(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from net_anh import gui
    w = gui.MainWindow()
    w.set_ui_mode("simple")
    assert w.settings().look is None
    w.look_box2.setCurrentIndex(w.look_box2.findData("trong-sang"))
    assert w.look_box.currentData() == "trong-sang" and w.settings().look == fs.TRONG_SANG_SET.look
    w._simple_pick(kind="portrait", power=4)
    w._simple_pick(kind="id", power="auto")
    assert w.settings().look == fs.TRONG_SANG_SET.look  # đổi loại ảnh, mức nét vẫn giữ
    w.set_ui_mode("pro")
    w.sl_look.set(0.4)
    assert w.settings().look_amount == pytest.approx(0.4)
    w._apply_settings(w.settings().with_(look=None))
    assert w.look_box2.currentData() == "" and w.settings().look is None
    w.search.setText("trong sáng")
    assert any(not sec.isHidden() for sec in w.sections)
    w.close()
    app.processEvents()
