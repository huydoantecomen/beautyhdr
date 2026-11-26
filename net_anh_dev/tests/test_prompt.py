"""Bộ sinh prompt của bản nhà phát triển (chạy: python -m pytest net_anh_dev/tests)."""
import json
import os
import zipfile

import pytest

from net_anh import facestyle
from net_anh.levels import level_settings
from net_anh.settings import PRESETS, Settings
from net_anh_dev import prompt as P


def full_settings():
    s = Settings().with_(**PRESETS["restore"][1])
    ts = facestyle.TRONG_SANG_SET
    return s.with_(look=dict(ts.look), face_styles=facestyle.styles_to_settings(ts), deshine=0.5,
                   bg_color=(219, 142, 67), damage_mask=r"C:\Users\Nguyen Van A\Desktop\mask.png")


@pytest.mark.parametrize("target", list(P.TARGETS))
@pytest.mark.parametrize("lang", list(P.LANGS))
def test_every_target_fits_and_keeps_identity(target, lang):
    p = P.build(full_settings(), target, lang, P.Context(level=5, size=(800, 600), preset_name="Trong sáng"))
    assert 0 < p.chars <= p.limit
    key = "nhận ra 100" if lang == "vi" else "100% recognizable"
    assert key in p.text  # nhắc giữ danh tính luôn còn dù bị rút gọn
    assert ("đổi mặt hay danh tính" if lang == "vi" else "changing the face or identity") in p.text.lower()
    assert "#438EDB" in p.text  # đổi nền (ưu tiên 0) không bị cắt, kể cả Grok


def test_grok_is_compact_and_lists_dropped():
    p = P.build(full_settings(), "grok", "en", P.Context(level=5, size=(800, 600)))
    assert "\n" not in p.text and p.dropped


def test_no_private_data_in_prompt_or_json():
    s = full_settings().with_(phone="samsung")
    ctx = P.Context(level=5, size=(800, 600))
    for t in P.TARGETS:
        for lg in P.LANGS:
            txt = P.build(s, t, lg, ctx).text
            assert "Nguyen" not in txt and "mask.png" not in txt and "samsung" not in txt.lower()
    js = json.loads(P.export_json(s, ctx, "0.15.0"))
    assert js["settings"]["damage_mask"] is True
    assert "Nguyen" not in json.dumps(js, ensure_ascii=False)
    assert set(js["prompts"]) == set(P.TARGETS) and set(js["prompts"]["grok"]) == set(P.LANGS)


def test_settings_drive_the_text():
    base = Settings()
    gentle = P.build(level_settings(3, base), "chatgpt", "en", P.Context(level=3)).text
    strong = P.build(level_settings(5, base), "chatgpt", "en", P.Context(level=5)).text
    assert "level 3/6" in gentle and "level 5/6" in strong
    assert "Keep ≈70% of the original face" in gentle and "Keep ≈30% of the original face" in strong
    # bật/tắt từng nút có ảnh hưởng
    assert "oily shine" not in P.build(base, "chatgpt", "en").text
    assert "oily shine" in P.build(base.with_(deshine=0.6), "chatgpt", "en").text
    assert "Target look" in P.build(base.with_(look=facestyle.TRONG_SANG_SET.look), "chatgpt", "en").text
    assert "For women" not in P.build(base.with_(subject="off"), "chatgpt", "en").text
    one = P.build(base.with_(subject="woman"), "chatgpt", "en").text
    assert "Face retouch:" in one and "For men" not in one
    assert "colorize it realistically" in P.build(full_settings().with_(scene_ai=1.0), "chatgpt", "en").text
    assert "Colorizing a black-and-white" in P.build(full_settings(), "chatgpt", "en").text


def test_upscale_uses_image_size():
    t = P.build(Settings(mode="portrait"), "chatgpt", "en", P.Context(size=(500, 400))).text
    assert "4×" in t and "2000×1600 px" in t
    t = P.build(Settings(mode="portrait", upscale=1), "chatgpt", "en").text
    assert "Keep the original resolution" in t


def test_cli_prints_prompt(capsys):
    from net_anh_dev.cli import main
    assert main(["--preset", "max", "--level", "5", "--target", "gemini", "--lang", "vi", "--bg-color", "xanh"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Dùng ảnh đã tải lên") and "chỉ dán phần trên dòng này" in out
    assert main(["--json"]) == 0
    json.loads(capsys.readouterr().out)


def test_dev_package_contains_dev(tmp_path):
    from net_anh_dev import dong_goi
    zp = dong_goi.main(str(tmp_path))
    assert zp.name.endswith("-dev-TronBo.zip")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    top = names[0].split("/")[0]
    assert f"{top}/CHAY_BAN_DEV.bat" in names and f"{top}/HUONG_DAN_DEV.txt" in names
    assert f"{top}/ma-nguon/net_anh_dev/prompt.py" in names and f"{top}/ma-nguon/net_anh/gui.py" in names
    assert not [n for n in names if "/.git/" in n or n.endswith(".pyc")]


def test_dev_gui_menu_and_dialog():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("torch")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from net_anh.gui import MainWindow
    from net_anh_dev import gui_dev
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.qs.setValue(gui_dev.QS_ON, "0")
    w.qs.setValue(gui_dev.QS_PRIVACY, "1")  # bỏ hộp thoại đồng ý trong test
    gui_dev.install(w)
    assert not w.a_dev_prompt.isVisible() if hasattr(w.a_dev_prompt, "isVisible") else True
    w.a_dev_on.setChecked(True)
    w.set_ui_mode("simple")
    assert w.a_dev_prompt.isVisible() and not w.a_dev_prompt.isEnabled()  # chỉ ở Chuyên sâu
    w.set_ui_mode("pro")
    assert w.a_dev_prompt.isEnabled()
    d = gui_dev.PromptDialog(w)
    assert "100% recognizable" in d.text.toPlainText() or "nhận ra 100" in d.text.toPlainText()
    d.target.setCurrentIndex(d.target.findData("grok"))
    assert len(d.text.toPlainText()) <= P.TARGETS["grok"].limit
    d.copy()
    assert QApplication.clipboard().text() == d.text.toPlainText()
    d.close()
    w.a_dev_on.setChecked(False)
    assert not w.a_dev_prompt.isVisible()
    QSettings("BeautyHDR", "BeautyHDR").remove("dev")
    w.close()
    app.processEvents()
