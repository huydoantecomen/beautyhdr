"""Kiểm thử nhanh (không cần GPU). Các test cần model AI tự bỏ qua nếu model chưa tải."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from net_anh import classic, weights
from net_anh.analysis import Analysis, analyze, check_face, recommend
from net_anh.history import History
from net_anh.pipeline import (Cancelled, Enhancer, ImageError, blend_strength, load_image, save_image,
                              unique_output_path)
from net_anh.settings import PRESETS, Settings, apply_preset

RNG = np.random.default_rng(0)


def img(h=120, w=160, value=None):
    if value is not None:
        return np.full((h, w, 3), value, np.uint8)
    x = cv2.GaussianBlur(RNG.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)


# ---------- đọc / ghi ----------

def test_exif_orientation_applied(tmp_path):
    a = np.zeros((40, 80, 3), np.uint8)  # ngang 80×40
    p = tmp_path / "rot.jpg"
    im = Image.fromarray(a)
    exif = im.getexif()
    exif[0x0112] = 6  # xoay 90°
    im.save(p, exif=exif)
    assert load_image(p).shape[:2] == (80, 40)


def test_unsupported_and_corrupt(tmp_path):
    (tmp_path / "a.gif").write_bytes(b"GIF89a")
    with pytest.raises(ImageError):
        load_image(tmp_path / "a.gif")
    (tmp_path / "b.jpg").write_bytes(b"not an image")
    with pytest.raises(ImageError):
        load_image(tmp_path / "b.jpg")


def test_output_never_overwrites_source(tmp_path):
    src = tmp_path / "x.jpg"
    save_image(img(), src)
    p1 = unique_output_path(src)
    assert p1 != src
    save_image(img(), p1)
    p2 = unique_output_path(src)
    assert p2 not in (src, p1)
    # cùng đuôi với ảnh gốc, tag rỗng: vẫn không trùng
    assert unique_output_path(src, tag="") != src


def test_save_quality_and_formats(tmp_path):
    x = img()
    for ext in (".jpg", ".png", ".webp"):
        out = save_image(x, tmp_path / f"o{ext}", 94)
        assert load_image(out).shape == x.shape


# ---------- cổ điển ----------

@pytest.mark.parametrize("up", [1, 2, 4])
def test_quick_enhance_shapes(up):
    x = img()
    assert classic.quick_enhance(x, upscale=up).shape == (120 * up, 160 * up, 3)


def test_noise_estimate_monotonic():
    x = img(200, 200)
    est = [classic.estimate_noise(np.clip(x + RNG.normal(0, s, x.shape), 0, 255).astype(np.uint8)) for s in (0, 5, 15)]
    assert est[0] < est[1] < est[2]


def test_retinex_and_brightness_lift_dark_image():
    dark = (img(200, 200) * 0.2).astype(np.uint8)
    assert classic.retinex_l(dark).mean() > dark.mean()
    assert classic.brightness(dark, 0.5).mean() > dark.mean()
    assert classic.brightness(dark, -0.5).mean() <= dark.mean()


def test_strength_zero_returns_plain_upscale():
    x = img()
    out = classic.quick_enhance(x, 2)
    base = cv2.resize(x, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    assert np.array_equal(blend_strength(x, out, 0.0), base)


# ---------- phân tích + router ----------

def test_check_face_rules():
    lm = np.array([[40, 50], [60, 50], [50, 60], [42, 70], [58, 70]], np.float32)
    assert check_face(np.array([30, 30, 70, 90, 1]), lm)[0]
    assert not check_face(np.array([0, 0, 10, 10, 1]), lm)[0]  # mặt < 20 px
    side = lm.copy()
    side[2, 0] = 62  # mũi lệch hẳn sang một bên
    ok, why = check_face(np.array([30, 30, 70, 90, 1]), side)
    assert not ok and "45" in why


def test_router_dark_image_uses_exposure():
    """Ảnh tối: Tự động sửa phơi sáng (nhân sáng giữ màu), không bật Retinex (Retinex làm sáng quá tay, ΔL +40)."""
    a = analyze((img(300, 300) * 0.25).astype(np.uint8))
    preset, s, why = recommend(a)
    assert a.expo == "tối" and a.light_label == "tối"
    assert preset == "auto" and not s.low_light and s.exposure == 1.0
    assert any("thiếu sáng" in w for w in why)


def test_router_gray_noisy_is_old_photo():
    g = cv2.cvtColor(cv2.cvtColor(img(300, 300), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    noise = RNG.normal(0, 12, g.shape[:2])[..., None]  # nhiễu hạt xám như ảnh scan
    g = np.clip(g + noise, 0, 255).astype(np.uint8)
    a = analyze(g)
    preset, s, why = recommend(a)
    assert preset == "auto" and s.contrast >= 0.67 and "ảnh cũ" in why[0]
    # chưa dò mặt: mức 2 (AI nhẹ, khử hạt mạnh như Ảnh cũ); đã dò mặt (không có mặt): mức 3, chế độ Ảnh cũ
    assert s.mode == "upscale" and s.sr_dn == 1.0
    a.faces_checked = True
    assert recommend(a)[1].mode == "old"


def test_router_no_face_check_never_portrait():
    a = Analysis(500, 500, blur=20, noise=1, brightness=120, contrast=60, saturation=80)
    a.faces_checked = False
    assert recommend(a)[1].mode != "portrait"


def test_presets_apply():
    for k in PRESETS:
        apply_preset(Settings(), k)
    assert apply_preset(Settings(), "portrait").mode == "portrait"
    assert Settings(fidelity=0.5).ai_weight == 0.5
    assert Settings(contrast=2 / 3).clahe_clip == pytest.approx(2.0)


# ---------- lịch sử ----------

def test_history_roundtrip(tmp_path):
    h = History(tmp_path)
    out = save_image(img(), tmp_path / "o.jpg")
    eid = h.add(tmp_path / "s.jpg", out, img(), "quick", Settings().to_dict(), 1.2)
    assert [e.id for e in h.list()] == [eid]
    h.delete(eid, delete_output=True)
    assert h.list() == [] and not out.exists()
    h.close()


# ---------- quản lý model ----------

def test_model_status_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("NET_ANH_WEIGHTS", str(tmp_path))
    m = weights.MANIFEST["esrgan_x2"]
    assert weights.status("esrgan_x2") == "missing"
    (tmp_path / m.file).write_bytes(b"x" * 10)
    assert weights.status("esrgan_x2") == "corrupt"

    def fake_download(info, path, say, cancel):
        tmp = path.with_suffix(".part")
        tmp.write_bytes(b"wrong")
        return tmp, False

    monkeypatch.setattr(weights, "_download", fake_download)
    monkeypatch.setattr(weights.time, "sleep", lambda s: None)
    with pytest.raises(weights.ModelError):
        weights.ensure("esrgan_x2", retries=2)
    assert not (tmp_path / m.file).exists()  # file sai checksum bị xoá
    assert not list(tmp_path.glob("*.part"))  # không bao giờ đổi tên file chưa kiểm vào chỗ model


def test_model_download_size_guard(tmp_path, monkeypatch):
    """Server trả thừa byte thì dừng, không để file lại; trả thiếu byte (mất mạng) thì giữ .part để làm tiếp bằng Range."""
    import io

    monkeypatch.setenv("NET_ANH_WEIGHTS", str(tmp_path))
    monkeypatch.setenv("NET_ANH_DRIVE_JSON", str(tmp_path / "khong-co.json"))
    m = weights.MANIFEST["esrgan_compact"]

    class Resp(io.BytesIO):
        headers: dict = {}
        status = 200

    for n in (m.size - 5, m.size + (1 << 20) + 5):
        monkeypatch.setattr(weights.urllib.request, "urlopen", lambda *a, n=n, **k: Resp(b"\0" * n))
        for f in tmp_path.glob("*.part"):
            f.unlink()
        with pytest.raises(weights.ModelError):
            weights._download(m, tmp_path / m.file, lambda s: None, None)
        parts = list(tmp_path.glob("*.part"))
        assert bool(parts) == (n < m.size)


def test_load_16bit_gray_and_pixel_bomb(tmp_path, monkeypatch):
    """0.6.1: ảnh scan xám 16 bit trước đây ra gần trắng toàn bộ; ảnh quá nhiều điểm ảnh trả ImageError."""
    g = np.linspace(0, 65535, 64 * 256).reshape(64, 256).astype(np.uint16)
    p = tmp_path / "scan16.png"
    cv2.imwrite(str(p), g)
    x = load_image(p)
    assert x.dtype == np.uint8 and x.shape == (64, 256, 3)
    assert abs(float(x.mean()) - 127.5) < 2
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(ImageError):
        load_image(p)


def test_tiled_matches_untiled():
    """0.6.1: ghép ô thẳng vào uint8 cho kết quả giống hệt chạy cả ảnh (mạng tích chập 1 lớp, viền ô đủ rộng)."""
    import torch

    net = torch.nn.Sequential(torch.nn.Conv2d(3, 3 * 4, 3, padding=1), torch.nn.PixelShuffle(2)).eval()
    eng = Enhancer("cpu")
    x = (np.random.default_rng(3).random((70, 90, 3)) * 255).astype(np.uint8)
    a = eng._esrgan_run(x, 2, 0, net)
    b = eng._esrgan_run(x, 2, 32, net)
    assert a.shape == b.shape == (140, 180, 3)
    assert np.abs(a.astype(int) - b).max() <= 1


# ---------- pipeline ----------

def test_quick_mode_runs_on_cpu():
    out, rep = Enhancer("cpu").process(img(), Settings(mode="quick", upscale=2))
    assert out.shape == (240, 320, 3) and rep.mode == "quick"


def test_ai_failure_falls_back_to_quick(monkeypatch):
    eng = Enhancer("cpu")

    def boom(*a, **k):
        raise weights.ModelError("không có mạng")

    monkeypatch.setattr(eng, "_process_ai", boom)
    out, rep = eng.process(img(), Settings(mode="portrait", upscale=2))
    assert rep.mode == "quick" and "Cải thiện nhanh" in rep.notes[0] and out.shape == (240, 320, 3)


def test_cancel_raises():
    eng = Enhancer("cpu")
    eng.cancel_requested = lambda: True
    with pytest.raises(Cancelled):
        eng.process(img(), Settings(mode="quick"))


AI_READY = all(weights.status(m) == "ready" for m in ("esrgan_compact", "esrgan_compact_wdn", "gfpgan", "retinaface", "parsenet"))


@pytest.mark.skipif(not AI_READY, reason="chưa tải model AI")
def test_no_face_image_skips_face_restoration():
    out, rep = Enhancer("cpu").process(img(96, 128), Settings(mode="portrait", upscale=2))
    assert rep.faces_found == 0 and rep.faces_restored == 0 and out.shape == (192, 256, 3)


@pytest.mark.skipif(not AI_READY, reason="chưa tải model AI")
def test_source_file_unchanged_after_cli(tmp_path):
    from net_anh.cli import main

    src = tmp_path / "a.png"
    save_image(img(), src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    assert main([str(src), "--mode", "quick"]) == 0
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before
    assert (tmp_path / "a_net.jpg").exists()


def test_cli_unicode_paths_and_output_folder(tmp_path):
    """Đường dẫn có dấu tiếng Việt và dấu cách (như D:\\nét ảnh\\...), "-o thư mục/" ghi vào trong thư mục."""
    from net_anh.cli import main

    d = tmp_path / "nét ảnh" / "ảnh thử"
    d.mkdir(parents=True)
    src = d / "chân dung số 1.png"
    save_image(img(), src)
    assert load_image(src).shape == (120, 160, 3)
    out_dir = tmp_path / "nét ảnh" / "kết quả thử"
    assert main([str(src), "--mode", "quick", "--upscale", "2", "-o", str(out_dir) + "/"]) == 0
    out = out_dir / "chân dung số 1_net.jpg"
    assert out.is_file() and load_image(out).shape[:2] == (240, 320)
    assert main([str(src), "--mode", "quick", "--upscale", "2", "-o", str(out_dir)]) == 0  # thư mục đã có
    assert (out_dir / "chân dung số 1_net_2.jpg").is_file()
    assert main([str(src), "--mode", "quick", "--upscale", "1", "-o", str(out_dir / "tên mới.png")]) == 0
    assert load_image(out_dir / "tên mới.png").shape[:2] == (120, 160)


def test_sr_denoise_mapping():
    assert Settings().sr_dn == 0.5
    assert Settings(mode="old").sr_dn == 1.0
    assert Settings(denoise=3).sr_dn == 0.3
    assert Settings(sr_denoise=0.1, denoise=9).sr_dn == 0.1


@pytest.mark.skipif(not AI_READY, reason="chưa có model AI")
def test_compact_upscale_shape_and_dni_cache():
    from net_anh.pipeline import Enhancer
    eng = Enhancer("cpu")
    img = (np.random.default_rng(0).random((40, 52, 3)) * 255).astype(np.uint8)
    for up in (1, 2, 4):
        out, rep = eng.process(img, Settings(mode="upscale", upscale=up, tile=32))
        assert out.shape == (40 * up, 52 * up, 3)
    eng.process(img, Settings(mode="upscale", denoise=2))
    assert sum(isinstance(k, tuple) for k in eng._esrgan) == 1  # chỉ giữ một mức trộn trong bộ nhớ


def test_blur_factor_detects_blur_not_sharp():
    rng = np.random.default_rng(1)
    sharp = cv2.resize((rng.random((80, 80, 3)) * 255).astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
    assert classic.blur_factor(sharp) == 1.0
    assert classic.blur_factor(cv2.GaussianBlur(sharp, (0, 0), 3)) >= 2.0


def test_router_blurry_no_faces_uses_ai_deblur():
    a = Analysis(width=2000, height=1500, blur=60, noise=2, brightness=120, contrast=60, saturation=80, blur_d=2.5,
                 faces=[], faces_checked=True)
    preset, s, why = recommend(a)
    assert s.mode == "upscale" and any("khử mờ" in w for w in why)


def test_weights_dir_frozen_fallback(tmp_path, monkeypatch):
    """Bản .exe: dùng weights cạnh exe nếu ghi được, nếu không thì %LOCALAPPDATA%/NetAnh/weights."""
    import os
    import stat
    import sys

    from net_anh import weights

    monkeypatch.delenv("NET_ANH_WEIGHTS", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.setattr(sys, "executable", str(app / "NetAnh.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert weights.weights_dir() == app / "weights"
    if os.name != "nt" and os.geteuid() != 0:  # root ghi được mọi thư mục nên không thử được
        (app / "weights").chmod(stat.S_IRUSR | stat.S_IXUSR)
        assert weights.weights_dir() == tmp_path / "local" / "NetAnh" / "weights"


def test_selftest_quick(tmp_path, monkeypatch):
    from net_anh.selftest import dispatch

    monkeypatch.setenv("NET_ANH_WEIGHTS", str(tmp_path / "w"))  # không có model: chỉ chạy Nhanh
    log = tmp_path / "tk.log"
    assert dispatch(["--tu-kiem", str(log)]) == 0
    assert "KẾT QUẢ: ĐẠT" in log.read_text(encoding="utf-8")
    assert dispatch(["--tu-kiem", str(log), "--can-ai"]) == 1
    assert dispatch(["anh.jpg"]) is None


def test_color_gloss():
    from net_anh import color

    x = img(120, 160)
    assert np.array_equal(color.enhance(x, 0, 0), x)  # tắt cả hai: giữ nguyên
    skin = np.full((80, 80, 3), (120, 150, 200), np.uint8)  # BGR màu da
    assert color.skin_mask(skin).mean() > 0.9
    grey = np.full((80, 80, 3), (128, 128, 128), np.uint8)
    assert color.skin_mask(grey).max() < 0.1
    y = color.enhance(x, 0.6, 0.5)
    assert y.shape == x.shape and y.dtype == np.uint8 and np.abs(y.astype(int) - x).mean() > 1
    # độ rực: ảnh nhạt màu thì độ bão hoà tăng
    dull = cv2.cvtColor(np.dstack([np.full((60, 60), 100, np.uint8), np.full((60, 60), 110, np.uint8),
                                   np.full((60, 60), 150, np.uint8)]), cv2.COLOR_HSV2BGR)
    s0 = cv2.cvtColor(dull, cv2.COLOR_BGR2HSV)[..., 1].mean()
    s1 = cv2.cvtColor(color.enhance(dull, 1.0, 0), cv2.COLOR_BGR2HSV)[..., 1].mean()
    assert s1 > s0 + 5


def test_preset_max_is_remini_style():
    s = apply_preset(Settings(), "max")
    assert s.mode == "portrait" and s.detail_fidelity == 0.0 and s.fidelity < 0.5 and s.color > 0 and s.gloss == 0  # 0.11: huy chấm mù ưa bản tắt độ bóng da
    assert s.upscale == 0  # tự động: ít nhất 2×, ảnh nhỏ 4×
    assert Settings().fidelity == 0.7


# ---------- màu ảnh điện thoại ----------

def _cast(x, g):
    from net_anh import wb
    lin = np.power(x.astype(np.float32) / 255.0, wb.GAMMA) * np.array(g, np.float32)
    return (np.clip(np.power(np.clip(lin, 0, 1), 1 / wb.GAMMA), 0, 1) * 255 + 0.5).astype(np.uint8)


def test_wb_removes_warm_cast():
    from net_anh import wb
    x = img(160, 200)
    warm = _cast(x, (0.72, 1.0, 1.25))  # B, G, R: đèn vàng
    assert wb.cast_degrees(wb.estimate_illuminant(warm)) > wb.cast_degrees(wb.estimate_illuminant(x)) + 5
    out, notes = wb.correct(warm, 1.0)
    assert notes and "ám" in notes[0]
    def ab_err(a, b):
        la, lb = (cv2.cvtColor(v.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) for v in (a, b))
        return float(np.abs(la[..., 1:] - lb[..., 1:]).mean())
    ref, _ = wb.correct(x, 1.0)
    assert ab_err(out, ref) < 0.5 * ab_err(warm, ref)
    same, notes0 = wb.correct(np.full((50, 50, 3), 128, np.uint8), 1.0)  # ảnh xám: không có gì để sửa
    assert not notes0 and np.array_equal(same, np.full((50, 50, 3), 128, np.uint8))


def test_skin_tone_pulls_red_skin():
    from net_anh import wb
    red = np.zeros((200, 200, 3), np.uint8)
    red[:] = (100, 105, 200)  # BGR: da đỏ hẳn (góc màu ~28°; da hồng 36–44° được giữ)
    lab0 = cv2.cvtColor(red.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[100, 100]
    out, why = wb.skin_tone(red, [np.array([40, 30, 160, 170])], 1.0)
    lab1 = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[100, 100]
    h0, h1 = (np.degrees(np.arctan2(v[2], v[1])) for v in (lab0, lab1))
    assert "đỏ" in why and h1 > h0 + 3
    assert np.array_equal(wb.skin_tone(red, [], 1.0)[0], red)  # không có mặt: không đụng


def test_phone_brand_from_exif(tmp_path):
    from PIL import Image
    from net_anh import phone
    assert phone.brand_of("Apple", "iPhone 13") == "apple"
    assert phone.brand_of("samsung", "SM-S918B") == "samsung"
    assert phone.brand_of("Xiaomi", "Redmi Note 12") == "xiaomi"
    assert phone.brand_of("realme", "RMX3686") == "oppo"
    assert phone.brand_of("vivo", "V2250") == "vivo"
    assert phone.brand_of("HONOR", "ANY-NX1") == "huawei"
    assert phone.brand_of("Sony", "XQ-DQ72") == "generic" and phone.brand_of("SONY", "ILCE-7M3") == "camera"
    assert phone.brand_of("", "") == "generic"
    f = tmp_path / "a.jpg"
    im = Image.fromarray(img()[..., ::-1].copy())
    ex = im.getexif(); ex[0x010F] = "OPPO"; ex[0x0110] = "CPH2491"
    im.save(f, exif=ex)
    assert phone.detect(f)[0] == "oppo"
    g = tmp_path / "b.jpg"; im.save(g)
    assert phone.detect(g) [0] == "generic" and "EXIF" in phone.detect(g)[1]


def test_dark_suit_is_not_dark_image():
    """Áo tối + tường sáng: L trung bình thấp nhưng không phải ảnh thiếu sáng (lỗi bản 0.4 bật Retinex)."""
    x = np.full((300, 300, 3), 20, np.uint8)
    x[:, :60] = 235  # tường sáng (20 % ảnh)
    a = analyze(x)
    assert a.brightness < 80 and not a.dark
    assert analyze((img(300, 300) * 0.25).astype(np.uint8)).dark


def test_process_applies_phone_profile():
    eng = Enhancer("cpu")
    warm = _cast(img(96, 128), (0.72, 1.0, 1.25))
    _, rep = eng.process(warm, Settings(mode="quick", upscale=1, phone="apple"))
    assert any("ám" in n for n in rep.notes)
    out, rep = eng.process(warm, Settings(mode="quick", upscale=1, phone="off"))
    assert not any("ám" in n for n in rep.notes)


def test_wb_ignores_face_skin():
    """Ảnh cận mặt trên nền xám trung tính: da ấm không được tính là ám màu khi đã biết vùng mặt."""
    from net_anh import wb
    rng = np.random.default_rng(3)
    x = np.clip(rng.normal(150, 12, (300, 300, 3)), 0, 255).astype(np.uint8)
    box = [80.0, 50.0, 220.0, 230.0]
    region = wb.face_region_mask(x.shape[:2], [box]) > 0.5
    x[region] = (95, 140, 205)  # da (B, G, R)
    x = cv2.GaussianBlur(x, (0, 0), 1.0)
    assert wb.cast_degrees(wb.estimate_illuminant(x)) > 4
    assert wb.cast_degrees(wb.estimate_illuminant(x, exclude=wb.face_region_mask(x.shape[:2], [box]))) < 2
    out, notes = wb.correct(x, 0.7, 0.0, 0.0, [box])
    assert not notes and out is x


def test_twilight_scene_is_not_dark():
    """Trời chạng vạng có vài đèn sáng (~3 % điểm ảnh): không bật Retinex (bản 0.5 trước sửa làm trời thành tím nhạt)."""
    x = np.zeros((300, 400, 3), np.uint8)
    x[:] = (70, 35, 15)  # trời xanh thẫm (BGR)
    x[:, 180:220] = (60, 60, 60)  # cột tối
    rng = np.random.default_rng(1)
    for _ in range(14):  # đèn
        cx, cy = int(rng.integers(10, 390)), int(rng.integers(220, 290))
        cv2.circle(x, (cx, cy), 6, (200, 235, 255), -1)
    x[260:300, 150:250] = (230, 235, 240)  # bệ sáng
    a = analyze(x)
    assert a.brightness < 80 and not a.dark, (a.brightness, a.highlight)


def _analysis(faces: bool, checked: bool = True, **kw):
    from net_anh.analysis import Analysis, Face
    f = [Face(np.array([0, 0, 120, 120, 0.99], np.float32),
              np.array([[40, 50], [80, 50], [60, 70], [45, 95], [75, 95]], np.float32))] if faces else []
    d = dict(width=640, height=480, blur=80.0, noise=2.0, brightness=120.0, contrast=55.0, saturation=60.0, blur_d=2.0)
    d.update(kw)
    return Analysis(**d, faces=f, faces_checked=checked)


def test_level_ladder():
    """Thang mức nét: 1–2 không AI, 3–5 AI vẽ lại tăng dần; giữ hãng máy, model, Retinex; chế độ theo nội dung.
    Mức 6 = mức 5 với bộ model nặng nhất."""
    from net_anh.levels import BEST_BG, BEST_FACE, LEVELS, describe, level_for, level_settings
    base = Settings(phone="oppo", face_model="restoreformer", sr_model="rrdb", low_light=True, upscale=1)
    a = _analysis(True)
    assert list(LEVELS) == [1, 2, 3, 4, 5, 6]
    ss = [level_settings(n, base, a) for n in range(1, 6)]
    s6 = level_settings(6, base, a)
    assert (s6.face_model, s6.sr_model) == (BEST_FACE, BEST_BG) == ("restoreformer_pp", "dat2")
    assert s6.with_(face_model=base.face_model, sr_model=base.sr_model) == ss[4]
    assert level_for("max", apply_preset(Settings(), "max").with_(face_model=BEST_FACE, sr_model=BEST_BG)) == 6
    assert "RestoreFormer++" in describe(6, s6) and "DAT-2" in describe(6, s6) and "vài phút" in describe(6, s6)
    assert [s.mode for s in ss] == ["quick", "upscale", "portrait", "portrait", "portrait"]
    assert ss[0].sharpness < ss[1].sharpness
    assert [s.sharpness for s in ss[2:]] == sorted(s.sharpness for s in ss[2:])
    fids = [s.fidelity for s in ss[2:]]
    assert fids == sorted(fids, reverse=True) and fids[0] == 0.7 and fids[-1] == 0.3
    assert ss[2].detail_fidelity is None and ss[4].detail_fidelity == 0.0 and ss[4].upscale == 2 and ss[2].upscale == 1
    assert ss[1].contrast == 0.0  # ảnh tối có Retinex: mức 2 không thêm CLAHE
    for s in ss:
        assert (s.phone, s.face_model, s.sr_model, s.low_light) == ("oppo", "restoreformer", "rrdb", True)
    # không có mặt → Phóng to AI, không làm bóng da; ảnh cũ → Ảnh cũ; chưa dò được mặt → Chân dung
    s5 = level_settings(5, Settings(), _analysis(False))
    assert s5.mode == "upscale" and s5.gloss == 0.0 and s5.color > 0
    assert level_settings(4, Settings(mode="old"), _analysis(False)).mode == "old"
    assert level_settings(3, Settings(), _analysis(False, checked=False)).mode == "portrait"
    # preset ↔ mức
    assert level_for("max", apply_preset(Settings(), "max")) == 5
    assert level_for("natural", apply_preset(Settings(), "natural")) == 1
    assert level_for("sharp", apply_preset(Settings(), "sharp")) == 1  # cổ điển, làm nét mạnh
    assert level_for("portrait", apply_preset(Settings(), "portrait")) == 3
    assert "RestoreFormer" in describe(5, ss[4]) and "không cần tải" in describe(1, ss[0])


def test_auto_level_by_blur():
    """Tự động chọn mức theo độ mờ đo được: mờ ít dùng model nhẹ, mờ nhiều mới dùng model mặt."""
    from net_anh.levels import auto_level

    def big(**kw):  # mặt rộng 300 px (mặt nhỏ dưới 192 px luôn từ mức 3)
        a = _analysis(True, width=1200, height=900, **kw)
        a.faces[0].bbox = np.array([100, 100, 400, 420, 0.99], np.float32)
        return a

    assert auto_level(big(blur_d=1.0, blur=400.0))[0] == 1
    assert auto_level(big(blur_d=1.4))[0] == 2
    assert auto_level(big(blur_d=1.9))[0] == 3
    assert auto_level(big(blur_d=2.4))[0] == 4
    assert auto_level(big(blur_d=3.0))[0] == 5
    assert auto_level(_analysis(True, blur_d=1.0, blur=400.0))[0] == 3  # mặt nhỏ 120 px
    assert auto_level(_analysis(False, blur_d=3.0))[0] == 3  # không mặt: mức 4–5 chỉ thêm màu
    assert auto_level(_analysis(False, checked=False, blur_d=3.0))[0] == 2  # chưa dò mặt: không tự tải model mặt
    n, why = auto_level(big(blur_d=1.4))
    assert why.startswith("mức 2") and "mờ" in why


def test_protect_faces_keeps_source_on_face():
    from net_anh.pipeline import protect_faces
    src = img(100, 100)
    out = np.zeros((200, 200, 3), np.uint8)
    res = protect_faces(out, src, [np.array([30, 30, 70, 70])], w_ai=0.0)
    up = cv2.resize(src, (200, 200), interpolation=cv2.INTER_CUBIC)
    assert np.abs(res[100, 100].astype(int) - up[100, 100]).max() <= 2  # giữa mặt: ảnh gốc
    assert res[5, 5].max() == 0  # xa mặt: kết quả AI
    assert protect_faces(out, src, [], 0.0) is out


def _photo(h=240, w=320, seed=4):
    """Ảnh giống ảnh chụp (đủ sáng, có vùng sáng gần trắng): nền chuyển màu + vật thể + chi tiết nhỏ."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = np.stack([60 + 120 * xx / w, 80 + 100 * yy / h, 150 - 60 * xx / w], -1)
    cv2.circle(x, (w // 3, h // 2), h // 4, (40, 90, 200), -1)
    cv2.rectangle(x, (w // 2, h // 5), (w - 30, h // 2), (235, 240, 245), -1)
    x += rng.normal(0, 8, x.shape)
    return np.clip(cv2.GaussianBlur(x, (0, 0), 1.0), 0, 255).astype(np.uint8)


def _ev(x, ev):
    lin = np.power(x.astype(np.float32) / 255, 2.2) * 2 ** ev
    return (np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255 + 0.5).astype(np.uint8)


def test_exposure_fixes_dark_and_bright():
    from net_anh import exposure
    gt = _photo()
    dark = _ev(gt, -3.0)
    out, p = exposure.correct(dark)
    assert p.kind == "tối" and 1.5 < p.gain <= 3.0
    assert abs(float(out.mean()) - float(gt.mean())) < abs(float(dark.mean()) - float(gt.mean())) / 3
    bright = _ev(gt, 1.6)
    out, p = exposure.correct(bright)
    assert p.kind == "sáng" and p.gain < 0.9
    assert out.mean() < bright.mean() and out[60:110, 175:275].min() >= 250  # mảng trắng lớn vẫn trắng
    assert exposure.correct(gt)[1].kind == ""  # ảnh đủ sáng: giữ nguyên


def test_exposure_keeps_natural_bright_and_graphics():
    """Trang giấy trắng, logo nền trắng: sáng tự nhiên, không phải quá sáng."""
    from net_anh import exposure
    page = np.full((300, 240, 3), 238, np.uint8)
    for r in range(20, 280, 14):
        page[r:r + 6, 20:220] = 40  # dòng chữ
    assert exposure.plan(page).kind == ""
    logo = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(logo, (150, 150), 80, (30, 60, 200), -1)
    cv2.putText(logo, "LOGO", (70, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    assert exposure.plan(logo).kind == ""


def test_exposure_backlit_lifts_face_not_sky():
    from net_anh import exposure
    rng = np.random.default_rng(2)
    x = np.clip(rng.normal(225, 6, (300, 300, 3)), 0, 255).astype(np.uint8)  # trời sáng
    cv2.ellipse(x, (150, 130), (45, 60), 0, 0, 360, (45, 55, 70), -1)  # mặt tối
    cv2.rectangle(x, (70, 200), (230, 300), (40, 40, 45), -1)  # vai, áo
    box = np.array([105, 70, 195, 190], np.float32)
    out, p = exposure.correct(x, [box])
    assert p.kind == "ngược sáng" and 1.3 < p.lift <= 2.0
    face = (slice(110, 150), slice(130, 170))
    assert out[face].mean() > 1.3 * x[face].mean()
    assert abs(float(out[:40].mean()) - float(x[:40].mean())) < 2  # trời giữ nguyên
    # cùng ảnh nhưng mặt đủ sáng: không sửa
    y = x.copy()
    cv2.ellipse(y, (150, 130), (45, 60), 0, 0, 360, (120, 150, 190), -1)
    assert exposure.plan(y, [box]).kind != "ngược sáng"


def test_process_exposure_note_and_off_switch():
    eng = Enhancer("cpu")
    dark = _ev(_photo(), -3.0)
    out, rep = eng.process(dark, Settings(mode="quick", upscale=1, phone="off"))
    assert any("thiếu sáng" in n for n in rep.notes) and out.mean() > dark.mean() * 1.4
    for s in (Settings(mode="quick", upscale=1, phone="off", exposure=0.0),
              Settings(mode="quick", upscale=1, phone="off", low_light=True)):
        assert not any("thiếu sáng" in n for n in eng.process(dark, s)[1].notes)


def test_very_blurry_warning():
    rng = np.random.default_rng(1)
    sharp = cv2.resize((rng.random((80, 80, 3)) * 255).astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
    a = analyze(cv2.GaussianBlur(sharp, (0, 0), 6))
    assert a.blur_d >= 3.5 and any("mờ rất nặng" in w for w in a.warnings)
    assert not any("mờ rất nặng" in w for w in analyze(sharp).warnings)


# ---------- tông tươi sáng ----------

def _clipped(x):
    return float((x.max(axis=2) >= 254).mean())


def _colorfulness(x):
    b, g, r = (c.astype(np.float32) for c in cv2.split(x))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def _logo():
    logo = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(logo, (150, 150), 80, (30, 60, 200), -1)
    cv2.putText(logo, "LOGO", (70, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    return logo


def test_tone_brighter_more_colorful_no_new_clipping():
    from net_anh import exposure, tone
    x = _ev(_photo(), -0.7)  # hơi u ám
    for amt in (0.6, 1.0):
        out, note = tone.bright_clear(x, amt)
        assert note.startswith("Tông tươi sáng")
        assert exposure.luma(out).mean() > exposure.luma(x).mean() + 0.02
        assert _colorfulness(out) > _colorfulness(x)
        assert _clipped(out) < _clipped(x) + 0.01


def test_tone_soft_shoulder_keeps_bright_saturated_detail():
    """Đỏ rực sáng sau nâng tông vẫn còn chuyển sắc (cắt cứng ở 255: dải đỏ 220–255 còn 3 mức); màu trung tính đúng
    đường cong Y + aY(1 − Y)."""
    from net_anh import tone
    ramp = np.zeros((8, 256, 3), np.uint8)
    ramp[..., 2] = np.arange(256)
    ramp[..., 1] = np.arange(256) // 3
    red = tone.lift(ramp, tone.LIFT_MAX, 0.0)[0, :, 2].astype(int)
    assert (np.diff(red) >= 0).all() and len(set(red[220:])) >= 8 and red[200] < 250
    gray = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, 2)
    g = tone.lift(gray, 0.4, 0.0)[0, :, 0].astype(np.float32) / 255
    v = np.arange(256) / 255
    assert np.abs(g - (v + 0.4 * v * (1 - v))).max() < 2 / 255


def test_tone_skips_graphics_night_and_off():
    from net_anh import tone
    logo, x = _logo(), _photo()
    assert tone.bright_clear(logo, 1.0)[0] is logo
    assert tone.bright_clear(x, 0.0)[0] is x
    assert tone.bright_clear(x, 1.0, graphic=True)[0] is x  # pipeline xét đồ hoạ trên ảnh vào
    night = (x * 0.08).astype(np.uint8)  # ảnh đêm: không nâng tông giữa
    assert tone.plan(night, [], 1.0)[0] == 0.0


def test_process_applies_tone_by_default(monkeypatch):
    from net_anh import exposure
    eng = Enhancer("cpu")
    x = _ev(_photo(), -0.7)
    out, rep = eng.process(x, Settings(mode="quick", upscale=1, phone="off", exposure=0.0))
    assert any(n.startswith("Tông tươi sáng") for n in rep.notes)
    plain, rep0 = eng.process(x, Settings(mode="quick", upscale=1, phone="off", exposure=0.0, tone=0.0))
    assert not any(n.startswith("Tông tươi sáng") for n in rep0.notes)
    assert out.mean() > plain.mean() + 3
    # logo phóng to: ảnh ra có thêm nhiều màu trung gian (như ảnh ra của AI) nhưng vẫn là hình đồ hoạ (xét trên ảnh vào)
    def like_ai(img, s):
        up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return np.clip(up + np.random.default_rng(0).normal(0, 6, up.shape), 0, 255).astype(np.uint8)

    assert not exposure.is_graphic(exposure._small(like_ai(_logo(), None))[0])
    monkeypatch.setattr(eng, "_quick", like_ai)
    _, rep = eng.process(_logo(), Settings(mode="quick", upscale=2, phone="off"))
    assert not any(n.startswith("Tông tươi sáng") for n in rep.notes)


# ---------- ảnh nhỏ ----------

def test_auto_upscale_small_images():
    from net_anh.levels import auto_upscale, level_settings
    assert Settings().upscale == 0
    assert (auto_upscale(300, 200), auto_upscale(999, 600), auto_upscale(1000, 800), auto_upscale(4000, 3000)) == (4, 4, 2, 2)
    eng = Enhancer("cpu")
    small = _photo(150, 200)
    out, rep = eng.process(small, Settings(mode="quick", phone="off"))
    assert out.shape[:2] == (600, 800) and any(n.startswith("Ảnh nhỏ") for n in rep.notes)
    out, rep = eng.process(_photo(560, 1040), Settings(mode="quick", phone="off"))
    assert out.shape[:2] == (1120, 2080) and not any(n.startswith("Ảnh nhỏ") for n in rep.notes)
    out, _ = eng.process(small, Settings(mode="quick", upscale=2, phone="off"))  # chọn tay thì giữ nguyên
    assert out.shape[:2] == (300, 400)
    _, s, why = recommend(analyze(small), Settings())
    assert s.upscale == 0 and any("phóng 4×" in w for w in why)
    assert level_settings(5, Settings(), _analysis(False)).upscale == 0  # mức 5 không ép 2× khi đang tự động
    assert any("Ảnh rất nhỏ" in w for w in analyze(small).warnings)
    assert not any("Ảnh rất nhỏ" in w for w in analyze(_photo(420, 560)).warnings)



# ---------- model mới (RestoreFormer++, DAT-2 / RealPLKSR qua spandrel) và kiểm cấu hình máy ----------

def test_restoreformer_pp_layout_and_forward():
    """RestoreFormer++: bộ giải mã chú ý thêm ở tầng 32 px (khớp checkpoint gốc), chạy được với ảnh bất kỳ cỡ 2^5."""
    import torch
    from net_anh.archs.restoreformer_arch import RestoreFormer
    torch.manual_seed(0)
    rf, pp = RestoreFormer(), RestoreFormer(head_size=4, ex_multi_scale_num=1)
    assert [len(u.attn) for u in rf.decoder.up] == [0, 0, 0, 0, 0, 3]
    assert [len(u.attn) for u in pp.decoder.up] == [0, 0, 0, 0, 3, 3]
    assert [len(d.attn) for d in pp.encoder.down] == [0, 0, 0, 0, 0, 2]
    with torch.no_grad():
        y = pp.eval()(torch.zeros(1, 3, 64, 64))[0]
    assert y.shape == (1, 3, 64, 64)
    if weights.status("restoreformer_pp") == "ready":  # checkpoint thật: nạp đúng từng khoá
        ck = torch.load(weights.weights_dir() / weights.MANIFEST["restoreformer_pp"].file, map_location="cpu",
                        weights_only=True)
        pp.load_state_dict({k.removeprefix("vqvae."): v for k, v in ck["state_dict"].items()}, strict=True)


def test_new_models_in_manifest_and_needed():
    from net_anh.levels import BG_MODELS, FACE_MODELS, models_needed
    for mid in ("restoreformer_pp", "dat2_web", "plksr_web"):
        m = weights.MANIFEST[mid]
        assert m.url.startswith("https://github.com/") and len(m.sha256) == 64 and m.size > 1e7
        assert "Apache" in m.license or "CC-BY-4.0" in m.license
    assert "dat2" in BG_MODELS and "plksr" in BG_MODELS and "restoreformer_pp" in FACE_MODELS
    need = models_needed(Settings(mode="portrait", face_model="restoreformer_pp", sr_model="dat2"))
    assert "dat2_web" in need and "restoreformer_pp" in need and not any(m.startswith("esrgan") for m in need)
    assert models_needed(Settings(mode="upscale", sr_model="plksr")) == ["plksr_web"]
    assert not set(weights.RUNTIME) & {"restoreformer_pp", "dat2_web", "plksr_web"}  # chỉ tải khi chọn


def _tiny_spandrel_model(path):
    """Mạng SRVGG compact rất nhỏ (spandrel nhận ra kiến trúc, 4×) thay cho DAT-2 140 MB khi test."""
    import torch
    from safetensors.torch import save_file
    from net_anh.archs import SRVGGNetCompact
    torch.manual_seed(0)
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=4, act_type="prelu")
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(path))


def test_extra_sr_model_runs_tiled_and_falls_back(tmp_path, monkeypatch):
    """Model nền nạp bằng spandrel: chạy chia ô (chồng 16 px) ra đúng cỡ; lỗi trên thiết bị thì dùng compact, ghi lý do."""
    import dataclasses
    import torch
    f = tmp_path / "tiny.safetensors"
    _tiny_spandrel_model(f)
    info = dataclasses.replace(weights.MANIFEST["dat2_web"], file=f.name, size=f.stat().st_size, tile=64)
    monkeypatch.setitem(weights.MANIFEST, "dat2_web", info)
    eng = Enhancer("cpu")
    monkeypatch.setattr(eng, "_ensure", lambda mid: f if mid == "dat2_web" else weights.ensure(mid))
    x = _photo(90, 120)
    out, rep = eng.process(x, Settings(mode="upscale", sr_model="dat2", upscale=2, deblur=1, tone=0, phone="off",
                                       exposure=0))
    assert out.shape[:2] == (180, 240) and rep.mode == "upscale"
    net, dtype = eng._sr_extra["dat2_web"]
    assert dtype == torch.float32 and next(net.parameters()).device.type == "cpu"
    # nguyên ảnh so với chia ô 64 px: chỉ khác ở mép ô (mạng nhỏ, chồng 16 px là đủ)
    whole = eng._esrgan_run(x, 4, 0, net)
    tiled = eng._esrgan_run(x, 4, 64, net, pad=16)
    assert np.abs(whole.astype(int) - tiled.astype(int)).max() <= 2
    if not AI_READY:
        return

    def broken(_mid):
        raise RuntimeError("không hỗ trợ trên thiết bị này")
    monkeypatch.setattr(eng, "_load_extra_sr", broken)
    out, rep = eng.process(x, Settings(mode="upscale", sr_model="dat2", upscale=2, tone=0, phone="off", exposure=0))
    assert out.shape[:2] == (180, 240) and any("dùng Real-ESRGAN compact thay" in n for n in rep.notes)
    assert "thay cho" in rep.bg_model and "compact" in rep.bg_model


def test_hardware_estimate_and_warnings():
    from net_anh import hardware
    from net_anh.hardware import Gpu, Machine
    m = hardware.machine()
    assert m.threads >= 1 and m.cpu and isinstance(m.summary(), str)
    cpu = Machine("Test CPU", 4, 16.0, 8.0, (), False)
    t_c = hardware.estimate("compact", "gfpgan", 1600, 1200, 1, kind="cpu")
    t_d = hardware.estimate("dat2", "restoreformer_pp", 1600, 1200, 1, kind="cpu")
    assert t_d > 20 * t_c and hardware.estimate("dat2", None, 1600, 1200, kind="cuda") < t_d / 10
    assert hardware.estimate("dat2", None, 1600, 1200, blur_d=2.0, overhead=False) == pytest.approx(
        hardware.estimate("dat2", None, 800, 600, overhead=False))  # ảnh mờ 2× được thu nhỏ trước khi vào mạng
    # cả bước ngoài mạng AI: mức 5 ảnh 800×1000 một mặt đo thật 13,8 giây trên CPU 4 luồng
    assert 11 < hardware.estimate("compact", "gfpgan", 800, 1000, 1, kind="cpu", threads=4) < 17
    assert hardware.estimate("compact", None, 800, 1000, overhead=False) < hardware.estimate("compact", None, 800, 1000)
    assert hardware.warnings("compact", "gfpgan", 1600, 1200, 1, 1.0, "cpu", 4, cpu) == []
    w = hardware.warnings("dat2", "restoreformer_pp", 1600, 1200, 1, 1.0, "cpu", 4, cpu)
    assert len(w) == 1 and "CPU" in w[0] and "DAT-2" in w[0] and "phút" in w[0] and "nhẹ hơn" in w[0]
    w = hardware.warnings("compact", "gfpgan", 4000, 3000, 1, 1.0, "cpu", 4, cpu)  # ảnh 12 MP, model nhẹ nhất
    assert len(w) == 1 and "CPU" in w[0] and "nhẹ hơn" not in w[0]
    nv = Machine("Test CPU", 8, 16.0, 8.0, (Gpu("NVIDIA GeForce RTX 3060", 12.0, False),), False)
    assert any("chỉ chạy CPU" in x for x in hardware.warnings("compact", None, 800, 600, 0, 1.0, "cpu", 8, nv))
    small = Machine("Test CPU", 8, 6.0, 2.0, (Gpu("NVIDIA GeForce GTX 1050", 2.0, True),), True)
    w = hardware.warnings("dat2", None, 1600, 1200, 0, 1.0, "cuda", 8, small)
    assert any("GPU chỉ có 2 GB" in x for x in w) and any("6 GB RAM" in x for x in w)
    assert hardware.fmt_seconds(0.4) == "dưới 1 giây" and hardware.fmt_seconds(45) == "45 giây"
    assert hardware.fmt_seconds(600) == "10 phút"


def test_fp16_overflow_reruns_tile_in_fp32():
    """fp16 tràn số trên GPU (ra NaN hoặc ô đen): ô đó chạy lại fp32, mạng chuyển hẳn sang fp32, không còn mảng đen."""
    import torch

    class Flaky(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            up = torch.nn.functional.interpolate(x * self.w, scale_factor=2, mode="nearest")
            if self.w.dtype == torch.float16 and self.calls == 2:  # giả lập: ô thứ hai ra NaN hoặc toàn 0
                return up * self.bad
            return up

    eng = Enhancer("cpu")
    x = np.full((64, 128, 3), 180, np.uint8)
    for bad in (float("nan"), 0.0):
        net = Flaky().half()
        net.bad = bad
        eng._fp32_fallback = eng._nonfinite = False
        out = eng._esrgan_run(x, 2, 32, net, dtype=torch.float16, pad=0)
        assert out.shape == (128, 256, 3) and out.min() >= 170
        assert eng._fp32_fallback and net.w.dtype == torch.float32 and not eng._nonfinite


def test_resize_long_output_sizes():
    from net_anh.pipeline import OUT_SIZES, resize_long
    x = np.zeros((1000, 3000, 3), np.uint8)
    assert resize_long(x, OUT_SIZES["1k"]).shape == (640, 1920, 3)
    assert resize_long(x, OUT_SIZES["4k"]).shape == (1280, 3840, 3)
    assert resize_long(x, OUT_SIZES["goc"]) is x
    assert resize_long(np.zeros((2000, 1000, 3), np.uint8), 2560).shape == (2560, 1280, 3)


def test_white_balance_keeps_natural_skin(tmp_path):
    """Ảnh thẻ phông xanh, áo cam: ước lượng nguồn sáng bị lệch (ám tím hồng giả). Da tự nhiên thì không được làm da
    hồng thành vàng nhợt: góc màu da đổi không quá SKIN_KEEP_DEG."""
    from net_anh import wb
    h, w = 400, 360
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (215, 170, 60)  # phông xanh lơ (BGR)
    img[260:] = (90, 120, 230)  # áo cam đỏ
    cv2.ellipse(img, (180, 150), (70, 90), 0, 0, 360, (150, 165, 215), -1)  # da hồng (góc màu ~40°)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    box = [np.array([110, 60, 250, 240], np.float32)]
    h0 = wb.skin_hue(img, box)
    if h0 is None:
        pytest.skip("mặt giả không đủ điểm da")
    out, notes = wb.correct(img, 0.6, 0.0, 0.0, box)
    h1 = wb.skin_hue(out, box)
    assert h1[0] <= max(h0[0], wb.SKIN_CENTER + wb.SKIN_CROSS) + 1, (h0, h1, notes)


def test_skin_tone_keeps_rosy_skin():
    """0.8.1: da hồng hào (góc màu ~35–44°) là sắc da đẹp, không bị kéo sang vàng."""
    from net_anh import wb
    rosy = np.zeros((200, 200, 3), np.uint8)
    rosy[:] = (120, 135, 215)
    out, why = wb.skin_tone(rosy, [np.array([40, 30, 160, 170])], 1.0)
    h = wb.skin_hue(rosy, [np.array([40, 30, 160, 170])])
    assert wb.SKIN_RED_LO <= h[0] <= wb.SKIN_HUE_LO and "da đỏ" not in why


# ---------- preset theo đối tượng ----------

def test_preset_file_roundtrip_and_validation(tmp_path):
    import json
    from net_anh import facestyle as fs
    text = fs.dumps([fs.DEFAULT_SET], app="test")
    sets, warn = fs.loads(text)
    assert not warn and sets[0].styles == fs.DEFAULT_SET.styles and not sets[0].builtin
    d = json.loads(text)
    d["sets"][0]["styles"]["child"]["smooth"] = 5
    d["sets"][0]["styles"]["man"]["lạ"] = 1
    d["sets"][0]["styles"]["woman"]["warm"] = "abc"
    sets, warn = fs.loads(json.dumps(d))
    assert sets[0].styles["child"].smooth == 1.0 and len(warn) >= 2
    for bad in (b"not json", json.dumps({"format": "khac"}), json.dumps({"format": fs.FORMAT, "version": 99, "sets": []}),
                b"x" * (fs.MAX_FILE_BYTES + 1)):
        with pytest.raises(fs.PresetError):
            fs.loads(bad)


def test_preset_library_import_export(tmp_path):
    from net_anh import facestyle as fs
    lib = fs.Library(tmp_path / "lib")
    assert [s.id for s in lib.sets()] == ["default", "trong-sang", "da-dep-the"] and lib.active().builtin
    with pytest.raises(fs.PresetError):
        lib.save(fs.DEFAULT_SET)
    mine = fs.DEFAULT_SET.copy("Của tôi")
    mine.styles["woman"] = fs.FaceStyle(smooth=0.9)
    lib.save(mine)
    lib.set_active(mine.id)
    f = lib.export([mine.id], tmp_path / ("x" + fs.EXT))
    other = fs.Library(tmp_path / "other")
    got, _ = other.import_file(f)
    got2, _ = other.import_file(f)  # nạp lại: mã mới, tên thêm (2)
    names = [s.name for s in other.sets()]
    assert "Của tôi" in names and "Của tôi (2)" in names and got[0].id != got2[0].id
    assert other.get(got[0].id).styles["woman"].smooth == 0.9
    lib.delete(mine.id)
    assert lib.active_id() == "default"


def test_subject_classify_thresholds():
    from net_anh import subject as sj
    p = np.zeros(81, np.float32)
    p[5] = 1
    assert sj.classify(p, 0.9).subject == "child"
    p[:] = 0
    p[70] = 1
    assert sj.classify(p, 0.1).subject == "elder"
    p[:] = 0
    p[30] = 1
    assert sj.classify(p, 0.9).subject == "man" and sj.classify(p, 0.1).subject == "woman"
    assert sj.classify(p, 0.5).subject == "adult"


def test_face_style_weights_and_apply():
    from net_anh import facestyle as fs
    st = fs.FaceStyle(keep=0.5, detail=0.5)
    a, d = st.weights(0.6, 0.6)
    assert abs(a - 0.3) < 1e-6 and d > a
    face = np.full((512, 512, 3), (120, 140, 190), np.uint8)
    labels = np.ones((512, 512), np.uint8)
    out = fs.apply(face, fs.FaceStyle(bright=1.0, warm=0.5), fs.skin_mask(labels))
    L0 = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)[256, 256, 0]
    L1 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[256, 256, 0]
    assert L1 > L0 and np.array_equal(out[5, 5], face[5, 5])  # mép ảnh (ngoài mặt) không đổi
    assert fs.apply(face, fs.FaceStyle(), fs.skin_mask(labels)) is face


def test_models_needed_includes_age_gender():
    from net_anh.levels import models_needed
    assert "age_cydral" in models_needed(Settings(mode="portrait"))
    assert "age_cydral" not in models_needed(Settings(mode="portrait", subject="woman"))
    assert models_needed(Settings(mode="quick")) == []


@pytest.mark.skipif(not AI_READY or weights.status("age_cydral") != "ready", reason="chưa tải model")
def test_pipeline_applies_subject_preset():
    """Mặt thật (UTKFace không có trong repo: dùng ảnh mẫu skimage astronaut, nữ ~45 tuổi)."""
    import skimage.data as sd
    im = cv2.cvtColor(sd.astronaut(), cv2.COLOR_RGB2BGR)
    eng = Enhancer("cpu")
    base = Settings(mode="quick", upscale=1, tone=0, phone="off", exposure=0)
    off, r0 = eng.process(im, base.with_(subject="off"))
    on, r1 = eng.process(im, base.with_(subject="auto"))
    assert not r0.subjects and r1.subjects and r1.styled
    assert np.abs(on.astype(int) - off).mean() > 0.05
    forced, r2 = eng.process(im, base.with_(subject="child"))
    assert "chọn tay" in r2.subjects[0]


def test_all_modules_compile():
    """0.8.1 bị lỗi cú pháp ở gui.py mà các test khác không nhập gui: biên dịch mọi file .py của app và công cụ."""
    import py_compile
    import os
    root = Path(__file__).resolve().parents[1]
    for f in list((root / "net_anh").rglob("*.py")) + list((root / "tools").glob("*.py")) + [root / "app.py"]:
        py_compile.compile(str(f), doraise=True)


def test_gui_builds_offscreen(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from net_anh import gui
    w = gui.MainWindow()
    w._refresh_history()
    s = w.settings()
    assert s.subject == "auto" and set(s.face_styles) >= {"child", "man", "woman", "elder"}
    gui.PresetDialog(w, w.preset_lib)
    w.close()
    app.processEvents()


def test_gui_two_modes_share_settings(monkeypatch, tmp_path):
    """0.12: Đơn giản và Chuyên sâu dùng chung một bộ thông số; loại ảnh bật đúng phần riêng; ô tìm lọc nhóm."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from net_anh import gui
    w = gui.MainWindow()
    w.set_ui_mode("simple")
    w._simple_pick(kind="old")
    s = w.settings()
    assert s.repair > 0 and s.fade > 0 and s.bg_color is None
    w._simple_pick(kind="id", power=5)
    s = w.settings()
    assert s.bg_color is not None and s.repair == 0 and w.level_group.checkedId() == 5
    w._level_pick(6)
    assert w.level_group.checkedId() == 6 and w.settings().face_model == "restoreformer_pp"
    w.set_ui_mode("pro")  # đổi chế độ không đổi thông số; hai thanh mức nét cùng trạng thái
    assert w.settings().bg_color is not None and w.level_group.checkedId() == 6 and not w.left.isHidden()
    assert [b[6].isChecked() for _, b in w._level_bars] == [True, True]
    w._level_pick(3)  # chọn ở Chuyên sâu, thanh của Đơn giản cũng đổi
    assert all(b[3].isChecked() and not a.isChecked() for a, b in w._level_bars)
    w._level_pick(None)
    assert all(a.isChecked() for a, _ in w._level_bars)
    w.search.setText("bóng da")
    shown = [sec.title for sec in w.sections if not sec.isHidden()]
    assert shown == ["Khuôn mặt"]
    w.search.setText("")
    assert all(not sec.isHidden() for sec in w.sections)
    w.set_ui_mode("simple")
    w._simple_pick(kind="dark", power="auto")
    assert w.settings().low_light
    w.close()
    app.processEvents()


# ---------- đổi màu nền đồng màu ----------

def _id_photo(h=300, w=240, bg=(200, 140, 60), seed=1):
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = bg
    img += np.linspace(-10, 10, w, dtype=np.float32)[None, :, None]  # bóng đổ ngang của phông
    cv2.circle(img, (200, 60), 7, (150, 100, 30), -1)  # vết bẩn rời trên phông
    cv2.ellipse(img, (120, 130), (55, 70), 0, 0, 360, (150, 170, 220), -1)  # mặt (da)
    cv2.ellipse(img, (120, 85), (60, 45), 0, 180, 360, (30, 30, 40), -1)  # tóc
    cv2.rectangle(img, (40, 240), (200, h), (60, 60, 200), -1)  # áo đỏ chạm đáy
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def test_backdrop_replaces_uniform_background_only():
    from net_anh import backdrop
    img = _id_photo()
    bd = backdrop.detect(img)
    assert bd is not None and bd.uniform > 0.9
    out, note = backdrop.replace(img, (255, 255, 255), bd)
    assert "Đã đổi màu nền" in note
    assert out[10, 10].min() >= 250 and out[150, 20].min() >= 250  # nền thành trắng, kể cả chỗ có bóng đổ
    assert tuple(out[60, 200]) == (255, 255, 255)  # vết bẩn rời trên phông cũng thành nền
    for y, x in ((140, 120), (270, 100)):  # mặt, áo giữ nguyên
        assert np.abs(out[y, x].astype(int) - img[y, x]).max() <= 3
    a = backdrop.alpha_matte(img, bd)
    assert 0.2 < a.mean() < 0.8 and a[140, 120] > 0.99 and a[10, 10] < 0.01


def test_backdrop_refuses_non_uniform_background():
    from net_anh import backdrop
    img = _photo(240, 300)  # ảnh nhiễu ngẫu nhiên
    assert backdrop.detect(img) is None
    out, note = backdrop.replace(img, (255, 255, 255))
    assert out is img and "không đồng màu" in note
    assert backdrop.parse_color("#ff0000") == (0, 0, 255) and backdrop.parse_color("trang") == (255, 255, 255)
    with pytest.raises(ValueError):
        backdrop.parse_color("không phải màu")


def test_pipeline_background_option():
    img_ = _id_photo()
    out, rep = Enhancer("cpu").process(img_, Settings(mode="quick", upscale=1, tone=0, phone="off", exposure=0,
                                                      bg_color=(255, 255, 255)))
    assert out[10, 10].min() >= 250 and any("đổi màu nền" in n.lower() for n in rep.notes)


# ---------- 0.10.3: sửa lỗi review mã ----------

def test_load_float_tiff_normalized_0_1(tmp_path):
    """Ảnh TIFF số thực (mode F) chuẩn hoá 0..1 trước đây chia cho 255 nên ra ảnh 8 bit chỉ còn 0 hoặc 1."""
    g = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (32, 1))
    p = tmp_path / "ramp_f.tif"
    Image.fromarray(g, mode="F").save(p)
    x = load_image(p)
    assert x.dtype == np.uint8 and x.shape == (32, 256, 3)
    row = x[0, :, 0].astype(int)
    assert row[0] == 0 and row[-1] == 255
    assert len(np.unique(row)) > 200 and np.all(np.diff(row) >= 0)  # giữ đủ dải xám, không nhảy bậc


def test_load_float_tiff_0_255_and_nan(tmp_path):
    g = np.tile(np.linspace(0.0, 255.0, 256, dtype=np.float32), (8, 1))
    g[0, 5] = np.nan
    p = tmp_path / "ramp_f255.tif"
    Image.fromarray(g, mode="F").save(p)
    x = load_image(p)
    assert abs(int(x[1, 128, 0]) - 128) <= 1 and x[1, -1, 0] == 255 and x[0, 5, 0] == 0


def test_process_very_long_narrow_strip():
    """Ảnh dải rất dài và hẹp vượt MAX_SIDE: trước đây cạnh ngắn làm tròn thành 0 nên cv2.resize lỗi."""
    from net_anh.analysis import MAX_SIDE
    from net_anh.pipeline import resize_long
    strip = img(MAX_SIDE * 5, 1)
    small = resize_long(strip, MAX_SIDE)
    assert small.shape[:2] == (MAX_SIDE, 1)
    out, rep = Enhancer("cpu").process(strip, Settings(mode="quick", upscale=1))
    assert out.shape[0] >= MAX_SIDE and out.shape[1] >= 1 and any(str(MAX_SIDE) in n for n in rep.notes)


class _FakeHelper:
    """Thay FaceRestoreHelper: 3 mặt cố định, không cần model dò mặt."""
    use_parse = False
    upscale_factor = 1

    def __init__(self):
        self.restored_faces, self.inverse_affine_matrices = [], []

    def read_image(self, im):
        self.input_img = im

    def get_face_landmarks_5(self, **k):
        self.det_faces, self.all_landmarks_5 = [], []
        for x in (20, 140, 260):
            self.det_faces.append(np.array([x, 40, x + 100, 160, 0.99]))
            self.all_landmarks_5.append(np.array([[x + 30, 80], [x + 70, 80], [x + 50, 100], [x + 35, 130], [x + 65, 130]], float))

    def align_warp_face(self):
        self.cropped_faces = [np.full((512, 512, 3), 100 + 30 * i, np.uint8) for i in range(len(self.det_faces))]

    def add_restored_face(self, f):
        self.restored_faces.append(f)

    def get_inverse_affine(self, *a):
        pass


@pytest.mark.parametrize("fail", [(1, 2, 3), (2,), ()])
def test_face_count_excludes_model_failures(monkeypatch, fail):
    """Mặt lỗi model dùng lại mặt gốc: vẫn ghép lại (đủ danh sách) nhưng không được đếm là đã phục hồi."""
    from net_anh.pipeline import Report
    eng = Enhancer("cpu")
    helper = _FakeHelper()
    calls = {"n": 0}

    def net(t, return_rgb=False):
        calls["n"] += 1
        if calls["n"] in fail:
            raise RuntimeError("CUDA error")
        return [t[0] * 0 + 0.2]

    pasted = {}

    def paste(h, bg):
        pasted["faces"] = list(h.restored_faces)
        return bg

    monkeypatch.setattr(eng, "_background", lambda im, s, rep: (im.copy(), 1.0))
    monkeypatch.setattr(eng, "_load_helper", lambda up: helper)
    monkeypatch.setattr(eng, "_load_face_net", lambda name: net)
    monkeypatch.setattr(eng, "_paste_faces", paste)
    s = Settings(mode="portrait", upscale=1, subject="off", protect_faces=False)
    rep = Report()
    eng._process_ai(img(200, 400), s, rep)
    assert rep.faces_found == 3
    assert rep.faces_restored == 3 - len(fail) and rep.faces_kept == len(fail)
    assert len(pasted["faces"]) == 3  # vẫn ghép đủ 3 mặt
    for i in fail:  # mặt lỗi giữ đúng ảnh gốc, không bị khớp màu hay trộn
        assert np.array_equal(pasted["faces"][i - 1], helper.cropped_faces[i - 1])
    assert f"làm nét {3 - len(fail)}/3" in rep.summary()


def test_batch_tally_separates_ai_from_fallback(monkeypatch):
    """Tổng kết hàng loạt không chỉ đếm tệp xuất được: ảnh chỉ qua xử lý nhanh do AI lỗi phải tách riêng."""
    from net_anh.pipeline import BatchTally, Report
    eng = Enhancer("cpu")
    tally = BatchTally(4)

    out, rep = eng.process(img(), Settings(mode="quick", upscale=1))  # người dùng chọn Nhanh: không phải lỗi
    assert rep.kind == "quick" and not rep.ai_fallback
    tally.add(rep, "a.jpg")

    monkeypatch.setattr(eng, "_process_ai", lambda *a, **k: (_ for _ in ()).throw(weights.ModelError("không có mạng")))
    out, rep = eng.process(img(), Settings(mode="portrait", upscale=1))
    assert rep.ai_fallback and rep.kind == "fallback" and "AI lỗi" in rep.summary()
    tally.add(rep, "b.jpg")

    tally.add(Report(mode="portrait", faces_found=2, faces_restored=1, faces_kept=1), "c.jpg")
    tally.add(Report(mode="portrait", faces_found=1, faces_restored=1), "d.jpg")
    tally.fail()
    c = tally.counts
    assert (c["ai"], c["partial"], c["fallback"], c["quick"], tally.failed) == (1, 1, 1, 1, 1)
    assert tally.needs_attention and tally.exported == 4
    s = tally.summary()
    assert "CHỈ XỬ LÝ NHANH vì AI lỗi: 1" in s and "b.jpg" in s and "AI đủ: 1" in s


def test_group_photo_defaults():
    """Ảnh nhóm: mặc định xử lý tới 24 mặt (trước đây 5, mặt còn lại bị mạng nền vẽ thành mặt nạ); mặt quá nhỏ bị lọc."""
    from net_anh.settings import Settings
    from net_anh import pipeline
    assert Settings().max_faces == 24
    assert pipeline._eye_dist(np.array([[10, 10], [22, 10], [16, 16], [12, 20], [20, 20]])) == 12.0
    assert pipeline.AI_MIN_EYE <= 12


def test_progress_track_and_warnings():
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from net_anh import gui

    now = [0.0]
    t = gui.ProgressTrack(clock=lambda: now[0])
    t.start(60)
    t.feed("Đang làm nét toàn ảnh bằng DAT-2 trên CPU…")
    t.feed("DAT-2: 40%")
    assert t.step == "Làm nét nền 40%" and 30 <= t.pct <= 40
    t.feed("Đang làm nét khuôn mặt 2/5…")
    assert t.step == "Mặt 2/5" and t.pct > 70
    assert t.warning() == ""
    now[0] = 400.0
    assert "lâu hơn ước tính" in t.warning()
    t.start(0)
    now[0] = 600.0
    assert "chưa có tiến triển" in t.warning()


def test_vietnamese_output_survives_cp1252_console(tmp_path):
    """0.13.1: console Windows cp1252 làm print chữ Việt sập app (UnicodeEncodeError). Chạy các lối vào thật với
    PYTHONIOENCODING=cp1252 và không có PYTHONUTF8: không được sập, chữ Việt ra đúng UTF-8."""
    import subprocess
    import sys

    import os
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8",)}
    env["PYTHONIOENCODING"] = "cp1252"
    env["NET_ANH_WEIGHTS"] = str(tmp_path)
    env["PYTHONPATH"] = str(root)
    code = ("import net_anh,sys; print('Đã huỷ mọi điện thoại ủ ề ọ'); print('lỗi', file=sys.stderr); "
            "from net_anh import server; "
            "import io; from net_anh.selftest import dispatch; sys.exit(dispatch(['--tai-model','%s','--progress-file','%s']) if False else 0)"
            % (tmp_path, tmp_path / "p.log"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, cwd=root)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "Đã huỷ mọi điện thoại ủ ề ọ" in r.stdout.decode("utf-8")
    # máy chủ dòng lệnh: in thông báo tiếng Việt khi cổng bị chiếm, không sập vì mã hoá
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    r = subprocess.run([sys.executable, str(root / "app.py"), "--may-chu", "--dong-lenh", "--host", "127.0.0.1",
                        "--port", str(port), "--no-discovery"],
                       capture_output=True, env=env, cwd=root, timeout=110)
    s.close()
    out = r.stdout.decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in r.stderr.decode("utf-8", "replace")
    assert r.returncode == 2 and "Không mở được cổng" in out, (r.returncode, out, r.stderr[-300:])
# ---------- khử bóng loáng da (0.13.0) ----------

def _oily_face(h=320, w=260, seed=3):
    """Mảng da (màu da châu Á, có vân) với vệt bóng gần trắng ở giữa (trán/mũi) và một gò má sáng nhưng đủ màu."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lab = np.zeros((h, w, 3), np.float32)
    lab[..., 0] = 62 + 6 * np.cos((xx - w / 2) / w * np.pi)  # khối mặt nhẹ
    lab[..., 1], lab[..., 2] = 14.0, 18.0
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.0) * 4.0
    lab[..., 0] += tex
    spec = np.exp(-(((xx - 130) / 18) ** 2 + ((yy - 120) / 30) ** 2))  # vệt bóng dầu: sáng lên, nhạt màu
    lab[..., 0] += 26 * spec
    lab[..., 1] *= 1 - 0.7 * spec
    lab[..., 2] *= 1 - 0.7 * spec
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (130, 160), (110, 150), 0, 0, 360, 1.0, -1)
    return img, cv2.GaussianBlur(mask, (0, 0), 3), spec


def test_deshine_reduces_shine_keeps_texture_and_outside():
    from net_anh import deshine
    img, m, spec = _oily_face()
    out = deshine.remove_shine(img, m, 0.8, face_px=220)
    lab0 = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lab1 = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    hot = spec > 0.6
    assert lab1[..., 0][hot].mean() < lab0[..., 0][hot].mean() - 8  # vệt bóng tối đi rõ
    c0 = np.hypot(lab0[..., 1], lab0[..., 2])[hot].mean()
    c1 = np.hypot(lab1[..., 1], lab1[..., 2])[hot].mean()
    assert c1 > c0 + 2  # trả lại màu da
    # vân da (tầng mịn) trong vùng bóng còn ≥ 60 %
    hp = lambda L: L - cv2.GaussianBlur(L, (0, 0), 2.0)
    assert hp(lab1[..., 0])[hot].std() > 0.6 * hp(lab0[..., 0])[hot].std()
    # da không bóng gần như không đổi, ngoài mặt nạ giữ nguyên tuyệt đối
    calm = (spec < 0.02) & (m > 0.99)
    assert np.abs(lab1[..., 0][calm] - lab0[..., 0][calm]).mean() < 1.0
    assert np.array_equal(out[m <= 0], img[m <= 0])
    assert np.array_equal(deshine.remove_shine(img, m, 0.0, 220), img)
    assert deshine.shine_fraction(img, m, 220) > deshine.shine_fraction(out, m, 220)


def test_deshine_default_off_and_presets():
    from net_anh.settings import PRESETS, Settings, apply_preset
    assert Settings().deshine == 0.0
    assert all(apply_preset(Settings(), k).deshine == 0.0 for k in PRESETS)


def test_preset_file_matte_v2_and_reads_v1():
    from net_anh import facestyle as fs
    v1 = json.dumps({"format": fs.FORMAT, "version": 1, "sets": [
        {"name": "cũ", "styles": {"woman": {"smooth": 0.4, "glow": 0.2}}}]})
    sets, warn = fs.loads(v1)
    assert sets[0].styles["woman"].matte == 0.0 and not warn
    s = fs.PresetSet("mới", {"man": fs.FaceStyle(matte=0.3), "child": fs.FaceStyle(matte=-2)})
    text = fs.dumps([s])
    assert json.loads(text)["version"] == fs.VERSION
    back, warn = fs.loads(text)
    assert back[0].styles["man"].matte == pytest.approx(0.3)
    assert back[0].styles["child"].matte == -1.0 and warn  # ngoài khoảng: kẹp và báo


def test_gui_deshine_checkbox_shared(monkeypatch, tmp_path):
    """0.13: ô Khử bóng da ở Đơn giản và Chuyên sâu là một thông số; đổi loại ảnh, mức nét không làm mất."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from net_anh import gui
    w = gui.MainWindow()
    w.set_ui_mode("simple")
    assert w.settings().deshine == 0.0 and not w.cb_deshine2.isChecked()
    w.cb_deshine2.setChecked(True)
    assert w.cb_deshine.isChecked() and w.settings().deshine == pytest.approx(gui.DESHINE_DEFAULT)
    w._simple_pick(kind="portrait", power=4)
    w._simple_pick(kind="auto", power="auto")
    assert w.settings().deshine == pytest.approx(gui.DESHINE_DEFAULT)
    w.set_ui_mode("pro")
    assert w.sl_deshine.isEnabled()
    w.sl_deshine.set(0.85)
    assert w.settings().deshine == pytest.approx(0.85)
    w.cb_deshine.setChecked(False)
    assert w.settings().deshine == 0.0 and not w.cb_deshine2.isChecked() and not w.sl_deshine.isEnabled()
    w.cb_deshine.setChecked(True)  # bật lại: nhớ cường độ cũ
    assert w.settings().deshine == pytest.approx(0.85)
    w._apply_settings(w.settings().with_(deshine=0.0))  # lịch sử, hoàn tác: theo đúng thông số đã lưu
    assert not w.cb_deshine.isChecked()
    w.search.setText("bóng dầu")
    assert [sec.title for sec in w.sections if not sec.isHidden()] == ["Khuôn mặt"]
    w.close()
    app.processEvents()


def test_gpu_vendors_and_advice():
    """0.14.1: nhận diện hãng card từ tên, lời khuyên theo hãng, đọc danh sách card Windows (bỏ card ảo)."""
    from net_anh import hardware

    G = hardware.Gpu
    assert G("NVIDIA GeForce GTX 1660 Ti", 6, False).vendor == "nvidia"
    assert G("AMD Radeon RX 6600", 8, False).vendor == "amd"
    assert G("Intel(R) Arc(TM) A750 Graphics", 8, False).vendor == "intel"
    assert G("Intel(R) UHD Graphics 770", 0, False).vendor == "intel"
    assert "GPU Intel" in hardware.advice(G("Intel(R) Iris(R) Xe Graphics", 0, False))
    assert "chưa hỗ trợ card AMD" in hardware.advice(G("AMD Radeon(TM) Graphics", 0, False))
    assert "GPU NVIDIA" in hardware.advice(G("NVIDIA GeForce RTX 3060", 12, False))
    js = ('[{"Name":"Intel(R) UHD Graphics 630","AdapterRAM":1073741824},{"Name":"NVIDIA GeForce GTX 1050","AdapterRAM":4293918720},'
          '{"Name":"Microsoft Basic Display Adapter","AdapterRAM":0},{"Name":"Parsec Virtual Display Adapter","AdapterRAM":null}]')
    got = hardware.parse_adapters(js)
    assert [g.name for g in got] == ["Intel(R) UHD Graphics 630", "NVIDIA GeForce GTX 1050"]
    assert got[0].vram_gb == 1.0 and got[1].vram_gb == 0.0  # AdapterRAM 4 GB bị cắt ở 32 bit: coi là không rõ
    assert hardware.parse_adapters("không phải json") == []
    m = hardware.Machine("cpu", 4, 8, 4, (G("AMD Radeon RX 6600", 8, False), G("Intel UHD", 0, False)), False)
    assert m.unused_gpu.vendor == "intel"  # ưu tiên Intel (có gói xpu) hơn AMD
    hardware.machine.cache_clear()
    import net_anh.hardware as hw
    orig = hw.machine
    hw.machine = lambda: m
    try:
        assert "GPU Intel" in hardware.warnings("compact", None, 1000, 1000, 0, 1.0, "cpu", 4)[-1]
    finally:
        hw.machine = orig


def test_non_cuda_gpu_error_falls_back_to_cpu(monkeypatch):
    """Intel XPU / DirectML lỗi lúc chạy: chạy lại bằng CPU, ghi chú rõ, ảnh sau dùng CPU luôn."""
    from net_anh import pipeline
    from net_anh.device import DeviceInfo, pick_device
    from net_anh.settings import Settings

    eng = pipeline.Enhancer("cpu")
    fake = DeviceInfo(eng.dev.device, "GPU Intel Arc (8 GB, Intel XPU)", "xpu", False)
    eng.dev = fake
    calls = []

    def boom(self, img, s, rep):
        calls.append(self.dev.kind)
        if self.dev.kind == "xpu":
            raise RuntimeError("UR_RESULT_ERROR_OUT_OF_RESOURCES")
        return img.copy()

    monkeypatch.setattr(pipeline.Enhancer, "_process_ai", boom)
    img = np.full((64, 64, 3), 120, np.uint8)
    out, rep = eng.process(img, Settings(mode="portrait", upscale=1))
    assert calls == ["xpu", "cpu"] and eng.dev.kind == "cpu" and eng.gpu_failed
    assert any("GPU lỗi" in n for n in rep.notes) and not rep.ai_fallback and rep.device.startswith("CPU")
