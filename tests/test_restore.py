"""Khôi phục ảnh cũ (net_anh/restore.py): bộ dò đốm, phục hồi màu phai, ảnh đen trắng, preset."""
import cv2
import numpy as np

from net_anh import restore, weights
from net_anh.settings import PRESETS, Settings, apply_preset


def _scene():
    rng = np.random.default_rng(1)
    img = np.zeros((400, 500, 3), np.uint8)
    img[:] = (90, 120, 150)
    img[100:300, 150:350] = (60, 80, 200)
    img[:, :250, 0] = 200  # nửa trái xanh dương, nửa phải đỏ/cam: ảnh màu thật
    img[:, 250:, 2] = 230
    img = cv2.GaussianBlur(img, (0, 0), 2)
    return np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


def test_speck_mask_finds_gold_dot_and_ignores_clean_image():
    img = _scene()
    assert restore.speck_mask(img).mean() < 0.004
    cv2.circle(img, (80, 80), 5, (40, 190, 235), -1)  # đốm vàng sáng (BGR)
    m = restore.speck_mask(img)
    assert m[80, 80] and m.mean() < 0.02


def test_fade_restore_stretches_faded_photo():
    img = _scene()
    faded = (img.astype(np.float32) * 0.45 + 110).astype(np.uint8)
    out, note = restore.fade_restore(faded)
    assert out.std() > faded.std() * 1.5 and note
    assert restore.fade_restore(faded, 0.0)[0] is faded


def test_mono_photo_stays_gray():
    g = np.tile(np.linspace(40, 200, 300, dtype=np.uint8), (200, 1))
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    img[..., 0] = np.clip(img[..., 0].astype(int) - 20, 0, 255)  # giấy ngả vàng
    assert restore.is_mono(img)
    out, _ = restore.fade_restore(img)
    assert np.abs(out[..., 0].astype(int) - out[..., 2].astype(int)).max() <= 1


def test_inpaint_falls_back_to_opencv_without_lama(monkeypatch):
    class E:
        dev = type("D", (), {"kind": "cpu", "device": "cpu"})()
        def _ensure(self, m): raise weights.ModelError("thiếu")
        def _check(self): pass
    img = _scene()
    bad = img.copy(); bad[50:54, :] = 255
    mask = np.zeros(img.shape[:2], bool); mask[50:54, :] = True
    out, how = restore.inpaint(E(), bad, mask)
    assert "OpenCV" in how and np.abs(out[52].astype(int) - img[52].astype(int)).mean() < 12


def test_restore_preset_enables_all_steps():
    s = apply_preset(Settings(), "restore")
    assert s.repair > 0 and s.fade > 0 and s.auto_rotate and "restore" in PRESETS
    assert Settings().repair == 0 and not Settings().auto_rotate  # mặc định không đổi hành vi cũ


def test_manifest_has_restore_models():
    assert weights.MANIFEST["lama"].license == "Apache-2.0" and weights.MANIFEST["scratch"].zip_member


class _Eng:
    dev = type("D", (), {"kind": "cpu", "device": "cpu"})()
    def say(self, m): pass
    def _check(self): pass
    def _ensure(self, m): raise weights.ModelError("thiếu")


def test_user_damage_mask_is_inpainted_and_remembered(tmp_path):
    from net_anh.pipeline import Report
    img = _scene()
    bad = img.copy(); bad[100:140, 200:260] = (30, 200, 240)
    m = np.zeros(img.shape[:2], np.uint8); m[96:144, 196:264] = 255
    cv2.imwrite(str(tmp_path / "m.png"), m)
    e, rep = _Eng(), Report()
    out = restore.run(e, bad, 0.0, 0.0, False, rep, str(tmp_path / "m.png"))
    assert np.abs(out[120, 230].astype(int) - img[120, 230].astype(int)).mean() < 25  # chỗ hỏng đã được vá
    assert e._repair_mask is not None and e._repair_mask[120, 230] and "vùng bạn tô" in rep.notes[0]
    assert restore.load_damage_mask(str(tmp_path / "khong_co.png"), (10, 10)) is None
    assert restore.load_damage_mask("", (10, 10)) is None


def test_face_damage_weight_follows_alignment():
    class H:  # affine tịnh tiến: mặt 512×512 lấy từ ô (100,100) của ảnh
        affine_matrices = [np.array([[1, 0, -100], [0, 1, -100]], np.float32)]
    e = _Eng(); e._repair_mask = np.zeros((600, 600), bool); e._repair_mask[250:350, 250:350] = True
    w = restore.face_damage_weight(e, H(), 0)
    assert w is not None and w.shape == (512, 512) and w[200, 200] > 0.9 and w[20, 20] < 0.05
    e._repair_mask[:] = False
    assert restore.face_damage_weight(e, H(), 0) is None



def test_is_mono_keeps_color_near_edge():
    """Vùng giữa xám nhưng áo đỏ sát mép: trước đây is_mono() chỉ xét vùng giữa nên fade_restore() bỏ hết màu."""
    g = np.tile(np.linspace(60, 190, 400, dtype=np.uint8), (300, 1))
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    img[:, :60] = (40, 40, 200)  # áo đỏ ở mép trái (BGR), trong khoảng 4–20 % chiều rộng
    assert not restore.is_mono(img)
    out, note = restore.fade_restore(img, 1.0)
    assert "đen trắng" not in note
    red = out[150, 30].astype(int)
    assert red[2] - red[0] > 80  # áo vẫn đỏ


def test_is_mono_ignores_colored_scanner_border():
    """Ảnh đen trắng có viền máy scan màu mỏng (< 4 %) vẫn là ảnh đen trắng."""
    g = np.tile(np.linspace(40, 200, 400, dtype=np.uint8), (300, 1))
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    img[:, :8] = (200, 60, 20)
    img[:8] = (200, 60, 20)
    assert restore.is_mono(img)


def _warm(img):
    f = img.astype(np.float32) / 255
    return (np.clip(f ** np.array([1.25, 1.05, 0.88]), 0, 1) * 255).round().astype(np.uint8)


def test_fade_restore_keeps_intentional_warm_light():
    """Ảnh nắng chiều có chủ ý (đen vẫn đen, sáng nhất vẫn trắng, màu đủ rực): giữ sắc ấm, không gỡ như ám vàng."""
    img = _scene()
    img[20:60, 20:60] = 250  # có điểm trắng thật
    img[320:390, 10:80] = 5
    warm = _warm(img)
    cast, _ = restore.fade_cast(warm)
    assert not cast
    out, note = restore.fade_restore(warm)
    assert "giữ sắc" in note
    lab_in = cv2.cvtColor(warm, cv2.COLOR_BGR2LAB)[..., 2].mean()
    lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 2].mean()
    assert lab_out > lab_in - 2  # không mất sắc vàng ấm


def test_fade_restore_yellowed_paper_is_cast():
    img = _scene()
    img[20:60, 20:60] = 250
    yellow = (np.clip(img.astype(np.float32) * np.array([0.7, 0.95, 1.04]) + np.array([0, 5, 10]), 0, 255)).astype(np.uint8)
    cast, why = restore.fade_cast(yellow)
    assert cast and why


def test_fade_restore_vibrance_does_not_blow_up_saturated_red():
    """Vùng đã rực (môi, áo đỏ) chỉ thêm ít độ rực: bản 0.10 làm môi đỏ gắt (C +30)."""
    img = _scene()
    faded = (img.astype(np.float32) * 0.45 + 110).astype(np.uint8)
    faded[150:200, 200:260] = (120, 120, 175)  # mảng hồng đỏ như môi
    def chroma(x):
        lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) - 128
        return np.hypot(lab[150:200, 200:260, 1], lab[150:200, 200:260, 2]).mean()
    c_in, c_new, c_old = chroma(faded), chroma(restore.fade_restore(faded)[0]), chroma(restore.fade_restore_v1(faded)[0])
    assert c_new < c_old - 1


class _FakeFaces:
    """Thay Enhancer: một mặt có 5 điểm mốc cố định, không có ParseNet."""
    _helper = None

    def __init__(self, lm):
        self.lm = np.asarray(lm, np.float32)

    def detect_faces(self, img):
        return [(np.array([150, 120, 350, 360, 0.99]), self.lm)]


def test_feature_guard_drops_specks_on_eyes_and_lips():
    img = _scene()
    lm = [(210, 200), (290, 200), (250, 250), (220, 300), (280, 300)]
    enh = _FakeFaces(lm)
    prot, head = restore.feature_guard(enh, img)
    for x, y in lm:
        assert prot[y, x]
    assert head[230, 250] and not head[20, 20]
    assert not prot[380, 480]
