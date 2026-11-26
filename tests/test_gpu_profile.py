"""Hồ sơ theo dòng card (gpu_profile), chọn kiểu số có chạy thử, cắt ô theo VRAM, làm thư mục (batch).

Môi trường kiểm không có GPU: phần GPU được giả lập bằng một DeviceInfo kind="cuda" chạy trên CPU (PyTorch CPU chạy
được cả fp16 và bf16), nên đường chọn kiểu số, cắt ô và báo cáo đều đi đúng như trên card thật.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from net_anh import gpu_profile as gp
from net_anh import hardware
from net_anh.batch import BatchRunner, extra_engines
from net_anh.device import DeviceInfo


# ---------------------------------------------------------------- nhận dòng card

@pytest.mark.parametrize("name, cc, key", [
    ("NVIDIA GeForce GTX 1660 Ti", (7, 5), "gtx16"),
    ("NVIDIA GeForce GTX 1650", (7, 5), "gtx16"),
    ("NVIDIA GeForce MX450", (7, 5), "gtx16"),
    ("NVIDIA GeForce RTX 2060", (7, 5), "rtx20"),
    ("NVIDIA GeForce RTX 3060", (8, 6), "rtx30"),
    ("NVIDIA GeForce RTX 4070 Laptop GPU", (8, 9), "rtx40"),
    ("NVIDIA GeForce RTX 5080", (12, 0), "rtx50"),
    ("NVIDIA GeForce GTX 1060 6GB", (6, 1), "gtx10"),
    ("NVIDIA GeForce GTX 970", (5, 2), "maxwell"),
    ("NVIDIA TITAN V", (7, 0), "volta"),
    # chỉ có tên (nvidia-smi khi bản cài là CPU)
    ("NVIDIA GeForce GTX 1660 SUPER", None, "gtx16"),
    ("NVIDIA GeForce RTX 3070 Ti", None, "rtx30"),
    ("NVIDIA GeForce RTX 4090", None, "rtx40"),
    ("NVIDIA GeForce RTX 5060", None, "rtx50"),
    ("NVIDIA GeForce GTX 1080 Ti", None, "gtx10"),
    ("NVIDIA GeForce GTX 960", None, "maxwell"),
    ("NVIDIA Quadro P2000", None, "gtx10"),
    ("NVIDIA Tesla K80", None, "kepler"),
])
def test_nvidia_family(name, cc, key):
    assert gp.nvidia_family(name, cc).key == key


def test_precision_rules_per_family():
    assert gp.FAMILIES["gtx16"].precisions == ("fp32",)  # lỗi nền đen fp16 trên GTX 1660 Ti
    assert gp.FAMILIES["gtx10"].precisions == ("fp32",)  # fp16 Pascal chậm
    assert gp.FAMILIES["rtx20"].precisions == ("fp16", "fp32")  # không có bf16 thật
    for k in ("rtx30", "rtx40", "rtx50"):
        assert gp.FAMILIES[k].precisions[:2] == ("fp16", "bf16") and gp.FAMILIES[k].tf32
    for f in gp.FAMILIES.values():
        assert f.precisions[-1] == "fp32"


def test_other_vendors():
    assert gp.family_of("xpu", "Intel(R) Arc(TM) A770 Graphics").key == "intel_arc"
    assert gp.family_of("xpu", "Intel(R) Arc(TM) B580 Graphics").key == "intel_arc"
    assert gp.family_of("xpu", "Intel(R) Iris(R) Xe Graphics").key == "intel_igpu"
    assert gp.family_of("xpu", "Intel(R) Arc(TM) A770 Graphics").precisions == ("fp32",)  # giữ như 0.14.1
    rocm = gp.family_of("cuda", "AMD Radeon RX 7900 XTX", (11, 0), hip=True)
    assert rocm.key == "amd_rocm" and "fp16" in rocm.precisions
    assert gp.family_of("cuda", "AMD Radeon RX 6600", (10, 3), hip=True).precisions == ("fp32",)
    assert gp.family_of("directml", "AMD Radeon RX 6600").key == "amd_dml"
    assert gp.family_of("cpu", "").key == "cpu"


def test_arch_supported():
    new = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]  # PyTorch PyPI 2.15 cu132 (đọc trong môi trường này)
    assert not gp.arch_supported((6, 1), new)  # GTX 1060: phải dùng bản cuda_old
    assert not gp.arch_supported((5, 2), new)
    assert gp.arch_supported((7, 5), new) and gp.arch_supported((8, 9), new) and gp.arch_supported((12, 0), new)
    old = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]  # kiểu bản cu126
    assert gp.arch_supported((6, 1), old)
    assert not gp.arch_supported((12, 0), old)  # RTX 50 cần bản CUDA 12.8 trở lên
    assert gp.arch_supported((12, 0), old + ["compute_90"])  # có PTX: driver tự dịch
    assert gp.arch_supported((6, 1), [])


def test_candidates_filter_by_model(monkeypatch):
    monkeypatch.delenv("NETANH_PRECISION", raising=False)
    f30, f20, f16 = gp.FAMILIES["rtx30"], gp.FAMILIES["rtx20"], gp.FAMILIES["gtx16"]
    assert gp.candidates(f30) == ["fp16", "bf16", "fp32"]
    assert gp.candidates(f30, supports_half=False) == ["bf16", "fp32"]  # DAT-2
    assert gp.candidates(f20, supports_half=False) == ["fp32"]
    assert gp.candidates(f30, bf16_native=False, supports_half=False) == ["fp32"]
    assert gp.candidates(f16) == ["fp32"]
    monkeypatch.setenv("NETANH_PRECISION", "fp32")
    assert gp.candidates(f30) == ["fp32"]
    monkeypatch.setenv("NETANH_PRECISION", "fp16")
    assert gp.candidates(f16) == ["fp16", "fp32"]  # ép để đo: vẫn qua bước kiểm


# ---------------------------------------------------------------- chạy thử kiểu số

def test_compare_catches_nan_black_and_drift():
    ref = gp.probe_image(32)
    assert gp.compare(ref, ref + 0.001).ok
    bad = ref.clone()
    bad[0, 0, 0, 0] = float("nan")
    assert not gp.compare(ref, bad).ok and "NaN" in gp.compare(ref, bad).why
    assert not gp.compare(ref, torch.zeros_like(ref)).ok
    assert not gp.compare(ref, (ref + 0.05).clamp(0, 1)).ok  # lệch đều 13 mức xám


class _Overflow(torch.nn.Module):
    """Giống lỗi GTX 16: đúng ở fp32, fp16 tràn số."""

    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 3, 1)
        torch.nn.init.dirac_(self.c.weight)
        torch.nn.init.zeros_(self.c.bias)

    def forward(self, x):
        y = self.c(x)
        return y * 7e4 / 7e4 if y.dtype == torch.float16 else y


def test_probe_rejects_overflow_and_leaves_fp32():
    net = _Overflow()
    r = gp.probe_dtype(net, torch.device("cpu"), torch.float16)
    assert not r.ok
    assert next(net.parameters()).dtype == torch.float32


def _fake_gpu(key: str, cc=(8, 6), vram=12.0) -> DeviceInfo:
    fam = gp.FAMILIES[key]
    return DeviceInfo(torch.device("cpu"), f"GPU giả {fam.label} ({vram:.0f} GB)", "cuda", "fp16" in fam.precisions,
                      fam, vram, cc)


@pytest.fixture
def fake_engine(monkeypatch):
    from net_anh.pipeline import Enhancer

    def make(key: str, cc=(8, 6), vram=12.0):
        e = Enhancer("cpu")
        e.dev = _fake_gpu(key, cc, vram)
        return e

    return make


def test_place_picks_fp16_on_rtx30_and_fp32_on_gtx16(fake_engine):
    e = fake_engine("rtx30")
    net = e._load_compact(1.0)
    assert next(net.parameters()).dtype == torch.float16
    assert e.precision["compact"][0] == "fp16" and e.precision["compact"][1][0][1].ok
    g = fake_engine("gtx16", (7, 5), 6.0)
    assert next(g._load_compact(1.0).parameters()).dtype == torch.float32
    assert g.precision["compact"] == ("fp32", [])  # không thử fp16 trên dòng này


def test_place_falls_back_when_probe_fails(fake_engine):
    e = fake_engine("rtx40", (8, 9))
    net, dt = e._place(_Overflow(), "thu")
    assert dt == torch.float32 and next(net.parameters()).dtype == torch.float32
    name, tried = e.precision["thu"]
    assert name == "fp32" and [c for c, _ in tried] == ["fp16"]  # model không khai bf16: chỉ thử fp16, hỏng → fp32
    # model chỉ hỏng ở fp16: bf16 được chọn
    class HalfOnly(_Overflow):
        def forward(self, x):
            y = self.c(x)
            return y * float("inf") if y.dtype == torch.float16 else y
    _, dt2 = e._place(HalfOnly(), "thu2", half=True, bf16=True)
    assert dt2 == torch.bfloat16


def test_dat2_bf16_only_on_ampere(fake_engine):
    from net_anh import weights

    if weights.status("dat2_web") != "ready":
        pytest.skip("chưa tải DAT-2")
    e = fake_engine("rtx30")
    _, dt = e._load_extra_sr("dat2_web")
    assert dt == torch.bfloat16  # fp16 của DAT-2 ra NaN (đo thật), bf16 đạt
    t = fake_engine("rtx20", (7, 5), 8.0)
    _, dt = t._load_extra_sr("dat2_web")
    assert dt == torch.float32


def test_fake_gpu_result_matches_cpu(fake_engine):
    """Đường GPU (fp16, cắt ô tự động) cho ảnh gần như y hệt đường CPU fp32."""
    from net_anh.pipeline import Enhancer
    from net_anh.settings import Settings

    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(128, 40, (90, 120, 3)), 0, 255).astype(np.uint8)
    s = Settings(mode="upscale", upscale=4, tone=0.0, exposure=0.0)
    a, _ = Enhancer("cpu").process(img, s)
    e = fake_engine("rtx30", vram=4.0)
    b, rep = e.process(img, s)
    assert "(fp16)" in rep.bg_model
    assert e.last_tile >= 128
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.mean() < 0.6 and d.max() <= 12


# ---------------------------------------------------------------- cắt ô theo bộ nhớ

def test_auto_tile_scales_with_vram():
    t4 = gp.auto_tile("dat2", 3.5, 4, "fp32")
    t12 = gp.auto_tile("dat2", 11, 12, "bf16")
    t24 = gp.auto_tile("dat2", 22, 24, "bf16")
    assert gp.TILE_MIN <= t4 < t12 < t24 <= gp.TILE_MAX
    assert gp.auto_tile("compact", 11, 12, "fp16") == gp.TILE_MAX  # 12 GB: ảnh thường không phải cắt
    assert gp.auto_tile("compact", 5, 6, "fp32") >= 1024
    assert gp.auto_tile("dat2", 0.5, 2, "fp32") == gp.TILE_MIN
    assert gp.auto_tile("rrdb", 0, 0) == gp.CPU_TILE  # không đọc được bộ nhớ: như bản cũ
    for t in (t4, t12, t24):
        assert t % 32 == 0
    # bộ nhớ còn trống ít (app khác chiếm) thì ô nhỏ lại
    assert gp.auto_tile("dat2", 3, 12, "bf16") < t12


def test_auto_tile_used_on_gpu_and_cpu_stays_400(fake_engine):
    from net_anh.pipeline import Enhancer

    net = Enhancer("cpu")._load_compact(1.0)
    assert Enhancer("cpu")._auto_tile("compact", net) == 400
    e = fake_engine("rtx30", vram=6.0)
    assert e._auto_tile("compact", net) > 1000  # 6 GB: compact gần như không phải cắt
    e.vram_share = 0.5
    assert e._auto_tile("dat2", net) < fake_engine("rtx30", vram=6.0)._auto_tile("dat2", net)


# ---------------------------------------------------------------- hàng loạt

def test_batch_workers(monkeypatch):
    monkeypatch.delenv("NETANH_BATCH_WORKERS", raising=False)
    assert gp.batch_workers("cuda", 12, 32, 16) == 2
    assert gp.batch_workers("cuda", 8, 32, 16) == 1
    assert gp.batch_workers("cuda", 24, 8, 16) == 1  # thiếu RAM
    assert gp.batch_workers("cpu", 0, 64, 32) == 1
    assert gp.batch_workers("xpu", 16, 32, 16) == 1
    monkeypatch.setenv("NETANH_BATCH_WORKERS", "3")
    assert gp.batch_workers("cpu", 0, 8, 4) == 3


class _Eng:
    def __init__(self, name):
        self.name = name
        self.vram_share = 1.0
        self.busy = threading.Lock()
        self.done: list[int] = []


def test_batch_runner_single_engine_keeps_order_and_saves():
    eng = _Eng("a")
    saved, loaded_in = [], []

    def load(x):
        loaded_in.append(threading.current_thread().name)
        if x == 3:
            raise ValueError("hỏng")
        return x * 10

    r = BatchRunner([eng], load)
    errors = []

    def work(e, lo):
        try:
            v = lo.get()
        except ValueError:
            errors.append(lo.index)
            return
        e.done.append(v)
        r.later(lambda v=v: (time.sleep(0.01), saved.append(v)))

    r.run(list(range(6)), work)
    assert eng.done == [0, 10, 20, 40, 50] and errors == [3]
    assert sorted(saved) == [0, 10, 20, 40, 50]  # mọi file ghi xong trước khi run trả về
    assert all(n == "batch-read" for n in loaded_in)  # đọc ảnh ở luồng riêng


def test_batch_runner_two_engines_parallel_and_exclusive():
    engines = extra_engines(_Eng("a"), 2, lambda: _Eng("b"))
    assert [e.vram_share for e in engines] == [0.5, 0.5]
    first_end = {}
    starts = {}

    def work(e, lo):
        assert e.busy.acquire(blocking=False), "một engine bị hai luồng dùng cùng lúc"
        try:
            starts.setdefault(e.name, time.monotonic())
            time.sleep(0.03)
            e.done.append(lo.value)
            first_end.setdefault(e.name, time.monotonic())
        finally:
            e.busy.release()

    BatchRunner(engines, lambda x: x).run(list(range(12)), work)
    a, b = engines
    assert sorted(a.done + b.done) == list(range(12))
    assert a.done and b.done  # cả hai luồng cùng làm
    assert starts["b"] >= first_end["a"]  # engine 2 chờ engine 1 xong ảnh đầu (model đã có trên đĩa)


def test_batch_runner_cancel_stops_everything():
    from net_anh.pipeline import Cancelled

    eng = _Eng("a")

    def work(e, lo):
        if lo.index == 2:
            raise Cancelled()
        e.done.append(lo.index)

    with pytest.raises(Cancelled):
        BatchRunner([eng], lambda x: x, stop_errors=(Cancelled,)).run(list(range(50)), work)
    assert eng.done == [0, 1]


# ---------------------------------------------------------------- máy, cảnh báo, bảng khuyến nghị

def test_legacy_card_advice_and_speedup(monkeypatch):
    g = hardware.Gpu("NVIDIA GeForce GTX 1060 6GB", 6, False, legacy=True)
    assert g.family.key == "gtx10"
    assert "đời cũ" in hardware.advice(g)
    m = hardware.Machine("cpu", 8, 16, 12, (hardware.Gpu("NVIDIA GeForce RTX 4070", 12, True),), True)
    monkeypatch.setattr(hardware, "machine", lambda: m)
    t40 = hardware.estimate("dat2", None, 1600, 1200, kind="cuda")
    m2 = hardware.Machine("cpu", 8, 16, 12, (hardware.Gpu("NVIDIA GeForce GTX 1660 Ti", 6, True),), True)
    monkeypatch.setattr(hardware, "machine", lambda: m2)
    t16 = hardware.estimate("dat2", None, 1600, 1200, kind="cuda")
    assert t16 > 2 * t40  # ước tính theo dòng card, không còn một hệ số chung cho mọi GPU NVIDIA


def test_unsupported_arch_not_listed(monkeypatch):
    """GTX 1060 với PyTorch không còn mã sm_61: không liệt kê là GPU dùng được, báo lý do."""
    from net_anh import device

    props = SimpleNamespace(name="NVIDIA GeForce GTX 1060 6GB", total_memory=6 * 1024**3, major=6, minor=1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_86", "sm_120"])
    devs = device.list_devices()
    assert [d.kind for d in devs] == ["cpu"]
    assert device.UNSUPPORTED and "đời cũ" in device.UNSUPPORTED[0][1]
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_61", "sm_75"])
    devs = device.list_devices()
    assert devs[0].kind == "cuda" and devs[0].family.key == "gtx10" and not devs[0].half and devs[0].vram_gb == 6


def test_recommendation_table():
    rows = gp.table()
    assert len(rows) == len(gp.TABLE_ORDER)
    keys = {r["Dòng card"]: r for r in rows}
    g16 = keys[gp.FAMILIES["gtx16"].label]
    assert g16["Kiểu số (thử theo thứ tự)"] == "fp32" and "4 GB" in g16["Cắt ô (compact / DAT-2)"]
    r40 = keys[gp.FAMILIES["rtx40"].label]
    assert "song song" in r40["Hàng loạt"] and r40["Mức hỗ trợ"] == "đầy đủ"
    assert keys[gp.FAMILIES["amd_rocm"].label]["Mức hỗ trợ"] == "thử nghiệm"


def test_gpu_bench_report_on_fake_gpu(tmp_path, monkeypatch):
    """Script đo cho huy chạy trên máy thật: đi hết đường GPU (fp32/fp16/bf16, chạy thử, ghi file) trên thiết bị giả."""
    from net_anh import gpu_bench, pipeline, weights

    if weights.status("esrgan_compact") != "ready":
        pytest.skip("chưa tải compact")
    monkeypatch.setattr(pipeline, "pick_device", lambda pref="auto": _fake_gpu("rtx30"))
    txt = gpu_bench.run(lambda m: None, quick=True, out_dir=tmp_path)
    body = txt.read_text(encoding="utf-8")
    assert "compact fp16" in body and "compact bf16" in body and "ĐẠT" in body
    import json
    data = json.loads(txt.with_suffix(".json").read_text(encoding="utf-8"))
    rows = {r["dtype"]: r for r in data["sr"]}
    assert rows["fp16"]["probe_ok"] and rows["fp16"]["mean_err"] < 0.01 and rows["fp32"]["s_256"] > 0


def _app(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_device_note_and_cards_dialog(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    from net_anh import gui
    w = gui.MainWindow()
    w._devices = [_fake_gpu("rtx40", (8, 9), 12.0)] + w._devices[-1:]
    w._fill_devices()
    w.device.setCurrentIndex(0)
    w._device_changed()
    assert "GeForce RTX 40" in w.dev_note.text() and "fp16 → bf16 → fp32" in w.dev_note.text()
    d = gui.CardsDialog(w)
    t = d.findChild(gui.QTableWidget)
    assert t.rowCount() == len(gp.TABLE_ORDER)
    row = gp.TABLE_ORDER.index("rtx40")
    assert t.item(row, 0).font().bold() and not t.item(0, 0).font().bold()
    w.close()
    app.processEvents()


def test_gui_folder_batch_two_workers(monkeypatch, tmp_path):
    """Làm thư mục từ giao diện với 2 luồng song song: đủ ảnh ra, không trùng tên, ghi lịch sử, ảnh hỏng không dừng lô."""
    import cv2

    app = _app(monkeypatch, tmp_path)
    monkeypatch.setenv("NETANH_BATCH_WORKERS", "2")
    from net_anh import gui
    src, dst = tmp_path / "vao", tmp_path / "ra"
    src.mkdir()
    dst.mkdir()
    rng = np.random.default_rng(0)
    for i in range(5):
        cv2.imwrite(str(src / f"a{i}.jpg"), rng.integers(0, 255, (60, 80, 3), np.uint8))
    cv2.imwrite(str(src / "a0.png"), rng.integers(0, 255, (60, 80, 3), np.uint8))  # cùng tên gốc với a0.jpg
    (src / "hong.jpg").write_bytes(b"khong phai anh")
    dirs = iter([str(src), str(dst)])
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *a, **k: next(dirs))
    w = gui.MainWindow()
    monkeypatch.setattr(w, "settings", lambda: gui.Settings(mode="quick"))
    msgs = []
    w.say = lambda m, *a, **k: msgs.append(m)
    result = {}
    monkeypatch.setattr(w, "_batch_done", lambda r: result.setdefault("r", r))
    w.choose_folder()
    w.job.wait(120_000)
    for _ in range(50):
        app.processEvents()
        if "r" in result:
            break
        time.sleep(0.05)
    ok, n, tally = result["r"]
    assert (ok, n) == (6, 7) and tally.failed == 1
    outs = sorted(p.name for p in dst.iterdir())
    assert len(outs) == 6 and len(set(outs)) == 6
    assert any("2 ảnh song song" in m for m in msgs)
    w.close()
    app.processEvents()


# ---------------------------------------------------------------- kịch bản từng đời card (docs: bảng kịch bản)

@pytest.mark.parametrize("names, smi, flavor", [
    (["NVIDIA GeForce RTX 5090"], True, "cuda"),
    (["NVIDIA GeForce RTX 4060 Laptop GPU", "Intel(R) UHD Graphics"], True, "cuda"),  # laptop lai: theo NVIDIA
    (["NVIDIA GeForce RTX 3060"], True, "cuda"),
    (["NVIDIA GeForce RTX 2060 SUPER"], True, "cuda"),
    (["NVIDIA GeForce GTX 1660 Ti"], True, "cuda"),
    (["NVIDIA GeForce GTX 1650"], True, "cuda"),
    (["NVIDIA GeForce GTX 1060 6GB"], True, "cuda_old"),
    (["NVIDIA GeForce GTX 1050 Ti"], True, "cuda_old"),
    (["NVIDIA GeForce MX150", "Intel(R) UHD Graphics 620"], True, "cuda_old"),
    (["NVIDIA GeForce GT 1030"], True, "cuda_old"),
    (["NVIDIA GeForce GTX 970"], True, "cuda_old"),
    (["NVIDIA GeForce GTX 750 Ti"], True, "cuda_old"),
    (["NVIDIA Quadro P2000"], True, "cuda_old"),
    (["NVIDIA RTX A4000"], True, "cuda"),
    (["NVIDIA GeForce GTX 760"], True, "cpu"),  # Kepler: không bản GPU nào chạy
    (["NVIDIA GeForce GT 730"], True, "cpu"),
    (["NVIDIA GeForce GTX 780", "Intel(R) Iris(R) Xe Graphics"], True, "xpu"),  # Kepler + Iris Xe: dùng Intel
    (["NVIDIA GeForce GTX 1060 6GB", "NVIDIA GeForce RTX 3060"], True, "cuda"),  # card mới thắng
    (["NVIDIA GeForce GTX 760", "NVIDIA GeForce GTX 1070"], True, "cuda_old"),  # bỏ qua Kepler
    (["NVIDIA GeForce RTX 3070"], False, "cuda"),  # chưa cài driver: vẫn chọn gói GPU (app nhắc cài driver)
    (["Intel(R) Arc(TM) B580 Graphics"], False, "xpu"),
    (["Intel(R) Arc(TM) Graphics"], False, "xpu"),  # Core Ultra
    (["Intel(R) Iris(R) Xe Graphics"], False, "xpu"),
    (["Intel(R) UHD Graphics 630"], False, "cpu"),
    (["AMD Radeon RX 7900 XTX"], False, "cpu"),
    (["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 4050 Laptop GPU"], True, "cuda"),
    (["Microsoft Basic Display Adapter"], False, "cpu"),
    ([], False, "cpu"),
])
def test_install_scenarios(names, smi, flavor):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bl", Path(__file__).parents[1] / "launcher" / "beautyhdr_launcher.py")
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    names = [n for n in names if "microsoft" not in n.lower()]  # adapter_names() bỏ card ảo
    assert bl.recommend_flavor(names, smi, {"cpu", "cuda", "cuda_old", "xpu"})[0] == flavor


@pytest.mark.parametrize("name, key", [
    ("NVIDIA GeForce GTX 760", "kepler"), ("NVIDIA GeForce GT 730", "kepler"), ("NVIDIA Quadro K2200", "kepler"),
    ("NVIDIA GeForce GTX 750 Ti", "maxwell"), ("NVIDIA GeForce GTX TITAN X", "maxwell"),
    ("NVIDIA TITAN Xp", "gtx10"), ("NVIDIA Quadro P2000", "gtx10"), ("NVIDIA GeForce GT 1030", "gtx10"),
    ("NVIDIA Quadro RTX 4000", "rtx20"), ("NVIDIA RTX A4000", "rtx30"), ("NVIDIA RTX 4000 Ada Generation", "rtx40"),
    ("NVIDIA RTX PRO 6000 Blackwell", "rtx50"),
])
def test_name_only_families(name, key):
    assert gp.nvidia_family(name).key == key


def test_runtime_scenarios(monkeypatch):
    """Máy nhiều card: chọn card nhanh nhất; Kepler (cc 3.x) không có trong bản PyTorch nào nên không được liệt kê."""
    from net_anh import device

    assert gp.nvidia_family("NVIDIA GeForce GTX 780", (3, 5)).key == "kepler"
    cards = [SimpleNamespace(name="NVIDIA GeForce GTX 1050", total_memory=4 * 1024**3, major=6, minor=1),
             SimpleNamespace(name="NVIDIA GeForce RTX 3060", total_memory=12 * 1024**3, major=8, minor=6),
             SimpleNamespace(name="NVIDIA GeForce GTX 780", total_memory=3 * 1024**3, major=3, minor=5)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: cards[i])
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_50", "sm_60", "sm_61", "sm_75", "sm_86"])
    best = device.pick_device("auto")
    assert best.family.key == "rtx30" and str(best.device) == "cuda:1"
    assert [n for n, _ in device.UNSUPPORTED] == ["NVIDIA GeForce GTX 780"]
    assert device.pick_device("cpu").kind == "cpu"


def test_driver_and_kepler_advice(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    g = hardware.Gpu("NVIDIA GeForce RTX 3060", 12, False)
    assert "driver" in hardware.advice(g)
    k = hardware.Gpu("NVIDIA GeForce GTX 770", 2, False)
    assert "Kepler" in hardware.advice(k)
    monkeypatch.setattr(torch.version, "cuda", None)
    assert "chọn bản GPU NVIDIA" in hardware.advice(g)


# ---------------------------------------------------------------- tăng tốc thêm: kênh cuối, model mặt, cuDNN

def test_channels_last_rules(monkeypatch):
    monkeypatch.delenv("NETANH_CHANNELS_LAST", raising=False)
    assert gp.channels_last("cuda", (8, 6), "fp16", "compact")
    assert gp.channels_last("cuda", (7, 5), "fp16", "rrdb")  # RTX 20 có Tensor Core
    assert not gp.channels_last("cuda", (7, 5), "fp32", "compact")  # GTX 16 fp32: không lợi
    assert not gp.channels_last("cuda", (6, 1), "fp32", "compact")
    assert not gp.channels_last("cuda", (8, 9), "bf16", "dat2")  # model chú ý
    assert not gp.channels_last("cuda", (8, 9), "bf16", "gfpgan")  # GFPGAN lỗi .view() với kênh cuối
    assert not gp.channels_last("xpu", (8, 6), "fp16", "compact")
    monkeypatch.setenv("NETANH_CHANNELS_LAST", "0")
    assert not gp.channels_last("cuda", (8, 6), "fp16", "compact")
    monkeypatch.setenv("NETANH_CHANNELS_LAST", "1")
    assert gp.channels_last("cuda", (6, 1), "fp32", "compact")


def test_channels_last_same_result(fake_engine):
    e = fake_engine("rtx30")
    net = e._load_compact(1.0)
    assert net._netanh_cl
    conv = next(m for m in net.modules() if isinstance(m, torch.nn.Conv2d))
    assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    x = gp.probe_image(48)
    with torch.no_grad():
        a = e._forward(net, x)
        gp.to_contiguous(net)
        net._netanh_cl = False
        b = e._forward(net, x)
    assert float((a.float() - b.float()).abs().max()) < 1e-2
    g = fake_engine("gtx16", (7, 5), 6.0)
    assert not g._load_compact(1.0)._netanh_cl


def test_face_candidates_per_family():
    assert gp.face_candidates(gp.FAMILIES["rtx30"], "gfpgan") == ["bf16", "fp32"]
    assert gp.face_candidates(gp.FAMILIES["rtx50"], "restoreformer_pp") == ["bf16", "fp32"]
    for k in ("rtx20", "gtx16", "gtx10", "intel_arc", "cpu"):
        assert gp.face_candidates(gp.FAMILIES[k], "gfpgan") == ["fp32"], k


def test_gfpgan_bf16_on_ampere_fp32_on_turing(fake_engine):
    from net_anh import weights
    from net_anh.pipeline import Enhancer

    if weights.status("gfpgan") != "ready":
        pytest.skip("chưa tải GFPGAN")
    t = (gp.probe_image(512) - 0.5) / 0.5
    with torch.no_grad():
        ref = Enhancer("cpu")._load_gfpgan()
        torch.manual_seed(1)
        a = ref(t, return_rgb=False)[0]
        e = fake_engine("rtx30")
        f = e._load_gfpgan()
        assert next(f.parameters()).dtype == torch.bfloat16 and e.precision["gfpgan"][0] == "bf16"
        torch.manual_seed(1)
        b = e._face_run(f, t)
    assert b.dtype == torch.float32
    assert float(((a - b).abs() / 2).mean()) < 0.01
    r = fake_engine("rtx20", (7, 5), 8.0)
    assert next(r._load_gfpgan().parameters()).dtype == torch.float32


def test_face_run_falls_back_to_fp32_on_nan(fake_engine):
    class BadHalf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x, return_rgb=False):
            y = self.c(x)
            return (y * float("inf") if y.dtype != torch.float32 else y,)

    e = fake_engine("rtx30")
    net = BadHalf().to(torch.bfloat16)
    net._netanh_cl = False
    o = e._face_run(net, torch.rand(1, 3, 16, 16))
    assert torch.isfinite(o).all() and next(net.parameters()).dtype == torch.float32


def test_batch_turns_on_cudnn_benchmark_and_restores():
    from types import SimpleNamespace

    from net_anh.batch import BatchRunner

    old = torch.backends.cudnn.benchmark
    seen = []
    eng = SimpleNamespace(dev=SimpleNamespace(kind="cuda"))
    BatchRunner([eng], lambda i: i).run([1, 2, 3], lambda e, lo: seen.append(torch.backends.cudnn.benchmark))
    assert seen == [True] * 3 and torch.backends.cudnn.benchmark == old
    seen.clear()
    BatchRunner([eng], lambda i: i).run([1, 2], lambda e, lo: seen.append(torch.backends.cudnn.benchmark))
    assert seen == [old] * 2  # 1–2 ảnh: không đáng cho cuDNN đo
    cpu = SimpleNamespace(dev=SimpleNamespace(kind="cpu"))
    seen.clear()
    BatchRunner([cpu], lambda i: i).run([1, 2, 3], lambda e, lo: seen.append(torch.backends.cudnn.benchmark))
    assert seen == [old] * 3


@pytest.mark.parametrize("key", gp.TABLE_ORDER)
def test_model_rows_every_family(key):
    rows = gp.model_rows(gp.FAMILIES[key])
    assert [r["Model"] for r in rows] == [m[1] for m in gp.MODELS]
    for r in rows:
        assert set(r) == set(gp.MODEL_COLS) and r["Dự kiến"] != "?"
    dat = rows[3]
    if key in ("rtx30", "rtx40", "rtx50"):
        assert dat["Kiểu số"] == "bf16" and rows[0]["Kênh cuối"] == "có" and rows[4]["Kiểu số"] == "bf16"
    if key in ("gtx16", "gtx10", "cpu"):
        assert all(r["Kiểu số"] == "fp32" and r["Kênh cuối"] == "không" for r in rows)


def test_estimates_follow_card_order():
    f = gp.FAMILIES
    for m in ("compact", "rrdb", "dat2", "gfpgan"):
        ts = [gp.est_seconds(f[k], m) for k in ("rtx50", "rtx40", "rtx30", "rtx20", "gtx16", "gtx10", "cpu")]
        assert ts == sorted(ts), m


# ---------------------------------------------------------------- dữ liệu mở rộng: khai báo model, kho số đo

def test_model_db_covers_every_model_the_app_runs():
    from net_anh import weights

    for k in ("compact", "plksr", "rrdb", "dat2", "gfpgan", "restoreformer_pp"):
        assert k in gp.MODEL_DB, k
    for m in gp.MODEL_DB.values():
        assert m.role in ("sr", "face") and m.arch in ("conv", "attention", "stylegan", "diffusion")
        for w in m.weights:
            assert w in weights.MANIFEST, (m.key, w)
    assert gp.MODEL_OF_KEY["esrgan_x4"] == "rrdb" and gp.MODEL_OF_KEY["dat2_web"] == "dat2"
    assert gp.ACT_BYTES_PER_PX["dat2"] == 34_000 and "gfpgan" not in gp.ACT_BYTES_PER_PX


def _write_data(tmp_path, models: list[dict], records: list[dict]):
    import json

    base = json.loads((gp.DATA_DIR / "model_profiles.json").read_text(encoding="utf-8"))
    base["models"] += models
    (tmp_path / "model_profiles.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "so_do_gpu.json").write_text(json.dumps({"schema": 1, "records": records}), encoding="utf-8")


def test_new_heavy_model_needs_only_a_data_row(tmp_path, monkeypatch):
    """Thêm một model nặng (giả định, kiểu diffusion cần 12 GB, chỉ hợp lý với bf16): chỉ thêm dòng dữ liệu là bảng,
    cắt ô, ước lượng và kiểm phần cứng đều có."""
    import importlib

    heavy = {"key": "sieu_net", "label": "Siêu nét (thử)", "levels": "mức 7", "role": "sr", "arch": "diffusion",
             "weights": [], "fp16": False, "bf16": True, "act_bytes_per_px": 90000, "model_gb": 5.0, "cpu_s": 3000,
             "min_vram_gb": 12, "min_precision": "bf16", "slow_s": 60}
    _write_data(tmp_path, [heavy], [])
    monkeypatch.setenv("NETANH_DATA_DIR", str(tmp_path))
    g = importlib.reload(gp)
    try:
        assert "sieu_net" in g.MODEL_DB and any(k == "sieu_net" for k, _ in g.MODELS)
        assert g.fits(g.FAMILIES["rtx40"], "sieu_net", 8)[0] == "không"  # thiếu VRAM
        assert g.fits(g.FAMILIES["rtx40"], "sieu_net", 24)[0] == "ok"
        assert g.fits(g.FAMILIES["rtx20"], "sieu_net", 24)[0] == "chậm"  # không có bf16 thật
        assert g.best_dtype(g.FAMILIES["rtx30"], "sieu_net") == "bf16"
        assert g.auto_tile("sieu_net", 20, 24, "bf16") >= g.TILE_MIN
        row = next(r for r in g.model_rows(g.FAMILIES["rtx40"], 24) if r["Model"].startswith("Siêu nét"))
        assert row["Phần cứng"] == "ok" and "(ước lượng)" in row["Dự kiến"]
    finally:
        monkeypatch.delenv("NETANH_DATA_DIR")
        importlib.reload(gp)


def test_measured_numbers_replace_estimates(tmp_path, monkeypatch):
    import importlib

    recs = [{"family": "gtx16", "card": f"GTX 1660 Ti #{i}", "model": "compact", "dtype": "fp32", "metric": "s_per_mp",
             "value": v} for i, v in enumerate((0.9, 1.1, 1.0))]
    recs.append({"family": "gtx16", "model": "compact", "dtype": "fp32", "metric": "s_per_mp", "value": 50.0,
                 "app_default": False})  # cấu hình app không dùng: bỏ qua
    recs.append({"family": "gtx16", "model": "dat2", "dtype": "fp32", "metric": "bytes_per_px", "value": 60000})
    _write_data(tmp_path, [], recs)
    monkeypatch.setenv("NETANH_DATA_DIR", str(tmp_path))
    g = importlib.reload(gp)
    try:
        assert g.measured("gtx16", "compact", "fp32", "s_per_mp") == 1.0
        t, real = g.est(g.FAMILIES["gtx16"], "compact", 2.0)
        assert real and t == 2.0
        assert not g.est(g.FAMILIES["rtx30"], "compact")[1]
        assert g.auto_tile("dat2", 5, 6, "fp32", "gtx16") < g.auto_tile("dat2", 5, 6, "fp32")  # số đo thật lớn hơn
        row = g.model_rows(g.FAMILIES["gtx16"], 6)[0]
        assert row["Dự kiến"] == "1.0 s / 1 MP vào (đo)"
    finally:
        monkeypatch.delenv("NETANH_DATA_DIR")
        importlib.reload(gp)


def test_bench_records_and_import_tool(tmp_path):
    import importlib.util

    from net_anh import gpu_bench

    rep = {"family": "rtx30", "device": "GPU 0: RTX 3060", "kind": "cuda", "cc": [8, 6], "vram_gb": 12, "app": "0.15.0",
           "time": "2026-11-27 09:00",
           "sr": [{"model": "compact", "dtype": "fp16", "s_256": 0.01, "s_512": 0.03, "s_512_kenh_cuoi": 0.02,
                   "bytes_per_px_512": 1300, "probe_ok": True, "mean_err": 0.0001},
                  {"model": "dat2", "dtype": "fp16", "error": "NaN"}],
           "face": {"gfpgan_fp32_s": 0.05, "gfpgan_bf16_s": 0.03, "gfpgan_fp16_lech_muc_xam": 91.0},
           "pipeline": {"workers_1_s_per_image": 0.8, "workers_2_s_per_image": 0.5}}
    rs = gpu_bench.records(rep)
    by = {(r["model"], r["dtype"], r["metric"], r.get("tile"), r.get("channels_last")): r for r in rs}
    assert by[("compact", "fp16", "s_per_mp", 512, True)]["app_default"]  # RTX 30 fp16: app bật kênh cuối
    assert not by[("compact", "fp16", "s_per_mp", 512, False)]["app_default"]
    assert abs(by[("compact", "fp16", "s_per_mp", 512, True)]["value"] - 0.02 / 0.262144) < 1e-9
    assert ("gfpgan", "bf16", "s_per_face", None, None) in by and not any(r["model"] == "dat2" for r in rs)
    spec = importlib.util.spec_from_file_location("nhap_so_do", Path(__file__).resolve().parents[1] / "tools" / "nhap_so_do.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    store = {"schema": 1, "records": []}
    assert tool.merge(store, tool.records_of(rep)) == len(rs)
    assert tool.merge(store, tool.records_of(rep)) == 0  # gộp lại cùng file: không trùng
    assert "rtx30" in tool.summary(store)
