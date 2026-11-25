"""Tự kiểm tra bản cài / bản đóng gói, và tải sẵn model trước khi đóng gói.

    NetAnh.exe --tu-kiem [file_log]     kiểm thư viện, thiết bị, model, chạy thử Nhanh + AI trên ảnh tổng hợp
    NetAnh.exe --tai-model [thư_mục]    tải các model cần khi chạy (kiểm SHA-256) vào thư mục
Mã thoát 0 = đạt. Bản .exe không có cửa sổ dòng lệnh nên kết quả luôn được ghi thêm ra file log.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import traceback
from pathlib import Path


def _log_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    return base / "tu_kiem.log"


def tu_kiem(log_file: str | None = None, need_ai: bool = False) -> int:
    lines: list[str] = []
    ok = True

    def say(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    try:
        import cv2
        import numpy as np
        import torch

        from . import __version__, weights
        from .pipeline import Enhancer
        from .settings import Settings

        say(f"Beauty HDR {__version__} · Python {platform.python_version()} · {platform.platform()}")
        say(f"torch {torch.__version__} (CUDA {torch.version.cuda or 'không'}) · OpenCV {cv2.__version__} · numpy {np.__version__}")
        from . import hardware

        say("Máy: " + hardware.machine().summary())
        eng = Enhancer("auto")
        say(f"Thiết bị: {eng.dev.name}")
        import spandrel  # nạp model nền kiến trúc mới (DAT-2, RealPLKSR): bản đóng gói phải có thư viện này

        say(f"spandrel {spandrel.__version__}")
        # nạp thử module giao diện: lỗi trong đây (thiếu module, lỗi cú pháp) làm exe không mở được cửa sổ,
        # mà phần chạy thử bên dưới không đụng tới (lỗi 0.8.1 "No module named 'net_anh.gui'")
        import importlib

        for mod in ("net_anh.gui", "net_anh.server_gui"):
            try:
                importlib.import_module(mod)
            except ModuleNotFoundError as e:
                if sys.platform == "win32" or (e.name or "").startswith("net_anh"):
                    raise
                say(f"  (bỏ qua {mod} trên máy không màn hình: {e})")
            except ImportError as e:  # Linux không có thư viện đồ hoạ (libEGL…): chỉ bản Windows bắt buộc
                if sys.platform == "win32":
                    raise
                say(f"  (bỏ qua {mod} trên máy không màn hình: {e})")
        say("Giao diện (gui, server_gui): đạt")
        say(f"Thư mục model: {weights.weights_dir()}")
        for m in weights.MANIFEST:
            if m in weights.RUNTIME or weights.status(m) != "missing":
                say(f"  {weights.MANIFEST[m].file:38s} {weights.status(m)}")

        rng = np.random.default_rng(0)
        img = cv2.GaussianBlur((rng.random((96, 128, 3)) * 255).astype(np.uint8), (0, 0), 2)
        t0 = time.perf_counter()
        out, _ = eng.process(img, Settings(mode="quick", upscale=2))
        assert out.shape == (192, 256, 3), out.shape
        say(f"Cải thiện nhanh: đạt ({time.perf_counter() - t0:.2f} s)")

        if eng.models_ready():
            t0 = time.perf_counter()
            out, rep = eng.process(img, Settings(mode="portrait", upscale=2))
            assert out.shape == (192, 256, 3), out.shape
            say(f"Chân dung AI (Real-ESRGAN + dò mặt): đạt ({time.perf_counter() - t0:.2f} s, {rep.device})")
        elif need_ai:
            ok = False
            say("THIẾU model AI (cần cho bản đầy đủ)")
        else:
            say("Chưa có model AI: bỏ qua phần chạy thử AI (app sẽ tải khi cần)")
    except Exception:
        ok = False
        say("LỖI:\n" + traceback.format_exc())
    say("KẾT QUẢ: ĐẠT" if ok else "KẾT QUẢ: KHÔNG ĐẠT")
    try:
        _log_path(log_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass
    return 0 if ok else 1


def tai_model(folder: str | None = None, progress_file: str | None = None) -> int:
    """Tải các model cần để chạy. Tiến độ in ra console VÀ ghi vào progress_file (exe đóng gói không có console, nên
    BeautyHDR.exe đọc tiến độ từ file này)."""
    if folder:
        os.environ["NET_ANH_WEIGHTS"] = str(Path(folder).resolve())
    from . import weights

    def say(msg: str) -> None:
        print(msg, flush=True)
        if progress_file:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    for m in weights.RUNTIME:
        p = weights.ensure(m, progress=say)
        say(f"OK {p}")
    return 0


def dispatch(argv: list[str]) -> int | None:
    """Xử lý các cờ dòng lệnh đặc biệt; trả None nếu không có cờ nào (mở giao diện)."""
    if not argv:
        return None
    if argv[0] == "--tu-kiem":
        rest = [a for a in argv[1:] if a != "--can-ai"]
        return tu_kiem(rest[0] if rest else None, need_ai="--can-ai" in argv)
    if argv[0] == "--tai-model":
        rest = [x for x in argv[1:] if x != "--progress-file"]
        pf = argv[argv.index("--progress-file") + 1] if "--progress-file" in argv else None
        rest = [x for x in rest if x != pf]
        return tai_model(rest[0] if rest else None, pf)
    if argv[0] == "--may-chu":  # máy chủ cho điện thoại; --dong-lenh: không mở cửa sổ (in ra console)
        if "--dong-lenh" in argv:
            from .server import main as server_main
            return server_main([a for a in argv[1:] if a != "--dong-lenh"])
        from .server_gui import main as server_gui_main
        return server_gui_main(argv[1:])
    if argv[0] == "--cli":
        from .cli import main
        return main(argv[1:])
    return None
