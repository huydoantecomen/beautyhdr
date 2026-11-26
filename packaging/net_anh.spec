# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: đóng gói Beauty HDR thành thư mục dist/NetAnh (NetAnh.exe + thư viện).
#   pyinstaller --noconfirm packaging/net_anh.spec
# Model AI KHÔNG nằm trong gói này: build_windows.ps1 / bộ cài chép thư mục weights vào cạnh NetAnh.exe,
# hoặc app tự tải ở lần dùng đầu (xem net_anh/weights.py).
import os
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = re.search(r'__version__ = "([^"]+)"', (ROOT / "net_anh" / "__init__.py").read_text(encoding="utf-8"))[1]
ICON = str(ROOT / "packaging" / "net_anh.ico")
CONSOLE = os.environ.get("NET_ANH_CONSOLE") == "1"  # bật cửa sổ dòng lệnh để gỡ lỗi

# Thư viện không dùng khi chạy app (chỉ dùng trong tools/ hoặc phần tracking của facexlib): bỏ cho nhẹ.
EXCLUDES = [
    "matplotlib", "tensorboard", "IPython", "jupyter", "notebook", "pytest", "tkinter",
    "skimage", "scipy", "numba", "llvmlite", "filterpy", "pandas", "onnx", "onnxruntime",
    "transformers", "bitsandbytes", "pyarrow", "datasets", "accelerate", "sklearn", "timm", "huggingface_hub",
    # KHÔNG bỏ torch._dynamo / sympy: torchvision.ops (facexlib dùng) import torch._dynamo.utils, cái này cần
    # sympy; bỏ đi thì exe lỗi ngay khi nạp model dò mặt (đã thử chặn đúng danh sách này trên Linux: tự kiểm KHÔNG ĐẠT).
    "torch.utils.tensorboard", "triton",
    "PySide6.Qt3DCore", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
]


def _torchvision_binaries():
    """Phần C++ của torchvision (nms, roi_align, đọc ảnh) nạp bằng đường dẫn chứ không bằng import, nên PyInstaller
    không tự thấy. torchvision từ 0.29 đổi tên thành _C_stable.pyd / image_stable.pyd mà hook của PyInstaller chưa biết:
    thiếu file này exe báo "operator torchvision::nms does not exist" (lỗi 0.8.1, đã tái hiện trên Linux)."""
    import torchvision

    d = Path(torchvision.__file__).parent
    found = [(str(f), "torchvision") for f in d.iterdir()
             if f.suffix.lower() in (".pyd", ".so", ".dll") and f.name.split(".")[0] in ("_C", "_C_stable", "image", "image_stable")]
    if not any(Path(f).name.startswith("_C") for f, _ in found):
        raise SystemExit(f"Không thấy phần C++ của torchvision trong {d}: cài lại torch và torchvision")
    return found + collect_dynamic_libs("torchvision")


TV_BINARIES = _torchvision_binaries()

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=TV_BINARIES,
    datas=collect_data_files("facexlib", excludes=["**/*.pth"]) + [(str(ROOT / "net_anh" / "assets" / "icon.png"), "net_anh/assets"),
           (str(ROOT / "net_anh" / "drive_models.json"), "net_anh"),
           (str(ROOT / "net_anh" / "du_lieu"), "net_anh/du_lieu")],
    hiddenimports=collect_submodules("net_anh") + collect_submodules("spandrel"),  # spandrel: DAT-2, RealPLKSR
    excludes=EXCLUDES,
    noarchive=False,
)


def _check_own_modules():
    """PyInstaller lặng lẽ BỎ file .py có lỗi cú pháp mà vẫn build xong: exe chỉ báo lỗi khi mở cửa sổ
    ("No module named 'net_anh.gui'", lỗi 0.8.1 trên máy huy). Ở đây biên dịch thử từng file của net_anh và
    kiểm chắc chắn mọi module đã vào gói; thiếu thì dừng build ngay, kèm dòng lỗi."""
    pkg = ROOT / "net_anh"
    packed = {name for name, _, _ in a.pure}
    errors = []
    for f in sorted(pkg.rglob("*.py")):
        rel = f.relative_to(ROOT).with_suffix("")
        mod = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
        try:
            compile(f.read_bytes(), str(f), "exec")
        except SyntaxError as e:
            errors.append(f"{f.relative_to(ROOT)} dòng {e.lineno}: lỗi cú pháp: {e.msg}")
            continue
        if mod not in packed:
            errors.append(f"{mod}: không có trong gói")
    if errors:
        raise SystemExit("Mã nguồn net_anh lỗi, dừng build:\n" + "\n".join(errors))


_check_own_modules()
pyz = PYZ(a.pure)

if os.name == "nt":
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)
    nums = tuple(int(x) for x in (VERSION.split(".") + ["0", "0", "0"])[:4])
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[StringFileInfo([StringTable("040904B0", [
            StringStruct("ProductName", "Net Anh"),
            StringStruct("FileDescription", "Net Anh - lam net anh"),
            StringStruct("FileVersion", VERSION),
            StringStruct("ProductVersion", VERSION),
            StringStruct("OriginalFilename", "NetAnh.exe"),
        ])]), VarFileInfo([VarStruct("Translation", [1033, 1200])])],
    )
else:
    version_info = None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NetAnh",
    icon=ICON,
    version=version_info,
    console=CONSOLE,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="NetAnh")
