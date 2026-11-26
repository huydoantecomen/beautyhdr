"""Tạo file zip gửi người dùng Windows: ngoài cùng chỉ có TAO_FILE_CAI_DAT.bat, HUONG_DAN.txt và thư mục ma-nguon.

    python tools/dong_goi_nguoi_dung.py [thư_mục_ra]      → NetAnh-<phiên bản>-Windows.zip

Người dùng giải nén, bấm đúp TAO_FILE_CAI_DAT.bat là ra một file NetAnh-<phiên bản>-setup.exe (packaging/build_windows.ps1).
Không đóng gói môi trường Python, file build, model và kết quả đo (.csv) để thư mục gọn.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# net_anh_dev: bản nhà phát triển (xuất prompt cho AI ngoài), chỉ có trong gói Dev (net_anh_dev/dong_goi.py)
SKIP_DIRS = {"net_anh_dev", "__pycache__", ".pytest_cache", ".venv", "_build", "build", "dist", "output", "weights", ".idea", ".vscode"}
SKIP_SUFFIX = {".pyc", ".pth", ".onnx", ".csv", ".log", ".orig", ".rej"}


def files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(part in SKIP_DIRS or part.startswith(".venv") for part in rel.parts[:-1]):
            continue
        if p.suffix.lower() in SKIP_SUFFIX or p.name.endswith("-setup.exe"):
            continue
        yield p, rel


def main(out_dir: str = ".") -> Path:
    ver = re.search(r'__version__ = "([^"]+)"', (ROOT / "net_anh" / "__init__.py").read_text(encoding="utf-8"))[1]
    top = f"NetAnh-{ver}"
    dst = Path(out_dir) / f"NetAnh-{ver}-Windows.zip"
    guide = (ROOT / "packaging" / "nguoi_dung" / "HUONG_DAN.txt").read_text(encoding="utf-8").replace("{ver}", ver)
    n = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(ROOT / "packaging" / "nguoi_dung" / "TAO_FILE_CAI_DAT.bat", f"{top}/TAO_FILE_CAI_DAT.bat")
        # Notepad cũ cần BOM để đọc đúng UTF-8, và CRLF để xuống dòng
        z.writestr(f"{top}/HUONG_DAN.txt", "﻿" + guide.replace("\r\n", "\n").replace("\n", "\r\n"))
        for p, rel in files():
            z.write(p, f"{top}/ma-nguon/{rel.as_posix()}")
            n += 1
    print(f"{dst} ({n} file mã nguồn, {dst.stat().st_size / 1e6:.2f} MB)")
    return dst


if __name__ == "__main__":
    main(*sys.argv[1:2])
