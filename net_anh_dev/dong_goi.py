"""Tạo gói bản nhà phát triển: BeautyHDR-<phiên bản>-dev-TronBo.zip (PC Full bản dev)

    python net_anh_dev/dong_goi.py [thư_mục_ra]

Ngoài cùng: CHAY_BAN_DEV.bat, HUONG_DAN_DEV.txt, thư mục ma-nguon (mã nguồn đầy đủ, CÓ net_anh_dev). Không kèm môi
trường Python, model, kết quả đo. Đây là gói DUY NHẤT có net_anh_dev; gói thường dùng tools/dong_goi_nguoi_dung.files()
(bỏ net_anh_dev) và packaging/net_anh.spec (excludes net_anh_dev, dừng build nếu lọt vào).
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "_build", "build", "dist", "output", "weights", ".idea",
             ".vscode", ".git"}
SKIP_SUFFIX = {".pyc", ".pth", ".onnx", ".csv", ".log", ".orig", ".rej"}


def files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(part in SKIP_DIRS or part.startswith(".venv") for part in rel.parts[:-1]):
            continue
        if p.suffix.lower() in SKIP_SUFFIX or p.name.endswith("-setup.exe") or p.name.endswith(".zip"):
            continue
        yield p, rel


def main(out_dir: str = ".") -> Path:
    ver = re.search(r'__version__ = "([^"]+)"', (ROOT / "net_anh" / "__init__.py").read_text(encoding="utf-8"))[1]
    top = f"BeautyHDR-{ver}-dev"
    dst = Path(out_dir) / f"{top}-TronBo.zip"
    crlf = lambda s: s.replace("\r\n", "\n").replace("\n", "\r\n")  # noqa: E731
    n = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"{top}/CHAY_BAN_DEV.bat", crlf((HERE / "CHAY_BAN_DEV.bat").read_text(encoding="utf-8")))
        guide = (HERE / "HUONG_DAN_DEV.txt").read_text(encoding="utf-8").replace("{ver}", ver)
        z.writestr(f"{top}/HUONG_DAN_DEV.txt", "﻿" + crlf(guide))
        for p, rel in files():
            z.write(p, f"{top}/ma-nguon/{rel.as_posix()}")
            n += 1
    print(f"{dst} ({n} file mã nguồn, {dst.stat().st_size / 1e6:.2f} MB)")
    return dst


if __name__ == "__main__":
    main(*sys.argv[1:2])
