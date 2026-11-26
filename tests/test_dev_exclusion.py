"""Bản phát hành thường KHÔNG được chứa chức năng của bản nhà phát triển (huy chốt 2026-11-26): không mã, không menu,
không chữ. Chức năng dev nằm riêng ở gói net_anh_dev, chỉ vào gói PC Full bản dev (net_anh_dev/dong_goi.py)."""
import ast
import importlib.util
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# chữ đặc trưng của chức năng dev: không được có trong mã của bản thường
DEV_WORDS = re.compile(r"net_anh_dev|xuất prompt|xuat prompt|ChatGPT|Gemini|Grok|Nhà phát triển", re.IGNORECASE)


def test_normal_source_package_has_no_dev(tmp_path):
    import dong_goi_nguoi_dung as d
    rels = [rel.as_posix() for _, rel in d.files()]
    assert rels, "gói thường rỗng?"
    assert not [r for r in rels if r.startswith("net_anh_dev/") or "/net_anh_dev/" in r]
    zp = d.main(str(tmp_path))
    with zipfile.ZipFile(zp) as z:
        assert not [n for n in z.namelist() if "net_anh_dev" in n]


def test_exe_spec_excludes_dev():
    spec = (ROOT / "packaging" / "net_anh.spec").read_text(encoding="utf-8")
    excl = spec[spec.index("EXCLUDES = ["):spec.index("]", spec.index("EXCLUDES = ["))]
    assert '"net_anh_dev"' in excl
    assert "gói nhà phát triển lọt vào bản thường" in spec  # chặn build nếu lọt vào a.pure


def test_cloud_server_copies_only_net_anh():
    df = (ROOT / "packaging" / "cloud" / "Dockerfile").read_text(encoding="utf-8")
    assert "net_anh_dev" not in df
    assert not re.search(r"^\s*COPY\s+\.\s", df, re.M)


def test_net_anh_has_no_dev_code():
    """Không file nào của net_anh nhập net_anh_dev hay chứa chữ của chức năng xuất prompt."""
    bad = []
    for f in sorted((ROOT / "net_anh").rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "net_anh_dev" for a in node.names):
                bad.append(f"{f.name}: import net_anh_dev")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "net_anh_dev":
                bad.append(f"{f.name}: from net_anh_dev")
        for m in DEV_WORDS.finditer(src):
            bad.append(f"{f.relative_to(ROOT)}: “{m.group(0)}”")
    assert not bad, bad


def test_normal_launchers_do_not_start_dev():
    for name in ("app.py", "chay.bat", "cai_dat.bat", "cai_dat.sh", "may_chu.bat", "dong_goi_exe.bat",
                 "launcher/beautyhdr_launcher.py", "packaging/build_windows.ps1",
                 ".github/workflows/release-exe.yml", ".github/workflows/build-windows.yml"):
        p = ROOT / name
        if p.exists():
            assert "net_anh_dev" not in p.read_text(encoding="utf-8", errors="replace"), name


def test_normal_gui_has_no_dev_menu():
    """Cửa sổ bản thường (không móc setup) không có menu, lệnh nào của chức năng dev."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if importlib.util.find_spec("PySide6") is None or importlib.util.find_spec("torch") is None:
        import pytest
        pytest.skip("cần PySide6 và torch")
    from PySide6.QtWidgets import QApplication
    from net_anh.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    texts = [a.text() for a in w.findChildren(__import__("PySide6.QtGui", fromlist=["QAction"]).QAction)]
    texts += [m.title() for m in w.menuBar().findChildren(__import__("PySide6.QtWidgets", fromlist=["QMenu"]).QMenu)]
    assert not [t for t in texts if DEV_WORDS.search(t or "")], texts
    assert not hasattr(w, "a_dev_prompt")
    w.close()
    app.processEvents()
