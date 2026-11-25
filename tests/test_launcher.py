"""Kiểm bộ khởi động BeautyHDR.exe bằng máy chủ HTTP giả (có hỗ trợ Range) và một "NetAnh" giả bằng shell."""
import hashlib
import http.server
import io
import json
import os
import re
import stat
import sys
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launcher"))
pytestmark = pytest.mark.skipif(os.name == "nt", reason="NetAnh giả là script shell")
import beautyhdr_launcher as bl  # noqa: E402

FAKE = """#!/bin/sh
if [ "$1" = "--tai-model" ]; then
  PF=""; while [ $# -gt 0 ]; do [ "$1" = "--progress-file" ] && PF="$2"; shift; done
  for i in 1 2 3 4 5 6 7; do echo "Đang tải m$i: 50%" >> "$PF"; echo "OK /x/m$i" >> "$PF"; done
  [ -n "$FAKE_FAIL" ] && exit 3
  exit 0
fi
echo launched > "$BEAUTYHDR_HOME/launched.txt"
"""


def make_zip(ver: str, pad: int = 300_000) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        zi = zipfile.ZipInfo("NetAnh/NetAnh")
        zi.external_attr = (stat.S_IFREG | 0o755) << 16
        z.writestr(zi, FAKE)
        z.writestr("NetAnh/version.txt", ver)
        z.writestr("NetAnh/pad.bin", os.urandom(pad))
    return buf.getvalue()


class Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    drop_after: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = self.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d+)-", rng)):
            start = int(m[1])
        body = data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = self.drop_after.pop(self.path, None)
        self.wfile.write(body if cut is None else body[:cut])  # cắt ngang: giả lập mất mạng giữa chừng


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Handler.files, Handler.drop_after = {}, {}
    yield f"http://127.0.0.1:{srv.server_port}", Handler
    srv.shutdown()


def publish(base, H, ver, parts=2):
    z = make_zip(ver)
    n = -(-len(z) // parts)
    chunks = [z[i * n:(i + 1) * n] for i in range(parts)]
    plist = []
    for i, c in enumerate(chunks, 1):
        name = f"app-cpu.zip.{i:03d}"
        H.files[f"/{ver}/{name}"] = c
        plist.append({"name": name, "url": f"{base}/{ver}/{name}", "size": len(c), "sha256": hashlib.sha256(c).hexdigest()})
    upd = {"version": ver, "flavors": {"cpu": {"parts": plist}}}
    H.files["/update.json"] = json.dumps(upd).encode()
    return upd


class UI(bl.ConsoleUI):
    def __init__(self):
        super().__init__()
        self.logs = []

    def log(self, m):
        self.logs.append(m)


def test_fresh_install_models_and_update(server, tmp_path, monkeypatch):
    base, H = server
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    publish(base, H, "1.0.0")
    ui = UI()
    L = bl.Launcher(ui, update_url=base + "/update.json", flavor="cpu")
    assert L.run(launch=False) == 0
    assert (L.app_dir / "version.txt").read_text() == "1.0.0"
    assert L.state.get("models_ok") is True and L.state.get("version") == "1.0.0"
    assert not any((tmp_path / "home" / "downloads").glob("*"))
    # lần mở sau: không có bản mới, không tải lại gì
    n = len(ui.logs)
    bl.Launcher(ui, update_url=base + "/update.json").run(launch=False)
    assert len(ui.logs) == n
    # có bản mới: cập nhật, giữ thư mục weights, tải model lại 0 lần cần thiết (đánh dấu lại bằng một lượt kiểm nhanh)
    (L.weights).mkdir(exist_ok=True)
    (L.weights / "keep.pth").write_text("x")
    publish(base, H, "1.1.0", parts=3)
    L2 = bl.Launcher(ui, update_url=base + "/update.json")
    L2.run(launch=True)
    assert (L2.app_dir / "version.txt").read_text() == "1.1.0"
    assert (L2.weights / "keep.pth").exists()
    assert not (tmp_path / "home" / "app.old").exists() and not (tmp_path / "home" / "app.new").exists()


def test_resume_after_dropped_connection(server, tmp_path, monkeypatch):
    base, H = server
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    upd = publish(base, H, "1.0.0", parts=1)
    p = upd["flavors"]["cpu"]["parts"][0]
    H.drop_after["/1.0.0/" + p["name"]] = 100_000  # ngắt sau 100 KB
    L = bl.Launcher(UI(), update_url=base + "/update.json", flavor="cpu")
    L.run(launch=False)
    assert L.installed()


def test_offline_first_run_and_offline_update(server, tmp_path, monkeypatch):
    base, H = server
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    with pytest.raises(bl.Fail, match="mạng"):
        bl.Launcher(UI(), update_url=base + "/khong-co.json", flavor="cpu").run(launch=False)
    publish(base, H, "1.0.0")
    bl.Launcher(UI(), update_url=base + "/update.json", flavor="cpu").run(launch=False)
    # mất mạng khi mở lần sau: vẫn chạy bản đang có
    L = bl.Launcher(UI(), update_url=base + "/khong-co.json")
    assert L.run(launch=True) == 0


def test_disk_full_is_reported_and_bad_checksum_rejected(server, tmp_path, monkeypatch):
    base, H = server
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    publish(base, H, "1.0.0")
    monkeypatch.setattr(bl.shutil, "disk_usage", lambda p: type("U", (), {"free": 1000})())
    with pytest.raises(bl.Fail, match="trống"):
        bl.Launcher(UI(), update_url=base + "/update.json", flavor="cpu").run(launch=False)
    monkeypatch.undo()
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home2"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    upd = publish(base, H, "1.0.0")
    upd["flavors"]["cpu"]["parts"][0]["sha256"] = "0" * 64
    H.files["/update.json"] = json.dumps(upd).encode()
    with pytest.raises(bl.Fail, match="checksum"):
        bl.Launcher(UI(), update_url=base + "/update.json", flavor="cpu").run(launch=False)


def test_model_failure_is_reported_and_retried_next_launch(server, tmp_path, monkeypatch):
    base, H = server
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    publish(base, H, "1.0.0")
    monkeypatch.setenv("FAKE_FAIL", "1")
    L = bl.Launcher(UI(), update_url=base + "/update.json", flavor="cpu")
    with pytest.raises(bl.Fail, match="model"):
        L.run(launch=False)
    assert L.installed() and not L.state.get("models_ok")
    monkeypatch.delenv("FAKE_FAIL")
    bl.Launcher(UI(), update_url=base + "/update.json").run(launch=False)
    assert bl.State(tmp_path / "home").get("models_ok") is True


def test_offer_gpu_switch_keeps_models(server, tmp_path, monkeypatch):
    base, H = server
    monkeypatch.setenv("BEAUTYHDR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BEAUTYHDR_RETRY_SCALE", "0")
    upd = publish(base, H, "1.0.0")
    upd["flavors"]["cuda"] = upd["flavors"]["cpu"]
    H.files["/update.json"] = json.dumps(upd).encode()
    monkeypatch.setattr(bl, "has_nvidia", lambda: False)
    L = bl.Launcher(UI(), update_url=base + "/update.json", flavor="cpu")
    L.run(launch=False)
    assert L.state.get("flavor") == "cpu"
    # lần sau máy có card: hỏi một lần, đồng ý thì chuyển sang cuda, model không tải lại
    monkeypatch.setattr(bl, "has_nvidia", lambda: True)
    ui = UI()
    L2 = bl.Launcher(ui, update_url=base + "/update.json")
    L2.run(launch=False)
    assert L2.state.get("flavor") == "cuda" and L2.state.get("models_ok") is True
