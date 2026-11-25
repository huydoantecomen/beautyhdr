"""Tải model từ Google Drive giả (trang xác nhận virus, hạn mức, link riêng tư, file sai) rồi dự phòng nguồn gốc."""
import hashlib
import http.server
import json
import threading
from dataclasses import replace

import pytest

from net_anh import drive, weights

DATA = b"model-bytes-" * 5000
SHA = hashlib.sha256(DATA).hexdigest()
FID = "A" * 33


class H(http.server.BaseHTTPRequestHandler):
    mode = "confirm"
    hits: list = []
    cut = None

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/octet-stream", rng=True):
        start = 0
        r = self.headers.get("Range")
        if rng and r:
            start = int(r.split("=")[1].split("-")[0])
        part = body[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        if self.cut and not start:
            part = part[: self.cut]
        self.wfile.write(part)

    def do_GET(self):
        H.hits.append(self.path)
        if self.path.startswith("/orig"):
            return self._send(DATA)
        if self.path.startswith("/download"):
            if H.mode == "quota":
                return self._send(b"<html>Sorry, download quota exceeded</html>", "text/html", rng=False)
            if H.mode == "private":
                return self._send(b"<html>Sign in accounts.google.com</html>", "text/html", rng=False)
            if H.mode == "wrong":
                return self._send(b"x" * len(DATA))
            if H.mode == "confirm" and "uuid=" not in self.path:
                port = self.server.server_port
                form = (f'<form id="d" action="http://127.0.0.1:{port}/download" method="get">'
                        f'<input type="hidden" name="id" value="{FID}"><input type="hidden" name="export" value="download">'
                        f'<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u-1"></form>')
                return self._send(form.encode(), "text/html", rng=False)
            return self._send(DATA)
        self.send_error(404)


@pytest.fixture
def env(tmp_path, monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    H.mode, H.hits, H.cut = "confirm", [], None
    monkeypatch.setattr(drive, "BASE", base)
    monkeypatch.setenv("NET_ANH_WEIGHTS", str(tmp_path / "w"))
    cfg = tmp_path / "d.json"
    cfg.write_text(json.dumps({"files": {"t.bin": f"https://drive.google.com/file/d/{FID}/view?usp=sharing"}}))
    monkeypatch.setenv("NET_ANH_DRIVE_JSON", str(cfg))
    m = replace(next(iter(weights.MANIFEST.values())), model_id="t", file="t.bin", size=len(DATA), sha256=SHA,
                url=base + "/orig", zip_member="")
    monkeypatch.setitem(weights.MANIFEST, "t", m)
    weights._DRIVE_BAD.clear()
    yield base, tmp_path
    srv.shutdown()


def run(env_):
    msgs = []
    p = weights.ensure("t", progress=msgs.append, retries=3)
    assert p.read_bytes() == DATA
    return msgs


def test_file_id_parsing():
    assert drive.file_id(f"https://drive.google.com/file/d/{FID}/view?usp=sharing") == FID
    assert drive.file_id(f"https://drive.google.com/open?id={FID}") == FID
    assert drive.file_id(FID) == FID
    with pytest.raises(drive.DriveError):
        drive.file_id("https://example.com/x")


def test_drive_confirm_page_then_download(env):
    msgs = run(env)
    assert any("Google Drive" in m for m in msgs)
    assert not any(h.startswith("/orig") for h in H.hits)  # không đụng nguồn gốc


@pytest.mark.parametrize("mode,word", [("quota", "hạn mức"), ("private", "chưa chia sẻ")])
def test_drive_blocked_falls_back_to_origin(env, mode, word):
    H.mode = mode
    msgs = run(env)
    assert any(word in m for m in msgs) and any(h.startswith("/orig") for h in H.hits)


def test_drive_wrong_content_falls_back_to_origin(env):
    H.mode = "wrong"
    msgs = run(env)
    assert any("sai nội dung" in m or "sai checksum" in m for m in msgs) and any(h.startswith("/orig") for h in H.hits)


def test_resume_partial_file(env):
    base, tmp = env
    H.cut = 20000  # lần đầu bị ngắt sau 20 KB
    msgs = run(env)
    assert any("làm tiếp" in m for m in msgs)


def test_no_drive_link_uses_origin(env, monkeypatch):
    base, tmp = env
    (tmp / "d.json").write_text(json.dumps({"files": {"t.bin": ""}}))
    run(env)
    assert not any(h.startswith("/download") for h in H.hits)


def test_not_enough_disk(env, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda p: type("U", (), {"free": 1000})())
    with pytest.raises(weights.ModelError, match="trống"):
        weights.ensure("t", retries=1)


def test_parse_folder_page_and_folder_config(env, monkeypatch):
    base, tmp = env
    items = [[FID, ["root"], "t.bin", "application/octet-stream"], ["B" * 33, ["root"], "sub", "application/vnd.google-apps.folder"]]
    enc = json.dumps([items]).replace("\\", "\\\\").replace("'", "\\'")
    html = f"<html><script>window['_DRIVE_ivd'] = '{enc}';</script></html>"
    assert drive.parse_folder_html(html) == {"t.bin": FID}
    with pytest.raises(drive.DriveError, match="chưa chia sẻ"):
        drive.parse_folder_html("<html>login</html>")
    # cấu hình chỉ có folder: app liệt kê qua trang giả rồi tải bằng id
    class P(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(html.encode())
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), P)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("BEAUTYHDR_DRIVE_PAGE", f"http://127.0.0.1:{srv.server_port}")
    drive._FOLDER_CACHE.clear()
    (tmp / "d.json").write_text(json.dumps({"folder": "https://drive.google.com/drive/folders/" + "C" * 33, "files": {}}))
    assert drive.links(None) == {"t.bin": FID}
    msgs = run(env)
    assert any("Google Drive" in m for m in msgs)
    srv.shutdown()
