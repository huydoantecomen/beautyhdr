"""Chế độ máy chủ: điện thoại (app Beauty HDR Android) gửi ảnh lên PC này để xử lý bằng model nặng nhất trên GPU.

Dùng chung pipeline với giao diện và dòng lệnh (Enhancer, recommend, level_settings), chỉ thêm lớp HTTP. Chỉ dùng thư
viện chuẩn của Python (http.server), không thêm gói cài đặt nào.

    python -m net_anh.server                 chạy máy chủ (cổng 8765), in mã ghép nối
    python -m net_anh.server --new-code      đổi mã ghép nối
    python -m net_anh.server --unpair-all    huỷ mọi điện thoại đã ghép nối
    NetAnh.exe --may-chu                     cửa sổ máy chủ (server_gui.py)

API phiên bản 1 (chi tiết: docs/API_MAY_CHU.md). Mọi câu trả lời lỗi là JSON {"error": "...", "code": "..."}.
  GET    /api/v1/info                 không cần mã: tên máy, GPU, đang bận, hàng đợi (để app kiểm tra máy chủ)
  POST   /api/v1/pair                 {"code": "123456", "device": "Pixel 8"} → {"token": "..."}
  GET    /api/v1/ping                 kiểm tra mã (token) còn hợp lệ
  POST   /api/v1/jobs?level=6&phone=auto&upscale=0&max_side=4096   thân = file ảnh → 202 {"id": ..., "position": n}
         tuỳ chọn (từ 0.10.2, info["features"]): restore=1 (khôi phục ảnh cũ: xoay, vá xước/bụi LaMa, màu phai),
         bg=RRGGBB (đổi nền đồng màu), subject=auto|child|man|woman|elder|adult|off, styles=<JSON kiểu mặt của bộ preset>,
         deshine=0..1 (từ 0.13.0: khử bóng loáng da)
  GET    /api/v1/jobs/<id>            {"state": queued|running|done|error|cancelled, "progress": "...", "report": {...}}
  GET    /api/v1/jobs/<id>/result     ảnh JPEG kết quả
  DELETE /api/v1/jobs/<id>            huỷ
Các bước gửi việc → hỏi trạng thái → lấy ảnh tách riêng để chạy được sau proxy/GPU thuê có giới hạn thời gian mỗi
yêu cầu (Cloudflare 100 s, RunPod, Modal), và để điện thoại mất sóng giữa chừng vẫn lấy lại được kết quả.

Bảo mật: chỉ máy đã ghép nối bằng mã 6 số (hiện trên PC) mới gửi ảnh được; mã đúng một lần rồi đổi mã mới; sai 5 lần
khoá 60 giây. Token lưu dạng băm SHA-256 trong ~/.net_anh/server.json. Trong mạng nhà, HTTP không mã hoá; ở ngoài
dùng Tailscale (mã hoá sẵn) hoặc Cloudflare Tunnel (HTTPS). Ảnh chỉ giữ trong RAM, xoá sau 15 phút.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from . import __version__

API = 1
PORT = 8765
DISCOVERY_PORT = 8766  # UDP: điện thoại phát "NETANH_DISCOVER_V1" trong wifi, máy chủ trả lời JSON
DISCOVER_MAGIC = b"NETANH_DISCOVER_V1"
MAX_UPLOAD_MB = 40
MAX_QUEUE = 8  # việc chờ tối đa (cả đang chạy)
KEEP_SECONDS = 15 * 60  # giữ kết quả để điện thoại lấy lại
MAX_FAILS, LOCK_SECONDS = 5, 60
LEVEL_NAMES = {"auto": "Tự động", 1: "Nhẹ", 2: "Vừa", 3: "Rõ", 4: "Rất rõ", 5: "Tối đa", 6: "Cao nhất"}
FEATURES = ("restore", "bg", "subject", "styles", "deshine")  # tham số thêm của POST /api/v1/jobs (từ 0.10.2; deshine từ 0.13.0)
MAX_STYLES_CHARS = 8192
IMAGE_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp",
             "image/bmp": ".bmp", "image/tiff": ".tif"}

Say = Callable[[str], None]


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _sniff_ext(data: bytes) -> str | None:
    """Đuôi file theo vài byte đầu (không tin Content-Type của bên gửi)."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    return None


# ---------------------------------------------------------------- cấu hình, ghép nối

class Pairing:
    """Mã ghép nối 6 số và danh sách điện thoại đã ghép (token băm), lưu trong ~/.net_anh/server.json."""

    def __init__(self, path: Path | None = None, static_tokens: list[str] | None = None):
        if path is None:
            from .history import data_dir
            path = data_dir() / "server.json"
        self.path = path
        self.lock = threading.Lock()
        self.fails = 0
        self.locked_until = 0.0
        self.static = {_hash(t) for t in (static_tokens or []) if t}
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("devices", {})
        if not self.data.get("code"):
            self.new_code()

    @property
    def code(self) -> str:
        return self.data["code"]

    @property
    def devices(self) -> dict:
        return self.data["devices"]

    def _save(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def new_code(self) -> str:
        with self.lock:
            self.data["code"] = f"{secrets.randbelow(10 ** 6):06d}"
            self._save()
            return self.code

    def unpair_all(self):
        with self.lock:
            self.data["devices"] = {}
            self._save()

    def pair(self, code: str, device: str) -> str:
        """Mã đúng → token mới (mã đổi luôn, mỗi mã chỉ dùng một lần). Sai → PermissionError; sai nhiều → TimeoutError."""
        with self.lock:
            now = time.time()
            if now < self.locked_until:
                raise TimeoutError(int(self.locked_until - now) + 1)
            if not hmac.compare_digest(str(code).strip().encode(), self.code.encode()):
                self.fails += 1
                if self.fails >= MAX_FAILS:
                    self.fails, self.locked_until = 0, now + LOCK_SECONDS
                raise PermissionError()
            self.fails = 0
            token = secrets.token_urlsafe(32)
            self.devices[_hash(token)] = {"device": device[:60] or "Điện thoại", "paired": now, "seen": now}
            self.data["code"] = f"{secrets.randbelow(10 ** 6):06d}"
            self._save()
            return token

    def check(self, token: str | None) -> str | None:
        """Tên điện thoại nếu token hợp lệ."""
        if not token:
            return None
        h = _hash(token)
        if h in self.static:
            return "máy được cấp sẵn mã"
        with self.lock:
            d = self.devices.get(h)
            if d is None:
                return None
            if time.time() - d.get("seen", 0) > 3600:  # ghi đĩa tối đa mỗi giờ
                d["seen"] = time.time()
                self._save()
            return d["device"]


# ---------------------------------------------------------------- hàng đợi xử lý

@dataclass
class Job:
    id: str
    data: bytes
    ext: str
    level: str | int
    phone: str
    upscale: int
    max_side: int
    quality: int
    device: str
    extra: dict = field(default_factory=dict)  # thông số thêm (khôi phục ảnh cũ, đổi nền, preset đối tượng)
    created: float = field(default_factory=time.time)
    state: str = "queued"  # queued | running | done | error | cancelled
    progress: str = "Đang chờ"
    error: str = ""
    result: bytes | None = None
    report: dict = field(default_factory=dict)
    finished: float = 0.0
    cancel: bool = False

    def public(self, position: int) -> dict:
        d = {"id": self.id, "state": self.state, "progress": self.progress, "position": position,
             "elapsed": round((self.finished or time.time()) - self.created, 1)}
        if self.error:
            d["error"] = self.error
        if self.report:
            d["report"] = self.report
        return d


def build_settings(eng, img, path: Path, level, phone_key: str, upscale: int):
    """Thông số như GUI/CLI: phân tích ảnh → Tự động (sáng, ảnh cũ, CLAHE) → mức nét người dùng chọn → hãng máy."""
    from .analysis import recommend
    from .levels import level_settings
    from .phone import PROFILES, detect
    from .settings import Settings

    an = eng.analyze(img, with_faces=True)  # máy chủ: luôn dò mặt (tải model dò mặt nếu chưa có)
    _, s, why = recommend(an, Settings(upscale=upscale))
    if level != "auto":
        s = level_settings(int(level), s.with_(upscale=upscale), an)
        why = [f"mức {level} · {LEVEL_NAMES[int(level)]} do điện thoại chọn"] + why[1:]  # why[0]: mức Tự động
    if phone_key == "auto" or phone_key not in PROFILES:
        key, desc = detect(path)
    else:
        key, desc = phone_key, PROFILES[phone_key].label
    return s.with_(phone=key, wb=-1.0, skin_tone=-1.0), an, why, desc


def parse_extra(q: dict) -> dict:
    """Tham số thêm của việc (khôi phục ảnh cũ, đổi nền, preset đối tượng) → trường của Settings. Sai thì ApiError 400."""
    from . import facestyle
    extra: dict = {}
    r = q.get("restore", "0")
    if r not in ("0", "1"):
        raise ApiError(400, "bad_restore", "restore phải là 0 hoặc 1")
    if r == "1":  # như preset "Khôi phục ảnh cũ": xoay thẳng, vá vết hỏng, phục hồi màu phai, không cân bằng trắng
        extra.update(repair=1.0, fade=1.0, auto_rotate=True, wb=0.0)
    bg = q.get("bg", "")
    if bg:
        t = bg.lstrip("#").lower()
        if len(t) != 6 or any(c not in "0123456789abcdef" for c in t):
            raise ApiError(400, "bad_bg", "bg phải là màu RRGGBB")
        extra["bg_color"] = (int(t[4:6], 16), int(t[2:4], 16), int(t[0:2], 16))  # B, G, R
    ds = q.get("deshine", "")
    if ds:  # khử bóng loáng da 0..1 (0 = tắt); bản Android cũ không gửi thì tắt như mặc định
        try:
            v = float(ds)
        except ValueError:
            raise ApiError(400, "bad_deshine", "deshine phải là số 0..1") from None
        if not 0.0 <= v <= 1.0:
            raise ApiError(400, "bad_deshine", "deshine phải là số 0..1")
        extra["deshine"] = v
    subj = q.get("subject", "")
    if subj:
        if subj not in ("auto", "off") + facestyle.SUBJECT_KEYS:
            raise ApiError(400, "bad_subject", "subject phải là auto, off, child, man, woman, elder hoặc adult")
        extra["subject"] = subj
    st = q.get("styles", "")
    if st:
        if len(st) > MAX_STYLES_CHARS:
            raise ApiError(400, "bad_styles", "styles quá dài")
        try:
            d = json.loads(st)
            styles = facestyle.parse_set({"name": "điện thoại", "styles": d}, []).styles
        except (ValueError, RecursionError, facestyle.PresetError):
            raise ApiError(400, "bad_styles", "styles không phải kiểu mặt hợp lệ") from None
        extra["face_styles"] = {k: v.to_dict() for k, v in styles.items()}
    return extra


class Worker:
    """Một luồng duy nhất giữ Enhancer (model nằm sẵn trong VRAM giữa các ảnh), xử lý lần lượt từng việc."""

    def __init__(self, device: str = "auto", say: Say | None = None):
        from .pipeline import Enhancer

        self.say = say or (lambda m: None)
        self.eng = Enhancer(device)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []  # việc chờ, theo thứ tự
        self.cv = threading.Condition()
        self.current: Job | None = None
        self.stopped = False
        threading.Thread(target=self._loop, name="net-anh-worker", daemon=True).start()

    # --- hàng đợi
    def submit(self, job: Job) -> int:
        with self.cv:
            self._cleanup()
            if len(self.order) + (1 if self.current else 0) >= MAX_QUEUE:
                raise OverflowError()
            self.jobs[job.id] = job
            self.order.append(job.id)
            self.cv.notify()
            return self.position(job)

    def position(self, job: Job) -> int:
        """0 = đang chạy hoặc xong, n = còn n việc phía trước + chính nó."""
        try:
            return self.order.index(job.id) + 1 + (1 if self.current else 0)
        except ValueError:
            return 0

    def get(self, job_id: str) -> Job | None:
        with self.cv:
            return self.jobs.get(job_id)

    def cancel(self, job: Job):
        with self.cv:
            job.cancel = True
            if job.id in self.order:
                self.order.remove(job.id)
                job.state, job.progress, job.finished, job.data = "cancelled", "Đã huỷ", time.time(), b""

    def busy(self) -> tuple[bool, int]:
        with self.cv:
            return self.current is not None, len(self.order)

    def _cleanup(self):
        now = time.time()
        for jid in [j.id for j in self.jobs.values() if j.finished and now - j.finished > KEEP_SECONDS]:
            del self.jobs[jid]

    # --- xử lý
    def _loop(self):
        while not self.stopped:
            with self.cv:
                while not self.order and not self.stopped:
                    self.cv.wait(timeout=30)
                    self._cleanup()
                if self.stopped:
                    return
                job = self.jobs[self.order.pop(0)]
                self.current = job
            try:
                self._run(job)
            finally:
                with self.cv:
                    self.current = None
                    job.data = b""  # không giữ ảnh gốc của người dùng lâu hơn cần
                    job.finished = time.time()

    def _run(self, job: Job):
        import cv2

        from .pipeline import Cancelled, ImageError, load_image, resize_long

        job.state = "running"
        eng = self.eng
        eng.cancel_requested = lambda: job.cancel
        eng.say = lambda m: setattr(job, "progress", m)
        t0 = time.perf_counter()
        tmp = None
        try:
            # load_image cần file (kiểm tra định dạng, xoay theo EXIF, đọc EXIF hãng máy): ghi tạm rồi xoá ngay
            fd, tmp = tempfile.mkstemp(suffix=job.ext, prefix="netanh_")
            with os.fdopen(fd, "wb") as f:
                f.write(job.data)
            job.progress = "Đang phân tích ảnh…"
            img = load_image(tmp)
            s, an, why, phone_desc = build_settings(eng, img, Path(tmp), job.level, job.phone, job.upscale)
            if job.extra:
                s = s.with_(**job.extra)
            eng._check()
            self.say(f"Ảnh {img.shape[1]}×{img.shape[0]} từ {job.device}: mức {LEVEL_NAMES.get(job.level, job.level)}, "
                     f"model mặt {s.face_model}, nền {s.sr_model}")
            res, rep = eng.process(img, s)
            notes = list(rep.notes)
            h, w = res.shape[:2]
            if job.max_side and max(h, w) > job.max_side:
                res = resize_long(res, job.max_side)
                notes.append(f"Thu nhỏ còn cạnh dài {job.max_side} px theo yêu cầu của điện thoại")
            job.progress = "Đang nén ảnh…"
            ok, buf = cv2.imencode(".jpg", res, [cv2.IMWRITE_JPEG_QUALITY, job.quality])
            if not ok:
                raise IOError("không nén được ảnh JPEG")
            job.result = buf.tobytes()
            from .levels import auto_level
            job.report = {
                "mode": rep.mode, "device": rep.device, "seconds": round(time.perf_counter() - t0, 2),
                "faces_found": rep.faces_found, "faces_restored": rep.faces_restored, "ai_fallback": rep.ai_fallback,
                "notes": notes,
                "width": res.shape[1], "height": res.shape[0], "input_width": img.shape[1],
                "input_height": img.shape[0],
                "level": job.level if job.level != "auto" else auto_level(an)[0],
                "face_model": s.face_model if s.mode in ("portrait", "old") else None, "bg_model": s.sr_model,
                "phone": s.phone, "phone_desc": phone_desc, "reasons": why, "analysis": an.summary(),
                "server": socket.gethostname(), "summary": rep.summary(),
            }
            job.state, job.progress = "done", "Xong"
            self.say(f"  xong sau {job.report['seconds']:.1f} s ({rep.device}), {res.shape[1]}×{res.shape[0]}")
        except (Cancelled, InterruptedError):
            job.state, job.progress = "cancelled", "Đã huỷ"
            self.say("  điện thoại đã huỷ")
        except ImageError as e:
            job.state, job.error = "error", str(e)
            self.say(f"  ảnh lỗi: {e}")
        except Exception as e:  # hết VRAM/RAM, lỗi model…: báo cho điện thoại, máy chủ vẫn chạy tiếp
            job.state, job.error = "error", f"{e.__class__.__name__}: {e}"
            self.say("  LỖI: " + traceback.format_exc(limit=3))
        finally:
            eng.cancel_requested = lambda: False
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------- HTTP

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


class Server:
    def __init__(self, host: str = "0.0.0.0", port: int = PORT, device: str = "auto", say: Say | None = None,
                 pairing: Pairing | None = None, name: str | None = None, allow_pair: bool = True,
                 discovery: bool = True):
        self.say = say or (lambda m: print(m, flush=True))
        self.pairing = pairing or Pairing(static_tokens=os.environ.get("NET_ANH_TOKENS", "").split(","))
        self.name = name or socket.gethostname()
        self.allow_pair = allow_pair
        self.worker = Worker(device, self.say)
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.udp: socket.socket | None = None
        if discovery:
            try:
                self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.udp.bind(("", DISCOVERY_PORT))
                self.udp.settimeout(1.0)
            except OSError as e:
                self.say(f"Không mở được cổng tìm máy tự động UDP {DISCOVERY_PORT} ({e}); điện thoại vẫn nhập địa chỉ tay được")
                self.udp = None

    def info(self) -> dict:
        running, waiting = self.worker.busy()
        dev = self.worker.eng.dev
        return {"app": "netanh", "api": API, "version": __version__, "name": self.name, "device": dev.name,
                "gpu": dev.kind != "cpu", "busy": running, "queue": waiting, "port": self.port,
                "max_upload_mb": MAX_UPLOAD_MB, "levels": [1, 2, 3, 4, 5, 6], "pairing": self.allow_pair,
                "features": list(FEATURES)}

    # --- vòng chạy
    def serve_forever(self):
        if self.udp:
            threading.Thread(target=self._discovery_loop, name="net-anh-discovery", daemon=True).start()
        self.httpd.serve_forever(poll_interval=0.5)

    def start(self) -> "Server":
        threading.Thread(target=self.serve_forever, name="net-anh-http", daemon=True).start()
        return self

    def stop(self):
        self.worker.stopped = True
        with self.worker.cv:
            self.worker.cv.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.udp:
            self.udp.close()
            self.udp = None

    def _discovery_loop(self):
        while self.udp:
            try:
                data, addr = self.udp.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            if data.strip() == DISCOVER_MAGIC:
                reply = {k: v for k, v in self.info().items() if k in ("app", "api", "version", "name", "device", "gpu", "port")}
                try:
                    self.udp.sendto(json.dumps(reply, ensure_ascii=False).encode(), addr)
                except OSError:
                    pass

    # --- xử lý yêu cầu
    def _handler(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            server_version = f"NetAnh/{__version__}"
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # im lặng: log ý nghĩa do Worker ghi
                pass

            def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8", extra=None):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _json(self, status: int, obj: dict):
                self._send(status, json.dumps(obj, ensure_ascii=False).encode())

            def _auth(self) -> str:
                h = self.headers.get("Authorization", "")
                token = h[7:].strip() if h.lower().startswith("bearer ") else None
                dev = srv.pairing.check(token)
                if dev is None:
                    raise ApiError(401, "unauthorized", "Điện thoại chưa ghép nối với máy chủ này (hoặc đã bị huỷ). "
                                                        "Hãy nhập mã ghép nối hiện trên máy tính.")
                return dev

            def _body(self, limit_mb: float) -> bytes:
                try:
                    n = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    raise ApiError(411, "length_required", "Thiếu Content-Length")
                if n > limit_mb * 1e6:
                    raise ApiError(413, "too_large", f"Ảnh lớn hơn {limit_mb:g} MB")
                data = b""
                while len(data) < n:
                    chunk = self.rfile.read(min(1 << 20, n - len(data)))
                    if not chunk:
                        raise ApiError(400, "incomplete", "Ảnh gửi lên bị đứt giữa chừng")
                    data += chunk
                return data

            def _route(self, method: str):
                u = urlparse(self.path)
                parts = [p for p in u.path.split("/") if p]
                q = {k: v[-1] for k, v in parse_qs(u.query).items()}
                if parts[:2] != ["api", "v1"]:
                    if method == "GET" and not parts:
                        return self._send(200, f"Beauty HDR {__version__}: máy chủ đang chạy. Mở app Beauty HDR trên "
                                               f"điện thoại để kết nối.\n".encode(), "text/plain; charset=utf-8")
                    raise ApiError(404, "not_found", "Không có đường dẫn này")
                rest = parts[2:]
                if method == "GET" and rest == ["info"]:
                    return self._json(200, srv.info())
                if method == "POST" and rest == ["pair"]:
                    if not srv.allow_pair:
                        raise ApiError(403, "pairing_disabled", "Máy chủ này không cho ghép nối bằng mã")
                    try:
                        req = json.loads(self._body(0.01) or b"{}")
                        code, device = str(req.get("code", "")), str(req.get("device", ""))
                    except (ValueError, AttributeError):
                        raise ApiError(400, "bad_request", "Yêu cầu ghép nối không hợp lệ")
                    try:
                        token = srv.pairing.pair(code, device)
                    except PermissionError:
                        raise ApiError(403, "wrong_code", "Mã ghép nối không đúng. Xem mã trên cửa sổ máy chủ ở máy tính.")
                    except TimeoutError as e:
                        raise ApiError(429, "locked", f"Nhập sai nhiều lần, thử lại sau {e.args[0]} giây")
                    srv.say(f"Đã ghép nối: {device or 'điện thoại'} ({self.client_address[0]}). Mã mới: {srv.pairing.code}")
                    return self._json(200, {"token": token, "name": srv.name})
                device = self._auth()
                if method == "GET" and rest == ["ping"]:
                    return self._json(200, {"ok": True, "device": device, **srv.info()})
                if method == "POST" and rest == ["jobs"]:
                    return self._submit(q, device)
                if len(rest) >= 2 and rest[0] == "jobs":
                    job = srv.worker.get(rest[1])
                    if job is None:
                        raise ApiError(404, "no_job", "Không tìm thấy việc này (có thể đã quá 15 phút hoặc máy chủ "
                                                      "vừa khởi động lại)")
                    if method == "GET" and len(rest) == 2:
                        with srv.worker.cv:
                            pos = srv.worker.position(job)
                        return self._json(200, job.public(pos))
                    if method == "GET" and rest[2:] == ["result"]:
                        if job.state != "done" or job.result is None:
                            raise ApiError(409, "not_ready", "Ảnh chưa xử lý xong")
                        return self._send(200, job.result, "image/jpeg")
                    if method == "DELETE" and len(rest) == 2:
                        srv.worker.cancel(job)
                        return self._json(200, job.public(0))
                raise ApiError(404, "not_found", "Không có đường dẫn này")

            def _submit(self, q: dict, device: str):
                level = q.get("level", "6")
                if level != "auto":
                    if level not in {"1", "2", "3", "4", "5", "6"}:
                        raise ApiError(400, "bad_level", "level phải là auto hoặc 1–6")
                    level = int(level)
                try:
                    upscale = int(q.get("upscale", "0"))
                    max_side = int(q.get("max_side", "0"))
                    quality = int(q.get("quality", "94"))
                except ValueError:
                    raise ApiError(400, "bad_request", "upscale, max_side, quality phải là số")
                if upscale not in (0, 1, 2, 4):
                    raise ApiError(400, "bad_upscale", "upscale phải là 0, 1, 2 hoặc 4")
                extra = parse_extra(q)
                data = self._body(MAX_UPLOAD_MB)
                ext = _sniff_ext(data)
                if ext is None:
                    raise ApiError(415, "bad_image", "Không phải ảnh JPG, PNG, WebP, BMP hay TIFF")
                job = Job(id=uuid.uuid4().hex, data=data, ext=ext, level=level, phone=q.get("phone", "auto"),
                          upscale=upscale, max_side=max(0, max_side), quality=min(100, max(60, quality)),
                          device=device, extra=extra)
                try:
                    pos = srv.worker.submit(job)
                except OverflowError:
                    raise ApiError(503, "queue_full", "Máy chủ đang bận nhiều ảnh, thử lại sau")
                return self._json(202, job.public(pos))

            def _dispatch(self, method: str):
                try:
                    self._route(method)
                except ApiError as e:
                    self._json(e.status, {"error": str(e), "code": e.code})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    srv.say("LỖI máy chủ: " + traceback.format_exc(limit=3))
                    try:
                        self._json(500, {"error": f"Lỗi máy chủ: {e.__class__.__name__}", "code": "internal"})
                    except OSError:
                        pass

            def do_GET(self):
                self._dispatch("GET")

            def do_HEAD(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

            def do_DELETE(self):
                self._dispatch("DELETE")

        return Handler


def local_addresses() -> list[str]:
    """Địa chỉ IPv4 của máy trong mạng (wifi/LAN, Tailscale 100.x), để người dùng nhập vào điện thoại."""
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # không gửi gói nào, chỉ để hệ điều hành chọn card mạng chính
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(info[4][0])
    except OSError:
        pass
    out = []
    for ip in ips:
        if ip not in out and not ip.startswith("127."):
            out.append(ip)
    return out


def describe_addresses(port: int) -> list[str]:
    lines = []
    for ip in local_addresses():
        tag = " (Tailscale, dùng được khi ở ngoài)" if ip.startswith("100.") else ""
        lines.append(f"{ip}:{port}{tag}")
    return lines


def prepare_models(say: Say) -> None:
    """Tải sẵn model của mức 6 Cao nhất để lần gửi ảnh đầu không phải chờ tải ~450 MB."""
    from . import weights
    for m in ("retinaface", "parsenet", "esrgan_compact", "gfpgan", "restoreformer_pp", "dat2_web"):
        weights.ensure(m, say)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="net_anh.server", description="Máy chủ Beauty HDR cho điện thoại")
    p.add_argument("--host", default="0.0.0.0", help="mặc định 0.0.0.0 (mọi card mạng)")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--name", help="tên máy chủ hiện trên điện thoại (mặc định: tên máy tính)")
    p.add_argument("--new-code", action="store_true", help="đổi mã ghép nối rồi chạy")
    p.add_argument("--unpair-all", action="store_true", help="huỷ mọi điện thoại đã ghép nối rồi chạy")
    p.add_argument("--no-pair", action="store_true",
                   help="tắt ghép nối bằng mã; chỉ nhận token đặt sẵn trong biến môi trường NET_ANH_TOKENS (GPU thuê)")
    p.add_argument("--no-discovery", action="store_true", help="tắt trả lời tìm máy tự động (UDP 8766)")
    p.add_argument("--prepare", action="store_true", help="tải sẵn model mức 6 Cao nhất trước khi nhận ảnh")
    a = p.parse_args(argv)
    say = lambda m: print(m, flush=True)  # noqa: E731
    pairing = Pairing(static_tokens=os.environ.get("NET_ANH_TOKENS", "").split(","))
    if a.unpair_all:
        pairing.unpair_all()
        say("Đã huỷ mọi điện thoại đã ghép nối.")
    if a.new_code:
        pairing.new_code()
    if a.prepare:
        prepare_models(say)
    try:
        srv = Server(a.host, a.port, a.device, say, pairing, a.name, allow_pair=not a.no_pair,
                     discovery=not a.no_discovery)
    except OSError as e:
        say(f"Không mở được cổng {a.port}: {e}. Có thể máy chủ Beauty HDR đang chạy ở cửa sổ khác.")
        return 2
    say(f"Beauty HDR {__version__} · máy chủ đang chạy trên {srv.worker.eng.dev.name}")
    say("Địa chỉ để nhập vào điện thoại (cùng wifi):")
    for line in describe_addresses(srv.port) or [f"<IP máy này>:{srv.port}"]:
        say("   " + line)
    if not a.no_pair:
        say(f"MÃ GHÉP NỐI: {pairing.code}   (đã ghép {len(pairing.devices)} điện thoại)")
    say("Dừng: Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        say("Đang dừng…")
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
