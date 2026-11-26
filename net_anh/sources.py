"""Nguồn xử lý của app PC: Trên máy này (mặc định), Máy chủ GPU (một máy chủ Beauty HDR khác), API ngoài.

- **Trên máy** ("local"): pipeline của máy này như mọi bản trước.
- **Máy chủ GPU** ("server"): gửi ảnh tới một máy chủ Beauty HDR (PC khác trong nhà, GPU thuê) qua API v1 của
  server.py: ghép nối bằng mã 6 số một lần, sau đó gửi ảnh → hỏi trạng thái → lấy ảnh. Gửi kèm đủ bộ thông số của
  app (`settings=<JSON>`, tính năng "settings" từ 0.16.0) nên kết quả giống chạy trên máy; máy chủ cũ không có tính năng
  này thì chỉ gửi mức nét và các tuỳ chọn API cũ hiểu được.
- **API ngoài** ("api"): gửi thẳng tới fal.ai / Stability AI / Replicate bằng khoá API người dùng tự nhập (lưu trên máy
  này ở ~/.net_anh/providers.json, không gửi đi đâu khác). Ảnh gửi đi bỏ sạch EXIF (providers.encode_clean). Kết quả được
  kiểm giữ khuôn mặt (ArcFace, như máy chủ): mặt bị đổi hoặc API lỗi thì xử lý lại bằng máy này và ghi lý do.

Cả hai nguồn ngoài đều TẮT mặc định và phải hỏi người dùng đồng ý gửi ảnh ra khỏi máy (gui.py: ConsentDialog).
Mạng đi qua providers.transport (test thay bằng hàm giả), chỉ dùng thư viện chuẩn.
"""
from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import providers
from .pipeline import Report
from .settings import Settings

SOURCES = (
    ("local", "Trên máy này"),
    ("server", "Máy chủ GPU (Beauty HDR)"),
    ("api", "API ngoài (fal.ai, Stability, Replicate)"),
)
DEFAULT_PORT = 8765
POLL_SECONDS = 1.0
MAX_WAIT = 30 * 60
# nhà cung cấp API ngoài người dùng nhập khoá được trên PC (khoá trong providers.json)
API_PROVIDERS = (
    ("fal", "fal.ai Clarity Upscaler", "https://fal.ai/dashboard/keys"),
    ("stability", "Stability AI Upscale", "https://platform.stability.ai/account/keys"),
    ("replicate", "Replicate (model tự chọn)", "https://replicate.com/account/api-tokens"),
)

Progress = Callable[[str], None]
Cancelled = Callable[[], bool]


class SourceError(Exception):
    """Lỗi nguồn xử lý ngoài; `code` ngắn (server: mã lỗi API v1)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- máy chủ GPU (API v1)

def normalize_url(text: str) -> str:
    """'192.168.1.5' → 'http://192.168.1.5:8765'; giữ https và cổng nếu có; bỏ đường dẫn thừa."""
    t = (text or "").strip().rstrip("/")
    if not t:
        raise SourceError("bad_url", "Chưa nhập địa chỉ máy chủ")
    if "://" not in t:
        t = "http://" + t
    u = urllib.parse.urlsplit(t)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SourceError("bad_url", f"Địa chỉ máy chủ không hợp lệ: {text}")
    host = u.hostname if ":" not in u.hostname else f"[{u.hostname}]"
    port = u.port or (DEFAULT_PORT if u.scheme == "http" else None)
    return f"{u.scheme}://{host}" + (f":{port}" if port else "")


def _call(method: str, url: str, token: str = "", body: bytes | None = None, ctype: str = "",
          timeout: float = 30) -> tuple[int, bytes]:
    hdr = {}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    if ctype:
        hdr["Content-Type"] = ctype
    try:
        st, _, raw = providers._http(method, url, hdr, body, timeout)
    except providers.ProviderError as e:
        raise SourceError("network", f"Không kết nối được máy chủ ({e})") from None
    return st, raw


def _json(st: int, raw: bytes) -> dict:
    try:
        obj = json.loads(raw or b"{}")
    except ValueError:
        obj = {}
    if not isinstance(obj, dict):
        obj = {}
    if st >= 400:
        raise SourceError(obj.get("code") or f"http_{st}", obj.get("error") or f"Máy chủ trả lỗi {st}")
    return obj


def server_info(url: str) -> dict:
    st, raw = _call("GET", url + "/api/v1/info", timeout=8)
    info = _json(st, raw)
    if info.get("app") != "netanh" or info.get("api") != 1:
        raise SourceError("not_beautyhdr", "Địa chỉ này không phải máy chủ Beauty HDR")
    return info


def pair(url: str, code: str, device: str) -> str:
    body = json.dumps({"code": code.strip(), "device": device[:60]}).encode()
    st, raw = _call("POST", url + "/api/v1/pair", body=body, ctype="application/json", timeout=10)
    tok = _json(st, raw).get("token")
    if not tok:
        raise SourceError("bad_pair", "Máy chủ không trả mã kết nối")
    return tok


def ping(url: str, token: str) -> bool:
    st, raw = _call("GET", url + "/api/v1/ping", token, timeout=8)
    return st < 400


def server_params(s: Settings, level, features: list[str], quality: int = 95) -> dict:
    """Tham số POST /api/v1/jobs cho bộ thông số `s`. Máy chủ có "settings": gửi đủ (giống chạy trên máy); không có thì
    chỉ gửi các tuỳ chọn API cũ hiểu (mức nét, khôi phục, đổi nền, đối tượng, khử bóng)."""
    q = {"level": str(level) if level else "auto", "upscale": str(s.upscale if s.upscale in (0, 1, 2, 4) else 0),
         "quality": str(quality), "phone": s.phone or "auto"}
    if "settings" in features:
        d = s.to_dict()
        d.pop("damage_mask", None)
        d.pop("tile", None)
        q["settings"] = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        return q
    if "restore" in features and (s.repair > 0 or s.fade > 0):
        q["restore"] = "1"
    if "bg" in features and s.bg_color:
        b, g, r = s.bg_color
        q["bg"] = f"{r:02x}{g:02x}{b:02x}"
    if "subject" in features:
        q["subject"] = s.subject
    if "styles" in features and s.face_styles:
        q["styles"] = json.dumps(s.face_styles, separators=(",", ":"))
    if "deshine" in features and s.deshine > 0:
        q["deshine"] = f"{s.deshine:.2f}"
    return q


def lost_on_old_server(s: Settings, features: list[str]) -> list[str]:
    """Những tuỳ chọn máy chủ cũ (không có "settings") bỏ qua, để báo người dùng."""
    if "settings" in features:
        return []
    out = []
    if s.look:
        out.append("tông mẫu")
    if s.scene_vivid > 0:
        out.append("nâng màu cảnh" + (" và tô màu AI" if s.scene_ai > 0 else ""))
    if s.damage_mask:
        out.append("vùng hỏng tự tô")
    return out


def encode_for_upload(img: np.ndarray) -> bytes:
    """Ảnh gửi lên máy chủ: PNG không mất chất lượng nếu ≤ 30 MB, không thì JPEG 97 (giới hạn tải lên 40 MB)."""
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if ok and buf.nbytes <= 30 * 1024 * 1024:
        return buf.tobytes()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    if not ok:
        raise SourceError("encode", "Không nén được ảnh để gửi")
    return buf.tobytes()


def run_on_server(url: str, token: str, img: np.ndarray, params: dict, progress: Progress | None = None,
                  cancel: Cancelled | None = None) -> tuple[np.ndarray, dict]:
    """Gửi ảnh, chờ, lấy kết quả. Trả (ảnh BGR, report của máy chủ). Huỷ → providers.ProviderCancelled."""
    say = progress or (lambda m: None)
    say("Đang gửi ảnh lên máy chủ GPU…")
    st, raw = _call("POST", url + "/api/v1/jobs?" + urllib.parse.urlencode(params), token,
                    encode_for_upload(img), "application/octet-stream", timeout=300)
    job = _json(st, raw)
    jid = job.get("id")
    if not jid:
        raise SourceError("bad_reply", "Máy chủ không trả mã việc")
    t0 = time.time()
    last = ""
    while True:
        if cancel and cancel():
            try:
                _call("DELETE", f"{url}/api/v1/jobs/{jid}", token, timeout=8)
            except SourceError:
                pass
            raise providers.ProviderCancelled()
        st, raw = _call("GET", f"{url}/api/v1/jobs/{jid}", token, timeout=15)
        j = _json(st, raw)
        state = j.get("state")
        if state == "done":
            break
        if state in ("error", "cancelled"):
            raise SourceError("job_" + state, "Máy chủ: " + (j.get("error") or ("đã huỷ" if state == "cancelled"
                                                                                else "lỗi khi xử lý")))
        msg = j.get("progress") or ""
        if state == "queued" and j.get("position"):
            msg = f"Đang chờ trên máy chủ (vị trí {j['position']})"
        if msg and msg != last:
            say(msg if msg.startswith(("Đang", "Máy chủ")) else "Máy chủ: " + msg)
            last = msg
        if time.time() - t0 > MAX_WAIT:
            raise SourceError("timeout", f"Máy chủ xử lý quá {MAX_WAIT // 60} phút")
        time.sleep(POLL_SECONDS)
    say("Đang tải ảnh kết quả từ máy chủ…")
    st, raw = _call("GET", f"{url}/api/v1/jobs/{jid}/result", token, timeout=120)
    if st >= 400:
        _json(st, raw)
    try:
        out = providers.decode(raw)
    except providers.ProviderError as e:
        raise SourceError("bad_image", str(e)) from None
    return out, j.get("report") or {}


def report_from_server(rep: dict, url: str) -> Report:
    host = rep.get("server") or urllib.parse.urlsplit(url).hostname or url
    r = Report(mode=rep.get("mode", ""), device=f"máy chủ {host} ({rep.get('device', '?')})",
               seconds=float(rep.get("seconds") or 0.0), faces_found=int(rep.get("faces_found") or 0),
               faces_restored=int(rep.get("faces_restored") or 0), ai_fallback=bool(rep.get("ai_fallback")),
               notes=[str(n) for n in rep.get("notes") or []][:20], face_model=rep.get("face_model") or "",
               bg_model=rep.get("bg_model") or "", device_kind="remote")
    return r


def remote_path() -> Path:
    from .history import data_dir
    return data_dir() / "remote.json"


def load_remote(path: Path | None = None) -> dict:
    """{"url", "token", "name", "device", "features", "trusted": [url đã đồng ý không hỏi lại]}; lỗi/thiếu → {}."""
    try:
        d = json.loads((path or remote_path()).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_remote(d: dict, path: Path | None = None) -> None:
    path = path or remote_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)


# ---------------------------------------------------------------- API ngoài (khoá do người dùng nhập)

def save_api_key(provider: str, key: str | None, extra: dict | None = None, path: Path | None = None) -> Path:
    """Ghi khoá (và model Replicate) vào providers.json của máy này. Khoá rỗng = xoá khoá; None = giữ khoá cũ."""
    if provider not in {p for p, _, _ in API_PROVIDERS}:
        raise ValueError(provider)
    path = path or providers.config_path()
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except (OSError, ValueError):
        cfg = {}
    sec = cfg.get(provider) if isinstance(cfg.get(provider), dict) else {}
    if key is not None:  # None = giữ khoá đã lưu (chỉ đổi model)
        key = key.strip()
        if key:
            sec["key"] = key
        else:
            sec.pop("key", None)
    for k, v in (extra or {}).items():
        if v in (None, ""):
            sec.pop(k, None)
        else:
            sec[k] = v
    cfg[provider] = sec
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return path


def has_key(provider: str, cfg: dict | None = None) -> bool:
    cfg = providers.load_config() if cfg is None else cfg
    return bool((cfg.get(provider) or {}).get("key"))


def mask_key(key: str) -> str:
    return "" if not key else ("•" * 6 + key[-4:] if len(key) > 8 else "•" * len(key))


@dataclass
class ExternalResult:
    img: np.ndarray
    used_api: bool  # False = API lỗi hoặc đổi mặt → đã phải xử lý bằng máy này (gọi lại pipeline ở gui)
    note: str
    ids: list


def run_api(eng, e: "providers.Engine", img: np.ndarray, upscale: int, progress: Progress | None = None,
            cancel: Cancelled | None = None, guard_ids: float = providers.GUARD_IDS) -> ExternalResult:
    """Gửi ảnh tới API ngoài rồi kiểm giữ mặt bằng ArcFace trên máy này (`eng`: Enhancer). API lỗi hoặc mặt bị đổi →
    used_api=False và lý do (người gọi tự chạy pipeline của máy này). Huỷ → providers.ProviderCancelled."""
    say = progress or (lambda m: None)
    try:
        out = e.run(img, upscale, progress=say, cancel=cancel)
    except providers.ProviderCancelled:
        raise
    except providers.ProviderError as err:
        return ExternalResult(img, False, f"{e.label} lỗi ({err.code}): {err}. Đã xử lý bằng máy này.", [])
    except Exception as err:  # lỗi bất ngờ ở bộ nối: vẫn ra ảnh bằng máy này
        return ExternalResult(img, False, f"{e.label} lỗi: {err.__class__.__name__}: {err}. Đã xử lý bằng máy này.", [])
    say("Đang kiểm giữ khuôn mặt…")
    try:
        ids = providers.face_ids(eng, img, out)
    except Exception as err:
        return ExternalResult(out, True, f"Không kiểm được khuôn mặt ({err.__class__.__name__}); dùng ảnh của {e.label}", [])
    low = [x for x in ids if x < guard_ids]
    if low:
        return ExternalResult(img, False, f"{e.label} làm đổi khuôn mặt (IDS {min(low):.2f} dưới ngưỡng {guard_ids:.2f}). "
                                          "Đã xử lý bằng máy này.", ids)
    note = f"Xử lý bằng {e.label} ({e.price})"
    if ids:
        note += f"; giữ danh tính {len(ids)} mặt, IDS thấp nhất {min(ids):.2f}"
    return ExternalResult(out, True, note, ids)


def report_from_api(e: "providers.Engine", r: ExternalResult, seconds: float) -> Report:
    return Report(mode="external", device=f"{e.label} (dịch vụ ngoài)", seconds=seconds, faces_found=len(r.ids),
                  notes=[r.note], bg_model=e.label, device_kind="external")


__all__ = ["SOURCES", "API_PROVIDERS", "SourceError", "normalize_url", "server_info", "pair", "ping", "server_params",
           "lost_on_old_server", "run_on_server", "load_remote", "save_remote", "report_from_server", "save_api_key", "has_key", "mask_key", "run_api",
           "report_from_api", "ExternalResult"]
