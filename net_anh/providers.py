"""Cổng kết nối API xử lý ảnh bên ngoài (fal.ai, Stability AI, Replicate) cho máy chủ Beauty HDR.

Ý tưởng: app (PC, Android) chỉ nói chuyện với máy chủ Beauty HDR qua API v1. Máy chủ chọn "nguồn xử lý" cho từng
việc: `local` (pipeline của máy này, mặc định) hoặc một API ngoài. Khoá API chỉ nằm trên máy chủ (biến môi trường hoặc
~/.net_anh/providers.json), không bao giờ gửi cho điện thoại, nên thêm/bớt API không phải phát hành lại app.

Riêng tư: ảnh gửi ra ngoài được giải mã rồi nén lại (mất toàn bộ EXIF, vị trí GPS, tên máy), thu nhỏ theo giới hạn của
từng API, và chỉ gửi khi app báo người dùng đã đồng ý (consent=1). Kết quả từ API ngoài được kiểm giữ danh tính (IDS,
ArcFace) ở server.py: mặt bị đổi quá ngưỡng thì dùng kết quả của pipeline máy này thay thế.

Cấu hình (ưu tiên biến môi trường cho khoá):
    FAL_KEY=...                 fal.ai (Clarity Upscaler)
    STABILITY_API_KEY=...       Stability AI (Upscale Conservative, Fast)
    REPLICATE_API_TOKEN=...     Replicate (model tự chọn trong providers.json)
~/.net_anh/providers.json (mọi trường tuỳ chọn):
    {"fal": {"key": "...", "creativity": 0.25, "resemblance": 0.75},
     "stability": {"key": "...", "prompt": "..."},
     "replicate": {"key": "...", "model": "owner/name[:version]", "image_field": "image",
                   "input": {"scale": 2}, "label": "Tên hiện trên app", "price": "~0,01 USD/ảnh"},
     "disabled": ["stability-fast"], "guard_ids": 0.7}

Dòng lệnh:
    python -m net_anh.providers                         liệt kê nguồn đã cấu hình
    python -m net_anh.providers ANH.jpg --engine fal-clarity -o ra.png [--yes]   thử một API (không cần máy chủ)

Chỉ dùng thư viện chuẩn (urllib) + OpenCV/numpy đã có. Mạng đi qua proxy hệ thống nếu có (HTTPS_PROXY).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

LOCAL = "local"
GUARD_IDS = 0.70  # IDS dưới mức này coi như API đã đổi mặt (pipeline máy này đạt ~0,96 ở mặt lớn)
HTTP_TIMEOUT = 120
POLL_SECONDS = 1.5
MAX_WAIT = 600  # tối đa 10 phút cho một ảnh ở API ngoài
USER_AGENT = "BeautyHDR-gateway"

Progress = Callable[[str], None]
Cancelled = Callable[[], bool]


class ProviderError(Exception):
    """Lỗi từ API ngoài; `code` ngắn để app hiển thị/ghi log, `status` HTTP nếu có."""

    def __init__(self, code: str, message: str, status: int = 0):
        super().__init__(message)
        self.code, self.status = code, status


class ProviderCancelled(Exception):
    pass


# ---------------------------------------------------------------- HTTP (thay được trong test)

def _urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float):
    req = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()) if e.headers else {}, e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderError("network", f"Không kết nối được API ({getattr(e, 'reason', e)})") from None


transport = _urllib_transport  # test thay bằng hàm giả: (method, url, headers, body, timeout) → (status, headers, bytes)


def _http(method: str, url: str, headers: dict | None = None, body: bytes | None = None,
          timeout: float = HTTP_TIMEOUT) -> tuple[int, dict, bytes]:
    return transport(method, url, headers or {}, body, timeout)


def _json_or_error(status: int, data: bytes, who: str) -> dict:
    try:
        obj = json.loads(data or b"{}")
    except ValueError:
        obj = {}
    if status >= 400:
        raise ProviderError(_status_code(status), f"{who} trả lỗi {status}: {_error_text(obj, data)}", status)
    return obj if isinstance(obj, dict) else {"value": obj}


def _status_code(status: int) -> str:
    return {401: "bad_key", 403: "forbidden", 402: "no_credit", 413: "too_large", 422: "rejected",
            429: "rate_limited"}.get(status, "api_error")


def _error_text(obj, raw: bytes) -> str:
    if isinstance(obj, dict):
        for k in ("detail", "error", "errors", "message", "name"):
            if obj.get(k):
                v = obj[k]
                return (", ".join(map(str, v)) if isinstance(v, list) else str(v))[:300]
    return raw[:200].decode("utf-8", "replace")


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    b = "----netanh" + uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (fname, data, ctype) in files.items():
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


# ---------------------------------------------------------------- chuẩn bị ảnh

def fit_pixels(img: np.ndarray, max_pixels: int, max_side: int = 0, min_side: int = 0) -> np.ndarray:
    """Thu nhỏ (giữ tỉ lệ) để vừa giới hạn điểm ảnh / cạnh của API. Không phóng to."""
    h, w = img.shape[:2]
    f = 1.0
    if max_pixels and h * w > max_pixels:
        f = min(f, (max_pixels / (h * w)) ** 0.5)
    if max_side and max(h, w) > max_side:
        f = min(f, max_side / max(h, w))
    if f < 1.0:
        nw, nh = max(1, int(w * f)), max(1, int(h * f))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if min_side and min(img.shape[:2]) < min_side:
        raise ProviderError("too_small", f"Ảnh quá nhỏ cho API này (cạnh ngắn tối thiểu {min_side} px)")
    return img


def encode_clean(img: np.ndarray, max_bytes: int = 9 * 1024 * 1024) -> tuple[bytes, str]:
    """Nén lại từ điểm ảnh (bỏ sạch EXIF/GPS). PNG nếu nhỏ, không thì JPEG chất lượng giảm dần cho vừa giới hạn."""
    ok, buf = cv2.imencode(".png", img)
    if ok and len(buf) <= max_bytes // 3:
        return buf.tobytes(), "image/png"
    for q in (95, 90, 85, 80, 70):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(buf) <= max_bytes:
            return buf.tobytes(), "image/jpeg"
    raise ProviderError("too_large", "Ảnh vẫn quá lớn sau khi nén cho API")


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ProviderError("bad_output", "API trả về dữ liệu không phải ảnh")
    return img


def _data_uri(data: bytes, ctype: str) -> str:
    return f"data:{ctype};base64," + base64.b64encode(data).decode()


def _fetch_image(url: str, headers: dict | None = None) -> np.ndarray:
    if url.startswith("data:"):
        return decode(base64.b64decode(url.split(",", 1)[1]))
    st, _, data = _http("GET", url, headers)
    if st >= 400:
        raise ProviderError("bad_output", f"Không tải được ảnh kết quả ({st})", st)
    return decode(data)


def _wait(cancel: Cancelled | None, t0: float):
    if cancel and cancel():
        raise ProviderCancelled()
    if time.time() - t0 > MAX_WAIT:
        raise ProviderError("timeout", f"API xử lý quá {MAX_WAIT // 60} phút")
    time.sleep(POLL_SECONDS)


# ---------------------------------------------------------------- các nguồn xử lý

@dataclass
class Engine:
    id: str
    label: str
    provider: str  # fal | stability | replicate
    price: str  # chú thích giá cho người dùng (ước lượng theo bảng giá công khai)
    note: str
    key: str = field(repr=False, default="")
    cfg: dict = field(repr=False, default_factory=dict)

    def public(self) -> dict:
        """Thông tin gửi cho app: KHÔNG có khoá."""
        return {"id": self.id, "label": self.label, "provider": self.provider, "external": True,
                "price": self.price, "note": self.note}

    def run(self, img: np.ndarray, upscale: int = 0, progress: Progress | None = None,
            cancel: Cancelled | None = None) -> np.ndarray:
        say = progress or (lambda m: None)
        fn = {"fal-clarity": _run_fal_clarity, "stability-conservative": _run_stability_conservative,
              "stability-fast": _run_stability_fast, "replicate": _run_replicate}[self.id]
        return fn(self, img, upscale, say, cancel)


def _run_fal_clarity(e: Engine, img, upscale, say, cancel):
    factor = 4 if upscale == 4 else 2
    img = fit_pixels(img, max_pixels=16_000_000 // (factor * factor))  # ảnh ra ≤ 16 MP (giá tính theo MP)
    data, ctype = encode_clean(img, 8 * 1024 * 1024)
    body = {"image_url": _data_uri(data, ctype), "upscale_factor": factor,
            "creativity": float(e.cfg.get("creativity", 0.25)), "resemblance": float(e.cfg.get("resemblance", 0.75)),
            "prompt": e.cfg.get("prompt", "high quality photo, natural skin texture, sharp eyes"),
            "enable_safety_checker": True}
    hdr = {"Authorization": f"Key {e.key}", "Content-Type": "application/json"}
    say("Đang gửi ảnh lên fal.ai…")
    st, _, raw = _http("POST", "https://queue.fal.run/fal-ai/clarity-upscaler", hdr, json.dumps(body).encode())
    sub = _json_or_error(st, raw, "fal.ai")
    status_url, response_url, cancel_url = sub.get("status_url"), sub.get("response_url"), sub.get("cancel_url")
    if not status_url or not response_url:
        raise ProviderError("api_error", "fal.ai không trả địa chỉ theo dõi việc")
    t0 = time.time()
    auth = {"Authorization": f"Key {e.key}"}
    while True:
        st, _, raw = _http("GET", status_url, auth)
        s = _json_or_error(st, raw, "fal.ai")
        state = s.get("status")
        if state == "COMPLETED":
            if s.get("error"):
                raise ProviderError("api_error", f"fal.ai lỗi: {s.get('error')}")
            break
        if state == "IN_QUEUE":
            say(f"fal.ai: đang chờ (vị trí {s.get('queue_position', '?')})")
        elif state == "IN_PROGRESS":
            say("fal.ai: đang xử lý…")
        else:
            raise ProviderError("api_error", f"fal.ai trả trạng thái lạ: {state}")
        try:
            _wait(cancel, t0)
        except ProviderCancelled:
            if cancel_url:
                try:
                    _http("PUT", cancel_url, auth)
                except ProviderError:
                    pass
            raise
    st, _, raw = _http("GET", response_url, auth)
    out = _json_or_error(st, raw, "fal.ai")
    url = (out.get("image") or {}).get("url")
    if not url:
        raise ProviderError("bad_output", "fal.ai không trả ảnh")
    say("Đang tải ảnh kết quả từ fal.ai…")
    return _fetch_image(url)


def _stability(e: Engine, img, path: str, fields: dict, say):
    data, ctype = encode_clean(img, 9 * 1024 * 1024)  # giới hạn thân yêu cầu 10 MiB
    body, mp = _multipart({"output_format": "png", **fields}, {"image": ("anh" + (".png" if "png" in ctype else ".jpg"),
                                                                          data, ctype)})
    say("Đang gửi ảnh lên Stability AI…")
    st, h, raw = _http("POST", f"https://api.stability.ai/v2beta/stable-image/upscale/{path}",
                       {"Authorization": f"Bearer {e.key}", "Accept": "image/*", "Content-Type": mp}, body, 300)
    if st >= 400:
        _json_or_error(st, raw, "Stability AI")
    fin = {k.lower(): v for k, v in h.items()}.get("finish-reason", "SUCCESS")
    if fin == "CONTENT_FILTERED":
        raise ProviderError("rejected", "Stability AI chặn ảnh này theo bộ lọc nội dung")
    return decode(raw)


def _check_aspect(img):
    h, w = img.shape[:2]
    if max(h, w) / min(h, w) > 2.5:
        raise ProviderError("bad_aspect", "Stability AI chỉ nhận ảnh có tỉ lệ cạnh trong khoảng 1:2,5 đến 2,5:1")


def _run_stability_conservative(e: Engine, img, upscale, say, cancel):
    _check_aspect(img)
    img = fit_pixels(img, max_pixels=9_437_184, min_side=64)
    prompt = e.cfg.get("prompt", "high quality portrait photo, natural skin texture, sharp details")
    return _stability(e, img, "conservative", {"prompt": prompt, "creativity": str(e.cfg.get("creativity", 0.25))}, say)


def _run_stability_fast(e: Engine, img, upscale, say, cancel):
    _check_aspect(img)
    img = fit_pixels(img, max_pixels=1_048_576, max_side=1536, min_side=32)
    return _stability(e, img, "fast", {}, say)


def _run_replicate(e: Engine, img, upscale, say, cancel):
    model = e.cfg.get("model", "")
    img = fit_pixels(img, max_pixels=int(e.cfg.get("max_pixels", 4_000_000)))
    data, ctype = encode_clean(img, 8 * 1024 * 1024)
    inp = dict(e.cfg.get("input") or {})
    inp[e.cfg.get("image_field", "image")] = _data_uri(data, ctype)
    hdr = {"Authorization": f"Bearer {e.key}", "Content-Type": "application/json", "Prefer": "wait=55"}
    if ":" in model or "/" not in model:  # có mã phiên bản → /v1/predictions
        url, body = "https://api.replicate.com/v1/predictions", {"version": model, "input": inp}
    else:  # model chính thức owner/name
        url, body = f"https://api.replicate.com/v1/models/{model}/predictions", {"input": inp}
    say("Đang gửi ảnh lên Replicate…")
    st, _, raw = _http("POST", url, hdr, json.dumps(body).encode(), 90)
    p = _json_or_error(st, raw, "Replicate")
    auth = {"Authorization": f"Bearer {e.key}"}
    t0 = time.time()
    while p.get("status") not in ("succeeded", "failed", "canceled"):
        say(f"Replicate: {p.get('status', 'đang chờ')}…")
        try:
            _wait(cancel, t0)
        except ProviderCancelled:
            cu = (p.get("urls") or {}).get("cancel")
            if cu:
                try:
                    _http("POST", cu, auth)
                except ProviderError:
                    pass
            raise
        get = (p.get("urls") or {}).get("get") or f"https://api.replicate.com/v1/predictions/{p.get('id')}"
        st, _, raw = _http("GET", get, auth)
        p = _json_or_error(st, raw, "Replicate")
    if p["status"] != "succeeded":
        raise ProviderError("api_error", f"Replicate: {p.get('error') or p['status']}")
    out = p.get("output")
    if isinstance(out, list):
        out = out[-1] if out else None
    if isinstance(out, dict):
        out = out.get("image") or out.get("url")
    if not isinstance(out, str):
        raise ProviderError("bad_output", "Replicate không trả ảnh (kiểm tra model trong providers.json)")
    say("Đang tải ảnh kết quả từ Replicate…")
    return _fetch_image(out, auth)


# ---------------------------------------------------------------- cấu hình

def config_path() -> Path:
    from .history import data_dir
    return data_dir() / "providers.json"


def load_config(path: Path | None = None, env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    try:
        cfg = json.loads((path or config_path()).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except (OSError, ValueError):
        cfg = {}
    for prov, var in (("fal", "FAL_KEY"), ("stability", "STABILITY_API_KEY"), ("replicate", "REPLICATE_API_TOKEN")):
        sec = cfg.get(prov) if isinstance(cfg.get(prov), dict) else {}
        if env.get(var):
            sec = {**sec, "key": env[var]}
        cfg[prov] = sec
    if env.get("NETANH_REPLICATE_MODEL"):
        cfg["replicate"] = {**cfg["replicate"], "model": env["NETANH_REPLICATE_MODEL"]}
    return cfg


def guard_threshold(cfg: dict) -> float:
    try:
        return float(cfg.get("guard_ids", GUARD_IDS))
    except (TypeError, ValueError):
        return GUARD_IDS


def engines(cfg: dict | None = None) -> dict[str, Engine]:
    """Các nguồn API ngoài đã có khoá (và chưa bị tắt trong `disabled`), theo id."""
    cfg = load_config() if cfg is None else cfg
    off = set(cfg.get("disabled") or [])
    out: list[Engine] = []
    fal, sta, rep = cfg.get("fal", {}), cfg.get("stability", {}), cfg.get("replicate", {})
    if fal.get("key"):
        out.append(Engine("fal-clarity", "fal.ai Clarity (khuếch tán)", "fal", "~0,03 USD mỗi MP ảnh ra",
                          "Ảnh rất mờ; có thể thêm chi tiết không có trong ảnh gốc", fal["key"], fal))
    if sta.get("key"):
        out.append(Engine("stability-conservative", "Stability Conservative", "stability", "40 credit mỗi ảnh",
                          "Ảnh ra ~4 MP, ít bịa chi tiết", sta["key"], sta))
        out.append(Engine("stability-fast", "Stability Fast 4×", "stability", "2 credit mỗi ảnh",
                          "Ảnh vào tối đa 1 MP, phóng 4×", sta["key"], sta))
    if rep.get("key") and rep.get("model"):
        out.append(Engine("replicate", rep.get("label") or f"Replicate {rep['model'].split(':')[0]}", "replicate",
                          rep.get("price", "theo giây GPU"), rep.get("note", "Model tự chọn trên Replicate"),
                          rep["key"], rep))
    return {e.id: e for e in out if e.id not in off}


# ---------------------------------------------------------------- kiểm giữ mặt

def face_ids(eng, src: np.ndarray, out: np.ndarray) -> list[float]:
    """IDS từng mặt của ảnh gốc so với ảnh ra (ArcFace). Ảnh ra thu về cỡ ảnh gốc, cắt mặt cùng toạ độ căn chỉnh.
    Không có mặt (hoặc mặt quá nhỏ) → danh sách rỗng."""
    from .identity import Identity

    if out.shape[:2] != src.shape[:2]:
        out = cv2.resize(out, src.shape[1::-1], interpolation=cv2.INTER_AREA)
    helper = eng._load_helper(1)
    helper.clean_all()
    helper.read_image(src)
    helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=10)
    if not helper.all_landmarks_5:
        return []
    helper.align_warp_face()
    ident = getattr(eng, "_netanh_identity", None)
    if ident is None:
        ident = Identity(eng.dev.device)
        eng._netanh_identity = ident
    res = []
    for face, m in zip(helper.cropped_faces, helper.affine_matrices):
        fo = cv2.warpAffine(out, m, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        res.append(round(ident.similarity(face, fo), 4))
    helper.clean_all()
    return res


# ---------------------------------------------------------------- dòng lệnh

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="net_anh.providers", description="Cổng API xử lý ảnh ngoài của Beauty HDR")
    p.add_argument("image", nargs="?", help="ảnh để thử (bỏ trống: chỉ liệt kê nguồn đã cấu hình)")
    p.add_argument("--engine", help="id nguồn, ví dụ fal-clarity, stability-conservative")
    p.add_argument("--upscale", type=int, default=0, choices=[0, 2, 4])
    p.add_argument("-o", "--out", help="file ảnh ra (mặc định <tên>_<nguồn>.png)")
    p.add_argument("--yes", action="store_true", help="đồng ý gửi ảnh ra dịch vụ ngoài (không hỏi lại)")
    a = p.parse_args(argv)
    eng_map = engines()
    if not a.image:
        if not eng_map:
            print("Chưa cấu hình API ngoài nào. Đặt FAL_KEY, STABILITY_API_KEY hoặc REPLICATE_API_TOKEN, "
                  f"hoặc sửa {config_path()}")
        for e in eng_map.values():
            print(f"{e.id:24s} {e.label}  · {e.price}  · {e.note}")
        return 0
    if a.engine not in eng_map:
        print(f"Không có nguồn '{a.engine}'. Có: {', '.join(eng_map) or '(chưa cấu hình)'}")
        return 2
    e = eng_map[a.engine]
    if not a.yes:
        ans = input(f"Ảnh sẽ được gửi tới {e.label} ({e.provider}), giá {e.price}. Đồng ý? [c/K] ").strip().lower()
        if ans not in ("c", "co", "có", "y", "yes"):
            print("Đã huỷ.")
            return 1
    from .pipeline import load_image
    img = load_image(a.image)
    t0 = time.time()
    try:
        out = e.run(img, a.upscale, progress=print)
    except ProviderError as err:
        print(f"Lỗi [{err.code}]: {err}")
        return 3
    dst = a.out or str(Path(a.image).with_name(f"{Path(a.image).stem}_{e.id}.png"))
    cv2.imwrite(dst, out)
    print(f"Xong sau {time.time() - t0:.1f} s: {out.shape[1]}×{out.shape[0]} → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
