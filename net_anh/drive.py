"""Tải file model từ Google Drive (link chia sẻ công khai "Bất kỳ ai có đường liên kết").

Drive không cho tải thẳng file lớn: nó trả về trang HTML "không quét được virus, vẫn tải" có form ẩn. Ở đây gọi
drive.usercontent.google.com/download?confirm=t, nếu vẫn nhận HTML thì đọc form (id, export, confirm, uuid) rồi gọi lại.
Gặp quá hạn mức tải trong ngày hay file riêng tư thì báo rõ để weights.py chuyển sang nguồn gốc (GitHub).
Danh sách file và link nằm trong net_anh/drive_models.json (tên file → link hoặc id), ghi đè được bằng
biến NET_ANH_DRIVE_JSON hoặc file drive_models.json trong thư mục model.
"""
from __future__ import annotations

import codecs
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("BEAUTYHDR_DRIVE_BASE", "https://drive.usercontent.google.com")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BeautyHDR"


class DriveError(RuntimeError):
    pass


def file_id(link: str) -> str:
    """Lấy id từ link chia sẻ (…/file/d/<id>/view, …?id=<id>) hoặc chính id."""
    link = link.strip()
    for pat in (r"/file/d/([\w-]{20,})", r"[?&]id=([\w-]{20,})", r"^([\w-]{20,})$"):
        m = re.search(pat, link)
        if m:
            return m[1]
    raise DriveError(f"Không đọc được id Drive từ: {link[:80]}")


def _get(url: str, start: int, timeout: float):
    h = {"User-Agent": UA}
    if start:
        h["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _is_html(resp) -> bool:
    return "text/html" in (resp.headers.get("Content-Type") or "").lower()


def open_file(fid: str, start: int = 0, timeout: float = 60.0):
    """Mở luồng tải của file; trả (response, tổng dung lượng). Làm tiếp từ byte `start` nếu Drive hỗ trợ Range."""
    url = f"{BASE}/download?id={urllib.parse.quote(fid)}&export=download&confirm=t"
    try:
        resp = _get(url, start, timeout)
        if _is_html(resp):
            html = resp.read(300_000).decode("utf-8", "replace")
            resp.close()
            action = re.search(r'<form[^>]+action="([^"]+)"', html)
            params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))
            if action and params.get("id"):
                url2 = action.group(1).replace("&amp;", "&") + "?" + urllib.parse.urlencode(params)
                resp = _get(url2, start, timeout)
            else:
                raise DriveError(_reason(html))
            if _is_html(resp):
                raise DriveError(_reason(resp.read(300_000).decode("utf-8", "replace")))
    except urllib.error.HTTPError as e:
        raise DriveError(f"Drive trả lỗi {e.code}") from e
    length = int(resp.headers.get("Content-Length") or 0)
    total = (start + length) if getattr(resp, "status", 200) == 206 else length
    return resp, total, getattr(resp, "status", 200) == 206 or not start


def _reason(html: str) -> str:
    low = html.lower()
    if "quota" in low or "too many users" in low or "download quota" in low:
        return "Drive hết hạn mức tải trong ngày cho file này (thử lại sau hoặc dùng nguồn gốc)"
    if "sign in" in low or "accounts.google.com" in low or "request access" in low:
        return "File Drive chưa chia sẻ công khai (đặt 'Bất kỳ ai có đường liên kết')"
    if "not found" in low or "404" in low:
        return "Không thấy file trên Drive (link sai hoặc đã xoá)"
    return "Drive trả trang web thay vì file"


def _config_paths(weights_dir: Path | None) -> list[Path]:
    out: list[Path] = []
    if os.environ.get("NET_ANH_DRIVE_JSON"):
        out.append(Path(os.environ["NET_ANH_DRIVE_JSON"]))
    if weights_dir is not None:
        out.append(weights_dir / "drive_models.json")
    out.append(Path(__file__).with_name("drive_models.json"))
    return out


_FOLDER_CACHE: dict[str, dict[str, str]] = {}


def folder_id(link: str) -> str:
    m = re.search(r"/folders/([\w-]{20,})", link) or re.search(r"[?&]id=([\w-]{20,})", link) or re.match(r"^([\w-]{20,})$", link.strip())
    if not m:
        raise DriveError(f"Không đọc được id thư mục Drive từ: {link[:80]}")
    return m[1]


def parse_folder_html(html: str) -> dict[str, str]:
    """Tên file → id từ trang thư mục Drive công khai (cách gdown đọc biến window['_DRIVE_ivd'], tối đa ~50 file)."""
    m = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'(.*?)';", html, re.S)
    if not m:
        raise DriveError("Trang thư mục Drive không có danh sách file (thư mục chưa chia sẻ công khai?)")
    arr = json.loads(codecs.decode(m[1].encode("utf-8"), "unicode_escape"))
    out: dict[str, str] = {}
    for e in (arr[0] or []):
        try:
            if e[3] != "application/vnd.google-apps.folder":
                out[e[2]] = e[0]
        except (IndexError, TypeError):
            continue
    return out


def list_folder(link: str, timeout: float = 15.0) -> dict[str, str]:
    fid = folder_id(link)
    if fid in _FOLDER_CACHE:
        return _FOLDER_CACHE[fid]
    base = os.environ.get("BEAUTYHDR_DRIVE_PAGE", "https://drive.google.com")
    req = urllib.request.Request(f"{base}/drive/folders/{fid}?hl=en", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            html = r.read(3_000_000).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        raise DriveError(f"Không mở được thư mục Drive: {e}") from e
    _FOLDER_CACHE[fid] = res = parse_folder_html(html)
    return res


def links(weights_dir: Path | None = None) -> dict[str, str]:
    """Tên file model → id Drive. Nguồn đứng trước thắng nguồn đứng sau; link không đọc được bị bỏ qua.
    Có "folder" trong cấu hình thì liệt kê thư mục công khai (lỗi thì bỏ qua, dùng link từng file hoặc nguồn gốc)."""
    merged: dict[str, str] = {}
    for p in reversed(_config_paths(weights_dir)):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            d = cfg.get("files", {})
        except (OSError, ValueError, AttributeError):
            continue
        if cfg.get("folder"):
            try:
                merged.update(list_folder(cfg["folder"]))
            except DriveError:
                pass
        for name, link in d.items():
            try:
                if link:
                    merged[name] = file_id(link)
            except DriveError:
                pass
    return merged
