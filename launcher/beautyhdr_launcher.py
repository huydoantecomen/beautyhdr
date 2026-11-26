"""BeautyHDR.exe: một file duy nhất cho người dùng. Mở lên là tự lo mọi thứ rồi chạy app:

  1. Kiểm bản mới ở trang Releases (update.json); chưa cài hoặc có bản mới thì tải gói app (CPU hoặc GPU NVIDIA).
  2. Tải các model AI cần thiết (dùng chính NetAnh.exe --tai-model: kiểm SHA-256, model vá xước tách từ zip trong app).
  3. Chạy NetAnh.exe.

Dữ liệu nằm trong %LOCALAPPDATA%\\BeautyHDR (app\\, weights\\, downloads\\, state.json); xoá thư mục này là gỡ sạch.
Chỉ dùng thư viện chuẩn của Python (tkinter cho cửa sổ tiến độ), để file exe nhỏ. Mọi chỗ gọi mạng, đĩa, tiến trình con
đều tách thành hàm nhỏ để kiểm bằng máy chủ giả trên Linux (tests/test_launcher.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Kho PHẢI công khai: app trên máy người dùng không có token để tải từ kho riêng tư.
UPDATE_REPO = os.environ.get("BEAUTYHDR_REPO", "huydoantecomen/beautyhdr")
UPDATE_URL = os.environ.get("BEAUTYHDR_UPDATE_URL", f"https://github.com/{UPDATE_REPO}/releases/latest/download/update.json")
APP_EXE = "NetAnh.exe" if os.name == "nt" else "NetAnh"
MODEL_MARK = "models_ok"  # state: đã tải đủ model cho bản app này
GB = 1 << 30


class Fail(Exception):
    """Lỗi có thể nói cho người dùng hiểu (hết ổ, mất mạng, file hỏng)."""


class Cancelled(Exception):
    pass


def base_dir() -> Path:
    env = os.environ.get("BEAUTYHDR_HOME")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local/share") / "BeautyHDR"


def vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def adapter_names() -> list[str]:
    """Tên các card đồ hoạ thật (bỏ card ảo) theo Windows; máy khác Windows thì rỗng."""
    if os.name != "nt":
        return []
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    bad = re.compile(r"microsoft|remote|virtual|basic render|displaylink|parsec", re.I)
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and not bad.search(ln)]


# Pascal / Maxwell: GTX 10, GTX 900, GTX 750 (Ti), MX1xx–3xx, GT 1030, Titan X / Xp, Quadro P / M, Tesla P / M
LEGACY_NVIDIA = re.compile(r"gtx\s*(10[5-8]0|9[5-8]0|75\d)|\bmx\s*[1-3][1-5]0|\bgt\s*10[3-9]0|titan\s*x(p)?\b"
                           r"|quadro\s*[pm]\d|tesla\s*[pm]\d", re.I)
# Kepler và cũ hơn: GTX 600, GTX 760–780, GT 6xx / 7xx, GTX Titan (đời đầu), Quadro K, Tesla K: không bản PyTorch nào chạy
TOO_OLD_NVIDIA = re.compile(r"gtx\s*(6\d0|7[0-46-9]0)\b|\bgt\s*[4-7]\d0\b|gtx\s*titan(?!\s*x)|quadro\s*k\d|tesla\s*k\d", re.I)


def _nvidia_names(names: list[str]) -> list[str]:
    return [n for n in names if re.search(r"nvidia|geforce|quadro|titan|tesla", n, re.I)]


def nvidia_too_old(names: list[str]) -> bool:
    """Mọi card NVIDIA trên máy đều là Kepler hoặc cũ hơn: chỉ chạy được CPU."""
    nv = _nvidia_names(names)
    return bool(nv) and all(TOO_OLD_NVIDIA.search(n) for n in nv)


def nvidia_legacy(names: list[str]) -> bool:
    """Card NVIDIA đời Pascal / Maxwell (GTX 10, GTX 900, MX1xx–3xx): PyTorch CUDA 12.8 trở lên không chạy trên các
    dòng này, cần gói cuda_old (CUDA 12.6). Máy có cả card mới và card cũ thì theo card mới; card Kepler bỏ qua."""
    nv = [n for n in _nvidia_names(names) if not TOO_OLD_NVIDIA.search(n)]
    return bool(nv) and all(LEGACY_NVIDIA.search(n) for n in nv)


def recommend_flavor(names: list[str], nvidia_smi: bool, available: set[str]) -> tuple[str, str]:
    """(gói nên tải, hãng card). Gói: cuda (NVIDIA từ GTX 16 / RTX 20), cuda_old (GTX 10 / 900), xpu (Intel Arc, Core
    Ultra, Iris Xe), cpu. UHD Graphics đời cũ và card AMD chưa có gói GPU: dùng CPU. Gói không có trong bản phát hành
    thì rơi về cpu (card cũ không rơi về cuda: cuda không chạy trên card cũ)."""
    low = [n.lower() for n in names]
    if nvidia_smi or any("nvidia" in n or "geforce" in n for n in low):
        if nvidia_too_old(names):
            pass  # Kepler: không gói GPU nào chạy, xét tiếp card Intel trên máy (laptop), không có thì CPU
        elif nvidia_legacy(names):
            if "cuda_old" in available:
                return "cuda_old", "NVIDIA đời cũ"
        elif "cuda" in available:
            return "cuda", "NVIDIA"
    if "xpu" in available and any(re.search(r"\barc\b|iris|ultra|intel\(r\) graphics$", n) for n in low):
        return "xpu", "Intel"
    if any("amd" in n or "radeon" in n for n in low):
        return "cpu", "AMD"
    return "cpu", ""


def has_nvidia() -> bool:
    """Có card NVIDIA không (nvidia-smi có sẵn khi cài driver)."""
    for exe in ("nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe"):
        try:
            r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode == 0 and "GPU" in r.stdout:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


# ------------------------------------------------------------------ mạng + đĩa

def fetch_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "BeautyHDR-launcher"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check_disk(path: Path, need: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need:
        raise Fail(f"Ổ đĩa chứa {path.anchor or path} còn trống {free / GB:.1f} GB, cần khoảng {need / GB:.1f} GB. "
                   "Hãy dọn bớt ổ rồi mở lại, phần đã tải được giữ để làm tiếp.")


def download(url: str, dest: Path, size: int, sha256: str, progress, cancel, retries: int = 5) -> Path:
    """Tải có thể làm tiếp (HTTP Range, file .part), kiểm dung lượng và SHA-256. Tải dở mở lại thì đi tiếp từ chỗ dừng."""
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            if have < size:
                req = urllib.request.Request(url, headers={"User-Agent": "BeautyHDR-launcher",
                                                           **({"Range": f"bytes={have}-"} if have else {})})
                with urllib.request.urlopen(req, timeout=30) as r:
                    if have and r.status != 206:  # máy chủ bỏ qua Range: tải lại từ đầu
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        t0, b0 = time.monotonic(), have
                        while True:
                            if cancel():
                                raise Cancelled()
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            dt = max(time.monotonic() - t0, 0.001)
                            progress(have / size, f"{dest.name}: {have / 1e6:.0f}/{size / 1e6:.0f} MB "
                                                  f"({(have - b0) / dt / 1e6:.1f} MB/s)")
            if part.stat().st_size != size:
                raise Fail(f"{dest.name} tải thiếu ({part.stat().st_size}/{size} byte)")
            h = hashlib.sha256()
            with open(part, "rb") as f:
                for blk in iter(lambda: f.read(1 << 22), b""):
                    h.update(blk)
            if sha256 and h.hexdigest() != sha256:
                part.unlink()
                raise Fail(f"{dest.name} sai checksum, sẽ tải lại")
            part.replace(dest)
            return dest
        except Cancelled:
            raise
        except OSError as e:
            if getattr(e, "errno", None) == 28:  # ENOSPC
                raise Fail("Hết chỗ trên ổ đĩa khi tải. Dọn bớt ổ rồi mở lại, phần đã tải được giữ để làm tiếp.") from e
            last_err = e
        except Fail as e:
            last_err = e
        progress(0, f"Mạng lỗi ({last_err}), thử lại lần {attempt}/{retries}…")
        time.sleep(min(2 ** attempt, 10) * float(os.environ.get("BEAUTYHDR_RETRY_SCALE", "1")))
    raise Fail(f"Không tải được {dest.name}: {last_err}")


def extract_parts(parts: list[Path], out: Path, progress, cancel) -> None:
    """Ghép các phần .zip.001… thành một zip rồi giải nén vào `out` (thư mục mới, rỗng)."""
    joined = parts[0] if len(parts) == 1 else parts[0].with_name(re.sub(r"\.\d+$", "", parts[0].name) + ".joined")
    if len(parts) > 1:
        with open(joined, "wb") as w:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, w, 1 << 22)
    try:
        with zipfile.ZipFile(joined) as z:
            names = z.namelist()
            for i, n in enumerate(names):
                if cancel():
                    raise Cancelled()
                target = z.extract(n, out)
                mode = z.getinfo(n).external_attr >> 16
                if mode and os.name != "nt":  # giữ quyền chạy (chỉ cần khi thử trên Linux)
                    os.chmod(target, mode & 0o777)
                if i % 50 == 0:
                    progress(i / len(names), f"Giải nén {i}/{len(names)}")
    finally:
        if len(parts) > 1:
            joined.unlink(missing_ok=True)


# ------------------------------------------------------------------ trạng thái

class State:
    def __init__(self, home: Path):
        self.home = home
        self.file = home / "state.json"
        try:
            self.d = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.d = {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, **kw):
        self.d.update(kw)
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.file)


class Launcher:
    def __init__(self, ui, home: Path | None = None, update_url: str = UPDATE_URL, flavor: str | None = None):
        self.ui = ui
        self.home = home or base_dir()
        self.state = State(self.home)
        self.update_url = update_url
        self.flavor_arg = flavor
        self.app_dir = self.home / "app"
        self.weights = self.home / "weights"
        self._info: dict | None = None

    def cancel(self) -> bool:
        return self.ui.cancelled()

    def installed(self) -> bool:
        return (self.app_dir / APP_EXE).exists() and bool(self.state.get("version"))

    def ask_gpu(self, info: dict | None) -> str:
        """Nhận diện card, hỏi một lần có cài bản GPU không. Trả cuda / xpu / cpu."""
        avail = set(info["flavors"]) if info else {"cpu"}
        flavor, vendor = recommend_flavor(adapter_names(), has_nvidia(), avail)
        if flavor == "cuda_old":
            ok = self.ui.ask("Máy có card NVIDIA đời cũ",
                             "Cài bản GPU cho card đời cũ (GTX 10 / 900) + CPU (khuyên dùng)?\n"
                             "Dùng được CẢ HAI: ô Thiết bị trong app cho chọn GPU hoặc CPU, GPU lỗi thì tự chạy bằng CPU. "
                             "Nặng hơn khoảng 3 GB. Card dòng này luôn chạy fp32.\n\n"
                             "Chọn Không: chỉ cài bản CPU nhẹ.")
            return "cuda_old" if ok else "cpu"
        if flavor == "cuda":
            ok = self.ui.ask("Máy có card NVIDIA",
                             "Cài bản GPU + CPU (khuyên dùng)?\n"
                             "Một bản dùng được CẢ HAI: ô Thiết bị trong app cho chọn GPU hoặc CPU, mặc định GPU "
                             "(nhanh hơn khoảng 10 lần), GPU lỗi thì tự chạy bằng CPU. Nặng hơn khoảng 3 GB.\n\n"
                             "Chọn Không: chỉ cài bản CPU nhẹ.")
            return "cuda" if ok else "cpu"
        if flavor == "xpu":
            ok = self.ui.ask("Máy có card Intel",
                             "Cài bản GPU + CPU (khuyên dùng)?\n"
                             "Một bản dùng được CẢ HAI: ô Thiết bị trong app cho chọn GPU Intel hoặc CPU, mặc định GPU, "
                             "GPU lỗi thì tự chạy bằng CPU. Nặng hơn khoảng 2 GB. Bản Intel chưa thử trên nhiều máy.\n\n"
                             "Chọn Không: chỉ cài bản CPU nhẹ.")
            return "xpu" if ok else "cpu"
        if vendor == "AMD":
            self.ui.log("Máy có card AMD: bản hiện tại chưa hỗ trợ GPU AMD, dùng CPU.")
        return "cpu"

    def choose_flavor(self) -> str:
        if self.flavor_arg in ("cpu", "cuda", "cuda_old", "xpu"):
            return self.flavor_arg
        cur = self.state.get("flavor")
        if cur:
            return cur
        return self.ask_gpu(self._info)

    def latest(self) -> dict | None:
        try:
            self._info = fetch_json(self.update_url)
            return self._info
        except (OSError, ValueError) as e:
            self.ui.log(f"Không kiểm được bản mới ({e})")
            return None

    def install(self, info: dict, flavor: str) -> None:
        fl = info["flavors"].get(flavor) or info["flavors"].get("cpu")
        if fl is None:
            raise Fail("Bản phát hành không có gói app phù hợp.")
        parts = fl["parts"]
        total = sum(p["size"] for p in parts)
        dl = self.home / "downloads"
        # tải xong ghép + giải nén cần thêm ~2,2 lần dung lượng gói (zip, bản ghép, thư mục giải nén)
        need = int(total * 2.2) + (0 if self.state.get(MODEL_MARK) else int(1.0 * GB))
        done_bytes = sum((dl / p["name"]).stat().st_size for p in parts if (dl / p["name"]).exists())
        check_disk(self.home, max(need - done_bytes, 0))
        files, base = [], 0
        for p in parts:
            def prog(frac, msg, base=base, size=p["size"]):
                self.ui.progress((base + frac * size) / total * 0.8, "Tải ứng dụng · " + msg)
            files.append(download(p["url"], dl / p["name"], p["size"], p.get("sha256", ""), prog, self.cancel))
            base += p["size"]
        new = self.home / "app.new"
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir(parents=True)
        extract_parts(files, new, lambda f, m: self.ui.progress(0.8 + f * 0.2, "Cài đặt · " + m), self.cancel)
        root = new / "NetAnh" if (new / "NetAnh").is_dir() else new
        if not (root / APP_EXE).exists():
            shutil.rmtree(new, ignore_errors=True)
            raise Fail(f"Gói app thiếu {APP_EXE}.")
        old = self.home / "app.old"
        shutil.rmtree(old, ignore_errors=True)
        if self.app_dir.exists():
            self.app_dir.replace(old)
        try:
            root.replace(self.app_dir)
        except OSError:
            if old.exists():
                old.replace(self.app_dir)  # khôi phục bản cũ
            raise
        shutil.rmtree(new, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
        for p in files:
            p.unlink(missing_ok=True)
        self.state.set(version=info["version"], flavor=flavor, **{MODEL_MARK: False})
        self.ui.log(f"Đã cài Beauty HDR {info['version']} ({flavor}).")

    def offer_gpu(self, info: dict | None) -> None:
        """Đang dùng bản CPU mà máy có card dùng được (NVIDIA, Intel Arc/Iris Xe): hỏi một lần cho mỗi loại card có
        chuyển sang bản GPU không (giữ nguyên model)."""
        if info is None:
            return
        cur = self.state.get("flavor")
        if cur == "cuda" and "cuda_old" in info["flavors"] and nvidia_legacy(adapter_names()):
            # bản ≤ 0.14 cài gói cuda cho GTX 10: PyTorch mới không chạy trên card này, app chỉ dùng được CPU
            if not self.state.get("legacy_asked"):
                self.state.set(legacy_asked=True)
                if self.ui.ask("Card NVIDIA đời cũ", "Bản GPU đang cài không chạy được trên card này (GTX 10 / 900) nên "
                               "app đang dùng CPU. Chuyển sang bản GPU cho card đời cũ (tải khoảng 3 GB, model giữ nguyên)?"):
                    self.install(info, "cuda_old")
                    self.state.set(**{MODEL_MARK: True})
            return
        if cur != "cpu":
            return
        flavor, vendor = recommend_flavor(adapter_names(), has_nvidia(), set(info["flavors"]))
        asked = self.state.get("gpu_asked") or []
        if isinstance(asked, bool):  # state cũ (0.13): chỉ có cờ chung
            asked = ["cuda"] if asked else []
        if flavor == "cpu" or flavor in asked:
            return
        self.state.set(gpu_asked=asked + [flavor])
        name = {"cuda": "NVIDIA", "cuda_old": "NVIDIA đời cũ"}.get(flavor, "Intel")
        if self.ui.ask(f"Máy có card {name}", f"Bản đang cài chỉ chạy bằng CPU nên chậm. Chuyển sang bản GPU {name} + CPU "
                       "(chọn GPU hoặc CPU trong ô Thiết bị, nhanh hơn nhiều lần, tải thêm vài GB, model giữ nguyên)?"):
            self.install(info, flavor)
            self.state.set(**{MODEL_MARK: True})  # model không đổi theo bản app

    def models(self) -> None:
        if self.state.get(MODEL_MARK):
            return
        self.weights.mkdir(parents=True, exist_ok=True)
        check_disk(self.weights, int(0.8 * GB))
        exe = self.app_dir / APP_EXE
        env = {**os.environ, "NET_ANH_WEIGHTS": str(self.weights), "PYTHONUTF8": "1"}
        self.ui.progress(0, "Tải model AI (chỉ lần đầu, khoảng 560 MB)…")
        pf = self.home / "tai_model.log"
        pf.unlink(missing_ok=True)
        # NetAnh.exe không có console: tiến độ đọc từ file --progress-file (stdout chỉ là phần thêm khi chạy thử)
        proc = subprocess.Popen([str(exe), "--tai-model", str(self.weights), "--progress-file", str(pf)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        seen, total, lastline, pos = 0, 7, "", 0

        def pump() -> None:
            nonlocal seen, lastline, pos
            try:
                data = pf.read_bytes()[pos:]
            except OSError:
                return
            pos += len(data)
            for line in data.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                lastline = line
                if line.startswith("OK "):
                    seen += 1
                m = re.search(r"(\d+)%", line)
                self.ui.progress(min((seen + (int(m[1]) / 100 if m else 0)) / total, 0.99), "Tải model AI · " + line[:90])

        while proc.poll() is None:
            pump()
            if self.cancel():
                proc.kill()
                raise Cancelled()
            time.sleep(0.3)
        pump()
        if proc.returncode != 0:
            raise Fail(f"Tải model lỗi: {lastline}. Kiểm tra mạng rồi mở lại, phần đã tải được giữ để làm tiếp.")
        self.state.set(**{MODEL_MARK: True})

    def run(self, launch: bool = True) -> int:
        """Trả 0 nếu app đã được mở."""
        info = self.latest()
        cur = self.state.get("version")
        if not self.installed():
            if info is None:
                raise Fail("Chưa cài Beauty HDR và không kết nối được mạng để tải. Kiểm tra mạng rồi mở lại.")
            self.install(info, self.choose_flavor())
        elif info and vtuple(info["version"]) > vtuple(cur or "0"):
            self.ui.log(f"Có bản mới {info['version']} (đang dùng {cur}), đang cập nhật…")
            try:
                self.install(info, self.state.get("flavor", "cpu"))
            except Cancelled:
                raise
            except Fail as e:  # cập nhật lỗi không được chặn người dùng dùng bản đang có
                self.ui.log(f"Cập nhật lỗi, dùng bản hiện tại: {e}")
        self.offer_gpu(info)
        self.models()
        if launch:
            env = {**os.environ, "NET_ANH_WEIGHTS": str(self.weights)}
            subprocess.Popen([str(self.app_dir / APP_EXE)], env=env, cwd=str(self.app_dir))
        return 0


# ------------------------------------------------------------------ giao diện

class ConsoleUI:
    def __init__(self, yes: bool = True):
        self.yes = yes
        self._last = -1

    def cancelled(self) -> bool:
        return False

    def log(self, msg: str):
        print(msg, flush=True)

    def progress(self, frac: float, msg: str):
        p = int(frac * 100)
        if p != self._last:
            self._last = p
            print(f"[{p:3d}%] {msg}", flush=True)

    def ask(self, title: str, msg: str) -> bool:
        return self.yes

    def error(self, msg: str):
        print("LỖI: " + msg, file=sys.stderr, flush=True)

    def close(self):
        pass


class TkUI:
    """Cửa sổ nhỏ: dòng trạng thái, thanh % và nút Huỷ. Việc nặng chạy ở luồng riêng, giao diện chỉ vẽ lại."""

    def __init__(self):
        import tkinter as tk
        from tkinter import messagebox, ttk
        self.tk, self.mb = tk, messagebox
        self.root = tk.Tk()
        self.root.title("Beauty HDR")
        self.root.geometry("520x170")
        self.root.resizable(False, False)
        self.msg = tk.StringVar(value="Đang chuẩn bị…")
        tk.Label(self.root, text="Beauty HDR", font=("Segoe UI", 14, "bold")).pack(pady=(14, 2))
        tk.Label(self.root, textvariable=self.msg, wraplength=480, justify="left").pack(padx=16, pady=4)
        self.bar = ttk.Progressbar(self.root, length=480, maximum=100)
        self.bar.pack(pady=6)
        self.btn = tk.Button(self.root, text="Huỷ", command=self._cancel)
        self.btn.pack()
        self._cancel_flag = False
        self._state = (0.0, "Đang chuẩn bị…")
        self._asks: list = []
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self.root.after(100, self._tick)

    def _cancel(self):
        self._cancel_flag = True
        self._state = (self._state[0], "Đang huỷ…")

    def cancelled(self) -> bool:
        return self._cancel_flag

    def log(self, msg: str):
        self._state = (self._state[0], msg)

    def progress(self, frac: float, msg: str):
        self._state = (frac, msg)

    def ask(self, title: str, msg: str) -> bool:
        ev = threading.Event()
        box = {"r": True}
        self._asks.append((title, msg, ev, box))
        ev.wait()
        return box["r"]

    def error(self, msg: str):
        self._asks.append(("Beauty HDR", msg, None, None))

    def _tick(self):
        self.bar["value"] = self._state[0] * 100
        self.msg.set(self._state[1])
        while self._asks:
            title, msg, ev, box = self._asks.pop(0)
            if ev is None:
                self.mb.showerror(title, msg)
            else:
                box["r"] = self.mb.askyesno(title, msg)
                ev.set()
        self.root.after(150, self._tick)

    def close(self):
        try:
            self.root.after(0, self.root.destroy)
        except Exception:
            pass

    def run_in_thread(self, fn) -> int:
        out = {"code": 1}

        def work():
            out["code"] = fn()
            self.root.after(300, self.root.destroy)

        threading.Thread(target=work, daemon=True).start()
        self.root.mainloop()
        return out["code"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flavor = None
    if "--flavor" in argv:
        flavor = argv[argv.index("--flavor") + 1]
    console = "--console" in argv or (os.name != "nt" and not os.environ.get("DISPLAY"))
    if "--reset-flavor" in argv:  # đổi giữa CPU và GPU: xoá app, giữ model
        shutil.rmtree(base_dir() / "app", ignore_errors=True)
        State(base_dir()).set(version="", flavor="")
    ui = ConsoleUI() if console else TkUI()
    L = Launcher(ui, flavor=flavor)

    def job() -> int:
        try:
            return L.run()
        except Cancelled:
            ui.log("Đã huỷ. Phần đã tải được giữ để làm tiếp lần sau.")
            return 2
        except Fail as e:
            ui.error(str(e))
            return 1
        except Exception as e:  # noqa: BLE001
            ui.error(f"Lỗi không mong đợi: {e.__class__.__name__}: {e}")
            return 1

    if isinstance(ui, TkUI):
        return ui.run_in_thread(job)
    return job()


if __name__ == "__main__":
    sys.exit(main())
