"""Quản lý model: manifest (phiên bản, dung lượng, SHA-256, giấy phép), tải theo nhu cầu, kiểm tra checksum, thử lại."""
from __future__ import annotations

import hashlib
import http.client
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Progress = Callable[[str], None]


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    file: str
    version: str
    size: int  # byte
    sha256: str
    license: str
    url: str
    purpose: str
    input_shape: str
    label: str = ""  # tên ngắn hiện trong app (model nền kiến trúc mới)
    tile: int = 0  # ô lớn nhất khi chia ô (0 = theo Settings.tile)
    # Chỉ có trong một file zip lớn trên máy chủ: lấy đúng thành viên này bằng HTTP Range (không tải cả zip), rồi đổi
    # sang fp16 gọn (đã khớp `size`/`sha256` của file kết quả).
    zip_member: str = ""


MANIFEST: dict[str, ModelInfo] = {m.model_id: m for m in [
    ModelInfo("gfpgan", "GFPGANv1.4.pth", "1.4", 348632874,
              "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad", "Apache-2.0",
              "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
              "Phục hồi khuôn mặt", "1×3×512×512, RGB [-1, 1]"),
    ModelInfo("restoreformer", "RestoreFormer.pth", "GFPGAN 1.3.4", 290785322,
              "07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0", "Apache-2.0",
              "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
              "Phục hồi khuôn mặt, vẽ chi tiết mạnh hơn (kiểu Remini)", "1×3×512×512, RGB [-1, 1]"),
    # RestoreFormer++ (Wang và cộng sự, TPAMI 2023): checkpoint gốc của tác giả, nạp bằng archs/restoreformer_arch.py
    ModelInfo("restoreformer_pp", "RestoreFormer++.ckpt", "1.0.0", 294028815,
              "613fe52805f86bf8c2bffff08ae9f7a0b99f408be1bf221767af6183038be3a2", "Apache-2.0",
              "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt",
              "Phục hồi khuôn mặt thế hệ mới (RestoreFormer++, 2023)", "1×3×512×512, RGB [-1, 1]",
              "RestoreFormer++"),
    ModelInfo("esrgan_x2", "RealESRGAN_x2plus.pth", "0.2.1", 67061725,
              "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb", "BSD-3-Clause",
              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
              "Làm nét và phóng to 2× toàn ảnh", "1×3×H×W, RGB [0, 1], H và W chẵn"),
    ModelInfo("esrgan_x4", "RealESRGAN_x4plus.pth", "0.1.0", 67040989,
              "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", "BSD-3-Clause",
              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
              "Làm nét và phóng to 4× toàn ảnh", "1×3×H×W, RGB [0, 1]"),
    ModelInfo("esrgan_compact", "realesr-general-x4v3.pth", "0.2.5.0", 4885111,
              "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292", "BSD-3-Clause",
              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
              "Phóng to 4× nhanh (mạng compact, 5 MB), hợp điện thoại", "1×3×H×W, RGB [0, 1]"),
    ModelInfo("esrgan_compact_wdn", "realesr-general-wdn-x4v3.pth", "0.2.5.0", 4885111,
              "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca", "BSD-3-Clause",
              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
              "Bản giữ nhiễu của mạng compact, trộn trọng số để chỉnh mức khử nhiễu", "1×3×H×W, RGB [0, 1]"),
    # Model của Philip Hofmann (github.com/Phhofm/models), giấy phép CC-BY-4.0: dùng, sửa, bán được, phải ghi tên tác
    # giả (đã ghi ở cột Giấy phép trong Quản lý model và README). Nạp bằng spandrel (MIT).
    ModelInfo("dat2_web", "4xRealWebPhoto_v4_dat2.safetensors", "v4", 139792588,
              "819047acf43073f63ea7289ed94d6991ba8446a3bdf38e274fadfcbaa2b7381d", "CC-BY-4.0, tác giả Philip Hofmann",
              "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_dat2/4xRealWebPhoto_v4_dat2.safetensors",
              "Phóng to 4× chất lượng cao nhất (DAT-2, học trên ảnh mạng: mờ, nhiễu, nén JPEG/WebP), rất nặng",
              "1×3×H×W, RGB [0, 1]", "DAT-2", 256),
    ModelInfo("plksr_web", "4xNomosWebPhoto_RealPLKSR.safetensors", "1.0", 29590440,
              "9be0228f98156a100d6636d99b373ed2785b999723f9adc4cca504329ab157f2", "CC-BY-4.0, tác giả Philip Hofmann",
              "https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.safetensors",
              "Phóng to 4× (RealPLKSR, học trên ảnh mạng), nặng vừa", "1×3×H×W, RGB [0, 1]", "RealPLKSR", 400),
    ModelInfo("retinaface", "detection_Resnet50_Final.pth", "facexlib 0.1.0", 109497761,
              "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d", "MIT",
              "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
              "Tìm khuôn mặt + 5 landmark", "ảnh BGR, cạnh ngắn 640"),
    ModelInfo("parsenet", "parsing_parsenet.pth", "facexlib 0.2.2", 85331193,
              "3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2", "MIT",
              "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
              "Tách vùng da, tóc, ngũ quan", "1×3×512×512"),
    ModelInfo("arcface", "recognition_arcface_ir_se50.pth", "facexlib 0.1.0", 175367323,
              "a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28", "MIT",
              "https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth",
              "Đo độ giữ danh tính (IDS)", "1×3×112×112, RGB [-1, 1]"),
    # Cydral Technology (github.com/davisking/dlib-models), CC0 1.0: public domain, dùng thương mại được. Bản ONNX của
    # gói age-and-gender 2.0.0 (mowshon, MIT), ghim theo thẻ v2.0.0. Dùng để tự chọn preset theo đối tượng (subject.py).
    ModelInfo("age_cydral", "age-v1.onnx", "dlib-models v1 (age-and-gender 2.0.0)", 10768819,
              "4f9f0075200bd8ea6cd65ae6e6893cf820c513169382c08e82e533158358ed1c", "CC0-1.0, Cydral Technology",
              "https://raw.githubusercontent.com/mowshon/age-and-gender/v2.0.0/src/age_and_gender/models/age-v1.onnx",
              "Đoán tuổi khuôn mặt (chọn preset theo đối tượng)", "N×3×64×64, RGB trừ trung bình / 256",
              label="Tuổi (Cydral)"),
    ModelInfo("gender_cydral", "gender-v1.onnx", "dlib-models v1 (age-and-gender 2.0.0)", 530620,
              "d4c4116cf720cde18de28a09071d8e1826bcff7935ef3470052163f3f6b5d734", "CC0-1.0, Cydral Technology",
              "https://raw.githubusercontent.com/mowshon/age-and-gender/v2.0.0/src/age_and_gender/models/gender-v1.onnx",
              "Đoán giới tính khuôn mặt (chọn preset theo đối tượng)", "N×3×32×32, RGB trừ trung bình / 256",
              label="Giới tính (Cydral)"),
    # ----- khôi phục ảnh cũ (restore.py) -----
    # LaMa big-lama (Suvorov và cộng sự, WACV 2022; bản TorchScript của Sanster), Apache-2.0: vá vết xước, bụi, vết ố.
    ModelInfo("lama", "big-lama.pt", "big-lama (TorchScript)", 205669692,
              "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9", "Apache-2.0",
              "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
              "Vá vết xước, bụi, vết ố trên ảnh cũ (LaMa inpainting)", "1×3×H×W [0,1] + mặt nạ 1×1×H×W (H, W chia hết 8)",
              "LaMa"),
    # Microsoft "Bringing Old Photos Back to Life", MIT: U-Net dò vết xước/nếp gãy. File gốc nằm trong zip 2 GB:
    # tải đúng thành viên FT_Epoch_latest.pt (450 MB, có cả optimizer) rồi giữ lại 75 MB trọng số fp16.
    ModelInfo("scratch", "scratch_detect_fp16.pth", "BOPBTL v1.0 (fp16)", 75346399,
              "13eec13c4032dcff91117631f0f0168b77d34b2507b4b98d4c8cf4b3035881b6", "MIT, Microsoft",
              "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip",
              "Dò vết xước, nếp gãy, vết nhăn màng bọc (U-Net)", "1×1×H×W xám [-1, 1], cạnh chia hết 16", "Dò xước",
              zip_member="checkpoints/detection/FT_Epoch_latest.pt"),
    # DDColor (Kang và cộng sự, ICCV 2023), Apache-2.0: đoán màu theo ngữ nghĩa cho ảnh cảnh cũ (restore.scene_ai_color).
    ModelInfo("ddcolor", "ddcolor_modelscope.pth", "ddcolor_modelscope", 911914869,
              "d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7", "Apache-2.0, Kang và cộng sự",
              "https://huggingface.co/piddnad/ddcolor_modelscope/resolve/main/pytorch_model.bin",
              "Tô màu AI cho ảnh cảnh cũ (nâng màu sặc sỡ)", "1×3×512×512 RGB [0,1] (xám) → a, b của Lab",
              "DDColor"),
]}


class ModelError(RuntimeError):
    pass


class ModelTruncated(ModelError):
    """Tải bị ngắt giữa chừng: giữ file .part, lần thử sau làm tiếp bằng Range (không đổi sang nguồn khác)."""


# Model cần khi chạy app (không gồm ArcFace chỉ dùng để đo, và RRDB chỉ tải khi chọn mạng nền RRDB).
RUNTIME = ("esrgan_compact", "esrgan_compact_wdn", "retinaface", "parsenet", "gfpgan", "age_cydral", "gender_cydral")
# Settings.sr_model của model nền kiến trúc mới (nạp bằng spandrel) → khoá trong MANIFEST
SPANDREL_SR = {"dat2": "dat2_web", "plksr": "plksr_web"}


def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".ghi_thu"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def weights_dir() -> Path:
    """Thứ tự ưu tiên: biến NET_ANH_WEIGHTS; khi chạy bản đóng gói: thư mục weights cạnh file .exe nếu ghi được
    hoặc đã có đủ model đi kèm bộ cài, nếu không thì %LOCALAPPDATA%/NetAnh/weights; khi chạy mã nguồn: ~/.net_anh/weights."""
    env = os.environ.get("NET_ANH_WEIGHTS")
    if env:
        d = Path(env)
    elif getattr(sys, "frozen", False):
        d = Path(sys.executable).parent / "weights"
        bundled = all((d / MANIFEST[m].file).exists() for m in RUNTIME)
        if not bundled and not _writable(d):
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
            d = Path(base) / "NetAnh" / "weights"
    else:
        d = Path.home() / ".net_anh" / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(model_id: str) -> str:
    """'ready' | 'missing' | 'corrupt' (dung lượng sai)."""
    m = MANIFEST[model_id]
    p = weights_dir() / m.file
    if not p.exists():
        return "missing"
    if m.zip_member:  # file tự đổi sang fp16 sau khi tải: kích thước/băm có thể lệch vài byte tuỳ phiên bản torch
        return "ready" if abs(p.stat().st_size - m.size) < m.size // 20 else "corrupt"
    return "ready" if p.stat().st_size == m.size else "corrupt"


def delete(model_id: str) -> None:
    (weights_dir() / MANIFEST[model_id].file).unlink(missing_ok=True)


def ensure(model_id: str, progress: Progress | None = None, retries: int = 3,
           cancel: Callable[[], bool] | None = None) -> Path:
    """Trả về đường dẫn model; tải nếu thiếu, kiểm SHA-256 sau khi tải, thử lại tối đa `retries` lần."""
    m = MANIFEST[model_id]
    path = weights_dir() / m.file
    if status(model_id) == "ready":
        return path
    say = progress or (lambda s: None)
    need = int(m.size * (2.2 if m.zip_member else 1.15)) + (50 << 20)
    try:
        import shutil

        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        free = shutil.disk_usage(path.parent).free + (part.stat().st_size if part.exists() else 0)
        if free < need:
            raise ModelError(f"Ổ đĩa chứa {path.parent} còn trống {free / 2**30:.1f} GB, cần khoảng {need / 2**30:.1f} GB "
                             f"để tải {m.file}. Dọn bớt ổ rồi thử lại (phần đã tải được giữ).")
    except OSError:
        pass
    path.unlink(missing_ok=True)  # file hỏng (sai dung lượng) từ lần trước
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            tmp, from_drive = _download(m, path, say, cancel)
            say(f"Đang kiểm tra {m.file}…")
            # kiểm SHA-256 trên file tạm rồi mới đổi tên: tắt app giữa chừng không để lại file sai nội dung mà
            # đúng dung lượng (status() chỉ so dung lượng nên sẽ coi là "ready" mãi)
            if (not m.zip_member or from_drive) and sha256_of(tmp) != m.sha256:
                tmp.unlink(missing_ok=True)
                if from_drive:
                    _DRIVE_BAD.add(m.file)
                    say(f"{m.file} trên Drive sai nội dung, chuyển sang nguồn gốc.")
                raise ModelError(f"{m.file} sai checksum")
            tmp.replace(path)
            say(f"Đã tải và kiểm tra xong {m.file}")
            return path
        except InterruptedError:
            raise
        except Exception as e:  # mạng lỗi, file hỏng
            last = e
            say(f"Tải {m.file} lỗi (lần {attempt}/{retries}): {e}")
            time.sleep(min(2 ** attempt, 8))
    raise ModelError(f"Không tải được {m.file}: {last}")


class _RangeFile:
    """File chỉ đọc qua HTTP Range, cho zipfile đọc đúng một thành viên của file zip lớn."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" not in cr:
                raise ModelError("máy chủ không hỗ trợ tải một phần (Range)")
            self.size = int(cr.split("/")[1])

    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if self.pos >= end:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        self.pos += len(data)
        return data


def _download_zip_member(m: ModelInfo, path: Path, say: Progress, cancel) -> Path:
    import zipfile

    import torch

    tmp, raw = path.with_suffix(".part"), path.with_suffix(".raw")
    say(f"Đang tải {m.zip_member.rsplit('/', 1)[-1]} từ gói của tác giả (~450 MB, chỉ lần đầu)…")
    with zipfile.ZipFile(_RangeFile(m.url)) as z, z.open(m.zip_member) as src, open(raw, "wb") as out:
        done = 0
        while True:
            if cancel and cancel():
                out.close()
                raw.unlink(missing_ok=True)
                raise InterruptedError("Đã huỷ")
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done % (50 << 20) < (1 << 20):
                say(f"Đang tải {m.file}: {done >> 20} MB")
    try:
        ck = torch.load(raw, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", ""): (v.half() if v.is_floating_point() else v) for k, v in ck["model_state"].items()}
        torch.save(sd, tmp)
    finally:
        raw.unlink(missing_ok=True)
    return tmp


_DRIVE_BAD: set[str] = set()  # file mà bản trên Drive sai checksum: lần thử lại dùng nguồn gốc


def _stream_to(m: ModelInfo, opener, path: Path, say: Progress, cancel, label: str) -> Path:
    """Ghi luồng tải vào file .part, làm tiếp từ chỗ dừng nếu nguồn hỗ trợ Range. opener(start) → (resp, total, ranged)."""
    tmp = path.with_suffix(".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    if have >= m.size:
        tmp.unlink(missing_ok=True)
        have = 0
    resp, total, ranged = opener(have)
    try:
        if have and not ranged:  # nguồn bỏ qua Range: tải lại từ đầu
            have = 0
        if total and total != m.size:
            raise ModelError(f"{m.file} từ {label} có dung lượng {total} khác manifest {m.size}")
        say(f"Đang tải {m.file} từ {label} ({m.size / 1e6:.0f} MB" + (f", làm tiếp từ {have / 1e6:.0f} MB" if have else "") + ")…")
        done, last = have, -1
        with open(tmp, "ab" if have else "wb") as out:
            while True:
                if cancel and cancel():
                    raise InterruptedError("Đã huỷ")  # giữ file .part để lần sau làm tiếp
                try:
                    chunk = resp.read(1 << 20)
                except (http.client.IncompleteRead, ConnectionError, TimeoutError, OSError) as e:
                    out.flush()  # mất mạng giữa chừng: giữ phần đã tải
                    raise ModelTruncated(f"{m.file} bị ngắt khi tải ({e.__class__.__name__})") from e
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if done > m.size:  # server trả file khác (lớn hơn manifest): dừng sớm, không ghi tiếp ra đĩa
                    out.close()
                    tmp.unlink(missing_ok=True)
                    raise ModelError(f"{m.file} lớn hơn dung lượng trong manifest")
                pct = done * 100 // m.size
                if pct != last and pct % 10 == 0:
                    say(f"Đang tải {m.file}: {pct}%")
                    last = pct
    finally:
        resp.close()
    if done != m.size:  # tải thiếu: giữ .part để thử lại bằng Range
        raise ModelTruncated(f"{m.file} tải thiếu ({done}/{m.size} byte)")
    return tmp


def _http_opener(url: str):
    def opener(start: int):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        r = urllib.request.urlopen(req, timeout=60)
        total = int(r.headers.get("Content-Length") or 0)
        if start and getattr(r, "status", 200) == 206:
            return r, start + total, True
        return r, total, not start
    return opener


def _download(m: ModelInfo, path: Path, say: Progress, cancel) -> tuple[Path, bool]:
    """Thử Google Drive của huy trước (net_anh/drive_models.json), lỗi thì dùng nguồn gốc. Trả (file tạm, từ Drive?).
    File từ Drive luôn được kiểm SHA-256 ở ensure(); file từ nguồn gốc dạng zip_member thì đã tự dựng lại."""
    from . import drive

    fid = None if m.file in _DRIVE_BAD else drive.links(weights_dir()).get(m.file)
    if fid:
        try:
            return _stream_to(m, lambda start: drive.open_file(fid, start), path, say, cancel, "Google Drive"), True
        except (InterruptedError, ModelTruncated):
            raise
        except Exception as e:  # Drive chặn, hết hạn mức, link hỏng, thiếu dung lượng: chuyển sang nguồn gốc
            say(f"Google Drive không tải được {m.file} ({e}); chuyển sang nguồn gốc.")
            path.with_suffix(".part").unlink(missing_ok=True)
    if m.zip_member:
        return _download_zip_member(m, path, say, cancel), False
    return _stream_to(m, _http_opener(m.url), path, say, cancel, "nguồn gốc"), False
