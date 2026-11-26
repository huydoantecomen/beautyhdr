"""Đo tốc độ và độ chính xác trên card thật, để chỉnh bảng gpu_profile (tốc độ từng dòng, kiểu số, bộ nhớ mỗi điểm ảnh).

Chạy một trong các cách (mất khoảng 1–5 phút trên GPU, CPU lâu hơn):
    Trong app: Trợ giúp → Đo tốc độ card…
    NetAnh.exe --do-gpu                (bản cài)
    python app.py --do-gpu             (mã nguồn; thêm --nhanh để chỉ đo model nhẹ)
Kết quả ghi ra file BeautyHDR-do-gpu-<card>-<ngày>.txt và .json trong thư mục người dùng (Desktop nếu có). Gửi file
.txt cho người phát triển.

Đo gì:
  1. Máy: CPU, RAM, card, driver, bản PyTorch / CUDA, mã máy (sm_XX) có trong PyTorch, dòng card app nhận ra.
  2. Mỗi model nền đã tải (compact, x4plus, RealPLKSR, DAT-2) ở từng kiểu số fp32 / fp16 / bf16: lệch so với fp32 trên
     ảnh mẫu, thời gian ô 256 và 512 px, đỉnh bộ nhớ GPU (byte mỗi điểm ảnh ô vào).
  3. Model mặt GFPGAN ở fp32 / fp16 / bf16: lệch so với fp32 và thời gian mỗi mặt; kiểu app tự chọn.
  Model tích chập ở fp16 / bf16: thêm thời gian khi xếp dữ liệu kênh cuối (channels_last), để biết có nên bật.
  4. Cả quy trình: làm nét 6 ảnh 1600×1200 ở mức 3 (compact + GFPGAN nếu có mặt), một luồng và hai luồng song song.
"""
from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import __version__, gpu_profile, hardware, weights

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SR = (("compact", "esrgan_compact"), ("rrdb", "esrgan_x4"), ("plksr", "plksr_web"), ("dat2", "dat2_web"))


def _sync(dev) -> None:
    """Chờ GPU chạy xong (GPU chạy bất đồng bộ: không chờ thì đo sai)."""
    try:
        if dev.kind == "cuda":
            torch.cuda.synchronize(dev.device)
        elif dev.kind == "xpu":
            torch.xpu.synchronize(dev.device)
    except Exception:  # thiết bị giả khi kiểm (tests) hoặc driver không hỗ trợ
        pass


def _reset_peak(dev) -> None:
    try:
        if dev.kind == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev.device)
        elif dev.kind == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats(dev.device)
    except Exception:
        pass


def _peak(dev) -> int:
    try:
        if dev.kind == "cuda":
            return int(torch.cuda.max_memory_allocated(dev.device))
        if dev.kind == "xpu":
            return int(torch.xpu.max_memory_allocated(dev.device))
    except Exception:
        pass
    return 0


def _driver() -> str:
    try:
        flags = 0x08000000 if sys.platform == "win32" else 0
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=8, creationflags=flags)
        return out.stdout.strip()
    except Exception:
        return ""


def _time_forward(net, x, dev, reps: int) -> float:
    with torch.no_grad():
        net(x)  # chạy nóng (cuDNN chọn thuật toán, nạp kernel)
        _sync(dev)
        best = math.inf
        for _ in range(reps):
            t = time.perf_counter()
            net(x)
            _sync(dev)
            best = min(best, time.perf_counter() - t)
    return best


def _load_sr(eng, key: str):
    """Model nền ở fp32 trên CPU (bench tự đưa lên thiết bị theo từng kiểu số)."""
    import spandrel

    from .archs import RRDBNet, SRVGGNetCompact

    def sd(mid: str) -> dict:
        ck = torch.load(weights.weights_dir() / weights.MANIFEST[mid].file, map_location="cpu", weights_only=True)
        return ck.get("params_ema", ck.get("params", ck))

    if key == "compact":
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        net.load_state_dict(sd("esrgan_compact"), strict=True)
        return net.eval(), True, False
    if key == "rrdb":
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        net.load_state_dict(sd("esrgan_x4"), strict=True)
        return net.eval(), True, False
    mid = dict(SR)[key]
    desc = spandrel.ModelLoader(device="cpu").load_from_file(str(weights.weights_dir() / weights.MANIFEST[mid].file))
    return desc.model.eval(), bool(desc.supports_half), bool(desc.supports_bfloat16)


def run(say: Callable[[str], None] = print, quick: bool = False, out_dir: Path | None = None,
        device: str = "auto", pipeline_images: int = 6) -> Path:
    from .device import UNSUPPORTED, list_devices
    from .pipeline import Enhancer

    eng = Enhancer(device)
    dev = eng.dev
    m = hardware.machine()
    rep: dict = {
        "app": __version__, "time": time.strftime("%Y-%m-%d %H:%M"), "os": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None), "machine": m.summary(), "driver": _driver(),
        "device": dev.name, "kind": dev.kind, "family": dev.family.key, "cc": dev.cc, "vram_gb": round(dev.vram_gb, 2),
        "free_vram_gb": round(dev.free_vram_gb(), 2), "unsupported": UNSUPPORTED[:],
        "devices": [d.name for d in list_devices()], "sr": [], "face": {}, "pipeline": {},
    }
    try:
        rep["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    except Exception:
        rep["arch_list"] = []
    say(f"Đo trên {dev.name} · dòng {dev.family.label} · PyTorch {torch.__version__}")

    # ---- 2. model nền × kiểu số
    gpu = dev.kind != "cpu"
    for key, mid in SR:
        if quick and key != "compact":
            continue
        if weights.status(mid) != "ready":
            say(f"{key}: chưa tải model, bỏ qua")
            continue
        net, half_ok, bf16_ok = _load_sr(eng, key)
        net = net.to(dev.device)
        dtypes = ["fp32", "fp16", "bf16"] if gpu else ["fp32"]
        sizes = (256, 512) if gpu else (128,)
        for name in dtypes:
            row = {"model": key, "dtype": name, "model_says": {"fp16": half_ok, "bf16": bf16_ok}.get(name, True)}
            if name != "fp32":
                r = gpu_profile.probe_dtype(net, dev.device, DT[name])
                row.update(probe_ok=r.ok, mean_err=r.mean_err, p999_err=r.max_err, why=r.why)
            try:
                net.to(DT[name])
                for s in sizes:
                    x = torch.rand(1, 3, s, s, device=dev.device).to(DT[name])
                    _reset_peak(dev)
                    t = _time_forward(net, x, dev, 2 if (key == "dat2" or not gpu) else 3)
                    row[f"s_{s}"] = round(t, 4)
                    row[f"mp_per_s_{s}"] = round(s * s / 1e6 / t, 3)
                    pk = _peak(dev)
                    if pk:
                        row[f"bytes_per_px_{s}"] = round(pk / (s * s))
                    if t > 20:  # máy chậm: khỏi đo ô lớn hơn
                        break
                if gpu and gpu_profile.MODEL_ARCH.get(key) == "conv":  # kênh cuối (channels_last) nhanh hơn bao nhiêu
                    s = sizes[-1]
                    x = torch.rand(1, 3, s, s, device=dev.device).to(DT[name]).contiguous(memory_format=torch.channels_last)
                    gpu_profile.to_channels_last(net)
                    try:
                        row[f"s_{s}_kenh_cuoi"] = round(_time_forward(net, x, dev, 3), 4)
                    finally:
                        gpu_profile.to_contiguous(net)
            except Exception as e:  # thiếu bộ nhớ, kiểu số không chạy
                row["error"] = f"{e.__class__.__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
                _reset_peak(dev)
            finally:
                net.float()
            rep["sr"].append(row)
            say(_fmt_sr(row))
        del net
        _reset_peak(dev)

    # ---- 3. model mặt: fp32 và bf16 (fp16 lệch nặng ở GFPGAN), kiểu app tự chọn
    if weights.status("gfpgan") == "ready":
        try:
            from .archs import GFPGANv1Clean
            ck = torch.load(weights.weights_dir() / weights.MANIFEST["gfpgan"].file, map_location="cpu", weights_only=True)
            f = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                              fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
            f.load_state_dict(ck.get("params_ema", ck), strict=True)
            f = f.eval().to(dev.device)
            x = torch.rand(1, 3, 512, 512, device=dev.device) * 2 - 1
            for name in (["fp32", "fp16", "bf16"] if gpu else ["fp32"]):
                if name != "fp32":
                    r = gpu_profile.probe_dtype(f, dev.device, DT[name], size=512, fn=gpu_profile.face_forward)
                    rep["face"][f"gfpgan_{name}_lech_muc_xam"] = round(r.mean_err * 255, 2) if math.isfinite(r.mean_err) else "NaN"
                try:
                    f.to(DT[name])
                    xx = x.to(DT[name])
                    with torch.no_grad():
                        t = _time_forward(lambda z: f(z, return_rgb=False)[0], xx, dev, 3)
                    rep["face"][f"gfpgan_{name}_s"] = round(t, 4)
                    say(f"GFPGAN ({name}): {t:.3f} s mỗi mặt")
                except Exception as e:
                    rep["face"][f"gfpgan_{name}_loi"] = f"{e.__class__.__name__}: {str(e)[:100]}"
                finally:
                    f.float()
            eng._place_face(f, "gfpgan")  # kiểu app tự chọn trên card này (chạy thử như khi xử lý thật)
            rep["face"]["app_chon"] = eng.precision.get("gfpgan", ("fp32",))[0]
            del f
        except Exception as e:
            rep["face"]["error"] = f"{e.__class__.__name__}: {e}"

    # ---- 4. cả quy trình, 1 và 2 luồng
    if not quick or not gpu:
        rep["pipeline"] = _pipeline(eng, device, say, pipeline_images if gpu else 2)

    rep["schema"] = SCHEMA
    rep["records"] = records(rep)
    out_dir = out_dir or _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    card = re.sub(r"[^A-Za-z0-9]+", "-", dev.name.replace("GPU ", "").split(" (")[0]).strip("-") or "CPU"
    base = out_dir / f"BeautyHDR-do-gpu-{card}-{time.strftime('%Y%m%d-%H%M')}"
    base.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    txt = base.with_suffix(".txt")
    txt.write_text(summary(rep), encoding="utf-8")
    say(f"Đã ghi kết quả: {txt}")
    return txt


def _pipeline(eng, device: str, say, n: int) -> dict:
    import cv2

    from .batch import BatchRunner, extra_engines
    from .pipeline import Enhancer
    from .settings import Settings

    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(np.clip(rng.normal(128, 50, (1200, 1600, 3)), 0, 255).astype(np.uint8), (0, 0), 2)
    s = Settings(mode="upscale", upscale=2)
    out: dict = {"images": n, "size": "1600x1200", "mode": "upscale x2 compact"}
    eng.process(base, s)  # nạp model, chạy nóng
    for workers in ((1, 2) if eng.dev.kind == "cuda" else (1,)):
        engines = extra_engines(eng, workers, lambda: Enhancer(device))
        if workers > 1:
            engines[1].process(base, s)
        t = time.perf_counter()
        BatchRunner(engines, lambda i: base).run(list(range(n)), lambda e, lo: e.process(lo.value, s))
        dt = time.perf_counter() - t
        out[f"workers_{workers}_s_per_image"] = round(dt / n, 3)
        out[f"workers_{workers}_tile"] = engines[0].last_tile
        say(f"Cả quy trình, {workers} luồng: {dt / n:.2f} s mỗi ảnh (ô {engines[0].last_tile} px)")
        for e in engines:
            e.vram_share = 1.0
    return out


SCHEMA = 1  # đổi khi đổi cấu trúc bản ghi; tools/nhap_so_do.py đọc được mọi bản cũ hơn


def records(rep: dict) -> list[dict]:
    """Số đo dạng bản ghi phẳng (một số một dòng) để gộp vào net_anh/du_lieu/so_do_gpu.json bằng tools/nhap_so_do.py.
    Thêm loại số đo mới = thêm metric mới, không đổi cấu trúc. app_default: đúng cấu hình app dùng trên card này."""
    base = {"family": rep.get("family"), "card": rep.get("device"), "kind": rep.get("kind"), "cc": rep.get("cc"),
            "vram_gb": rep.get("vram_gb"), "app": rep.get("app"), "torch": rep.get("torch"), "cuda": rep.get("cuda"),
            "driver": rep.get("driver"), "date": rep.get("time")}
    cc = tuple(rep["cc"]) if rep.get("cc") else None
    out: list[dict] = []

    def add(model, dtype, metric, value, **kw):
        if isinstance(value, (int, float)) and math.isfinite(value):
            out.append(dict(base, model=model, dtype=dtype, metric=metric, value=value, **kw))

    for r in rep.get("sr", []):
        if "error" in r:
            continue
        m, dt = r["model"], r["dtype"]
        cl_app = gpu_profile.channels_last(rep.get("kind", ""), cc, dt, m)
        for k, v in r.items():
            mt = re.fullmatch(r"s_(\d+)(_kenh_cuoi)?", k)
            if mt:
                side, cl = int(mt.group(1)), bool(mt.group(2))
                add(m, dt, "s_per_mp", v / (side * side / 1e6), tile=side, channels_last=cl, app_default=cl == cl_app)
            mb = re.fullmatch(r"bytes_per_px_(\d+)", k)
            if mb:
                add(m, dt, "bytes_per_px", v, tile=int(mb.group(1)))
        if dt != "fp32" and "mean_err" in r:
            add(m, dt, "probe_err", r["mean_err"], ok=bool(r.get("probe_ok")))
    for k, v in rep.get("face", {}).items():
        mt = re.fullmatch(r"gfpgan_(fp32|fp16|bf16)_s", k)
        if mt:
            add("gfpgan", mt.group(1), "s_per_face", v)
        me = re.fullmatch(r"gfpgan_(fp16|bf16)_lech_muc_xam", k)
        if me and isinstance(v, (int, float)):
            add("gfpgan", me.group(1), "probe_err", v / 255)
    for k, v in rep.get("pipeline", {}).items():
        mw = re.fullmatch(r"workers_(\d+)_s_per_image", k)
        if mw:
            add("pipeline_compact_x2", "auto", "pipeline_s_per_image", v, workers=int(mw.group(1)))
    return out


def _fmt_sr(r: dict) -> str:
    if "error" in r:
        return f"{r['model']} {r['dtype']}: LỖI {r['error']}"
    p = "" if r["dtype"] == "fp32" else (f" · {'ĐẠT' if r.get('probe_ok') else 'KHÔNG ĐẠT'} lệch {r.get('mean_err', 0) * 255:.2f} "
                                         f"mức xám{(' (' + r['why'] + ')') if r.get('why') else ''}")
    speeds = " · ".join(f"ô {k[2:]}: {v:.3f} s" for k, v in r.items() if k.startswith("s_"))
    mem = " · ".join(f"{v / 1000:.1f} KB/px" for k, v in r.items() if k.startswith("bytes_per_px_"))
    return f"{r['model']} {r['dtype']}: {speeds}{(' · ' + mem) if mem else ''}{p}"


def summary(rep: dict) -> str:
    lines = [f"Beauty HDR {rep['app']} · đo card {rep['time']}", f"Hệ điều hành: {rep['os']} · Python {rep['python']}",
             f"PyTorch {rep['torch']} · CUDA {rep['cuda']} · HIP {rep['hip']} · mã máy {', '.join(rep['arch_list'])}",
             f"Máy: {rep['machine']}", f"Driver: {rep['driver'] or '(không có nvidia-smi)'}",
             f"Thiết bị: {rep['device']} · dòng {rep['family']} · cc {rep['cc']} · VRAM {rep['vram_gb']} GB, còn trống "
             f"{rep['free_vram_gb']} GB", f"Các thiết bị: {', '.join(rep['devices'])}"]
    for n, why in rep["unsupported"]:
        lines.append(f"Không dùng được: {n}: {why}")
    lines.append("")
    lines += [_fmt_sr(r) for r in rep["sr"]]
    if rep["face"]:
        lines.append(f"Mặt: {rep['face']}")
    if rep["pipeline"]:
        lines.append(f"Cả quy trình: {rep['pipeline']}")
    return "\n".join(lines) + "\n"


def _out_dir() -> Path:
    home = Path.home()
    for d in (home / "Desktop", home / "OneDrive" / "Desktop", home):
        if d.is_dir():
            return d
    return Path.cwd()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    dev = "cpu" if "--cpu" in argv else "auto"
    out = None
    if "--ra" in argv:
        out = Path(argv[argv.index("--ra") + 1])
    run(quick="--nhanh" in argv, device=dev, out_dir=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
