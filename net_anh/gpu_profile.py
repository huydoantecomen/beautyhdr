"""Hồ sơ theo dòng card: nhận dòng card và bộ nhớ (VRAM), chọn kiểu số (fp32 / fp16 / bf16), cỡ ô cắt ảnh và số ảnh
xử lý song song khi làm cả thư mục.

Nguyên tắc: card nào chưa chắc chạy đúng ở fp16 / bf16 thì không đoán. Mỗi model khi nạp lên GPU được chạy thử một ảnh
mẫu nhỏ ở fp32 và ở kiểu số nhanh hơn (probe_dtype); kết quả lệch, ra NaN hay ra đen thì dùng fp32 cho model đó. Khi
chạy thật, từng ô ảnh vẫn được kiểm như trước (Enhancer._forward): ô nào hỏng thì chạy lại fp32.

Các số tốc độ GPU (SPEEDUP) là ước lượng thô theo thông số card, CHƯA đo trên máy thật (môi trường build không có GPU):
tools/gpu_bench.py đo trên máy có card và in ra để cập nhật bảng này.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

GB = 1024**3


@dataclass(frozen=True)
class Family:
    key: str
    label: str  # tên hiện cho người dùng
    examples: str
    precisions: tuple[str, ...]  # thứ tự thử cho model nền; luôn kết thúc bằng fp32
    tf32: bool  # card có TF32 (Ampere trở lên): phép nhân ma trận fp32 nhanh hơn, sai số không thấy được trên ảnh
    speedup: float  # nhanh hơn CPU 4 luồng khoảng bao nhiêu lần (ước lượng, xem đầu file)
    support: str  # "đầy đủ" | "thử nghiệm" | "chưa hỗ trợ"
    note: str
    vram_common: tuple[float, ...] = ()  # cỡ bộ nhớ hay gặp của dòng này (GB), cho bảng khuyến nghị


FAMILIES: dict[str, Family] = {f.key: f for f in (
    Family("rtx50", "GeForce RTX 50 (Blackwell)", "RTX 5060, 5070, 5080, 5090", ("fp16", "bf16", "fp32"), True, 60,
           "đầy đủ", "Cần driver NVIDIA 570 trở lên. Model nặng nhất (mức 6) chạy thoải mái.", vram_common=(8, 12, 16, 32)),
    Family("rtx40", "GeForce RTX 40 (Ada)", "RTX 4060, 4070, 4080, 4090", ("fp16", "bf16", "fp32"), True, 45,
           "đầy đủ", "Model nặng nhất (mức 6) chạy thoải mái.", vram_common=(8, 12, 16, 24)),
    Family("rtx30", "GeForce RTX 30 (Ampere)", "RTX 3050, 3060, 3070, 3080, 3090", ("fp16", "bf16", "fp32"), True, 25,
           "đầy đủ", "Mốc đo ước lượng là RTX 3060. Card 8 GB trở lên chạy tốt mức 6.", vram_common=(4, 8, 12, 24)),
    Family("rtx20", "GeForce RTX 20 (Turing có Tensor Core)", "RTX 2060, 2070, 2080, Titan RTX", ("fp16", "fp32"), False,
           15, "đầy đủ", "Không có bf16 thật: DAT-2 chạy fp32, chậm hơn RTX 30 rõ rệt.", vram_common=(6, 8, 11)),
    Family("gtx16", "GeForce GTX 16 (Turing không Tensor Core)", "GTX 1630, 1650, 1660, 1660 Ti, 1660 Super, MX450",
           ("fp32",), False, 9, "đầy đủ",
           "Luôn chạy fp32: fp16 trên dòng này hay tràn số làm nền ra mảng đen (lỗi đã gặp trên GTX 1660 Ti).", vram_common=(4, 6)),
    Family("volta", "Titan V / Tesla V100 (Volta)", "Titan V, V100", ("fp16", "fp32"), False, 18, "thử nghiệm",
           "Chưa thử trên máy thật.", vram_common=(12, 16)),
    Family("gtx10", "GeForce GTX 10 (Pascal)", "GTX 1050, 1050 Ti, 1060, 1070, 1080, MX150–MX350", ("fp32",), False, 6,
           "đầy đủ", "fp16 trên Pascal chậm hơn fp32 nhiều lần nên luôn chạy fp32. Cần bản GPU cho card đời cũ "
           "(PyTorch CUDA 12.6): bản GPU thường không còn chạy trên Pascal.", vram_common=(2, 4, 6, 8, 11)),
    Family("maxwell", "GeForce GTX 900 / 750 (Maxwell)", "GTX 750 Ti, 950, 960, 970, 980", ("fp32",), False, 3,
           "thử nghiệm", "Card rất cũ, ít bộ nhớ: dùng model nhẹ. Cần bản GPU cho card đời cũ.", vram_common=(2, 4)),
    Family("nvidia", "NVIDIA (dòng khác)", "Quadro, Tesla, card trạm", ("fp16", "fp32"), False, 15, "thử nghiệm",
           "Chọn theo thế hệ chip khi đọc được; fp16 có tự kiểm.", vram_common=()),
    Family("amd_rocm", "AMD Radeon qua ROCm", "RX 7600–7900, RX 9060–9070 (ROCm trên Linux, bản xem trước trên Windows)",
           ("fp16", "fp32"), False, 15, "thử nghiệm",
           "Chưa có gói cài sẵn: chỉ dùng khi tự cài PyTorch ROCm từ mã nguồn. fp16 có tự kiểm.", vram_common=(8, 16, 24)),
    Family("amd_dml", "AMD / Intel qua DirectML", "Radeon RX 5000–9000, Vega, card tích hợp", ("fp32",), False, 5,
           "thử nghiệm", "Chạy từ mã nguồn với torch-directml; chậm hơn CUDA, một số model có thể lỗi (app tự về CPU).", vram_common=()),
    Family("intel_arc", "Intel Arc (XPU)", "Arc A380–A770, B570, B580, Arc trong Core Ultra", ("fp32",), False, 6,
           "thử nghiệm", "Gói GPU Intel từ 0.14.1. Đang dùng fp32 cho chắc; fp16 chờ số đo tools/gpu_bench.py.", vram_common=(6, 8, 12, 16)),
    Family("intel_igpu", "Intel Iris Xe / UHD (XPU)", "Iris Xe, UHD 7xx", ("fp32",), False, 2, "thử nghiệm",
           "Card tích hợp dùng chung RAM: chỉ nhanh hơn CPU chút ít với model nhẹ.", vram_common=()),
    Family("apple", "Apple Silicon (Metal)", "M1–M4", ("fp32",), False, 8, "thử nghiệm", "Chạy từ mã nguồn trên macOS.", vram_common=()),
    Family("cpu", "CPU", "Mọi máy", ("fp32",), False, 1, "đầy đủ",
           "Model nhẹ (mặc định) vài giây đến vài chục giây mỗi ảnh; model nặng rất chậm.", vram_common=()),
)}


def nvidia_family(name: str, cc: tuple[int, int] | None = None) -> Family:
    """Dòng card NVIDIA theo compute capability (đọc từ driver qua PyTorch) hoặc theo tên khi không có."""
    n = name.upper()
    if re.search(r"GTX\s*16|MX\s*4[5-9]0|MX\s*5\d0|T\d{3,4}\b(?!.*RTX)", n) and "RTX" not in n:
        if not cc or cc == (7, 5):  # TU116 / TU117: không Tensor Core
            return FAMILIES["gtx16"]
    if cc:
        major, minor = cc
        if major >= 10:
            return FAMILIES["rtx50"]
        if (major, minor) == (8, 9):
            return FAMILIES["rtx40"]
        if major == 8:
            return FAMILIES["rtx30"]
        if (major, minor) == (7, 5):
            return FAMILIES["rtx20"]
        if major == 7:
            return FAMILIES["volta"]
        if major == 6:
            return FAMILIES["gtx10"]
        if major < 6:
            return FAMILIES["maxwell"]
    m = re.search(r"RTX\s*(?:PRO\s*)?([2-5])0[5-9]0", n)
    if m:
        return FAMILIES[f"rtx{m[1]}0"]
    if re.search(r"GTX\s*10[5-8]0|MX\s*[1-3][1-5]0|TITAN\s*X(?!.*RTX)", n):
        return FAMILIES["gtx10"]
    if re.search(r"GTX\s*(7[5-9]|9[5-8])0", n):
        return FAMILIES["maxwell"]
    if "TITAN V" in n or "V100" in n:
        return FAMILIES["volta"]
    return FAMILIES["nvidia"]


def amd_rdna3_plus(name: str) -> bool:
    """RX 7000 (RDNA3) / RX 9000 (RDNA4): có lệnh fp16 nhanh, ROCm hỗ trợ chính thức."""
    return bool(re.search(r"RX\s*[79]\d{3}|RADEON\s*PRO\s*W7|INSTINCT|MI\d{3}", name.upper()))


def family_of(kind: str, name: str, cc: tuple[int, int] | None = None, hip: bool = False) -> Family:
    if kind == "cpu":
        return FAMILIES["cpu"]
    if kind == "cuda":
        if hip:
            f = FAMILIES["amd_rocm"]
            return f if amd_rdna3_plus(name) else _replace(f, precisions=("fp32",))
        return nvidia_family(name, cc)
    if kind == "xpu":
        return FAMILIES["intel_arc"] if re.search(r"\bARC\b|ULTRA|B5[78]0|A[37]\d0", name.upper()) else FAMILIES["intel_igpu"]
    if kind == "directml":
        return FAMILIES["amd_dml"]
    if kind == "mps":
        return FAMILIES["apple"]
    return FAMILIES["cpu"]


def _replace(f: Family, **kw) -> Family:
    import dataclasses

    return dataclasses.replace(f, **kw)


def arch_supported(cc: tuple[int, int], arch_list: list[str]) -> bool:
    """Bản PyTorch có mã máy cho card này không. sm_XY: chạy đúng thế hệ XY (cùng major, minor ≥ Y); compute_XY (PTX):
    driver biên dịch lại cho mọi card ≥ XY. Danh sách rỗng (bản build lạ): coi là chạy được, lỗi thật sẽ về CPU."""
    if not arch_list:
        return True
    v = cc[0] * 10 + cc[1]
    for a in arch_list:
        m = re.match(r"(sm|compute)_(\d+)", a)
        if not m:
            continue
        x = int(m[2])
        if m[1] == "sm" and x // 10 == cc[0] and x <= v:
            return True
        if m[1] == "compute" and x <= v:
            return True
    return False


# ---------------------------------------------------------------- chọn kiểu số có kiểm tra

DTYPE_NAMES = ("fp16", "bf16", "fp32")


def candidates(fam: Family, supports_half: bool = True, supports_bf16: bool = True, bf16_native: bool = True) -> list[str]:
    """Kiểu số sẽ thử theo thứ tự, lọc theo model (DAT-2 không chạy fp16) và card (bf16 thật từ Ampere).
    NETANH_PRECISION=fp32|fp16|bf16 ép một kiểu (vẫn qua bước kiểm)."""
    forced = os.environ.get("NETANH_PRECISION", "").lower()
    out = []
    for p in ((forced, "fp32") if forced in DTYPE_NAMES else fam.precisions):
        if p == "fp16" and not supports_half:
            continue
        if p == "bf16" and not (supports_bf16 and bf16_native):
            continue
        if p not in out:
            out.append(p)
    if "fp32" not in out:
        out.append("fp32")
    return out


def probe_image(size: int = 96, seed: int = 7):
    """Ảnh mẫu (1×3×size×size, 0..1) có các vùng làm fp16 hay lỗi: mảng sáng phẳng gần trắng (lỗi nền đen GTX 16),
    mảng tối, dải màu, cạnh sắc, nhiễu nhẹ."""
    import torch

    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    img = torch.stack([x, y, (x + y) / 2])
    q = size // 2
    img[:, :q, :q] = 0.97  # nền sáng phẳng
    img[:, q:, q:] = 0.03
    img[0, q - 4:q + 4, :] = 1.0  # vạch đỏ sắc
    img = img + 0.02 * torch.randn(img.shape, generator=g)
    return img.clamp(0, 1).unsqueeze(0)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    mean_err: float  # sai lệch trung bình so với fp32 (thang 0..1)
    max_err: float  # phân vị 99,9 của sai lệch
    why: str = ""


# Đo trên CPU 2026-11-26 với model thật (lệch trung bình / phân vị 99,9): fp16 compact 0,0001 / 0,0009, x4plus 0,0001 /
# 0,0009, RealPLKSR 0,0002 / 0,0018; bf16 0,0009–0,0018 / 0,005–0,015; DAT-2 fp16 ra NaN (đúng như đã biết: bị loại).
# Ngưỡng rộng hơn sai số làm tròn 7–10 lần để chỉ bắt lỗi thật (tràn số, mảng đen, nhiễu lạ).
PROBE_MEAN = 0.012  # ≈ 3 mức xám
PROBE_P999 = 0.12


def compare(ref, out) -> ProbeResult:
    import torch

    out = out.float()
    if not torch.isfinite(out).all():
        return ProbeResult(False, math.inf, math.inf, "ra NaN / vô cực")
    ref = ref.float()
    if float(ref.amax()) > 0.2 and float(out.amax()) < 0.02:
        return ProbeResult(False, math.inf, math.inf, "ra ảnh đen")
    d = (out.clamp(0, 1) - ref.clamp(0, 1)).abs().flatten()
    mean = float(d.mean())
    k = max(1, int(d.numel() * 0.001))
    p999 = float(d.topk(k).values.min())
    ok = mean <= PROBE_MEAN and p999 <= PROBE_P999
    return ProbeResult(ok, mean, p999, "" if ok else f"lệch fp32 {mean * 255:.1f} mức xám")


def probe_dtype(net, device, dtype, size: int = 96) -> ProbeResult:
    """Chạy thử `net` (đang ở fp32 trên `device`) với ảnh mẫu ở fp32 và ở `dtype`, so kết quả. Trả về net ở trạng thái
    fp32 (người gọi tự đổi kiểu khi ok)."""
    import torch

    x = probe_image(size).to(device)
    try:
        with torch.no_grad():
            ref = net(x)
            net.to(dtype)
            out = net(x.to(dtype))
        return compare(ref, out)
    except Exception as e:  # kiểu số không chạy được trên card / bản PyTorch này
        return ProbeResult(False, math.inf, math.inf, f"{e.__class__.__name__}: {str(e)[:80]}")
    finally:
        net.float()


# ---------------------------------------------------------------- cỡ ô theo bộ nhớ

# Bộ nhớ trung gian mỗi điểm ảnh ô VÀO ở fp32 (byte): đỉnh RAM khi chạy một ô 192 và 320 px trên CPU (2026-11-26:
# compact 1 000, x4plus 13 100, RealPLKSR 1 500, DAT-2 17 000; khớp số cũ DAT-2 ô 256 px ≈ 1,1 GB) nhân 2 cho bộ đệm
# của cuDNN và bộ cấp phát GPU. fp16 / bf16: một nửa. Thiếu bộ nhớ thật thì app vẫn tự chia đôi ô (như trước).
ACT_BYTES_PER_PX = {"compact": 2_500, "rrdb": 26_000, "plksr": 4_000, "dat2": 34_000}
MODEL_GB = {"compact": 0.01, "rrdb": 0.07, "plksr": 0.03, "dat2": 0.15}
RESERVE_GB = 1.2  # để lại cho model mặt, ParseNet, tìm mặt, Windows và app khác dùng GPU
TILE_MIN, TILE_MAX = 128, 1536
BAND_BYTES = 768 * 1024**2  # trần RAM cho dải cộng dồn khi ghép ô (Enhancer._tiled)
CPU_TILE = 400  # CPU: ô vừa phải (RAM, thanh tiến độ), như các bản trước


def auto_tile(model: str, free_gb: float, total_gb: float, dtype: str = "fp32") -> int:
    """Cạnh ô lớn nhất (px ảnh vào) vừa bộ nhớ GPU còn trống. free_gb = 0: không đọc được bộ nhớ, dùng 400 như cũ."""
    if free_gb <= 0 and total_gb <= 0:
        return CPU_TILE
    budget = min(free_gb if free_gb > 0 else total_gb * 0.7, total_gb * 0.85 if total_gb > 0 else free_gb)
    budget -= RESERVE_GB + MODEL_GB.get(model, 0.1)
    per_px = ACT_BYTES_PER_PX.get(model, 30_000) * (0.5 if dtype in ("fp16", "bf16") else 1.0)
    if budget <= 0:
        return TILE_MIN
    side = int(math.sqrt(budget * GB / per_px))
    return max(TILE_MIN, min(TILE_MAX, side - side % 32))


def vram_class(total_gb: float) -> str:
    if total_gb <= 0:
        return "không rõ"
    if total_gb < 4.5:
        return "ít (≤ 4 GB)"
    if total_gb < 7:
        return "vừa (6 GB)"
    if total_gb < 11:
        return "khá (8–10 GB)"
    return "nhiều (12 GB trở lên)"


# ---------------------------------------------------------------- hàng loạt

def batch_workers(kind: str, vram_gb: float, ram_gb: float, threads: int) -> int:
    """Số ảnh xử lý song song khi làm cả thư mục. Mỗi luồng giữ một bộ model riêng trên GPU (nạp một lần), trong lúc
    một ảnh chạy phần CPU (tìm mặt, ghép mặt, màu, lưu file) thì ảnh kia dùng GPU. Chỉ bật trên card nhiều bộ nhớ và
    máy đủ RAM, CPU; NETANH_BATCH_WORKERS=1|2|3 để ép."""
    env = os.environ.get("NETANH_BATCH_WORKERS", "")
    if env.isdigit() and int(env) > 0:
        return min(int(env), 4)
    if kind == "cuda" and vram_gb >= 11 and ram_gb >= 15 and threads >= 8:
        return 2
    return 1


# ---------------------------------------------------------------- bảng khuyến nghị

def recommendation(fam: Family, vram_gb: float = 0.0) -> dict[str, str]:
    """Một dòng của bảng khuyến nghị theo card (ô Thiết bị → Khuyến nghị theo card, README)."""
    prec = " → ".join(p for p in fam.precisions)
    sizes = (vram_gb,) if vram_gb else fam.vram_common
    fast = fam.precisions[0]
    dat = "bf16" if "bf16" in fam.precisions else "fp32"

    def tiles(v: float) -> str:
        return f"{v:g} GB: {auto_tile('compact', v * 0.9, v, fast)} / {auto_tile('dat2', v * 0.9, v, dat)} px"

    if fam.key == "cpu":
        tile, level = "400 px", "mức 1–5 (model nhẹ); mức 6 rất chậm"
    elif sizes:
        tile = " · ".join(tiles(v) for v in sizes)
        lo = min(sizes)
        level = "mức 1–6" if lo >= 6 else f"mức 1–6 từ 6 GB; {lo:g} GB nên mức 1–5"
    else:
        tile, level = "theo bộ nhớ đọc được, không đọc được thì 400 px", "mức 1–5; mức 6 chậm"
    big = fam.key not in ("cpu", "amd_dml", "intel_igpu", "apple", "intel_arc") and any(v >= 11 for v in sizes or (12,))
    batch = "2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng" if big else "1 ảnh, đọc/ghi file song song"
    return {"Dòng card": fam.label, "Ví dụ": fam.examples, "Kiểu số (thử theo thứ tự)": prec,
            "Cắt ô (compact / DAT-2)": tile,
            "Nên dùng": level, "Hàng loạt": batch, "Mức hỗ trợ": fam.support, "Ghi chú": fam.note}


TABLE_ORDER = ("rtx50", "rtx40", "rtx30", "rtx20", "gtx16", "gtx10", "maxwell", "intel_arc", "intel_igpu",
               "amd_rocm", "amd_dml", "apple", "cpu")


def table() -> list[dict[str, str]]:
    return [recommendation(FAMILIES[k]) for k in TABLE_ORDER]


__all__ = ["Family", "FAMILIES", "nvidia_family", "family_of", "arch_supported", "candidates", "probe_image",
           "probe_dtype", "compare", "ProbeResult", "auto_tile", "vram_class", "batch_workers", "recommendation", "table"]
