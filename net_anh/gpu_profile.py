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
    Family("kepler", "GeForce GTX 600 / 700 (Kepler) và cũ hơn", "GTX 660, 760, 770, 780, GT 710, GT 730",
           ("fp32",), False, 1, "chưa hỗ trợ",
           "PyTorch không còn bản nào chạy trên Kepler (từ 2023): app dùng CPU. Nên nâng cấp card.", vram_common=()),
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
        if major == 5:
            return FAMILIES["maxwell"]
        if major < 5:
            return FAMILIES["kepler"]
    m = re.search(r"RTX\s*(?:PRO\s*)?([2-5])0[5-9]0", n)
    if m:
        return FAMILIES[f"rtx{m[1]}0"]
    # card trạm: RTX PRO … Blackwell, RTX … Ada, RTX A4000 (Ampere), Quadro RTX (Turing), Quadro P (Pascal), M (Maxwell), K (Kepler)
    if re.search(r"BLACKWELL", n):
        return FAMILIES["rtx50"]
    if re.search(r"RTX\s*\d+\s*ADA|\bL4\b|L40", n):
        return FAMILIES["rtx40"]
    if re.search(r"RTX\s*A\d{3,4}|\bA[1-9]0{1,2}\b|\bA100\b", n):
        return FAMILIES["rtx30"]
    if re.search(r"QUADRO\s*RTX|\bT4\b", n):
        return FAMILIES["rtx20"]
    if re.search(r"GTX\s*10[5-8]0|MX\s*[1-3][1-5]0|TITAN\s*XP|GT\s*10[3-9]0|QUADRO\s*P\d|TESLA\s*P\d", n):
        return FAMILIES["gtx10"]
    if re.search(r"GTX\s*(75\d|9[5-8]0)|TITAN\s*X\b|QUADRO\s*M\d|TESLA\s*M\d", n):
        return FAMILIES["maxwell"]
    if re.search(r"GTX\s*(6\d0|7[0-46-9]0)\b|\bGT\s*[67]\d0\b|QUADRO\s*K\d|GTX\s*TITAN(?!\s*X)|TESLA\s*K\d", n):
        return FAMILIES["kepler"]
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


def face_forward(net, x):
    """Model mặt (GFPGAN, RestoreFormer): vào ảnh 0..1, ra 0..1 (mạng dùng thang −1..1) để so như model nền."""
    return (net(x * 2 - 1, return_rgb=False)[0] + 1) / 2


def probe_dtype(net, device, dtype, size: int = 96, fn=None) -> ProbeResult:
    """Chạy thử `net` (đang ở fp32 trên `device`) với ảnh mẫu ở fp32 và ở `dtype`, so kết quả. Trả về net ở trạng thái
    fp32 (người gọi tự đổi kiểu khi ok). `fn(net, x)`: cách gọi mạng (mặc định net(x)); model mặt dùng face_forward
    với ảnh 512 px."""
    import torch

    fn = fn or (lambda n, z: n(z))
    x = probe_image(size).to(device)
    try:
        with torch.no_grad():
            torch.manual_seed(0)  # GFPGAN cộng nhiễu ngẫu nhiên: cùng hạt giống cho hai lần chạy
            ref = fn(net, x)
            net.to(dtype)
            torch.manual_seed(0)
            out = fn(net, x.to(dtype))
        return compare(ref, out)
    except Exception as e:  # kiểu số không chạy được trên card / bản PyTorch này
        return ProbeResult(False, math.inf, math.inf, f"{e.__class__.__name__}: {str(e)[:80]}")
    finally:
        net.float()


# ---------------------------------------------------------------- tăng tốc thêm theo card × model

# Kiến trúc từng model: tích chập (conv) hay chú ý (attention). Tensor Core (RTX 20 trở lên) chạy tích chập fp16 / bf16
# nhanh nhất khi dữ liệu xếp theo kênh cuối (channels_last, NHWC); để mặc định (NCHW) thì cuDNN phải đảo lại mỗi lớp.
# GFPGAN là "stylegan": có .view() trên đặc trưng nên không chạy được kênh cuối (đã thử, lỗi view size).
MODEL_ARCH = {"compact": "conv", "rrdb": "conv", "plksr": "conv", "dat2": "attention", "gfpgan": "stylegan",
              "restoreformer": "attention", "restoreformer_pp": "attention"}
MODEL_OF_KEY = {"compact": "compact", "esrgan_x4": "rrdb", "esrgan_x2": "rrdb", "plksr_web": "plksr",
                "dat2_web": "dat2"}


def _env_flag(name: str) -> bool | None:
    v = os.environ.get(name, "").strip().lower()
    return {"1": True, "on": True, "0": False, "off": False}.get(v)


def channels_last(kind: str, cc: tuple[int, int] | None, dtype: str, model: str) -> bool:
    """Xếp dữ liệu kênh cuối cho model này không. Chỉ bật ở nơi chắc có lợi: card NVIDIA có Tensor Core (cc ≥ 7.0),
    kiểu số fp16 / bf16, model tích chập. Kết quả ảnh không đổi (cùng phép tính, chỉ khác thứ tự trong bộ nhớ).
    NETANH_CHANNELS_LAST=0|1 để ép (đo so sánh: Trợ giúp → Đo tốc độ card)."""
    if kind != "cuda" or MODEL_ARCH.get(model) != "conv":
        return False
    env = _env_flag("NETANH_CHANNELS_LAST")
    if env is not None:
        return env
    return cc is not None and tuple(cc) >= (7, 0) and dtype in ("fp16", "bf16")


def cudnn_benchmark(kind: str) -> bool:
    """Cho cuDNN tự đo và chọn thuật toán nhanh nhất cho từng cỡ ô (tốn chút thời gian lần đầu mỗi cỡ). Chỉ bật khi làm
    cả thư mục trên NVIDIA: các ảnh sau cùng cỡ ô nên lợi; ảnh lẻ thì không đáng. NETANH_CUDNN_BENCHMARK=0|1 để ép."""
    if kind != "cuda":
        return False
    env = _env_flag("NETANH_CUDNN_BENCHMARK")
    return True if env is None else env


def face_candidates(fam: Family, model: str) -> list[str]:
    """Kiểu số thử cho model mặt: chỉ bf16 (card có bf16 thật, RTX 30 trở lên), không thì fp32. fp16 bị loại: đo trên
    CPU 2026-11-26 GFPGAN fp16 lệch fp32 tới 91 mức xám (chuẩn hoá trọng số StyleGAN tràn số), bf16 chỉ lệch 0,5 mức."""
    return (["bf16"] if "bf16" in fam.precisions else []) + ["fp32"]


def to_channels_last(net) -> None:
    """Đổi trọng số các lớp Conv2d sang kênh cuối (Module.to(memory_format) lỗi với tham số không phải 4 chiều)."""
    import torch

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.to(memory_format=torch.channels_last)


def to_contiguous(net) -> None:
    import torch

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.to(memory_format=torch.contiguous_format)


# Tốc độ dự kiến: thời gian mỗi megapixel ảnh VÀO trên CPU 4 luồng (đo 2026-11-26, ô 192 px, PyTorch 2.x) chia cho
# speedup của dòng card và hệ số kiểu số. CHƯA ĐO TRÊN GPU THẬT: số ước lượng, sẽ thay bằng file đo của huy.
CPU_S_PER_MP = {"compact": 4.8, "plksr": 34.5, "rrdb": 83.0, "dat2": 286.0}
CPU_S_PER_FACE = {"gfpgan": 1.15}
# nhanh thêm bao nhiêu lần so với fp32 trên cùng card có Tensor Core (tích chập fp16 + kênh cuối ≈ 2×, chú ý bf16 ≈ 1,5×)
DTYPE_GAIN = {("conv", "fp16"): 2.0, ("conv", "bf16"): 1.8, ("attention", "bf16"): 1.5, ("attention", "fp16"): 1.6,
              ("stylegan", "bf16"): 1.5}


def best_dtype(fam: Family, model: str) -> str:
    """Kiểu số dự kiến app sẽ chọn cho model trên dòng card (khi chạy thử đạt)."""
    if model in ("dat2",):  # DAT-2 fp16 ra NaN: chỉ bf16 thật (cc ≥ 8)
        return "bf16" if "bf16" in fam.precisions else "fp32"
    if model in CPU_S_PER_FACE or model.startswith("restoreformer"):  # model mặt
        return face_candidates(fam, model)[0]
    return fam.precisions[0]


def est_seconds(fam: Family, model: str, mp: float = 1.0) -> float | None:
    """Thời gian dự kiến cho `mp` megapixel ảnh vào (model nền) hoặc một mặt (model mặt). None khi chưa có số đo CPU."""
    base = CPU_S_PER_MP.get(model) if model in CPU_S_PER_MP else CPU_S_PER_FACE.get(model)
    if not base:
        return None
    mult = mp if model in CPU_S_PER_MP else 1.0
    gain = DTYPE_GAIN.get((MODEL_ARCH.get(model, "conv"), best_dtype(fam, model)), 1.0)
    return base * mult / max(fam.speedup, 1e-6) / gain


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

    if fam.key in ("cpu", "kepler"):
        tile, level = "400 px (chạy CPU)", "mức 1–5 (model nhẹ); mức 6 rất chậm"
    elif sizes:
        tile = " · ".join(tiles(v) for v in sizes)
        lo = min(sizes)
        level = "mức 1–6" if lo >= 6 else f"mức 1–6 từ 6 GB; {lo:g} GB nên mức 1–5"
    else:
        tile, level = "theo bộ nhớ đọc được, không đọc được thì 400 px", "mức 1–5; mức 6 chậm"
    big = fam.key not in ("cpu", "kepler", "amd_dml", "intel_igpu", "apple", "intel_arc") and any(v >= 11 for v in sizes or (12,))
    batch = "2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng" if big else "1 ảnh, đọc/ghi file song song"
    return {"Dòng card": fam.label, "Ví dụ": fam.examples, "Kiểu số (thử theo thứ tự)": prec,
            "Cắt ô (compact / DAT-2)": tile,
            "Nên dùng": level, "Hàng loạt": batch, "Mức hỗ trợ": fam.support, "Ghi chú": fam.note}


MODELS = (("compact", "Real-ESRGAN compact (mức 1–4, mặc định)"), ("plksr", "RealPLKSR (mức 5)"),
          ("rrdb", "Real-ESRGAN x4plus"), ("dat2", "DAT-2 (mức 6 Cao nhất)"), ("gfpgan", "GFPGAN (mỗi khuôn mặt)"))
MODEL_COLS = ("Model", "Kiểu số", "Kênh cuối", "Cắt ô", "Dự kiến", "Ghi chú")


def _fmt_s(t: float | None) -> str:
    if t is None:
        return "?"
    if t < 0.1:
        return f"{t * 1000:.0f} ms"
    return f"{t:.1f} s" if t < 60 else f"{t / 60:.1f} phút"


def model_rows(fam: Family, vram_gb: float = 0.0, cc: tuple[int, int] | None = None) -> list[dict[str, str]]:
    """Bảng card × model: kiểu số app sẽ chọn (khi chạy thử đạt), có xếp kênh cuối không, cỡ ô theo VRAM và thời gian
    dự kiến cho ảnh 1 MP vào (≈ 1200×800, ra 4 800×3 200) hoặc một mặt. Thời gian là ƯỚC LƯỢNG: số đo CPU / speedup dòng
    card / hệ số kiểu số; sẽ thay bằng số đo thật (Trợ giúp → Đo tốc độ card)."""
    gpu = fam.key not in ("cpu", "kepler")
    kind = "cuda" if fam.key in ("rtx50", "rtx40", "rtx30", "rtx20", "gtx16", "volta", "gtx10", "maxwell", "nvidia") else "other"
    cc = cc or {"rtx50": (12, 0), "rtx40": (8, 9), "rtx30": (8, 6), "rtx20": (7, 5), "gtx16": (7, 5), "volta": (7, 0),
                "gtx10": (6, 1), "maxwell": (5, 2)}.get(fam.key)
    v = vram_gb or (min(fam.vram_common) if fam.vram_common else 0.0)
    rows = []
    for m, label in MODELS:
        dt = best_dtype(fam, m) if gpu else "fp32"
        face = m == "gfpgan"
        if face:
            tile = "512 px cố định (một mặt)"
        elif not gpu:
            tile = f"{CPU_TILE} px"
        elif v:
            tile = f"{auto_tile(m, v * 0.9, v, dt)} px ({v:g} GB)"
        else:
            tile = "theo bộ nhớ, không đọc được thì 400 px"
        note = ""
        if m == "dat2":
            note = "fp16 ra NaN nên chỉ bf16 (RTX 30 trở lên) hoặc fp32" if dt == "fp32" and gpu else "Model chú ý: không dùng kênh cuối"
            if fam.key in ("gtx16", "gtx10", "maxwell", "intel_igpu", "amd_dml", "cpu", "kepler"):
                note = "Rất chậm trên dòng này: nên dùng mức 1–5"
        elif face:
            note = ("bf16 (lệch < 1 mức xám); fp16 bị loại vì lệch màu mặt 91 mức xám" if dt == "bf16" else
                    "fp32: fp16 làm lệch màu mặt (91 mức xám), dòng này không có bf16 thật") if gpu else ""
        elif m == "compact":
            note = "Nhẹ nhất, mặc định"
        cl = channels_last(kind, cc, dt, m)
        rows.append({"Model": label, "Kiểu số": dt, "Kênh cuối": "có" if cl else "không", "Cắt ô": tile,
                     "Dự kiến": _fmt_s(est_seconds(fam, m, 1.0)) + (" / mặt" if face else " / 1 MP vào"),
                     "Ghi chú": note})
    return rows


TABLE_ORDER = ("rtx50", "rtx40", "rtx30", "rtx20", "gtx16", "gtx10", "maxwell", "kepler", "intel_arc", "intel_igpu",
               "amd_rocm", "amd_dml", "apple", "cpu")


def table() -> list[dict[str, str]]:
    return [recommendation(FAMILIES[k]) for k in TABLE_ORDER]


__all__ = ["model_rows", "MODELS", "MODEL_COLS", "to_channels_last", "to_contiguous", "channels_last", "cudnn_benchmark", "face_candidates", "face_forward", "best_dtype", "est_seconds",
           "MODEL_ARCH", "Family", "FAMILIES", "nvidia_family", "family_of", "arch_supported", "candidates", "probe_image",
           "probe_dtype", "compare", "ProbeResult", "auto_tile", "vram_class", "batch_workers", "recommendation", "table"]
