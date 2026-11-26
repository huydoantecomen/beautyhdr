"""Chế độ, preset và thông số chỉnh tay (đặc tả mục 6, 15)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace

MODES = {
    "quick": "Cải thiện nhanh",
    "portrait": "Chân dung AI",
    "old": "Ảnh cũ",
    "upscale": "Phóng to",
    "manual": "Tự chỉnh",
}

MODE_HINTS = {
    "quick": "Ảnh hơi mờ, nhiễu, tối. Nhanh, không tạo chi tiết giả, không cần GPU.",
    "portrait": "Làm rõ khuôn mặt bị mờ, nhoè, độ phân giải thấp. Dùng AI, nên có GPU.",
    "old": "Ảnh scan cũ: giảm nhiễu, tăng tương phản, phóng to, làm rõ mặt. Không tô màu.",
    "upscale": "Phóng to 2× hoặc 4× bằng AI, giữ ảnh nét. Hợp với ảnh không có người.",
    "manual": "Tự chỉnh từng thông số bên dưới.",
}


@dataclass
class Settings:
    mode: str = "quick"
    upscale: int = 0  # 0 = tự động theo cỡ ảnh (levels.auto_upscale: ảnh nhỏ 4×, còn lại 2×), 1, 2, 4
    strength: float = 1.0  # 0..1: trộn kết quả với ảnh gốc phóng Lanczos
    sharpness: float = 0.6  # 0..1 → unsharp amount 0..1.5 (0.6 → 0.9 như tài liệu)
    denoise: float = -1.0  # -1 = tự động theo mức nhiễu; 0..10 → NLM h = 1.2 × mức
    contrast: float = 0.0  # 0..1 → CLAHE clipLimit 0..3 (0 = tắt)
    brightness: float = 0.0  # -1..1
    low_light: bool = False  # Retinex đa thang cho ảnh tối
    # Tự sửa phơi sáng (exposure.py): ảnh quá tối, quá sáng, ngược sáng; 0 = tắt, 1 = sửa đủ mức đo được. Giữ màu,
    # chạy trước mọi bước khác. Bỏ qua khi bật Retinex (low_light) để không làm sáng hai lần.
    exposure: float = 1.0
    # 0..1: độ trung thực AI; mặt ra = (1−f)·AI + f·mặt gốc. 0,7 từ bản 0.3: quét 83 mặt 0–88 tuổi
    # (tools/param_sweep.py) cho IDS/SSIM cao nhất, giữ tuổi và vân da sát ảnh gốc ở cả 6 nhóm tuổi.
    fidelity: float = 0.7
    max_faces: int = 24  # ảnh nhóm: mặt nào không qua model mặt thì bị mạng nền vẽ thành mặt nạ (đo trên ảnh đội bóng 21 mặt)
    tile: int = -1  # cạnh ô cắt ảnh cho mạng nền: −1 = tự động (GPU theo bộ nhớ còn trống, CPU 400), 0 = không cắt
    color_match: bool = True
    face_model: str = "gfpgan"  # "gfpgan" (GFPGAN 1.4) | "restoreformer" (RestoreFormer, chi tiết mạnh hơn)
    protect_faces: bool = True
    # độ trung thực riêng cho mắt, lông mày, môi (tách vùng bằng ParseNet); None = như fidelity.
    # Đặt thấp hơn fidelity: ngũ quan sắc như Remini mà da vẫn giữ người thật và nếp nhăn.
    detail_fidelity: float | None = None
    # Preset theo đối tượng (facestyle.py): "auto" = đoán tuổi, giới tính từng mặt; "child" | "man" | "woman" | "elder"
    # | "adult" = ép một nhóm cho mọi mặt; "off" = tắt. face_styles: kiểu mặt của bộ preset đang dùng (None = bộ mặc định).
    subject: str = "auto"
    face_styles: dict | None = None
    # Đổi màu nền đồng màu (backdrop.py): None = giữ nền; (B, G, R) = đổi sang màu này khi dò được nền đồng màu.
    bg_color: tuple | None = None
    keep_lips: bool = True  # giữ dáng, màu môi gốc (model mặt chỉ thêm độ nét), xem Enhancer._keep_lips
    # Sửa màu ảnh điện thoại (wb.py, phone.py): hãng máy lấy từ EXIF (khoá trong phone.PROFILES).
    phone: str = "generic"
    wb: float = -1.0  # cân bằng trắng: −1 = theo hãng, 0 = tắt, 0..1 = phần ám màu được gỡ
    skin_tone: float = -1.0  # sửa da đỏ/vàng trong vùng mặt: −1 = theo hãng, 0 = tắt, 0..1
    color: float = 0.0  # 0..1: tăng màu (cân bằng trắng, độ rực, rõ khối), xem color.py
    gloss: float = 0.0  # 0..1: độ bóng da (mịn tầng thấp, giữ vân da, ánh bóng nhẹ)
    # 0..1: khử bóng loáng da (deshine.py): giảm vệt sáng chói do da dầu ở trán, mũi, má, giữ vân da và khối mặt.
    # 0 = tắt (mặc định; ô "Khử bóng da" trên giao diện). Không preset nào bật sẵn: người dùng tự chọn.
    deshine: float = 0.0
    # Tông mẫu (look.py): thông số look.Look dạng dict (từ bộ preset có tông mẫu, vd "Trong sáng (ảnh thẻ)"); None = tắt
    # (mặc định). Đưa màu da, môi, tóc, vùng trắng, nền về gần tông của bộ ảnh mẫu. look_amount 0..1: độ mạnh chung.
    look: dict | None = None
    look_amount: float = 1.0
    # 0..1: tông tươi sáng, trong trẻo (tone.py): nâng tông giữa, bỏ lớp sương vùng tối, tăng độ rực, da sáng hơn.
    # Bật sẵn từ bản 0.7 vì huy thấy ảnh sau sửa u ám; 0 = giữ tông ảnh gốc.
    tone: float = 0.6
    # Mạng phóng to nền. "compact" (realesr-general-x4v3, 5 MB) là mặc định từ bản 0.3: đo trên 12 ảnh không mặt
    # SSIM cao hơn và độ nét sát ảnh gốc hơn RRDB, trên mặt IDS ngang nhau, nhanh hơn ~4 lần (tools/evaluate_general.py).
    sr_model: str = "compact"  # "compact" | "rrdb" (RealESRGAN x2plus/x4plus, 67 MB)
    deblur: float = -1.0  # khử mờ khi phóng AI: −1 = tự ước lượng (classic.blur_factor), 1 = tắt, 1.5–4 = mức tay
    sr_denoise: float | None = None  # compact: 0 = giữ hạt, 1 = khử mạnh; None = theo thanh Khử nhiễu / chế độ
    # Phóng to AI (không có model mặt) trên ảnh mờ có người: phần AI trên vùng mặt, còn lại là ảnh gốc phóng bicubic
    # (như faceAi của Android). Mạng nền vẽ lại mặt theo kiểu "trung bình" làm lệch nét người; 0,25 đo trên 6 ảnh có
    # mặt × 3 mức mờ: IDS 0,966 so với 0,934 khi không giữ (mờ nhẹ), SSIM cao nhất. 1 = không giữ mặt.
    face_ai: float = 0.25
    # Khôi phục ảnh cũ (restore.py), chạy trước mọi bước khác; 0 = tắt. repair: vá xước/nếp gãy/bụi (LaMa);
    # fade: phục hồi màu phai; auto_rotate: xoay thẳng ảnh nằm ngang/ngược theo hướng mặt.
    repair: float = 0.0
    fade: float = 0.0
    auto_rotate: bool = False
    # Cách phục hồi màu phai: "giu_da" (0.11: chỉ gỡ ám khi có dấu hiệu phai/giấy ố, giữ ánh sáng ấm có chủ ý, độ rực theo
    # màu da; huy chấm mù 25 cặp: hơn 0.10 ở 4 ảnh, ngang ở 5, kém ở 1) hoặc "0.10" (kéo giãn kênh + độ rực +40 %)
    fade_style: str = "giu_da"
    # Mặt nhỏ (hai mắt cách < 32 px): True = phóng AI vùng mặt rồi mới căn và dựng, ít bịa mắt xanh, tóc vàng hơn nhưng
    # ít nét hơn; False = như 0.10. Chưa bật mặc định: huy ưa bản dựng nhiều AI hơn bản giới hạn AI, chờ chấm vòng 2.
    small_face_sr: bool = False
    # Ảnh cũ chụp cảnh (không phải ảnh thẻ nền đồng màu, không phải ảnh đen trắng): 0..1 nâng tông sặc sỡ hơn (độ rực,
    # tương phản tông giữa), da được giữ ít đổi. Chỉ chạy khi bật khôi phục ảnh cũ. 0 = tắt (mặc định).
    scene_vivid: float = 0.0
    scene_ai: float = 0.0          # 0..1: ảnh cảnh cũ lấy màu từ AI tô màu DDColor (cần scene_vivid > 0, tải 912 MB)
    # Đường dẫn PNG mặt nạ vùng hỏng nặng do người dùng tô (trắng = hỏng): vá bằng LaMa rồi, nếu trúng mặt, dựng lại
    # mặt đó 100 % bằng model mặt trong vùng hỏng (nút "Vẽ vùng hỏng…", CLI --mask).
    damage_mask: str = ""

    @property
    def sr_dn(self) -> float:
        if self.sr_denoise is not None:
            return self.sr_denoise
        if self.denoise >= 0:
            return min(self.denoise / 10.0, 1.0)
        return 1.0 if self.mode == "old" else 0.5

    # quy đổi sang tham số thuật toán
    @property
    def unsharp_amount(self) -> float:
        return 1.5 * self.sharpness

    @property
    def nlm_h(self) -> float | None:
        return None if self.denoise < 0 else 1.2 * self.denoise

    @property
    def clahe_clip(self) -> float:
        return 3.0 * self.contrast

    @property
    def ai_weight(self) -> float:
        return 1.0 - self.fidelity

    def with_(self, **kw) -> "Settings":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, warn: list[str] | None = None) -> "Settings":
        """Settings từ dict lạ (máy khác gửi qua API, file): chỉ nhận trường đã biết, đúng kiểu, số kẹp vào khoảng
        hợp lý; sai thì bỏ (dùng mặc định) và ghi vào `warn`. Không bao giờ nhận đường dẫn file (damage_mask)."""
        warn = [] if warn is None else warn
        if not isinstance(d, dict):
            raise ValueError("thông số phải là JSON object")
        kw: dict = {}
        for f in fields(cls):
            if f.name not in d or f.name in _NO_REMOTE:
                continue
            v, default = d[f.name], getattr(cls, f.name, None)
            ok = _check_field(f.name, v, default)
            if ok is _BAD:
                warn.append(f"bỏ {f.name}={str(v)[:40]}")
            else:
                kw[f.name] = ok
        for k in d:
            if k not in _FIELD_NAMES:
                warn.append(f"bỏ khoá lạ {str(k)[:40]}")
        return cls(**kw)


_BAD = object()
_FIELD_NAMES = {f.name for f in fields(Settings)}
_NO_REMOTE = {"damage_mask", "tile"}  # đường dẫn trên máy gửi; ô cắt ảnh theo card của máy chạy
_CHOICES = {"mode": set(MODES), "face_model": {"gfpgan", "restoreformer", "restoreformer_pp"},
            "sr_model": {"compact", "rrdb", "plksr", "dat2"},
            "fade_style": {"giu_da", "0.10"}}
_NUM_RANGE = {"upscale": (0, 4), "max_faces": (0, 64), "denoise": (-1.0, 10.0), "deblur": (-1.0, 4.0),
              "brightness": (-1.0, 1.0), "wb": (-1.0, 1.0), "skin_tone": (-1.0, 1.0)}


def _check_field(name: str, v, default):
    """Giá trị hợp lệ (đã kẹp) hoặc _BAD."""
    import math
    if name in ("face_styles", "look"):
        if v is None:
            return None
        if not isinstance(v, dict) or len(json.dumps(v)) > 8192:
            return _BAD
        try:
            if name == "face_styles":
                from .facestyle import parse_set
                return {k: st.to_dict() for k, st in parse_set({"name": "x", "styles": v}, []).styles.items()}
            from .look import Look
            return asdict(Look.from_dict(v))
        except Exception:
            return _BAD
    if name == "bg_color":
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in v):
            return tuple(v)
        return _BAD
    if name in ("detail_fidelity", "sr_denoise"):
        if v is None:
            return None
        default = 0.0
    if isinstance(default, bool):
        return v if isinstance(v, bool) else _BAD
    if isinstance(default, (int, float)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return _BAD
        lo, hi = _NUM_RANGE.get(name, (0.0, 1.0))
        v = min(hi, max(lo, v))
        if isinstance(default, int):
            v = int(round(v))
            if name == "upscale" and v == 3:
                v = 4
        return v
    if isinstance(default, str):
        if not isinstance(v, str) or len(v) > 40:
            return _BAD
        if name in _CHOICES and v not in _CHOICES[name]:
            return _BAD
        return v
    return _BAD



# Preset: chỉ những trường khác mặc định. "auto" do bộ phân tích quyết định.
PRESETS: dict[str, tuple[str, dict]] = {
    "auto": ("Tự động", {}),
    "natural": ("Tự nhiên", dict(sharpness=0.4, contrast=0.0, fidelity=0.8, strength=0.85, tone=0.3)),
    "sharp": ("Rõ nét", dict(sharpness=0.8, contrast=0.5, fidelity=0.35)),
    "portrait": ("Chân dung", dict(mode="portrait", sharpness=0.5, fidelity=0.7)),
    # kiểu Remini: AI vẽ lại toàn bộ mặt, phóng ít nhất 2× (tự động: ảnh nhỏ 4×), thêm màu và độ bóng; đẹp và nét
    # nhất nhưng mặt có thể hơi khác
    "max": ("Rõ nét tối đa", dict(mode="portrait", upscale=0, sharpness=0.8, fidelity=0.3, detail_fidelity=0.0,
                                         color=0.6, gloss=0.0)),
    # ảnh in cũ chụp lại bằng điện thoại: xoay thẳng, vá xước/bụi/nếp gãy, phục hồi màu phai, rồi làm nét kiểu Rõ nét tối
    # đa (mặt giữ 70 % AI), phóng tự động; tông tươi sáng của bản 0.7 vẫn bật
    "restore": ("Khôi phục ảnh cũ", dict(mode="portrait", upscale=0, repair=1.0, fade=1.0, auto_rotate=True,
                                          sharpness=0.7, fidelity=0.4, detail_fidelity=0.1, color=0.4, gloss=0.2,
                                          wb=0.0, tone=0.5, max_faces=24)),
    "old_photo": ("Ảnh cũ", dict(mode="old", contrast=0.67, fidelity=0.7)),
    "low_light": ("Ảnh tối", dict(low_light=True, brightness=0.2, contrast=0.0, sharpness=0.4)),
    "custom": ("Tuỳ chỉnh", {}),
}


def apply_preset(base: Settings, preset: str) -> Settings:
    return base.with_(**PRESETS[preset][1])
