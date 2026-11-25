"""Chế độ, preset và thông số chỉnh tay (đặc tả mục 6, 15)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

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
    tile: int = 400
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

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "Settings":
        return replace(self, **kw)


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
