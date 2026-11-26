"""Sinh prompt cho hệ render ngoài (ChatGPT, Gemini, Grok…) từ đúng thông số tinh chỉnh đang dùng trong Beauty HDR.

Chỉ có trong bản nhà phát triển (gói net_anh_dev). Không gọi mạng, không đọc ảnh: chỉ dịch Settings (mức nét, tông mẫu,
da, màu, ánh sáng, preset đối tượng, khôi phục ảnh cũ, đổi nền…) và kết quả phân tích ảnh (nếu có) thành văn bản.

Mỗi ý là một Item có độ ưu tiên: 0 = luôn giữ (giữ danh tính, điều cấm), 1 = chính, 2 = chi tiết, 3 = phụ. Prompt dài hơn
giới hạn của hệ đích thì bỏ dần ý ưu tiên thấp nhất và liệt kê ý đã bỏ (PromptPack.dropped). Giới hạn độ dài là mức
app tự đặt cho prompt dễ dùng, không phải giới hạn chính thức của từng dịch vụ.

Prompt KHÔNG chứa tên file, đường dẫn, EXIF, hãng máy hay vị trí: chỉ mô tả việc cần làm.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from net_anh.facestyle import SUBJECT_KEYS, FaceStyle, styles_from_settings
from net_anh.levels import LEVELS, auto_upscale
from net_anh.look import Look
from net_anh.phone import profile as phone_profile
from net_anh.settings import Settings

LANGS = {"en": "English", "vi": "Tiếng Việt"}


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    limit: int  # số ký tự tối đa app tự đặt cho prompt
    style: str  # "sections" (mục có gạch đầu dòng) | "narrative" (đoạn văn) | "compact" (một đoạn ngắn)
    numbers: bool  # kèm số % (ChatGPT hiểu tốt; Grok, Gemini dễ hiểu nhầm số thành nội dung ảnh)
    note: str  # ghi chú cách dùng (tiếng Việt, hiện trong cửa sổ, không nằm trong prompt)


TARGETS: dict[str, Target] = {t.key: t for t in [
    Target("chatgpt", "ChatGPT", 3500, "sections", True,
           "Đính kèm ảnh gốc trong cùng tin nhắn rồi dán prompt (chế độ sửa ảnh, không phải tạo ảnh mới). Nếu mặt bị "
           "đổi, nhắn tiếp: “Keep the face exactly as in the original photo, only fix …”."),
    Target("gemini", "Gemini", 2500, "narrative", False,
           "Tải ảnh lên Gemini (sửa ảnh) rồi dán prompt. Gemini làm tốt với mô tả dạng đoạn văn; nếu kết quả lệch, sửa "
           "từng ý một trong các lượt sau thay vì gửi lại cả prompt."),
    Target("grok", "Grok", 900, "compact", False,
           "Tải ảnh lên Grok (sửa ảnh) rồi dán prompt. Prompt được rút gọn: các ý bị bỏ liệt kê bên dưới, cần thì "
           "gửi thêm ở lượt sau."),
    Target("generic", "Chung (hệ khác)", 3000, "sections", True,
           "Dạng chung cho hệ sửa ảnh khác (Stable Diffusion img2img, Flux Kontext, Midjourney editor…). Dùng chế độ "
           "sửa ảnh có ảnh đầu vào, độ thay đổi (denoise/strength) thấp để giữ người."),
]}

PRIVACY_VI = ("Gửi ảnh lên ChatGPT, Gemini, Grok… là gửi ảnh ra dịch vụ bên ngoài: ảnh khuôn mặt là dữ liệu sinh trắc, "
              "có thể được lưu và dùng theo điều khoản của dịch vụ (kể cả để huấn luyện). Chỉ gửi ảnh khi người trong "
              "ảnh đồng ý, không gửi ảnh giấy tờ tuỳ thân. Beauty HDR không tự gửi gì đi: prompt chỉ là văn bản, không "
              "chứa tên file, EXIF, hãng máy hay vị trí.")

_AMT_EN = ((0.15, "very subtle"), (0.35, "subtle"), (0.6, "moderate"), (0.85, "strong"), (9.0, "very strong"))
_AMT_VI = ((0.15, "rất nhẹ"), (0.35, "nhẹ"), (0.6, "vừa phải"), (0.85, "mạnh"), (9.0, "rất mạnh"))


def amt(v: float, lang: str) -> str:
    v = abs(float(v))
    return next(w for th, w in (_AMT_VI if lang == "vi" else _AMT_EN) if v < th)


@dataclass
class Item:
    prio: int
    group: str  # keep | do | face | avoid | output | diag
    en: str
    vi: str
    xen: str = ""  # phần số (vd " (≈70 % original)") chỉ thêm cho hệ hiểu số
    xvi: str = ""

    def text(self, lang: str, numbers: bool) -> str:
        base, extra = (self.vi, self.xvi) if lang == "vi" else (self.en, self.xen)
        return base + (extra if numbers else "")


@dataclass
class PromptPack:
    target: str
    lang: str
    text: str
    limit: int
    dropped: list[str] = field(default_factory=list)  # ý bị bỏ vì quá dài (theo ngôn ngữ của prompt)
    note: str = ""

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Context:
    """Thông tin thêm ngoài Settings: mức nét đang chọn, cỡ ảnh, kết quả phân tích, tên bộ preset."""
    level: int | None = None
    size: tuple[int, int] | None = None  # (rộng, cao) ảnh gốc
    analysis: object | None = None  # net_anh.analysis.Analysis
    preset_name: str = ""
    diagnosis: bool = True  # kèm dòng chẩn đoán ảnh

    @property
    def faces(self) -> int | None:
        a = self.analysis
        if a is None or not getattr(a, "faces_checked", False):
            return None
        return len(a.good_faces)


# ---------------------------------------------------------------- dịch thông số → ý

_GROUPS = {"child": ("children", "trẻ em"), "man": ("men", "nam giới"), "woman": ("women", "nữ giới"),
           "elder": ("elderly people", "người lớn tuổi"), "adult": ("other adults", "người lớn khác")}
_CAST = {"vàng": "yellow", "đỏ": "red", "cam": "orange", "xanh": "blue", "xanh lá": "green", "tím hồng": "magenta"}
_BG_NAMES = {(255, 255, 255): ("white", "trắng"), (219, 142, 67): ("blue", "xanh"),
             (235, 206, 135): ("light blue", "xanh nhạt"), (30, 30, 200): ("red", "đỏ"),
             (203, 212, 217): ("light warm grey", "xám kem")}
_LEVEL_TEXT = {
    1: ("Light touch-up only: reduce noise and sharpen slightly, no AI re-drawing; the photo must look untouched.",
        "Chỉ chỉnh nhẹ: giảm nhiễu, làm nét nhẹ, không vẽ lại bằng AI; ảnh phải trông như chưa chỉnh."),
    2: ("Gentle deblur of the whole photo; faces stay almost exactly as they are.",
        "Khử mờ nhẹ toàn ảnh; khuôn mặt giữ gần như nguyên vẹn."),
    3: ("Clear up faces and background moderately while keeping the same person.",
        "Làm rõ mặt và nền ở mức vừa phải, vẫn đúng người."),
    4: ("Strong clarity: noticeably sharper faces, crisper eyes and lips, slightly richer color.",
        "Làm rõ mạnh: mặt nét hơn hẳn, mắt và môi sắc hơn, màu đậm hơn một chút."),
    5: ("Maximum clarity like a professional restoration: rebuild fine detail in eyes, lips, skin and hair, richer "
        "color; the person must still be clearly recognizable.",
        "Rõ nét tối đa như phục chế chuyên nghiệp: dựng lại chi tiết mắt, môi, da, tóc, màu đậm hơn; người trong ảnh "
        "vẫn phải nhận ra rõ ràng."),
    6: ("Highest-quality restoration: maximum clarity with the most faithful face reconstruction and the cleanest "
        "detail across the whole photo; the person must still be clearly recognizable.",
        "Phục chế chất lượng cao nhất: rõ nét tối đa, dựng mặt trung thực nhất, chi tiết sạch trên toàn ảnh; người trong "
        "ảnh vẫn phải nhận ra rõ ràng."),
}


def _pct(v: float) -> str:
    return f"{round(v * 100):.0f}%"


def _face_style_parts(st: FaceStyle, lang: str, deshine: bool) -> list[str]:
    vi = lang == "vi"
    out = []
    if st.keep > 0.05:
        out.append("giữ vân da, lỗ chân lông thật" if vi else "keep real skin texture and pores")
    if st.detail > 0.05:
        out.append(f"mắt, lông mày, môi sắc nét {amt(st.detail, lang)}" if vi
                   else f"{amt(st.detail, lang)} sharpening of eyes, brows and lips")
    elif st.detail < -0.05:
        out.append("mắt, môi mềm tự nhiên" if vi else "keep eyes and lips soft and natural")
    if st.smooth > 0.05:
        out.append(f"mịn da {amt(st.smooth, lang)}, vẫn giữ vân da" if vi
                   else f"{amt(st.smooth, lang)} skin smoothing that keeps texture")
    if st.glow > 0.05:
        out.append(f"da căng bóng khoẻ {amt(st.glow, lang)}" if vi else f"{amt(st.glow, lang)} healthy dewy glow")
    if st.bright > 0.05:
        out.append(f"da sáng hơn {amt(st.bright, lang)}" if vi else f"{amt(st.bright, lang)} skin brightening")
    if st.warm > 0.05:
        out.append(f"da hồng hào {amt(st.warm, lang)}" if vi else f"{amt(st.warm, lang)} rosier skin")
    elif st.warm < -0.05:
        out.append(f"da bớt đỏ {amt(st.warm, lang)}" if vi else f"{amt(st.warm, lang)} less redness")
    if deshine and st.matte > 0.05:
        out.append(f"bớt bóng dầu {amt(st.matte, lang)}" if vi else f"{amt(st.matte, lang)} less oily shine")
    elif deshine and st.matte < -0.05:
        out.append("giữ độ bóng tự nhiên của da" if vi else "keep the natural skin sheen")
    return out


def _bg_name(bgr) -> tuple[str, str, str]:
    b, g, r = (int(x) for x in bgr)
    hexv = f"#{r:02X}{g:02X}{b:02X}"
    en, vi = _BG_NAMES.get((b, g, r), ("solid", "đồng màu"))
    return en, vi, hexv


def _look_items(lk: Look, amount: float) -> list[Item]:
    items = []
    k = max(0.0, min(1.0, amount))
    bright = lk.skin_L >= 72
    en = [("bright, luminous" if bright else "natural") + " skin", "clear and not yellowish" if lk.skin_b <= 16 else "",
          "slightly pink-neutral" if 6 <= lk.skin_a <= 14 else ""]
    vi = [("da sáng, trong" if bright else "da tự nhiên"), "không ngả vàng" if lk.skin_b <= 16 else "",
          "hồng trung tính" if 6 <= lk.skin_a <= 14 else ""]
    items.append(Item(1, "do", f"Target look “{lk.name}” ({amt(k * max(lk.skin, lk.skin_light), 'en')}): "
                      + ", ".join(x for x in en if x) + ".",
                      f"Tông mẫu “{lk.name}” ({amt(k * max(lk.skin, lk.skin_light), 'vi')}): "
                      + ", ".join(x for x in vi if x) + ".",
                      f" Skin target Lab ≈ L{lk.skin_L:.0f} a{lk.skin_a:.0f} b{lk.skin_b:.0f}.",
                      f" Đích da Lab ≈ L{lk.skin_L:.0f} a{lk.skin_a:.0f} b{lk.skin_b:.0f}."))
    if lk.clean > 0.05 or lk.even > 0.05 or lk.tidy > 0.05:
        e, v = [], []
        if lk.clean > 0.05:
            e.append(f"{amt(k * lk.clean, 'en')} cleanup of acne, dark spots and small red marks")
            v.append(f"làm sạch mụn, đốm sạm, vết đỏ nhỏ {amt(k * lk.clean, 'vi')}")
        if lk.even > 0.05 or lk.tidy > 0.05:
            e.append(f"{amt(k * max(lk.even, lk.tidy), 'en')} more even skin tone")
            v.append(f"đều màu da {amt(k * max(lk.even, lk.tidy), 'vi')}")
        items.append(Item(2, "do", ", ".join(e).capitalize() + ", while keeping natural pores and skin texture.",
                          ", ".join(v).capitalize() + ", vẫn giữ lỗ chân lông và vân da tự nhiên."))
    e, v = [], []
    if lk.lips > 0.05:
        e.append("fresh, naturally rosy lips")
        v.append("môi tươi hồng tự nhiên")
    if lk.hair > 0.05:
        e.append("neutral deep-black hair without color cast")
        v.append("tóc đen trung tính, không ám màu")
    if lk.white > 0.05:
        e.append("clean neutral whites" + (", slightly cool" if lk.cool > 0.5 else ""))
        v.append("vùng trắng sạch, trung tính" + (", hơi lạnh" if lk.cool > 0.5 else ""))
    if lk.bg_clean > 0.05:
        e.append("an even, clean, flat backdrop")
        v.append("nền phẳng, sạch, đều màu")
    if e:
        items.append(Item(2, "do", "Also: " + ", ".join(e) + ".", "Thêm: " + ", ".join(v) + "."))
    return items


def collect(s: Settings, ctx: Context | None = None) -> list[Item]:
    """Mọi ý của prompt theo thứ tự xuất hiện."""
    ctx = ctx or Context()
    it: list[Item] = []
    faces = ctx.faces
    face_ai = s.mode in ("portrait", "old")
    restore = s.repair > 0 or s.fade > 0 or s.auto_rotate
    many = faces is not None and faces > 1

    # --- việc chính
    it.append(Item(0, "task", "Edit the attached photo. Do not generate a new image: retouch this exact photo.",
                   "Sửa chính ảnh đính kèm. Không tạo ảnh mới: chỉ chỉnh sửa trên đúng ảnh này."))
    who_en = f"all {faces} people" if many else "the person" if faces == 1 else "every person"
    who_vi = f"cả {faces} người" if many else "người trong ảnh" if faces == 1 else "mọi người trong ảnh"
    it.append(Item(0, "keep",
                   f"Keep {who_en} 100% recognizable: same face shape, eyes, eyebrows, nose, mouth, ears, age, "
                   "skin color, expression, hairstyle, pose and clothing.",
                   f"Giữ {who_vi} nhận ra 100 %: đúng dáng mặt, mắt, lông mày, mũi, miệng, tai, tuổi, màu da, nét mặt, "
                   "kiểu tóc, tư thế và quần áo."))
    if s.bg_color is not None:
        it.append(Item(0, "keep", "Keep the same framing, composition and aspect ratio (only the background color "
                                  "changes, as described below).",
                       "Giữ nguyên khung hình, bố cục và tỉ lệ ảnh (chỉ đổi màu nền như bên dưới)."))
    else:
        it.append(Item(0, "keep", "Keep the same framing, composition, background and aspect ratio.",
                       "Giữ nguyên khung hình, bố cục, nền và tỉ lệ ảnh."))

    lv = ctx.level if ctx.level in LEVELS else None
    if lv:
        en, vi = _LEVEL_TEXT[lv]
        it.append(Item(1, "do", en, vi, f" (Beauty HDR level {lv}/6 “{LEVELS[lv].name}”.)",
                       f" (Beauty HDR mức {lv}/6 “{LEVELS[lv].name}”.)"))

    # --- khôi phục ảnh cũ
    if restore:
        if s.auto_rotate:
            it.append(Item(2, "do", "Straighten the photo if it is rotated or upside down.",
                           "Xoay thẳng ảnh nếu bị nghiêng, nằm ngang hoặc ngược."))
        if s.repair > 0:
            it.append(Item(1, "do", "Repair scratches, dust, cracks, creases, stains and torn edges of this old print.",
                           "Vá vết xước, bụi, vết nứt, nếp gấp, vết ố và mép rách của ảnh in cũ."))
        if s.fade > 0:
            if s.fade_style == "giu_da":
                it.append(Item(1, "do", "Restore faded colors and yellowed paper, but keep intentional warm light "
                                        "and natural skin tones.",
                               "Phục hồi màu phai và giấy ố vàng, nhưng giữ ánh sáng ấm có chủ ý và màu da tự nhiên."))
            else:
                it.append(Item(1, "do", "Restore faded colors: stretch contrast and bring back saturation.",
                               "Phục hồi màu phai: kéo giãn tương phản, trả lại độ bão hoà."))
        if s.damage_mask:
            it.append(Item(1, "do", "Some areas are heavily damaged (marked white in the second attached image): "
                                    "reconstruct them plausibly; if they cover a face, rebuild it consistent with "
                                    "the rest of that face.",
                           "Một số vùng hỏng nặng (tô trắng ở ảnh đính kèm thứ hai): dựng lại hợp lý; nếu trúng mặt, "
                           "dựng lại cho khớp phần còn lại của khuôn mặt đó."))
        if s.scene_vivid > 0:
            it.append(Item(2, "do", "For scenery and clothing, make colors more vivid while keeping skin tones.",
                           "Với cảnh và quần áo, nâng màu sặc sỡ hơn nhưng giữ màu da."))
        if s.scene_ai > 0:
            it.append(Item(1, "do", "If the photo is black-and-white or nearly monochrome, colorize it realistically "
                                    "with period-appropriate, natural colors.",
                           "Nếu ảnh đen trắng hoặc gần đơn sắc, tô màu chân thực, tự nhiên, hợp thời kỳ."))

    # --- khuôn mặt
    if face_ai:
        f = s.fidelity
        if f >= 0.65:
            en = "Clarify faces gently: keep facial features almost exactly as in the original, only add sharpness."
            vi = "Làm rõ mặt nhẹ nhàng: giữ ngũ quan gần như y nguyên ảnh gốc, chỉ thêm độ nét."
        elif f >= 0.4:
            en = "Clarify faces noticeably while keeping the same person."
            vi = "Làm rõ mặt rõ rệt nhưng vẫn đúng người."
        else:
            en = "Reconstruct facial detail strongly, but the face must remain the same person."
            vi = "Dựng lại chi tiết mặt mạnh tay, nhưng vẫn phải đúng người đó."
        it.append(Item(1, "face", en, vi, f" Keep ≈{_pct(f)} of the original face.", f" Giữ ≈{_pct(f)} mặt gốc."))
        if s.detail_fidelity is not None and s.detail_fidelity < f - 0.05:
            it.append(Item(2, "face", "Make eyes, eyebrows and lips crisper than the rest of the face.",
                           "Mắt, lông mày, môi sắc nét hơn phần còn lại của mặt.",
                           f" (≈{_pct(1 - s.detail_fidelity)} reconstructed)", f" (≈{_pct(1 - s.detail_fidelity)} dựng lại)"))
        if s.keep_lips:
            it.append(Item(2, "face", "Keep lip shape and lip color as in the original.",
                           "Giữ dáng môi và màu môi như ảnh gốc."))
    elif s.mode in ("quick", "manual"):
        it.append(Item(1, "face", "Do not redraw faces; only clean noise and sharpen.",
                       "Không vẽ lại khuôn mặt; chỉ khử nhiễu và làm nét."))
    elif s.mode == "upscale" and s.face_ai < 0.95:
        it.append(Item(1, "face", "Faces stay mostly original pixels; only enlarge cleanly.",
                       "Khuôn mặt giữ gần như nguyên điểm ảnh gốc; chỉ phóng to sạch.",
                       f" (≈{_pct(1 - s.face_ai)} original)", f" (≈{_pct(1 - s.face_ai)} gốc)"))

    # preset đối tượng (kiểu mặt theo tuổi, giới tính)
    if s.subject != "off":
        styles = styles_from_settings(s.face_styles)
        deshine = s.deshine > 1e-3
        keys = [s.subject] if s.subject in SUBJECT_KEYS else list(SUBJECT_KEYS)
        pre = f" (preset “{ctx.preset_name}”)" if ctx.preset_name else ""
        for k in keys:
            st = styles.get(k) or styles.get("adult") or FaceStyle()
            pe, pv = _face_style_parts(st, "en", deshine), _face_style_parts(st, "vi", deshine)
            if not pe:
                continue
            ge, gv = _GROUPS[k]
            if len(keys) == 1:
                it.append(Item(2, "face", "Face retouch: " + "; ".join(pe) + ".", "Chỉnh mặt: " + "; ".join(pv) + ".",
                               pre, pre))
            else:
                it.append(Item(2 if k == "adult" else 3, "face", f"For {ge}: " + "; ".join(pe) + ".",
                               f"Với {gv}: " + "; ".join(pv) + "."))

    # --- da (chung)
    prof = phone_profile(s.phone)
    skin = prof.skin if s.skin_tone < 0 else s.skin_tone
    if skin > 0.05 and face_ai:
        it.append(Item(2, "do", f"Correct red or yellow cast on skin ({amt(skin, 'en')}).",
                       f"Sửa da ám đỏ, ám vàng ({amt(skin, 'vi')})."))
    if s.gloss > 0.05:
        it.append(Item(3, "do", f"Add a {amt(s.gloss, 'en')} soft, healthy skin glow.",
                       f"Thêm độ bóng da khoẻ, mềm ({amt(s.gloss, 'vi')})."))
    if s.deshine > 1e-3:
        it.append(Item(1, "do", f"Reduce oily shine on forehead, nose and cheeks ({amt(s.deshine, 'en')}), keep skin "
                                "texture and facial volume.",
                       f"Khử bóng dầu ở trán, mũi, má ({amt(s.deshine, 'vi')}), giữ vân da và khối mặt."))
    if s.look:
        it.extend(_look_items(Look.from_dict(s.look), s.look_amount))

    # --- ánh sáng
    a = ctx.analysis
    if s.low_light:
        it.append(Item(1, "do", "Brighten this dark photo cleanly: lift shadows without adding noise or a grey haze.",
                       "Làm sáng ảnh tối thật sạch: nâng vùng tối, không thêm nhiễu hay lớp sương xám."))
    elif s.exposure > 0.05:
        expo = getattr(a, "expo", "") if a is not None else None
        if expo == "tối":
            it.append(Item(1, "do", "The photo is underexposed: brighten it naturally.",
                           "Ảnh thiếu sáng: làm sáng tự nhiên."))
        elif expo == "sáng":
            it.append(Item(1, "do", "The photo is overexposed: recover highlights and skin detail.",
                           "Ảnh dư sáng: lấy lại chi tiết vùng sáng và da."))
        elif expo == "ngược sáng":
            it.append(Item(1, "do", "The photo is backlit: brighten faces and shadows, keep the bright background.",
                           "Ảnh ngược sáng: làm sáng mặt và vùng tối, giữ nền sáng."))
        elif expo is None:
            it.append(Item(3, "do", "Fix exposure if the photo is too dark, too bright or backlit.",
                           "Sửa phơi sáng nếu ảnh quá tối, quá sáng hoặc ngược sáng."))
    if abs(s.brightness) > 0.05:
        up = s.brightness > 0
        it.append(Item(2, "do", f"{'Brighten' if up else 'Darken'} the whole photo ({amt(s.brightness, 'en')}).",
                       f"{'Làm sáng' if up else 'Làm tối'} toàn ảnh ({amt(s.brightness, 'vi')})."))
    if s.contrast > 0.05:
        it.append(Item(2, "do", f"Increase local contrast ({amt(s.contrast, 'en')}) without halos.",
                       f"Tăng tương phản cục bộ ({amt(s.contrast, 'vi')}), không tạo quầng sáng."))
    if s.tone > 0.05:
        it.append(Item(2, "do", f"Fresh, bright, airy tone ({amt(s.tone, 'en')}): lift midtones, clear the haze in "
                                "shadows, slightly more vivid colors.",
                       f"Tông tươi sáng, trong trẻo ({amt(s.tone, 'vi')}): nâng tông giữa, bỏ lớp sương vùng tối, màu "
                       "rực hơn một chút."))

    # --- màu
    wb = prof.wb if s.wb < 0 else s.wb
    cast = getattr(a, "cast_name", "") if a is not None else ""
    if wb > 0.05:
        ce = "/".join(_CAST.get(c, c) for c in cast.split("/")) if cast and cast != "nhẹ" else ""
        it.append(Item(2, "do", f"Neutralize the {ce + ' ' if ce else ''}color cast ({amt(wb, 'en')}), keep warm light "
                                "that is intentional.",
                       f"Gỡ ám màu{' ' + cast if ce else ''} ({amt(wb, 'vi')}), giữ ánh sáng ấm có chủ ý."))
    if s.color > 0.05:
        it.append(Item(2, "do", f"Richer color and depth ({amt(s.color, 'en')}): cleaner white balance, more vibrance, "
                                "more dimensional light.",
                       f"Màu đậm và có chiều sâu hơn ({amt(s.color, 'vi')}): cân bằng trắng sạch hơn, màu rực hơn, "
                       "rõ khối hơn."))

    # --- độ nét, nhiễu
    if s.sharpness > 0.05:
        it.append(Item(2, "do", f"Sharpen details ({amt(s.sharpness, 'en')}) without halos or crunchy edges.",
                       f"Làm nét chi tiết ({amt(s.sharpness, 'vi')}), không tạo viền sáng hay răng cưa."))
    if s.denoise < 0:
        it.append(Item(2, "do", "Remove noise and JPEG compression artifacts as needed.",
                       "Khử nhiễu và vết nén JPEG khi cần."))
    elif s.denoise > 0.05:
        it.append(Item(2, "do", f"Reduce noise ({amt(s.denoise / 10, 'en')}).",
                       f"Khử nhiễu ({amt(s.denoise / 10, 'vi')})."))
    else:
        it.append(Item(3, "do", "Keep the natural film/sensor grain.", "Giữ hạt ảnh tự nhiên."))

    # --- đổi nền
    if s.bg_color is not None:
        en, vi, hexv = _bg_name(s.bg_color)
        it.append(Item(0, "do", f"Replace the plain background with a flat, even {en} color ({hexv}); clean edges "
                                "around hair and shoulders; do not change the person.",
                       f"Thay nền trơn bằng màu {vi} phẳng, đều ({hexv}); viền tóc và vai sạch; không đổi người."))

    # --- ảnh ra
    w, h = ctx.size if ctx.size else (0, 0)
    if s.mode in ("quick", "manual") and s.upscale in (0, 1) or s.upscale == 1:
        it.append(Item(2, "output", "Keep the original resolution.", "Giữ nguyên độ phân giải."))
    else:
        k = s.upscale or (auto_upscale(w, h) if w else 0)
        if s.upscale == 0 and lv in (5, 6):
            k = max(k, 2)
        if k:
            px_en = f" (about {w * k}×{h * k} px)" if w else ""
            px_vi = f" (khoảng {w * k}×{h * k} px)" if w else ""
            it.append(Item(1, "output", f"Output at {k}× the original resolution with real fine detail.",
                           f"Ảnh ra lớn gấp {k} lần ảnh gốc, có chi tiết thật.", px_en, px_vi))
        else:
            it.append(Item(1, "output", "Output at 2–4× the original resolution with real fine detail.",
                           "Ảnh ra lớn gấp 2–4 lần ảnh gốc, có chi tiết thật."))
    it.append(Item(1, "output", "Photorealistic result that looks like a real photograph, not a painting or CGI.",
                   "Kết quả chân thực như ảnh chụp, không giống tranh vẽ hay ảnh dựng 3D."))

    # --- điều cấm
    it.append(Item(0, "avoid", "changing the face or identity, beautifying into a different person, changing age, "
                               "skin color or ethnicity",
                   "đổi mặt hay danh tính, làm đẹp thành người khác, đổi tuổi, màu da hay sắc tộc"))
    it.append(Item(1, "avoid", "adding makeup, changing hairstyle, clothing or expression, adding or removing people "
                               "or objects, text, watermarks or borders",
                   "thêm trang điểm, đổi kiểu tóc, quần áo hay nét mặt, thêm hoặc bớt người, đồ vật, chữ, hình mờ, "
                   "khung viền"))
    it.append(Item(1, "avoid", "plastic or waxy skin, over-smoothing, HDR halos, cropping",
                   "da nhựa, da sáp, làm mịn quá tay, quầng HDR, cắt ảnh"))
    if (restore or s.mode == "old") and s.scene_ai <= 0:
        it.append(Item(1, "avoid", "colorizing a black-and-white photo", "tô màu ảnh đen trắng"))

    # --- chẩn đoán ảnh
    if ctx.diagnosis and (a is not None or ctx.size):
        e, v = [], []
        if ctx.size:
            e.append(f"{w}×{h} px")
            v.append(f"{w}×{h} px")
        if a is not None:
            bl = {"mờ nặng": "very blurry", "hơi mờ": "slightly blurry", "nét": "sharp"}.get(a.blur_label, a.blur_label)
            d = getattr(a, "blur_d", 1.0)
            e.append(bl + (f" (~{d:g}× blur)" if d > 1 else ""))
            v.append(a.blur_label + (f" (~{d:g}×)" if d > 1 else ""))
            e.append({"nhiều nhiễu": "noisy", "ít nhiễu": "low noise"}.get(a.noise_label, a.noise_label))
            v.append(a.noise_label)
            ll = a.light_label
            if ll != "bình thường":
                e.append({"tối": "dark", "quá sáng": "overexposed", "sáng gắt": "harsh light",
                          "ngược sáng": "backlit"}.get(ll, ll))
                v.append(ll)
            if cast and cast != "nhẹ":
                ce = "/".join(_CAST.get(c, c) for c in cast.split("/"))
                e.append(f"{ce} cast")
                v.append(f"ám {cast}")
            if faces is not None:
                e.append(f"{faces} face{'s' if faces != 1 else ''}")
                v.append(f"{faces} khuôn mặt")
        it.append(Item(3, "diag", "Photo diagnosis: " + ", ".join(e) + ".", "Chẩn đoán ảnh: " + ", ".join(v) + "."))
    return it


# ---------------------------------------------------------------- định dạng theo hệ đích

_HEAD = {
    "en": {"keep": "MUST KEEP", "do": "IMPROVE", "face": "FACES", "output": "OUTPUT", "avoid": "AVOID",
           "diag": "CONTEXT"},
    "vi": {"keep": "BẮT BUỘC GIỮ", "do": "CẢI THIỆN", "face": "KHUÔN MẶT", "output": "ẢNH RA", "avoid": "TRÁNH",
           "diag": "BỐI CẢNH"},
}
_ORDER = ("keep", "face", "do", "output", "avoid", "diag")


def _render(items: list[Item], t: Target, lang: str) -> str:
    vi = lang == "vi"
    txt = lambda i: i.text(lang, t.numbers)  # noqa: E731
    task = [txt(i) for i in items if i.group == "task"]
    by = {g: [txt(i) for i in items if i.group == g] for g in _ORDER}
    avoid = by.pop("avoid")
    if t.style == "sections":
        out = task[:]
        for g in _ORDER:
            lines = avoid if g == "avoid" else by.get(g, [])
            if lines:
                out += ["", _HEAD[lang][g] + ":"] + [f"- {x[:1].upper() + x[1:]}" + ("." if g == "avoid" else "")
                                                       for x in lines]
        return "\n".join(out).strip()
    if t.style == "narrative":
        paras = ["Dùng ảnh đã tải lên, chỉnh sửa chính ảnh này thay vì tạo ảnh mới." if vi
                 else "Using the provided photo, retouch this exact image instead of generating a new one."]
        paras.append(" ".join(by["keep"]))
        body = by["face"] + by["do"]
        if body:
            paras.append(" ".join(body))
        if by["output"]:
            paras.append(" ".join(by["output"]))
        if avoid:
            paras.append(("Tránh: " if vi else "Avoid ") + "; ".join(avoid) + ".")
        if by["diag"]:
            paras.append(" ".join(by["diag"]))
        return "\n\n".join(p for p in paras if p.strip())
    # compact: một đoạn
    parts = task + by["keep"] + by["face"] + by["do"] + by["output"]
    s = " ".join(parts)
    if avoid:
        s += (" Không: " if vi else " No ") + "; ".join(avoid) + "."
    if by["diag"]:
        s += " " + " ".join(by["diag"])
    return s.strip()


def build(s: Settings, target: str = "chatgpt", lang: str = "en", ctx: Context | None = None) -> PromptPack:
    t = TARGETS[target]
    lang = lang if lang in LANGS else "en"
    items = collect(s, ctx)
    dropped: list[str] = []
    text = _render(items, t, lang)
    while len(text) > t.limit:
        cand = [i for i in items if i.prio > 0]
        if not cand:
            break
        worst = max(range(len(items)), key=lambda j: (items[j].prio, j) if items[j].prio > 0 else (-1, j))
        dropped.append(items[worst].text(lang, t.numbers))
        del items[worst]
        text = _render(items, t, lang)
    return PromptPack(target, lang, text, t.limit, dropped, t.note)


def settings_for_export(s: Settings) -> dict:
    """Settings dạng dict cho file xuất: bỏ đường dẫn mặt nạ (có thể chứa tên người dùng Windows)."""
    d = s.to_dict()
    d["damage_mask"] = bool(d.get("damage_mask"))
    if d.get("bg_color") is not None:
        d["bg_color"] = list(d["bg_color"])
    return d


def export_json(s: Settings, ctx: Context | None = None, app_version: str = "") -> str:
    """Mọi biến thể (hệ × ngôn ngữ) kèm thông số, để nhà phát triển so sánh hoặc nạp vào công cụ khác."""
    ctx = ctx or Context()
    prompts = {}
    for tk in TARGETS:
        prompts[tk] = {}
        for lg in LANGS:
            p = build(s, tk, lg, ctx)
            prompts[tk][lg] = {"text": p.text, "chars": p.chars, "limit": p.limit, "dropped": p.dropped}
    doc = {"format": "beautyhdr-prompt", "version": 1, "app": app_version, "created": time.strftime("%Y-%m-%d %H:%M"),
           "level": ctx.level, "preset_set": ctx.preset_name, "size": list(ctx.size) if ctx.size else None,
           "settings": settings_for_export(s), "prompts": prompts, "privacy": PRIVACY_VI}
    return json.dumps(doc, ensure_ascii=False, indent=2)


def export_text(p: PromptPack) -> str:
    """File .txt: prompt ở trên, ghi chú cách dùng và ý bị bỏ ở dưới dòng kẻ (không dán phần dưới)."""
    t = TARGETS[p.target]
    tail = [f"--- Beauty HDR · {t.name} · {LANGS[p.lang]} · {p.chars}/{p.limit} ký tự · chỉ dán phần trên dòng này",
            "Cách dùng: " + p.note, "Riêng tư: " + PRIVACY_VI]
    if p.dropped:
        tail.append("Ý bị bỏ vì quá dài: " + " | ".join(p.dropped))
    return p.text + "\n\n" + "\n".join(tail) + "\n"


__all__ = ["TARGETS", "LANGS", "PRIVACY_VI", "Context", "Item", "PromptPack", "Target", "build", "collect",
           "export_json", "export_text", "settings_for_export"]
