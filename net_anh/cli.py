"""Dòng lệnh: xử lý một ảnh hoặc cả thư mục. Không bao giờ ghi đè ảnh gốc.

Ví dụ:
    python -m net_anh.cli anh.jpg                      # tự phân tích và chọn chế độ
    python -m net_anh.cli anh.jpg --mode portrait --fidelity 0.6
    python -m net_anh.cli thu_muc -o ket_qua --mode upscale --upscale 4
    python -m net_anh.cli thu_muc --level 2                    # thang mức nét 1–6, chế độ theo nội dung từng ảnh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import recommend
from .levels import BG_MODELS, FACE_MODELS, LEVELS, level_settings
from . import backdrop, facestyle
from .pipeline import OUT_SIZES, SUPPORTED, BatchTally, Enhancer, ImageError, load_image, resize_long, save_image, unique_output_path
from .settings import MODES, PRESETS, Settings, apply_preset

EXTS = SUPPORTED


def _preset_styles(a) -> dict | None:
    """Kiểu mặt của bộ preset chọn bằng --preset-file / --preset-set (None = bộ mặc định)."""
    if a.preset_file:
        sets, warn = facestyle.load_file(a.preset_file)
        for w in warn:
            print(f"  Bộ preset: {w}", file=sys.stderr)
    else:
        sets = facestyle.Library().sets()
        if not a.preset_set:
            sets = [facestyle.Library().active()]
    if a.preset_set:
        sets = [st for st in sets if a.preset_set in (st.id, st.name)]
        if not sets:
            raise facestyle.PresetError(f"không có bộ preset “{a.preset_set}”")
    a._styles = facestyle.styles_to_settings(sets[0])
    # tông mẫu của bộ: chỉ dùng khi chọn bộ rõ ràng (--preset-file / --preset-set); --look ghi đè
    a._look = sets[0].look if (a.preset_file or a.preset_set) else None
    print(f"Bộ preset: {sets[0].name}", file=sys.stderr)
    return a._styles


def build_settings(a, eng: Enhancer | None, img) -> tuple[Settings, str]:
    s = Settings(upscale=a.upscale, max_faces=a.max_faces, tile=a.tile)
    for k in ("face_model", "sr_model"):
        if getattr(a, k):
            s = s.with_(**{k: getattr(a, k)})
    note = ""
    if a.level:
        an = eng.analyze(img, with_faces=eng.models_ready() or a.download)
        _, auto, why = recommend(an, s)
        s = level_settings(a.level, auto.with_(upscale=a.upscale, face_model=s.face_model, sr_model=s.sr_model), an)
        note = f"Mức nét {a.level} · {LEVELS[a.level].name}: {MODES[s.mode]} ({'; '.join(why)})"
    elif a.preset == "auto" and a.mode == "auto":
        an = eng.analyze(img, with_faces=eng.models_ready() or a.download)
        preset, s, why = recommend(an, s)
        note = f"Tự động: {PRESETS[preset][0]} / {MODES[s.mode]} ({'; '.join(why)})"
    elif a.preset not in ("auto", "custom"):
        s = apply_preset(s, a.preset)
    if a.mode != "auto":
        s = s.with_(mode=a.mode)
    for k in ("strength", "sharpness", "denoise", "contrast", "brightness", "fidelity", "color", "gloss", "face_model",
              "wb", "skin_tone", "exposure", "tone", "deshine"):
        v = getattr(a, k)
        if v is not None:
            s = s.with_(**{k: v})
    if getattr(a, "mask", None):
        s = s.with_(damage_mask=str(Path(a.mask).resolve()))
        if not (s.repair > 0 or s.fade > 0):
            s = s.with_(repair=1.0, fade=1.0, auto_rotate=True)
    s = s.with_(fade_style=getattr(a, "fade_style", "giu_da"))
    if getattr(a, "small_face_sr", False):
        s = s.with_(small_face_sr=True)
    if a.low_light:
        s = s.with_(low_light=True)
    if getattr(a, "ai_lips", False):
        s = s.with_(keep_lips=False)
    s = s.with_(subject=getattr(a, "subject", "auto"), face_styles=getattr(a, "_styles", None),
                bg_color=getattr(a, "bg_color", None) or None)
    lk = getattr(a, "_look", None)
    if getattr(a, "look", None):
        from .look import LOOKS
        lk = None if a.look == "off" else LOOKS[a.look].to_dict()
    s = s.with_(look=lk)
    if getattr(a, "look_amount", None) is not None:
        s = s.with_(look_amount=float(min(max(a.look_amount, 0.0), 1.0)))
    return s, note


def hw_warnings(s: Settings, img, eng: Enhancer) -> list[str]:
    """Cảnh báo model nặng / máy yếu cho ảnh đầu tiên (hardware.warnings)."""
    import torch

    from . import hardware
    from .levels import auto_upscale
    h, w = img.shape[:2]
    faces = 0 if s.mode == "upscale" else 1
    return hardware.warnings(s.sr_model, s.face_model, w, h, faces, 1.0, eng.dev.kind, torch.get_num_threads(),
                             upscale=s.upscale or auto_upscale(w, h))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="net_anh", description="Làm nét ảnh bằng GPU/CPU của máy")
    p.add_argument("input", help="file ảnh hoặc thư mục")
    p.add_argument("-o", "--output", help="file hoặc thư mục ra (mặc định: cạnh ảnh gốc, thêm _net)")
    p.add_argument("--mode", choices=["auto", *MODES], default="auto")
    p.add_argument("--preset", choices=list(PRESETS), default="auto")
    p.add_argument("--mask", metavar="PNG", help="mặt nạ vùng hỏng nặng (trắng = hỏng): vá rồi dựng lại mặt hỏng "
                                                 "(dùng với --preset restore)")
    p.add_argument("--upscale", type=int, choices=[0, 1, 2, 4], default=0,
                   help="hệ số phóng; 0 = tự động (cạnh dài dưới 1000 px phóng 4×, còn lại 2×)")
    p.add_argument("--strength", type=float, help="0..1")
    p.add_argument("--sharpness", type=float, help="0..1")
    p.add_argument("--denoise", type=float, help="-1 = tự động, 0..10")
    p.add_argument("--contrast", type=float, help="0..1 (CLAHE)")
    p.add_argument("--brightness", type=float, help="-1..1")
    p.add_argument("--fidelity", type=float, help="0..1, cao = giữ mặt gốc nhiều hơn")
    from .phone import PROFILES
    p.add_argument("--phone", choices=["auto", *PROFILES], default="auto",
                   help="bộ lọc màu theo hãng máy; auto = đọc EXIF (Make/Model) của từng ảnh")
    p.add_argument("--wb", type=float, help="cân bằng trắng 0..1 (mặc định theo hãng)")
    p.add_argument("--skin-tone", type=float, help="sửa da đỏ/vàng 0..1 (mặc định theo hãng)")
    p.add_argument("--color", type=float, help="0..1 tăng màu (cân bằng trắng, độ rực, rõ khối)")
    p.add_argument("--gloss", type=float, help="0..1 độ bóng da")
    p.add_argument("--deshine", type=float, help="0..1 khử bóng loáng da (da dầu bóng ở trán, mũi, má); mặc định tắt")
    p.add_argument("--level", type=int, choices=list(LEVELS),
                   help="mức nét cần: 1 Nhẹ, 2 Vừa … 5 Tối đa (kiểu Remini), 6 Cao nhất (model mới nhất, nặng nhất: "
                        "RestoreFormer++, DAT-2); chế độ theo nội dung từng ảnh")
    p.add_argument("--face-model", choices=list(FACE_MODELS),
                   help="model mặt: gfpgan (mặc định, nhanh), restoreformer_pp (RestoreFormer++, mới nhất, giữ đúng "
                        "người nhất), restoreformer")
    p.add_argument("--bg-model", dest="sr_model", choices=list(BG_MODELS),
                   help="model làm nét nền: compact (nhẹ, nhanh, mặc định), rrdb (Real-ESRGAN x2plus/x4plus), plksr "
                        "(RealPLKSR, nặng vừa), dat2 (DAT-2, chất lượng cao nhất, rất nặng: nên có GPU)")
    p.add_argument("--exposure", type=float,
                   help="tự sửa sáng 0..1 cho ảnh quá tối, quá sáng, ngược sáng (mặc định 1, 0 = tắt)")
    p.add_argument("--low-light", action="store_true", help="bật Retinex cho ảnh tối (thay cho --exposure)")
    p.add_argument("--tone", type=float,
                   help="tông tươi sáng, trong trẻo 0..1 (mặc định 0.6, 0 = giữ tông ảnh gốc)")
    p.add_argument("--max-faces", type=int, default=24)
    p.add_argument("--fade-style", choices=["0.10", "giu_da"], default="giu_da",
                   help="phục hồi màu phai: giu_da (mặc định, giữ màu da và ánh sáng ấm có chủ ý) hoặc 0.10 (bản cũ)")
    p.add_argument("--small-face-sr", action="store_true",
                   help="mặt nhỏ trong ảnh nhóm: phóng AI vùng mặt trước khi dựng (thử nghiệm, ít bịa hơn nhưng ít nét)")
    p.add_argument("--tile", type=int, default=400, help="kích thước ô Real-ESRGAN, 0 = không chia")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg")
    p.add_argument("--quality", type=int, default=94)
    p.add_argument("--ai-lips", action="store_true", help="để model mặt vẽ lại môi (mặc định giữ dáng, màu môi gốc)")
    p.add_argument("--size", choices=["goc", "1k", "2k", "4k"], default="goc",
                   help="cỡ xuất theo cạnh dài: 1k = 1920, 2k = 2560, 4k = 3840 px (giữ tỉ lệ); goc = giữ cỡ kết quả")
    p.add_argument("--download", action="store_true", help="cho phép tải model AI khi phân tích tự động")
    p.add_argument("--subject", choices=["auto", "child", "man", "woman", "elder", "adult", "off"], default="auto",
                   help="preset theo đối tượng: auto = đoán tuổi, giới tính từng mặt; hoặc ép một nhóm; off = tắt")
    p.add_argument("--bg-color", help="đổi màu nền đồng màu (ảnh thẻ, phông trơn): #RRGGBB hoặc trang, xanh, xanh-nhat, do, xam")
    p.add_argument("--preset-file", help="bộ preset (.netanh.json) dùng cho lần chạy này (mặc định: bộ đang dùng)")
    p.add_argument("--look", choices=["trong_sang", "da_dep_the", "off"],
                   help="tông mẫu: trong_sang = da sáng trong, ít vàng đỏ, môi tươi, tóc đen, nền phẳng (ảnh thẻ); "
                        "off = tắt. Mặc định: theo bộ preset chọn bằng --preset-file/--preset-set, không có thì tắt")
    p.add_argument("--look-amount", type=float, help="0..1 độ mạnh tông mẫu (mặc định 1)")
    p.add_argument("--preset-set", help="tên hoặc mã bộ preset trong file / thư viện (mặc định bộ đầu / bộ đang dùng)")
    a = p.parse_args(argv)
    if a.bg_color:
        try:
            a.bg_color = backdrop.parse_color(a.bg_color)
        except ValueError as e:
            print(f"--bg-color: {e}", file=sys.stderr)
            return 2
    try:
        styles = _preset_styles(a)
    except (OSError, UnicodeDecodeError, facestyle.PresetError) as e:
        print(f"Bộ preset lỗi: {e}", file=sys.stderr)
        return 2

    src = Path(a.input)
    ext = "." + a.format
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in EXTS)
        out_dir = Path(a.output) if a.output else src.with_name(src.name + "_net")
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(f, unique_output_path(f, out_dir, ext)) for f in files]
    else:
        out = Path(a.output) if a.output else unique_output_path(src, None, ext)
        # "-o thư_mục/" (có dấu / cuối, thư mục đã có, hoặc không có đuôi file): ghi vào trong thư mục đó
        if a.output and (a.output.endswith(("/", "\\")) or out.is_dir() or not out.suffix):
            out.mkdir(parents=True, exist_ok=True)
            out = unique_output_path(src, out, ext)
        if out.exists() and out.resolve() == src.resolve():
            print("Không ghi đè ảnh gốc: chọn tên file ra khác.", file=sys.stderr)
            return 2
        jobs = [(src, out)]

    eng = Enhancer(a.device, progress=lambda m: print("  " + m, file=sys.stderr))
    print(f"Thiết bị: {eng.dev.name}", file=sys.stderr)
    failed = 0
    tally = BatchTally(len(jobs))
    for i, (f, out) in enumerate(jobs, 1):
        try:
            img = load_image(f)
            s, note = build_settings(a, eng, img)
            if note:
                print("  " + note, file=sys.stderr)
            if a.phone == "auto":
                from .phone import PROFILES, detect
                key, desc = detect(f)
                s = s.with_(phone=key)
                print(f"  Máy chụp: {desc} → bộ lọc {PROFILES[key].label}", file=sys.stderr)
            else:
                s = s.with_(phone=a.phone)
            if i == 1 and s.mode not in ("quick", "manual"):  # cảnh báo theo cấu hình máy, không chặn
                for msg in hw_warnings(s, img, eng):
                    print("  Cảnh báo: " + msg, file=sys.stderr)
            res, rep = eng.process(img, s)
            save_image(resize_long(res, OUT_SIZES[a.size]), out, a.quality)
            tally.add(rep, f.name)
            print(f"[{i}/{len(jobs)}] {f.name} → {out}  ({rep.summary()})")
        except ImageError as e:
            failed += 1
            tally.fail()
            print(f"[{i}/{len(jobs)}] {f.name}: {e}", file=sys.stderr)
        except Exception as e:  # một ảnh lỗi không dừng cả lô
            failed += 1
            tally.fail()
            print(f"[{i}/{len(jobs)}] {f.name}: LỖI {e}", file=sys.stderr)
    if len(jobs) > 1 or tally.needs_attention:
        print("Tổng kết: " + tally.summary(), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
