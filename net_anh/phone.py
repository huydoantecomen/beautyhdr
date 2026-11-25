"""Bộ lọc theo hãng điện thoại: nhận hãng từ EXIF (Make/Model), mỗi hãng một bộ thông số sửa màu.

Mỗi hãng xử lý ảnh (ISP) theo "gu" riêng, ví dụ iPhone giữ ánh vàng ấm của đèn trong nhà, Oppo/Vivo/Xiaomi làm
da hồng đỏ khi bật làm đẹp, Samsung màu rực và làm nét mạnh. Thông số dưới đây là điểm xuất phát theo đặc điểm
chung của từng hãng (chưa hiệu chỉnh trên ảnh thật của từng đời máy); chỉnh lại khi có ảnh mẫu.

  wb      phần ám màu được gỡ (0 = giữ nguyên, 1 = gỡ hết về trung tính)
  skin    mức kéo màu da đỏ/vàng về tông da tự nhiên (chỉ trong vùng mặt dò được)
  sat     chỉnh độ bão hoà toàn ảnh (−0,1 = giảm 10 %), cho máy vốn đã rực màu
  sharpen hệ số nhân độ nét (máy đã làm nét mạnh thì giảm để tránh viền trắng)

Ảnh gửi qua Zalo, Messenger, Facebook thường bị xoá EXIF → không nhận được hãng → dùng bộ "Khác".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhoneProfile:
    key: str
    label: str
    wb: float = 0.6
    skin: float = 0.5
    sat: float = 0.0
    sharpen: float = 1.0


PROFILES: dict[str, PhoneProfile] = {p.key: p for p in [
    PhoneProfile("generic", "Khác / không rõ", wb=0.6, skin=0.5),
    PhoneProfile("apple", "iPhone", wb=0.7, skin=0.5, sharpen=0.9),
    PhoneProfile("samsung", "Samsung", wb=0.5, skin=0.4, sat=-0.08, sharpen=0.75),
    PhoneProfile("xiaomi", "Xiaomi / Redmi / POCO", wb=0.7, skin=0.6, sat=-0.05),
    PhoneProfile("oppo", "Oppo / realme / OnePlus", wb=0.7, skin=0.8, sat=-0.05, sharpen=1.1),
    PhoneProfile("vivo", "Vivo / iQOO", wb=0.7, skin=0.8, sat=-0.05, sharpen=1.1),
    PhoneProfile("huawei", "Huawei / Honor", wb=0.6, skin=0.5),
    PhoneProfile("google", "Google Pixel", wb=0.4, skin=0.3, sharpen=0.9),
    PhoneProfile("camera", "Máy ảnh (Canon, Nikon, Sony, Fujifilm…)", wb=0.4, skin=0.3),
    PhoneProfile("off", "Tắt sửa màu", wb=0.0, skin=0.0),
]}

# chuỗi con trong EXIF Make (viết thường) → hãng
_MAKES = [
    ("apple", "apple"), ("samsung", "samsung"),
    ("xiaomi", "xiaomi"), ("redmi", "xiaomi"), ("poco", "xiaomi"),
    ("oppo", "oppo"), ("realme", "oppo"), ("oneplus", "oppo"),
    ("vivo", "vivo"), ("iqoo", "vivo"),
    ("huawei", "huawei"), ("honor", "huawei"),
    ("google", "google"),
    ("canon", "camera"), ("nikon", "camera"), ("fujifilm", "camera"), ("olympus", "camera"), ("om digital", "camera"),
    ("panasonic", "camera"), ("leica", "camera"), ("pentax", "camera"), ("ricoh", "camera"),
]
_SONY_PHONE = ("xperia", "xq-", "so-", "sov", "g8", "j9", "h8")


def brand_of(make: str, model: str = "") -> str:
    m = (make or "").strip().lower()
    mo = (model or "").strip().lower()
    if not m and mo.startswith("iphone"):
        return "apple"
    if m == "sony":
        return "generic" if mo.startswith(_SONY_PHONE) else "camera"
    for sub, key in _MAKES:
        if sub in m:
            return key
    # một số máy ghi hãng con trong Model (vd. Make "Xiaomi", Model "Redmi Note 12"; hoặc Make rỗng)
    for sub, key in _MAKES:
        if sub in mo:
            return key
    return "generic"


def read_make_model(path: str | Path) -> tuple[str, str]:
    """Đọc EXIF Make/Model bằng Pillow; không có thì trả chuỗi rỗng."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            return str(ex.get(0x010F, "") or "").strip("\x00 "), str(ex.get(0x0110, "") or "").strip("\x00 ")
    except Exception:
        return "", ""


def detect(path: str | Path) -> tuple[str, str]:
    """→ (khoá hãng, mô tả máy) từ EXIF của file ảnh."""
    make, model = read_make_model(path)
    if not make and not model:
        return "generic", "không có EXIF (ảnh qua Zalo/Facebook hoặc ảnh chụp màn hình)"
    desc = f"{make} {model}".strip()
    if model.lower().startswith(make.lower()) and make:
        desc = model
    return brand_of(make, model), desc


def profile(key: str) -> PhoneProfile:
    return PROFILES.get(key, PROFILES["generic"])
