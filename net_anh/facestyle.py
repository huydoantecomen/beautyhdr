"""Preset theo đối tượng (trẻ em, nam, nữ, người lớn tuổi): kiểu chỉnh riêng cho từng khuôn mặt, bộ preset, file chia sẻ.

Một **bộ preset** gồm một kiểu mặt (FaceStyle) cho mỗi đối tượng. Khi xử lý, mỗi khuôn mặt được đoán tuổi và giới tính
(subject.py) rồi chỉnh theo kiểu của nhóm mình: ảnh gia đình thì bố, mẹ, con mỗi người một kiểu. Người dùng cũng có thể
ép một đối tượng cho mọi mặt, hoặc tắt.

Kiểu mặt chỉ đổi vùng mặt (trên mặt đã căn 512×512, trước khi ghép lại), cộng thêm vào mức nét chứ không thay mức nét:
  keep    0..1   giữ nét thật: thêm phần mặt gốc vào phần AI của mức nét (giữ đúng người, giữ tuổi, giữ nếp nhăn)
  detail  −1..1  ngũ quan (mắt, lông mày, môi): + thêm AI cho sắc nét, − bớt AI
  smooth  0..1   mịn da: làm mịn tầng thấp và vết sạm, vẫn giữ một phần lỗ chân lông
  glow    0..1   bóng da: ánh sáng loang nhẹ ở vùng da sáng
  bright  0..1   sáng da
  warm    −1..1  hồng hào (+) hoặc bớt đỏ, mát hơn (−)
  matte   −1..1  khử bóng da: cộng vào thanh Khử bóng chung (vd nam da dầu +0,2); chỉ có tác dụng khi ô Khử bóng da
                 được tích. Có từ định dạng 2; file định dạng 1 đọc vẫn được (matte = 0).
Vùng da lấy từ ParseNet (da, mũi, tai), không đụng mắt, môi, tóc; chỉnh nhạt dần ra mép mặt để không lộ đường ghép.

File chia sẻ: JSON UTF-8, đuôi .netanh.json, dùng chung cho bản PC và Android. Chỉ có số và chữ, không có mã chạy được;
khi nạp, mọi giá trị được kiểm tra, ngoài khoảng thì kẹp lại, khoá lạ thì bỏ qua và báo lại.
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import cv2
import numpy as np

from .look import LOOKS, Look

FORMAT = "net-anh-preset-set"
VERSION = 3  # 2: thêm nút matte (khử bóng da); 3: bộ có thể kèm tông mẫu "look" (look.py). Đọc được file 1, 2
EXT = ".netanh.json"
MAX_FILE_BYTES = 512 * 1024
MAX_SETS = 50
MAX_NAME = 60

SUBJECT_KEYS = ("child", "man", "woman", "elder", "adult")
# khoảng hợp lệ của từng nút chỉnh
RANGES = {"keep": (0.0, 1.0), "detail": (-1.0, 1.0), "smooth": (0.0, 1.0), "glow": (0.0, 1.0), "bright": (0.0, 1.0),
          "warm": (-1.0, 1.0), "matte": (-1.0, 1.0)}
KNOB_NAMES = {"keep": "Giữ nét thật", "detail": "Ngũ quan", "smooth": "Mịn da", "glow": "Bóng da", "bright": "Sáng da",
              "warm": "Hồng hào", "matte": "Khử bóng"}


@dataclass(frozen=True)
class FaceStyle:
    keep: float = 0.0
    detail: float = 0.0
    smooth: float = 0.0
    glow: float = 0.0
    bright: float = 0.0
    warm: float = 0.0
    matte: float = 0.0

    def is_neutral(self) -> bool:
        return all(abs(getattr(self, k)) < 1e-3 for k in RANGES)

    def clamped(self) -> "FaceStyle":
        return FaceStyle(**{k: float(np.clip(getattr(self, k), *RANGES[k])) for k in RANGES})

    def to_dict(self) -> dict:
        return {k: round(float(getattr(self, k)), 3) for k in RANGES}

    def weights(self, w_ai: float, w_det: float) -> tuple[float, float]:
        """Phần AI cho da và cho ngũ quan sau khi áp kiểu này lên phần AI của mức nét."""
        a = w_ai * (1.0 - self.keep)
        d = w_det * (1.0 - self.keep)
        d = d + self.detail * (1.0 - d) if self.detail >= 0 else d * (1.0 + self.detail)
        return float(np.clip(a, 0, 1)), float(np.clip(d, 0, 1))


@dataclass
class PresetSet:
    name: str
    styles: dict[str, FaceStyle]
    author: str = ""
    note: str = ""
    id: str = ""
    created: str = ""
    builtin: bool = False
    look: dict | None = None  # tông mẫu kèm bộ (thông số look.Look, đã kẹp); None = bộ không có tông mẫu

    def style(self, subject: str) -> FaceStyle:
        return self.styles.get(subject) or self.styles.get("adult") or FaceStyle()

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "author": self.author, "note": self.note, "created": self.created,
                "styles": {k: self.styles[k].to_dict() for k in SUBJECT_KEYS if k in self.styles},
                **({"look": dict(self.look)} if self.look else {})}

    def copy(self, name: str | None = None) -> "PresetSet":
        return PresetSet(name or self.name, dict(self.styles), self.author, self.note, new_id(), today(), False,
                         dict(self.look) if self.look else None)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def today() -> str:
    return time.strftime("%Y-%m-%d")


# Bộ mặc định: chỉnh bằng tools/preset_bench.py (số đo trong README, phần "Preset theo đối tượng").
DEFAULT_SET = PresetSet(
    "Mặc định (Nét Ảnh)",
    {
        "child": FaceStyle(keep=0.3, detail=0.0, smooth=0.35, glow=0.1, bright=0.2, warm=0.2, matte=-0.2),
        "man": FaceStyle(keep=0.0, detail=0.2, smooth=0.1, glow=0.0, bright=0.1, warm=0.0, matte=0.1),
        "woman": FaceStyle(keep=0.0, detail=0.3, smooth=0.45, glow=0.3, bright=0.25, warm=0.2),
        "elder": FaceStyle(keep=0.35, detail=0.3, smooth=0.0, glow=0.0, bright=0.15, warm=0.1),
        "adult": FaceStyle(keep=0.0, detail=0.2, smooth=0.25, glow=0.1, bright=0.15, warm=0.1),
    },
    author="Nét Ảnh", note="Bộ preset có sẵn, đo trên ảnh mặt theo nhóm tuổi và giới tính.", id="default",
    created="2026-11-25", builtin=True)

# Bộ "Trong sáng": kiểu mặt như bộ mặc định, kèm tông mẫu trong sáng (look.py, đo từ 4 ảnh thẻ mẫu huy gửi 2026-11-25).
# Nữ, trẻ em mịn và sáng da thêm một chút như ảnh mẫu; khử bóng cộng thêm (chỉ khi ô Khử bóng da được tích).
TRONG_SANG_SET = PresetSet(
    "Trong sáng (ảnh thẻ)",
    {**DEFAULT_SET.styles,
     "woman": FaceStyle(keep=0.0, detail=0.3, smooth=0.55, glow=0.3, bright=0.3, warm=0.0, matte=0.2),
     "child": FaceStyle(keep=0.3, detail=0.0, smooth=0.4, glow=0.1, bright=0.25, warm=0.1, matte=-0.1),
     "adult": FaceStyle(keep=0.0, detail=0.2, smooth=0.3, glow=0.1, bright=0.2, warm=0.0, matte=0.1)},
    author="Beauty HDR", note="Da sáng, trong, ít vàng đỏ; môi tươi; tóc đen; nền phẳng sạch. Kèm tông mẫu Trong sáng.",
    id="trong-sang", created="2026-11-25", builtin=True, look=Look().to_dict())

# Bộ "Da đẹp ảnh thẻ": như Trong sáng, kèm tông mẫu Da đẹp ảnh thẻ (làm sạch mụn, đốm, vết đỏ; đều màu) và mịn da hơn.
DA_DEP_THE_SET = PresetSet(
    "Da đẹp ảnh thẻ",
    {**TRONG_SANG_SET.styles,
     "woman": FaceStyle(keep=0.0, detail=0.3, smooth=0.65, glow=0.2, bright=0.3, warm=0.0, matte=0.3),
     "man": FaceStyle(keep=0.0, detail=0.25, smooth=0.35, glow=0.0, bright=0.15, warm=0.0, matte=0.2),
     "adult": FaceStyle(keep=0.0, detail=0.2, smooth=0.45, glow=0.1, bright=0.2, warm=0.0, matte=0.2)},
    author="Beauty HDR", note="Da sạch như ảnh thẻ mẫu: bớt mụn, đốm sạm, vết đỏ; đều màu, giữ vân da. Kèm tông mẫu Da đẹp ảnh thẻ.",
    id="da-dep-the", created="2026-11-25", builtin=True, look=LOOKS["da_dep_the"].to_dict())

BUILTIN_SETS = (DEFAULT_SET, TRONG_SANG_SET, DA_DEP_THE_SET)
BUILTIN_IDS = {s.id for s in BUILTIN_SETS}


def builtin_sets() -> tuple[PresetSet, ...]:
    return BUILTIN_SETS


# ----------------------------------------------------------------- áp kiểu lên mặt đã căn 512×512

SKIN_CLASSES = (1, 2, 8, 9)  # da, mũi, hai tai (nhãn ParseNet của facexlib, xem Enhancer.DETAIL_CLASSES)


def face_window(size: int = 512) -> np.ndarray:
    """Cửa sổ elip mềm trên mặt căn kiểu FFHQ: 1 ở giữa mặt, nhạt dần về 0 ở mép mặt (cằm, chân tóc, tai)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size / 512.0)
    r = np.sqrt(((xx - 256) / 150.0) ** 2 + ((yy - 300) / 175.0) ** 2)
    return np.clip((1.25 - r) / 0.45, 0, 1).astype(np.float32)


_WINDOW: dict[int, np.ndarray] = {}


def skin_mask(labels: np.ndarray) -> np.ndarray:
    """Mặt nạ da 0..1 (đã nhạt dần về mép mặt) từ nhãn ParseNet của mặt căn."""
    size = labels.shape[0]
    if size not in _WINDOW:
        _WINDOW[size] = face_window(size)
    m = np.isin(labels, SKIN_CLASSES).astype(np.float32)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 4)
    return m * _WINDOW[size]


def apply(face_bgr: np.ndarray, st: FaceStyle, skin: np.ndarray) -> np.ndarray:
    """Áp phần màu và da của kiểu (smooth, glow, bright, warm) lên mặt căn. `skin`: mặt nạ từ skin_mask."""
    if all(abs(getattr(st, k)) < 1e-3 for k in ("smooth", "glow", "bright", "warm")):
        return face_bgr
    k = face_bgr.shape[0] / 512.0
    f = face_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)  # L 0..100
    L, a, b = cv2.split(lab)
    m = skin.astype(np.float32)
    if st.smooth > 1e-3:
        s = float(st.smooth)
        base = cv2.bilateralFilter(L, d=0, sigmaColor=4.0 + 6.0 * s, sigmaSpace=(4.0 + 8.0 * s) * k)
        fine = L - cv2.GaussianBlur(L, (0, 0), 1.2 * k)  # lỗ chân lông, vân da mịn
        mid = L - base - fine  # vết sạm, mụn, gồ ghề
        L2 = base + fine * (1.0 - 0.45 * s) + mid * (1.0 - 0.85 * s)
        L = L + m * (L2 - L)
        ab_s = (3.0 + 5.0 * s) * k  # đốm đỏ, vùng da loang màu
        a = a + m * 0.7 * s * (cv2.GaussianBlur(a, (0, 0), ab_s) - a)
        b = b + m * 0.7 * s * (cv2.GaussianBlur(b, (0, 0), ab_s) - b)
    if st.glow > 1e-3:
        hl = np.clip((L - 55.0) / 40.0, 0, 1) * m
        bloom = cv2.GaussianBlur(hl, (0, 0), 10.0 * k)
        L = 100.0 - (100.0 - L) * (1.0 - 0.35 * float(st.glow) * bloom)
    if st.bright > 1e-3:
        L = L + m * float(st.bright) * 9.0 * np.sqrt(np.clip(1.0 - L / 100.0, 0, 1))
    if abs(st.warm) > 1e-3:
        a = a + m * float(st.warm) * 3.0
        b = b + m * float(st.warm) * 2.0
    lab = cv2.merge([np.clip(L, 0, 100), a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------- file chia sẻ

class PresetError(ValueError):
    pass


def _clean_text(v, limit: int) -> str:
    if not isinstance(v, str):
        return ""
    v = re.sub(r"[\x00-\x1f\x7f‪-‮⁦-⁩]", " ", v)  # ký tự điều khiển, đảo chiều chữ
    return " ".join(v.split())[:limit]


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def parse_set(d: dict, warn: list[str]) -> PresetSet:
    if not isinstance(d, dict):
        raise PresetError("bộ preset không đúng dạng")
    name = _clean_text(d.get("name"), MAX_NAME) or "Bộ không tên"
    styles: dict[str, FaceStyle] = {}
    raw = d.get("styles")
    if not isinstance(raw, dict):
        raise PresetError(f"bộ \"{name}\" không có kiểu mặt nào")
    for subj, knobs in raw.items():
        if subj not in SUBJECT_KEYS:
            warn.append(f"{name}: bỏ qua đối tượng lạ \"{_clean_text(subj, 20)}\"")
            continue
        if not isinstance(knobs, dict):
            warn.append(f"{name}: kiểu mặt \"{subj}\" không đúng dạng, bỏ qua")
            continue
        vals = {}
        for k, v in knobs.items():
            if k not in RANGES:
                warn.append(f"{name}/{subj}: bỏ qua thông số lạ \"{_clean_text(k, 20)}\"")
                continue
            x = _num(v)
            if x is None:
                warn.append(f"{name}/{subj}: \"{k}\" không phải số, dùng 0")
                continue
            lo, hi = RANGES[k]
            if not lo <= x <= hi:
                warn.append(f"{name}/{subj}: \"{k}\" = {x:g} ngoài khoảng {lo:g}..{hi:g}, đã kẹp lại")
            vals[k] = float(np.clip(x, lo, hi))
        styles[subj] = FaceStyle(**vals)
    if not styles:
        raise PresetError(f"bộ \"{name}\" không có kiểu mặt hợp lệ")
    sid = d.get("id")
    sid = sid if isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", sid) else new_id()
    created = d.get("created")
    created = created if isinstance(created, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", created) else today()
    look = None
    if "look" in d and d["look"] is not None:
        if isinstance(d["look"], dict):
            look = Look.from_dict(d["look"], warn).to_dict()
        else:
            warn.append(f"{name}: tông mẫu không đúng dạng, bỏ qua")
    return PresetSet(name, styles, _clean_text(d.get("author"), MAX_NAME), _clean_text(d.get("note"), 300), sid,
                     created, False, look)


def loads(text: str | bytes) -> tuple[list[PresetSet], list[str]]:
    """Đọc nội dung file chia sẻ → (các bộ preset, cảnh báo). Sai định dạng thì PresetError."""
    if isinstance(text, bytes):
        if len(text) > MAX_FILE_BYTES:
            raise PresetError("file quá lớn để là file preset")
        text = text.decode("utf-8-sig", errors="strict")
    if len(text) > MAX_FILE_BYTES:
        raise PresetError("file quá lớn để là file preset")
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as e:
        raise PresetError(f"không đọc được JSON ({e.__class__.__name__})") from None
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise PresetError("không phải file preset của Nét Ảnh")
    warn: list[str] = []
    ver = d.get("version")
    if not isinstance(ver, int) or isinstance(ver, bool):
        raise PresetError("file preset thiếu số phiên bản")
    if ver > VERSION:
        warn.append(f"file tạo bằng bản Nét Ảnh mới hơn (định dạng {ver}), phần mới sẽ bị bỏ qua")
    sets = d.get("sets")
    if not isinstance(sets, list) or not sets:
        raise PresetError("file không có bộ preset nào")
    if len(sets) > MAX_SETS:
        warn.append(f"file có {len(sets)} bộ, chỉ nạp {MAX_SETS} bộ đầu")
    out = []
    for s in sets[:MAX_SETS]:
        try:
            out.append(parse_set(s, warn))
        except PresetError as e:
            warn.append(str(e))
    if not out:
        raise PresetError("không có bộ preset hợp lệ trong file")
    return out, warn


def dumps(sets: list[PresetSet], app: str = "") -> str:
    d = {"format": FORMAT, "version": VERSION, "app": app, "sets": [s.to_dict() for s in sets]}
    return json.dumps(d, ensure_ascii=False, indent=2) + "\n"


def load_file(path: str | Path) -> tuple[list[PresetSet], list[str]]:
    p = Path(path)
    if p.stat().st_size > MAX_FILE_BYTES:
        raise PresetError("file quá lớn để là file preset")
    return loads(p.read_bytes())


def save_file(sets: list[PresetSet], path: str | Path, app: str = "") -> Path:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(dumps(sets, app), encoding="utf-8")
    tmp.replace(p)
    return p


def safe_filename(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().strip(".")
    return (s or "bo-preset")[:60] + EXT


# ----------------------------------------------------------------- thư viện bộ preset trên máy

class Library:
    """Các bộ preset trên máy: bộ mặc định có sẵn (không sửa, không xoá) + bộ người dùng tạo hoặc nạp, mỗi bộ một file
    trong thư mục `presets`. Bộ đang dùng ghi ở file active.txt."""

    def __init__(self, root: Path | None = None):
        if root is None:
            from .history import data_dir
            root = data_dir() / "presets"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}{EXT}"

    def sets(self) -> list[PresetSet]:
        out = list(builtin_sets())
        for p in sorted(self.root.glob(f"*{EXT}")):
            try:
                got, _ = load_file(p)
            except (OSError, UnicodeDecodeError, PresetError):
                continue
            s = got[0]
            s.id = p.name[: -len(EXT)]
            out.append(s)
        return out

    def looks(self) -> list[PresetSet]:
        """Các bộ có kèm tông mẫu (cho ô chọn Tông mẫu)."""
        return [s for s in self.sets() if s.look]

    def get(self, sid: str | None) -> PresetSet:
        for s in self.sets():
            if s.id == sid:
                return s
        return DEFAULT_SET

    def save(self, s: PresetSet) -> PresetSet:
        if s.builtin:
            raise PresetError("bộ mặc định không sửa được: hãy tạo bản sao")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", s.id or ""):
            s.id = new_id()
        save_file([s], self._path(s.id))
        return s

    def delete(self, sid: str) -> None:
        if sid in BUILTIN_IDS:
            raise PresetError("không xoá được bộ có sẵn")
        self._path(sid).unlink(missing_ok=True)
        if self.active_id() == sid:
            self.set_active(DEFAULT_SET.id)

    def import_file(self, path: str | Path) -> tuple[list[PresetSet], list[str]]:
        """Nạp file chia sẻ vào thư viện. Trùng mã với bộ đã có thì tạo mã mới; trùng tên thì thêm "(2)"."""
        got, warn = load_file(path)
        have = self.sets()
        ids = {s.id for s in have}
        names = {s.name for s in have}
        for s in got:
            if s.id in ids or s.id in BUILTIN_IDS:
                s.id = new_id()
            base, n = s.name, 2
            while s.name in names:
                s.name = f"{base} ({n})"[:MAX_NAME]
                n += 1
            self.save(s)
            ids.add(s.id)
            names.add(s.name)
        return got, warn

    def export(self, sids: list[str], path: str | Path, app: str = "") -> Path:
        sets = [s for s in self.sets() if s.id in sids]
        if not sets:
            raise PresetError("chưa chọn bộ preset nào")
        out = []
        for s in sets:
            c = replace(s)
            c.builtin = False
            out.append(c)
        return save_file(out, path, app)

    def active_id(self) -> str:
        try:
            sid = (self.root / "active.txt").read_text(encoding="utf-8").strip()
        except OSError:
            return DEFAULT_SET.id
        return sid if sid in BUILTIN_IDS or self._path(sid).exists() else DEFAULT_SET.id

    def set_active(self, sid: str) -> None:
        (self.root / "active.txt").write_text(sid, encoding="utf-8")

    def active(self) -> PresetSet:
        return self.get(self.active_id())


def styles_to_settings(s: PresetSet) -> dict:
    """Dạng gọn để lưu trong Settings.face_styles (và lịch sử)."""
    return {k: v.to_dict() for k, v in s.styles.items()}


def styles_from_settings(d: dict | None) -> dict[str, FaceStyle]:
    if not d:
        return dict(DEFAULT_SET.styles)
    warn: list[str] = []
    return parse_set({"name": "x", "styles": d}, warn).styles


__all__ = ["FaceStyle", "PresetSet", "Library", "DEFAULT_SET", "TRONG_SANG_SET", "DA_DEP_THE_SET", "BUILTIN_SETS", "SUBJECT_KEYS", "RANGES", "KNOB_NAMES", "PresetError",
           "apply", "skin_mask", "loads", "dumps", "load_file", "save_file", "safe_filename", "FORMAT", "VERSION", "EXT",
           "styles_to_settings", "styles_from_settings"]
