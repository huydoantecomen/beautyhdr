"""Tông mẫu (look): đưa màu da, môi, tóc, nền và vùng trắng của ảnh chân dung về gần tông của một bộ ảnh mẫu.

huy gửi 4 ảnh thẻ mẫu tông "trong sáng" (2026-11-25) và muốn ảnh gốc ra gần tông đó. Đo bằng tools/look_stats.py
(ParseNet tách vùng, Lab, L 0..100):

    vùng   ảnh mẫu (trung bình 4 ảnh)            ảnh gốc áo cam            ảnh gốc cầu lông
    da     L 74 (61–87)  a 10  b 13  h ~52°       L 73  a 19  b 15 (đỏ)     L 64  a 15  b 30 (tối, vàng)
    môi    L 52  a 32  b 16  C 36  h ~26°         a 41  b 16                a 31  b 29 (môi vàng)
    tóc    L 15  sắc độ 2–5 (đen trung tính)       L 22  sắc độ 15 (nâu đỏ)   L 11  sắc độ 6
    nền    trắng L 100 hoặc xanh L 53, lệch L 0,4–1,4 (phẳng, sạch)   lệch L 3,2 (vết bẩn)   lệch L 4,5
    tương phản p5–p95 của L: 85–96               52 (mờ sương)             67

Ảnh mẫu khác ảnh gốc chủ yếu ở màu da (bớt đỏ, bớt vàng, góc màu ~50°), độ sáng da, môi hồng đỏ không ngả vàng, tóc
đen trung tính, nền phẳng, vùng trắng trung tính và dải sáng tối rộng. Vân da của mẫu (độ lệch tầng mịn ~2) gần bằng
ảnh gốc: mẫu không xoá vân da mà chỉ đều màu da.

Cách làm (không model; mặt nạ vùng lấy từ ParseNet ở Enhancer._look):

1. Da: đo màu da giữa mặt (trung vị), rồi trên toàn vùng da (mặt, tai, cổ và chỗ da cùng màu như tay) cộng một độ
   lệch a, b *không đổi* để màu trung bình về đích: vân da, bóng khối giữ nguyên vì cộng hằng số. Độ sáng nâng bằng
   L' = 100 − (100 − L)·k (sáng thêm nhiều ở tông giữa, gần trắng thì gần như không đổi, không cháy). Đều màu da: làm
   mờ a, b ở thang vết đỏ/sạm (không đụng L nên vân da còn).
2. Môi: cộng lệch a, b về màu môi đích (hồng đỏ, bớt vàng).
3. Tóc: bớt sắc độ (nâu đỏ, ám vàng thành đen trung tính) và hạ nhẹ phần sáng của tóc.
4. Vùng trắng gần trung tính (áo, nền trắng): gỡ ám, đưa về trắng hơi lạnh.
5. Dải sáng tối: kéo giãn L của cả ảnh để p5–p95 tới gần đích (ảnh mờ sương rõ ra), giữ trung bình da.
6. Nền đồng màu (backdrop.detect): thay bằng chính màu nền đó cho phẳng (hết vết bẩn, loang), sáng thêm một chút.

Mọi bước nhân với `amount` (0..1) và độ mạnh từng phần của Look; Look ghi được vào file preset .netanh.json.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np


@dataclass
class Look:
    """Thông số một tông mẫu. Đích màu theo Lab (L 0..100). Độ mạnh từng phần 0..1."""
    name: str = "Trong sáng"
    skin_L: float = 75.0  # độ sáng da đích (trung vị da giữa mặt)
    skin_a: float = 11.5  # đỏ – xanh lá
    skin_b: float = 14.0  # vàng – xanh dương
    skin: float = 0.8  # phần lệch màu da được kéo về đích
    skin_light: float = 0.8  # phần lệch độ sáng da được kéo về đích (chỉ làm sáng, không làm tối)
    even: float = 0.4  # đều màu da (bớt vết đỏ, sạm), không đụng vân da
    clean: float = 0.0  # làm sạch da: mụn, đốm sạm, vết đỏ nhỏ (tầng giữa), giữ vân da; xem clean_skin
    tidy: float = 0.0  # làm đều độ sáng da thang rộng (vùng sạm, loang) – chỉ làm sáng chỗ tối, không làm tối chỗ sáng
    lips_a: float = 33.0
    lips_b: float = 15.0
    lips: float = 0.7
    hair: float = 0.5  # bớt sắc độ tóc (về đen trung tính)
    white: float = 0.7  # gỡ ám vùng trắng
    cool: float = 1.5  # vùng trắng đích hơi lạnh: b* = −cool
    contrast: float = 0.6  # kéo dải p5–p95 về đích
    range_target: float = 88.0
    bg_clean: float = 0.9  # làm phẳng nền đồng màu
    bg_lift: float = 3.0  # nền sạch sáng thêm (L)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, warn: list[str] | None = None) -> "Look":
        lk = cls()
        known = {f.name: f for f in fields(cls)}
        for k, v in (d or {}).items():
            if k not in known:
                if warn is not None:
                    warn.append(f"tông mẫu: bỏ qua khoá lạ “{k}”")
                continue
            if k == "name":
                if isinstance(v, str):
                    lk.name = v.strip()[:60] or lk.name
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                if warn is not None:
                    warn.append(f"tông mẫu: “{k}” không phải số")
                continue
            lo, hi = RANGES[k]
            setattr(lk, k, float(np.clip(v, lo, hi)))
        return lk


RANGES = {"skin_L": (40.0, 90.0), "skin_a": (-5.0, 30.0), "skin_b": (-5.0, 40.0), "skin": (0.0, 1.0),
          "skin_light": (0.0, 1.0), "even": (0.0, 1.0), "clean": (0.0, 1.0), "tidy": (0.0, 1.0), "lips_a": (0.0, 60.0), "lips_b": (-10.0, 40.0),
          "lips": (0.0, 1.0), "hair": (0.0, 1.0), "white": (0.0, 1.0), "cool": (-5.0, 5.0),
          "contrast": (0.0, 1.0), "range_target": (50.0, 100.0), "bg_clean": (0.0, 1.0), "bg_lift": (-10.0, 15.0)}

# Tông mẫu có sẵn. Khoá ghi vào Settings.look.
LOOKS: dict[str, Look] = {
    "trong_sang": Look(),
    # Da đẹp ảnh thẻ: tông Trong sáng + da sạch như ảnh thẻ mẫu (đo: vân da mẫu 1,9–2,6 so với 2,2–3,0 của ảnh gốc,
    # vết sạm, đốm đỏ gần như không có) – làm sạch mụn, đốm, vết đỏ, đều màu mạnh hơn, da sáng thêm một chút
    "da_dep_the": Look(name="Da đẹp ảnh thẻ", skin_light=0.9, even=0.7, clean=0.8, tidy=0.6, lips=0.6),
}

BG_DEV_MEAN, BG_DEV_P90 = 5.5, 11.0  # nền chỉ được làm phẳng khi độ lệch (ΔE) so với phông phẳng nhỏ: trung bình, phân vị 90
BG_DEV_KEEP = (6.0, 14.0)  # điểm lệch hơn mức đầu bớt được làm phẳng, quá mức sau giữ nguyên (vết bẩn lệch ~6–10 vẫn sửa)
CLEAN_CAP = (4.0, 11.0)  # độ lệch L (tầng giữa so với da quanh) được coi là vết: dưới 4 sửa hết, trên 11 là ngũ quan, không đụng
CLEAN_CAP_A = (2.5, 8.0)  # tương tự cho độ đỏ a*

SKIN_LIGHT_MAX = 12.0  # nâng sáng da tối đa (L): ảnh rất tối để bước sửa sáng lo, không kéo mặt thành mảng trắng
LOW_T, HIGH_T = 9.0, 97.0  # đích p5, p95 của L khi kéo giãn dải sáng tối
GAIN_MAX = 1.35
SKIN_REACH = 0.45  # da ngoài vùng ParseNet chỉ tính trong khoảng này (bề rộng mặt) quanh vùng da đã tách
SKIN_SIM = 8.0  # da ngoài mặt (tay, vai): khoảng cách a, b tới màu da giữa mặt, thang Gauss
SKIN_SIM_L = 32.0  # và lệch độ sáng tối đa coi là da


@dataclass
class Masks:
    """Mặt nạ 0..1 cỡ ảnh. core: da giữa mặt (đo màu); skin: da mặt, tai, cổ (áp màu); lips; hair; person: người."""
    core: np.ndarray
    skin: np.ndarray
    lips: np.ndarray
    hair: np.ndarray


def _q(x: np.ndarray, m: np.ndarray, q: float = 50.0) -> float:
    v = x[m > 0.5]
    return float(np.percentile(v, q)) if v.size else float("nan")


def skin_weight(lab: np.ndarray, mk: Masks, core_ab: tuple[float, float], core_L: float, face_px: float = 200.0
                ) -> np.ndarray:
    """Trọng số áp màu da: vùng da ParseNet, cộng chỗ cùng màu da giữa mặt (tay, vai, cổ ngoài khung mặt)."""
    a, b, L = lab[..., 1], lab[..., 2], lab[..., 0]
    d2 = (a - core_ab[0]) ** 2 + (b - core_ab[1]) ** 2
    sim = np.exp(-d2 / (2 * SKIN_SIM ** 2)) * np.clip(1 - (np.abs(L - core_L) - SKIN_SIM_L) / 10, 0, 1)
    sim *= np.clip(np.sqrt(a * a + b * b) / 6.0, 0, 1)  # trắng, xám không phải da
    # chỉ vùng sát da đã tách (cổ, vai, tay gần mặt): đá be, váy hồng nhạt, tường cùng tông da ở xa mặt không được coi là da
    # (trước 0.14.0 thử nghiệm: coi là da ở mọi nơi trong ảnh → mảng loang màu khắp ảnh)
    h, w_ = mk.skin.shape
    reach = max(3.0, SKIN_REACH * face_px)
    near = cv2.GaussianBlur(cv2.dilate((mk.skin > 0.3).astype(np.uint8),
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(reach) + 1,) * 2)
                                       ).astype(np.float32), (0, 0), reach / 3.0)
    sim = sim * near
    w = np.maximum(mk.skin, sim.astype(np.float32))
    w *= (1 - mk.lips) * (1 - mk.hair)
    return cv2.GaussianBlur(w.astype(np.float32), (0, 0), 1.5)


def _nblur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _soft(d: np.ndarray, cap: tuple[float, float]) -> np.ndarray:
    """Phần của độ lệch `d` (≥ 0) được sửa: đủ khi d ≤ cap[0], giảm dần về 0 ở cap[1] (vết nhỏ sửa hết, nét lớn như lỗ mũi,
    nếp mắt, chân mày giữ nguyên)."""
    return np.clip(1.0 - (d - cap[0]) / (cap[1] - cap[0]), 0, 1)


def clean_skin(L: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, face_px: float, clean: float, tidy: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Làm sạch da (tách tần số, chỉ trong vùng da `w`, đã có trọng số 0..1):

    - tầng mịn L − Gauss(σ ≈ 0,6 % mặt) (lỗ chân lông, vân da) giữ nguyên;
    - tầng giữa (Gauss σ nhỏ so với tầng nền σ ≈ 7 % mặt, tính có trọng số vùng da): chỗ tối hơn da quanh một đoạn ngắn
      (mụn thâm, đốm sạm, vết muỗi đốt) được nâng về da quanh, chỗ đỏ hơn kéo bớt đỏ; chỗ lệch lớn (lỗ mũi, mí mắt, nếp
      mũi má) không đụng (_soft);
    - tidy: vùng sạm rộng (quầng, loang) nâng về mức da quanh ở thang lớn, không hạ chỗ sáng.
    """
    wf = w.astype(np.float32) + 1e-4
    s_mid = max(1.2, 0.009 * face_px)
    s_base = max(5.0, 0.07 * face_px)
    Lm, am, bm = _nblur(L, wf, s_mid), _nblur(a, wf, s_mid), _nblur(b, wf, s_mid)
    Lb, ab, bb = _nblur(Lm, wf, s_base), _nblur(am, wf, s_base), _nblur(bm, wf, s_base)
    if clean > 1e-3:
        # nền tính lại bỏ qua các vết sơ bộ để vết không kéo nền xuống
        spot = np.clip((Lb - Lm - 1.5) / 3.0, 0, 1)
        w2 = wf * (1 - spot) ** 2 + 1e-4
        Lb2 = _nblur(Lm, w2, s_base)
        dark = np.maximum(Lb2 - Lm, 0)
        L = L + clean * w * dark * _soft(dark, CLEAN_CAP)
        red = np.maximum(am - ab, 0)
        a = a - clean * w * red * _soft(red, CLEAN_CAP_A) * 0.9
        brown = np.maximum(bm - bb, 0)
        b = b - clean * w * brown * _soft(brown, CLEAN_CAP_A) * 0.5
    if tidy > 1e-3:
        s_wide = max(10.0, 0.18 * face_px)
        Lw = _nblur(Lm, wf, s_wide)
        low = np.maximum(Lw - Lb, 0)  # vùng rộng tối hơn mức chung của da: nâng lên (a, b không đổi)
        L = L + tidy * w * low * 0.6
    return L, a, b


def apply(img: np.ndarray, mk: Masks, lk: Look, amount: float = 1.0, bg=None) -> tuple[np.ndarray, dict]:
    """Áp tông mẫu `lk` lên ảnh BGR uint8. Trả (ảnh, số đo trước/sau cho ghi chú, test)."""
    amount = float(np.clip(amount, 0, 1))
    info: dict = {}
    if amount <= 1e-3 or mk.core.sum() < 50:
        return img, info
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    core = mk.core
    cL, ca, cb = _q(L, core), _q(a, core), _q(b, core)
    info["da_truoc"] = (cL, ca, cb)
    face_px = float(np.sqrt(core.sum()))

    # 1. da: lệch a, b hằng số, đều màu, nâng sáng kiểu "screen"
    ws = skin_weight(lab, mk, (ca, cb), cL, face_px) * amount
    da, db = lk.skin * (lk.skin_a - ca), lk.skin * (lk.skin_b - cb)
    if lk.even > 1e-3:
        s = max(3.0, 0.06 * face_px)
        a_s = cv2.GaussianBlur(a, (0, 0), s)
        b_s = cv2.GaussianBlur(b, (0, 0), s)
        # chỉ bớt phần đỏ hơn / vàng hơn chung quanh (vết đỏ, sạm); phần xanh hơn (mạch máu) để nguyên
        a = a - lk.even * ws * np.maximum(a - a_s, 0)
        b = b - lk.even * ws * 0.5 * np.maximum(b - b_s, 0)
    if lk.clean > 1e-3 or lk.tidy > 1e-3:  # trên màu gốc, trước khi dời màu (đo vết đúng ở màu thật)
        L, a, b = clean_skin(L, a, b, np.clip(ws / amount, 0, 1), face_px, lk.clean * amount, lk.tidy * amount)
    a = a + ws * da
    b = b + ws * db
    dL = float(np.clip(lk.skin_light * (lk.skin_L - cL), 0, SKIN_LIGHT_MAX))
    if dL > 0.05:
        k = (100 - (cL + dL)) / max(100 - cL, 1e-3)
        L = L + ws * ((100 - (100 - L) * k) - L)

    # 2. môi
    if lk.lips > 1e-3 and mk.lips.sum() > 20:
        la, lb = _q(a, mk.lips), _q(b, mk.lips)
        wl = mk.lips * amount * lk.lips
        a = a + wl * (lk.lips_a - la)
        b = b + wl * (lk.lips_b - lb)
        info["moi_truoc"] = (la, lb)

    # 3. tóc
    if lk.hair > 1e-3 and mk.hair.sum() > 20:
        wh = mk.hair * amount * lk.hair
        a = a * (1 - wh)
        b = b * (1 - wh)
        L = L - wh * 0.25 * np.maximum(L - 15.0, 0)

    # 4. vùng trắng gần trung tính → trắng hơi lạnh
    if lk.white > 1e-3:
        C = np.sqrt(a * a + b * b)
        # chỉ trắng thật (áo, phông trắng): sắc độ < 4 đầy đủ, hết ở 10. Váy kem, đá be, trời nhạt (sắc độ 8–18) không đụng,
        # nếu không trọng số đổi mượt theo sắc độ sẽ tạo mảng loang trên các vùng nhạt màu
        ww = np.clip((L - 78.0) / 12.0, 0, 1) * np.clip(1 - (C - 4.0) / 6.0, 0, 1) * (1 - ws) * amount * lk.white
        a = a * (1 - ww)
        b = b + ww * (-lk.cool - b)

    # 5. dải sáng tối: kéo giãn hai phía của độ sáng da riêng rẽ (da giữ nguyên): p5 về LOW_T, p95 về HIGH_T, chỉ giãn
    #    chứ không nén, và có giới hạn để tóc đen không thành mảng đen đặc (tóc mẫu L ~15, p10 ~4)
    if lk.contrast > 1e-3:
        small = cv2.resize(L, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        p5, p95 = (float(v) for v in np.percentile(small, (5, 95)))
        piv = _q(L, core)
        rng = p95 - p5
        t = amount * lk.contrast * float(np.clip((lk.range_target - rng) / 25.0, 0, 1))
        lo_t = piv - (piv - p5) - t * max(p5 - LOW_T, 0)
        hi_t = p95 + t * max(HIGH_T - p95, 0)
        g_lo = float(np.clip((piv - lo_t) / max(piv - p5, 1.0), 1.0, GAIN_MAX))
        g_hi = float(np.clip((hi_t - piv) / max(p95 - piv, 1.0), 1.0, GAIN_MAX))
        if g_lo > 1.005 or g_hi > 1.005:
            d = L - piv
            L = piv + np.where(d < 0, d * g_lo, d * g_hi)
            hi = L > 94
            L = np.where(hi, 94 + 6 * np.tanh((L - 94) / 6), L)
            lo = L < 4
            L = np.where(lo, 4 * np.exp((L - 4) / 4), L)
        info["dai_sang"] = (rng, g_lo, g_hi)

    lab = cv2.merge([np.clip(L, 0, 100).astype(np.float32), a.astype(np.float32), b.astype(np.float32)])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    info["da_sau"] = (_q(L, core), _q(a, core), _q(b, core))

    # 6. nền phẳng
    if lk.bg_clean > 1e-3:
        out, note = clean_backdrop(out, amount * lk.bg_clean, lk.bg_lift * amount, np.maximum(mk.hair, mk.skin))
        if note:
            info["nen"] = note
    return out, info


def clean_backdrop(img: np.ndarray, amount: float, lift: float, protect: np.ndarray | None = None
                   ) -> tuple[np.ndarray, str]:
    """Nền đồng màu (dò từ viền): vùng nền chắc chắn (cách người một khoảng) được thay bằng màu nền trơn (giữ chút chuyển
    sáng tối rất rộng của phông), sáng thêm `lift` L. Không đụng viền tóc, người: phông tối gần màu tóc (xanh đen) mà
    thay cả vùng sát tóc thì sợi tóc bị ăn thành nền. `protect`: mặt nạ 0..1 không được đụng (tóc, da)."""
    from . import backdrop
    bd = backdrop.detect(img)
    if bd is None or amount <= 1e-3:
        return img, ""
    h, w = img.shape[:2]
    alpha = backdrop.alpha_matte(img, bd)
    if float(alpha.mean()) > 0.97:
        return img, ""
    r = max(3, int(0.012 * max(h, w)))
    m = (alpha < 0.02).astype(np.uint8)
    if protect is not None:
        m &= (cv2.dilate((protect > 0.05).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) == 0)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if m.sum() < 0.02 * h * w:
        return img, ""
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    mf = m.astype(np.float32)
    s = max(8.0, 0.08 * max(h, w))
    num = cv2.GaussianBlur(lab * mf[..., None], (0, 0), s)
    den = cv2.GaussianBlur(mf, (0, 0), s)[..., None]
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-3), bd.color_lab.reshape(1, 1, 3))
    flat = 0.5 * smooth + 0.5 * bd.color_lab.reshape(1, 1, 3)  # phông trơn, còn một nửa chuyển sáng tối rộng
    flat[..., 0] = np.clip(flat[..., 0] + lift, 0, 100)
    # phông thật sự trơn thì mọi điểm gần màu phông phẳng; cây, trời, tường có hoa văn thì lệch nhiều: không làm phẳng
    # (làm phẳng sẽ phủ một lớp sương xám lên cây, trời, lộ rõ ranh giới), và điểm nào lệch xa vẫn giữ nguyên
    dev = np.sqrt(((lab - flat) ** 2).sum(axis=2))
    inside = mf > 0.5
    if float(dev[inside].mean()) > BG_DEV_MEAN or float(np.percentile(dev[inside], 90)) > BG_DEV_P90:
        return img, ""
    keep = np.clip(1.0 - (dev - BG_DEV_KEEP[0]) / (BG_DEV_KEEP[1] - BG_DEV_KEEP[0]), 0, 1)[..., None]
    wgt = cv2.GaussianBlur(mf, (0, 0), r / 2.0)[..., None] * amount * keep
    new = cv2.cvtColor(flat.astype(np.float32), cv2.COLOR_LAB2BGR) * 255.0
    out = img.astype(np.float32) * (1 - wgt) + new * wgt
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), "làm phẳng nền"


def masks_from_labels(labels: np.ndarray) -> Masks:
    """Mặt nạ từ ảnh nhãn ParseNet cỡ ảnh (−1 = ngoài mặt căn); dùng cho test và tools."""
    def soft(cls, er=0, blur=2.0):
        m = np.isin(labels, cls).astype(np.uint8)
        if er:
            m = cv2.erode(m, np.ones((er, er), np.uint8))
        return cv2.GaussianBlur(m.astype(np.float32), (0, 0), blur)
    return Masks(core=(soft((1, 2), 9, 0.1) > 0.5).astype(np.float32), skin=soft((1, 2, 8, 9, 17), 3),
                 lips=soft((11, 12), 0, 1.5), hair=soft((13,), 0, 2.0))


def get(key: str, custom: dict | None = None) -> Look | None:
    """Look theo khoá Settings.look; `custom` (dict từ file preset) thắng khoá."""
    if custom:
        return Look.from_dict(custom)
    return LOOKS.get(key or "")


__all__ = ["Look", "LOOKS", "Masks", "clean_skin", "RANGES", "apply", "clean_backdrop", "get", "masks_from_labels", "skin_weight"]
