"""Khôi phục ảnh cũ (chụp lại ảnh in, ảnh scan): bước làm sạch trước khi làm nét bằng AI.

  1. xoay thẳng ảnh nằm ngang/ngược (dò mặt ở 4 hướng)
  2. dò vết hư hỏng: vết xước, nếp gãy, nếp nhăn màng bọc (U-Net của Microsoft "Bringing Old Photos Back to Life", MIT)
     + đốm bụi, đốm vàng, mảnh vụn (bộ dò top-hat theo độ sáng và màu vàng)
  3. vá bằng LaMa (Apache-2.0), chia ô quanh vùng hỏng; không có LaMa thì dùng cv2.inpaint
  4. phục hồi màu phai: kéo giãn từng kênh theo phân vị vùng giữa ảnh (viền giấy không tính), cân bằng trắng mềm,
     tăng độ rực; ảnh đen trắng chỉ chuẩn mức sáng và gỡ ám màu giấy (không tô màu giả)
Mọi bước đều có thể tắt hoặc giảm bằng Settings.repair / Settings.fade.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from . import weights

SCRATCH_SHORT = 512  # cạnh ngắn khi chạy U-Net dò xước (đo trên 6 ảnh thật: 512 bắt vết nhăn màng bọc tốt hơn 256)
SCRATCH_THR = 0.4  # ngưỡng của tác giả
MAX_MASK = 0.12  # vùng vá tối đa (tỉ lệ diện tích): quá thì nâng ngưỡng, tránh "vẽ lại" cả ảnh
LAMA_WIN, LAMA_STRIDE = 640, 512  # ô vá, bước ô (mỗi ô có ≥ 64 px ngữ cảnh quanh phần lõi)


# ---------- 1. xoay thẳng ----------

def auto_rotate(enh, img: np.ndarray) -> tuple[np.ndarray, int]:
    """Trả (ảnh đã xoay, số độ xoay theo chiều kim đồng hồ: 0/90/180/270). Chỉ xoay khi một hướng có số mặt
    nhiều hơn hẳn hướng hiện tại (tránh xoay nhầm ảnh phong cảnh)."""
    if any(weights.status(m) != "ready" for m in ("retinaface", "parsenet")):
        return img, 0
    s = 640 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    codes = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    score = {}
    for deg, code in codes.items():
        x = small if code is None else cv2.rotate(small, code)
        try:
            fs = enh.detect_faces(x)
        except Exception:
            return img, 0
        score[deg] = sum(float(b[4]) if len(b) > 4 else 1.0 for b, _ in fs)
    best = max(score, key=score.get)
    if best != 0 and score[best] >= 1.0 and score[best] >= 2.0 * score[0] + 0.5:
        return cv2.rotate(img, codes[best]), best
    return img, 0


# ---------- 2. dò vết hư hỏng ----------

def _load_scratch(enh):
    net = getattr(enh, "_scratch", None)
    if net is None:
        from .archs.scratch_unet_arch import UNet

        path = enh._ensure("scratch")
        net = UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6, padding=True, batch_norm=True,
                   up_mode="upsample", with_tanh=False, sync_bn=False, antialiasing=True)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        # U-Net này hỏng khi chạy fp16 trên GPU cũ, và chỉ nặng 37 M tham số: chạy fp32 trên thiết bị của máy
        dev = enh.dev.device if enh.dev.kind == "cuda" else torch.device("cpu")
        net = net.to(dev).eval()
        enh._scratch = net
        enh._scratch_dev = dev
    return net


@torch.no_grad()
def scratch_prob(enh, img: np.ndarray) -> np.ndarray:
    """Xác suất vết xước/nếp gãy theo từng điểm ảnh (float32 0..1, cùng cỡ ảnh)."""
    net = _load_scratch(enh)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    k = SCRATCH_SHORT / min(h, w)
    H, W = max(16, round(h * k / 16) * 16), max(16, round(w * k / 16) * 16)
    x = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    t = torch.from_numpy((x - 0.5) / 0.5)[None, None].to(enh._scratch_dev)
    p = torch.sigmoid(net(t))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def speck_mask(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Đốm sáng/vàng nhỏ (bụi, mảnh vụn, đốm ố). Top-hat: chỉ bắt vật nhỏ hơn cỡ cửa sổ nên không dính cạnh,
    vân vải hay nếp áo; yêu cầu đủ sáng hơn quanh VÀ ngả vàng, hoặc rất sáng."""
    h, w = img.shape[:2]
    s = min(1.0, 1400 / max(h, w))
    x = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = max(3, int(max(x.shape[:2]) * 0.006))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    th_l = lab[..., 0] - cv2.morphologyEx(lab[..., 0], cv2.MORPH_OPEN, se)
    th_b = lab[..., 2] - cv2.morphologyEx(lab[..., 2], cv2.MORPH_OPEN, se)
    t = 1.0 / max(strength, 0.2)
    m = (((th_l > 30 * t) & (th_b > 8 * t)) | (th_l > 55 * t))
    # đốm vàng/nâu sẫm trên da hay nền tối: màu ngả vàng hơn hẳn vùng xung quanh (so với trung vị cục bộ)
    kk = (5 * r + 1) | 1
    med_b = cv2.medianBlur(np.clip(lab[..., 2], 0, 255).astype(np.uint8), kk).astype(np.float32)
    med_a = cv2.medianBlur(np.clip(lab[..., 1], 0, 255).astype(np.uint8), kk).astype(np.float32)
    gold = ((lab[..., 2] - med_b) > 7 * t) & (np.hypot(lab[..., 1] - med_a, lab[..., 2] - med_b) > 10 * t)
    m = (m | gold).astype(np.uint8)
    num, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(num, bool)
    # đốm bụi là hạt tròn: bỏ vật dài/mảnh (cạnh áo, ngón tay, viền màu) và vật quá đặc (mảng màu thật)
    bw, bh, ar = st[1:, 2].astype(np.float32), st[1:, 3].astype(np.float32), st[1:, 4].astype(np.float32)
    roundish = (np.maximum(bw, bh) <= 2.5 * np.maximum(np.minimum(bw, bh), 1)) & (ar >= 0.4 * bw * bh)
    keep[1:] = (ar <= 0.0035 * m.size) & (ar >= 2) & roundish
    m = keep[lbl].astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def detect_defects(enh, img: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, str]:
    """Mặt nạ vùng cần vá (bool, cùng cỡ ảnh) + ghi chú. strength 0..1: độ nhạy."""
    h, w = img.shape[:2]
    notes = []
    mask = np.zeros((h, w), bool)
    try:
        p = scratch_prob(enh, img)
        thr = SCRATCH_THR / max(strength, 0.3)
        while True:
            sm = p >= thr
            if sm.mean() <= MAX_MASK or thr >= 0.95:
                break
            thr += 0.1
        sm = sm.astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(sm, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, 4] >= max(6, 1e-6 * h * w)
        sm = keep[lbl].astype(np.uint8)
        d = max(3, int(max(h, w) * 0.0025)) | 1
        sm = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
        mask |= sm.astype(bool)
        notes.append(f"vết xước/nếp gãy {sm.mean() * 100:.1f} %")
    except (weights.ModelError, OSError, RuntimeError, MemoryError) as e:
        notes.append(f"không dò được vết xước ({e.__class__.__name__}), chỉ dò đốm")
    sp = speck_mask(img, strength)
    prot, head = feature_guard(enh, img)
    if prot is not None:
        # ngũ quan (mắt, lòng đen, lông mày, mũi, môi, kính): không coi là đốm bụi. Trên đầu (da mặt, tóc) chỉ nhận đốm
        # trắng rõ: nhánh "đốm vàng/nâu" bắt nhầm bóng cằm, má, chân tóc của mặt nhỏ trên ảnh phai (đo: 13–25 % da mặt)
        sp &= ~prot
        sp &= ~head | _bright_specks(img)
        # vết xước dài đi qua mắt, môi vẫn vá, nhưng chỉ vá đúng nét xước (không nới rộng) để LaMa không vẽ lại ngũ quan
        mask = np.where(prot, mask & _thin(mask), mask)
        notes.append("giữ nguyên ngũ quan")
    mask |= sp
    notes.append(f"đốm bụi {sp.mean() * 100:.1f} %")
    return mask, ", ".join(notes)


def _thin(m: np.ndarray) -> np.ndarray:
    """Phần lõi của vết (co lại 1 lần), bỏ phần nới rộng quanh nét xước."""
    return cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _bright_specks(img: np.ndarray) -> np.ndarray:
    """Điểm sáng hẳn hơn quanh nó (bụi trắng): phép top-hat như speck_mask nhưng chỉ nhánh rất sáng."""
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    r = max(3, int(max(img.shape[:2]) * 0.006))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = (l - cv2.morphologyEx(l, cv2.MORPH_OPEN, se)) > 55
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


GUARD_PARSE_MIN = 24  # khoảng cách hai mắt (px) tối thiểu để tách vùng bằng ParseNet; mặt nhỏ hơn chỉ dùng hình học


def feature_guard(enh, img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Vùng ngũ quan cần giữ (mắt, lông mày, mũi, miệng, kính) và vùng đầu (mặt + tóc), trên toạ độ ảnh. Đo trên 10 ảnh: bộ dò đốm
    bắt nhầm 22–63 % diện tích mắt + lông mày (lòng đen, ánh mắt, khoé mắt), 16–55 % môi trên mặt nhỏ, rồi LaMa vẽ
    lại thành da trơn. Vùng hình học theo 5 điểm mốc (đúng cả với mặt 15 px), cộng nhãn ParseNet cho mặt đủ lớn."""
    try:
        faces = enh.detect_faces(img)
    except Exception:
        return None, None
    if not faces:
        return None, None
    h, w = img.shape[:2]
    prot = np.zeros((h, w), np.uint8)
    head = np.zeros((h, w), np.uint8)
    helper = enh._helper
    for box, lm in faces:
        bx = np.asarray(box, np.float32)[:4]
        cx, cy, bw, bh = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, bx[2] - bx[0], bx[3] - bx[1]
        cv2.ellipse(head, (int(cx), int(cy - 0.1 * bh)), (int(0.65 * bw) + 2, int(0.7 * bh) + 2), 0, 0, 360, 1, -1)
        lm = np.asarray(lm, np.float32).reshape(5, 2)
        ed = float(np.linalg.norm(lm[0] - lm[1]))
        if ed < 3:
            continue
        up = np.array([lm[1][1] - lm[0][1], lm[0][0] - lm[1][0]]) / ed  # hướng lên trán (vuông góc trục mắt)
        if up[1] > 0:
            up = -up
        for e in lm[:2]:  # mắt + lông mày
            c = e + up * 0.18 * ed
            cv2.ellipse(prot, (int(c[0]), int(c[1])), (int(0.34 * ed) + 2, int(0.36 * ed) + 2),
                        float(np.degrees(np.arctan2(lm[1][1] - lm[0][1], lm[1][0] - lm[0][0]))), 0, 360, 1, -1)
        cv2.circle(prot, (int(lm[2][0]), int(lm[2][1])), int(0.22 * ed) + 2, 1, -1)  # mũi, lỗ mũi
        mc = (lm[3] + lm[4]) / 2
        mw = float(np.linalg.norm(lm[3] - lm[4]))
        cv2.ellipse(prot, (int(mc[0]), int(mc[1])), (int(0.5 * mw + 0.15 * ed) + 2, int(0.28 * ed) + 2),
                    float(np.degrees(np.arctan2(lm[4][1] - lm[3][1], lm[4][0] - lm[3][0]))), 0, 360, 1, -1)
        if ed >= GUARD_PARSE_MIN and helper is not None and getattr(helper, "face_parse", None) is not None:
            try:
                M = cv2.estimateAffinePartial2D(lm, helper.face_template, method=cv2.LMEDS)[0]
                face = cv2.warpAffine(img, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
                lab = enh._parse(helper, face)
                Mi = cv2.invertAffineTransform(M)
                feat = np.isin(lab, (2, 3, 4, 5, 6, 7, 10, 11, 12)).astype(np.uint8)
                feat = cv2.dilate(feat, np.ones((9, 9), np.uint8))
                prot |= cv2.warpAffine(feat, Mi, (w, h), flags=cv2.INTER_NEAREST)
            except Exception:
                pass
    return prot.astype(bool), head.astype(bool)


# ---------- 3. vá ----------

def _load_lama(enh):
    net = getattr(enh, "_lama", None)
    if net is None:
        path = enh._ensure("lama")
        dev = enh.dev.device if enh.dev.kind == "cuda" else torch.device("cpu")
        net = torch.jit.load(str(path), map_location=dev).eval()  # luôn fp32: FFT trong LaMa không an toàn ở fp16
        enh._lama, enh._lama_dev = net, dev
    return net


@torch.no_grad()
def _lama_window(enh, win: np.ndarray, m: np.ndarray) -> np.ndarray:
    net = _load_lama(enh)
    h, w = win.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        win = cv2.copyMakeBorder(win, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        m = cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = torch.from_numpy(cv2.cvtColor(win, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(enh._lama_dev)
    k = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(enh._lama_dev)
    y = net(x, k)[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)[:h, :w]


def inpaint(enh, img: np.ndarray, mask: np.ndarray, say=None) -> tuple[np.ndarray, str]:
    """Vá vùng `mask` bằng LaMa theo ô; lỗi/thiếu model thì dùng cv2.inpaint (Telea). Trả (ảnh, tên cách vá)."""
    if not mask.any():
        return img, ""
    h, w = img.shape[:2]
    m8 = mask.astype(np.uint8) * 255
    try:
        _load_lama(enh)
    except (weights.ModelError, OSError, RuntimeError) as e:
        r = max(3, int(max(h, w) * 0.002))
        return cv2.inpaint(img, m8, r, cv2.INPAINT_TELEA), f"OpenCV (không nạp được LaMa: {e.__class__.__name__})"
    out = img.copy()
    ys = list(range(0, max(1, h), LAMA_STRIDE))
    xs = list(range(0, max(1, w), LAMA_STRIDE))
    total = sum(1 for y in ys for x in xs if m8[y:y + LAMA_STRIDE, x:x + LAMA_STRIDE].any())
    done = 0
    pad = (LAMA_WIN - LAMA_STRIDE) // 2
    for y in ys:
        for x in xs:
            core = m8[y:y + LAMA_STRIDE, x:x + LAMA_STRIDE]
            if not core.any():
                continue
            enh._check()
            done += 1
            if say:
                say(f"Đang vá vết hỏng bằng LaMa… {done}/{total}")
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(h, y + LAMA_STRIDE + pad), min(w, x + LAMA_STRIDE + pad)
            res = _lama_window(enh, img[y0:y1, x0:x1], m8[y0:y1, x0:x1])
            cy0, cx0 = y - y0, x - x0
            ch, cw = core.shape
            dst = out[y:y + ch, x:x + cw]
            src = res[cy0:cy0 + ch, cx0:cx0 + cw]
            sel = core > 0
            dst[sel] = src[sel]
    # làm mềm mép vá: trộn theo mặt nạ nhoè nhẹ để không thấy đường viền
    soft = cv2.GaussianBlur(m8.astype(np.float32) / 255, (0, 0), 1.2)[..., None]
    soft = np.maximum(soft, (m8 > 0)[..., None].astype(np.float32) * 0.0)
    res = img.astype(np.float32) * (1 - soft) + out.astype(np.float32) * soft
    return res.round().clip(0, 255).astype(np.uint8), "LaMa"


# ---------- 4. phục hồi màu phai ----------

MONO_SPREAD = 22.0  # độ lệch hướng màu (Lab a/b, phân vị 5–95 %) dưới mức này coi là không có màu thật


def _chroma_spread(lab: np.ndarray) -> float:
    a, b = lab[..., 1].ravel() - 128, lab[..., 2].ravel() - 128
    return float(np.hypot(np.percentile(a, 95) - np.percentile(a, 5), np.percentile(b, 95) - np.percentile(b, 5)))


def is_mono(img: np.ndarray) -> bool:
    """Ảnh đen trắng (hoặc nâu đỏ một màu): không vùng nào của ảnh có màu thật.
    Đo trên độ lệch hướng màu giữa các điểm ảnh (chứ không phải độ ám chung của giấy). Xét vùng giữa VÀ lưới 4×4 ô
    phủ cả ảnh (bỏ 4 % sát mép: viền máy scan, mép giấy): chỉ cần một ô có màu rõ là giữ màu, vì nhánh đen trắng của
    fade_restore() bỏ hết màu của cả ảnh (vd vùng giữa xám nhưng áo đỏ sát mép)."""
    h, w = img.shape[:2]
    core = img[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
    if core.size == 0:
        core = img
    lab = cv2.cvtColor(cv2.resize(core, (160, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    if _chroma_spread(lab) >= MONO_SPREAD:
        return False
    my, mx = int(h * .04), int(w * .04)
    body = img[my:h - my, mx:w - mx] if h - 2 * my > 0 and w - 2 * mx > 0 else img
    n, t = 4, 64
    lab = cv2.cvtColor(cv2.resize(body, (n * t, n * t), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(n):
        for j in range(n):
            if _chroma_spread(lab[i * t:(i + 1) * t, j * t:(j + 1) * t]) >= MONO_SPREAD:
                return False
    return True


FADE_SKIN_C = 22.0  # độ rực (Lab C) đích của da khi phục hồi ảnh phai; da ảnh sạch đo được 19–32 (4 ảnh chân dung, nhóm)
FADE_SKIN_HUE = (35.0, 65.0)  # góc màu da tự nhiên (Lab, độ); da ảnh sạch đo được 38–64°

FADE_LOW_C = 30.0  # độ rực phân vị 90 % dưới mức này là ảnh phai: 5 ảnh phai thật của huy 16–28, ảnh sạch 39–55


def fade_cast(img: np.ndarray) -> tuple[bool, str]:
    """Ảnh có bị phai/ám do thời gian không (khác ánh sáng ấm có chủ ý). Dấu hiệu, đo trên 1 % điểm sáng nhất và tối
    nhất: đen bị nâng (thuốc phai, L > 15), hoặc vùng sáng nhất ngả vàng mà không phải da (giấy ố: b > 12 với a < 6,
    hoặc b > 25). Ảnh nắng chiều / đèn vàng có chủ ý: đen vẫn đen, chỗ sáng nhất trắng hoặc là da (a cao)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0] * (100 / 255)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    top, dark = l >= np.percentile(l, 99), l <= np.percentile(l, 1)
    ta, tb, dl = float(a[top].mean()), float(b[top].mean()), float(l[dark].mean())
    if dl > 15:
        return True, f"đen bị nâng (L {dl:.0f})"
    c90 = float(np.percentile(np.hypot(a, b), 90))
    if c90 < FADE_LOW_C:  # cả ảnh nhạt màu: thuốc ảnh phai (ảnh in chụp lại bằng điện thoại vẫn có đen sâu)
        return True, f"màu nhạt (C90 {c90:.0f})"
    if (tb > 12 and ta < 6) or tb > 25:
        return True, f"chỗ sáng ngả vàng (b {tb:.0f})"
    return False, ""


def cheek_mask(shape: tuple[int, int], faces) -> np.ndarray | None:
    """Hai mảng má (dưới mắt, cạnh mũi) theo 5 điểm mốc: đủ để đo màu da, không dính mắt, môi, tóc."""
    if not faces:
        return None
    m = np.zeros(shape, np.uint8)
    for lm in faces:
        lm = np.asarray(lm, np.float32).reshape(5, 2)
        ed = float(np.linalg.norm(lm[0] - lm[1]))
        if ed < 8:
            continue
        for e, mo in ((lm[0], lm[3]), (lm[1], lm[4])):
            c = e + 0.55 * (mo - e)
            cv2.circle(m, (int(c[0]), int(c[1])), max(2, int(0.18 * ed)), 1, -1)
    return m.astype(bool) if m.sum() >= 40 else None


def _ab_stats(lab: np.ndarray, sel: np.ndarray) -> tuple[float, float, float]:
    a, b = lab[..., 1][sel], lab[..., 2][sel]
    return float(np.median(a)), float(np.median(b)), float(np.median(np.hypot(a, b)))


def fade_restore(img: np.ndarray, amount: float = 1.0, sat: float = 0.4, faces=None) -> tuple[np.ndarray, str]:
    """Phục hồi ảnh phai/ám màu, giữ màu da. amount 0..1: mức sửa; sat: độ rực thêm tối đa khi không đo được da;
    faces: 5 điểm mốc của các mặt (đo màu da ở hai má).

    Khác bản 0.10 (fade_restore_v1, đo trên ảnh giả lập có đáp án + 5 ảnh phai thật):
      - chỉ gỡ ám (cân bằng từng kênh, gray-world) khi điểm trắng/điểm đen lệch màu: giấy ố, phai thuốc. Ảnh chụp dưới
        ánh nắng chiều, đèn vàng có chủ ý có trắng vẫn trắng, đen vẫn đen: chỉ kéo tương phản, giữ sắc ấm
        (bản cũ gỡ sắc ấm: ΔE 9,3 so với ảnh vào);
      - độ rực thêm theo màu da đo được (đích FADE_SKIN_C) thay vì +40 % cho mọi ảnh, kiểu vibrance (vùng đã rực như
        môi, áo đỏ thêm ít): bản cũ làm da ảnh phai xám đi (C −9) mà môi ảnh ám vàng đỏ gắt (C +30);
      - da lệch khỏi góc màu tự nhiên FADE_SKIN_HUE (ám tím hồng, ám xanh) thì xoay cả ảnh về phía đó một phần."""
    amount = float(np.clip(amount, 0, 1))
    if amount <= 0:
        return img, ""
    h, w = img.shape[:2]
    if is_mono(img):
        return fade_restore_v1(img, amount, sat)
    f = img.astype(np.float32) / 255
    my, mx = int(h * 0.15), int(w * 0.15)
    core = f[my:h - my, mx:w - mx].reshape(-1, 3)
    lo = np.percentile(core, 0.5, axis=0) * 0.9
    hi = np.percentile(core, 99.5, axis=0)
    cast, why = fade_cast(img)
    notes = []
    if cast:  # điểm trắng/đen lệch màu: giấy ố, phai thuốc → cân từng kênh
        hi = np.maximum(hi, lo + 0.3)
        st = np.clip((f - lo) / (hi - lo), 0, 1)
        notes.append(f"{why}, gỡ ám: kéo giãn kênh B/G/R ×" + "/".join(f"{1 / (b - a):.2f}" for a, b in zip(lo, hi)))
    else:  # ánh sáng có chủ ý: cùng một điểm đen/trắng cho cả 3 kênh
        l0, l1 = float(lo.min()), max(float(hi.max()), float(lo.min()) + 0.3)
        st = np.clip((f - l0) / (l1 - l0), 0, 1)
        notes.append("giữ sắc ánh sáng, chỉ kéo tương phản")
    out = f + (st - f) * amount
    if cast:
        lum = out.mean(-1)
        sel = (lum > 0.15) & (lum < 0.85)
        mean = out[sel].mean(0) if sel.any() else out.reshape(-1, 3).mean(0)
        gw = np.clip(mean.mean() / np.maximum(mean, 1e-3), 0.8, 1.25)
        out = np.clip(out * (1 + (gw - 1) * 0.5 * amount), 0, 1)
    u8 = (out * 255).round().astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] -= 128
    skin = cheek_mask((h, w), faces)
    if skin is not None:
        a, b, c = _ab_stats(lab, skin)
        hue = float(np.degrees(np.arctan2(b, a)))
        lo_h, hi_h = FADE_SKIN_HUE
        if cast and not lo_h <= hue <= hi_h and c > 3:  # da ám tím hồng / xanh: dời cả ảnh một phần về góc da gần nhất
            tgt = np.radians(lo_h) if abs(((hue - lo_h) + 180) % 360 - 180) < abs(((hue - hi_h) + 180) % 360 - 180) else np.radians(hi_h)
            da, db = c * np.cos(tgt) - a, c * np.sin(tgt) - b
            lab[..., 1] += 0.7 * amount * da
            lab[..., 2] += 0.7 * amount * db
            notes.append(f"da {hue:.0f}° → gần {np.degrees(tgt):.0f}°")
            a, b, c = _ab_stats(lab, skin)
        gain = float(np.clip(FADE_SKIN_C / max(c, 1.0), 1.0 + sat, 2.2)) if cast else 1.0
    else:
        C90 = float(np.percentile(np.hypot(lab[..., 1], lab[..., 2]), 90))
        gain = float(np.clip(38.0 / max(C90, 1.0), 1.0 + sat, 1.0 + 2 * sat)) if cast else 1.0
    if gain > 1.01:
        C = np.hypot(lab[..., 1], lab[..., 2])
        # vibrance: điểm đã rực (môi, áo đỏ) thêm ít; không đẩy quá C 60
        g = 1 + (gain - 1) * amount * np.clip(1 - C / 60.0, 0.15, 1)
        g = np.minimum(g, np.maximum(60.0 / np.maximum(C, 1e-3), 1.0))
        lab[..., 1] *= g
        lab[..., 2] *= g
        notes.append(f"độ rực ×{gain:.2f} (theo màu da)" if skin is not None else f"độ rực ×{gain:.2f}")
    lab[..., 1:] += 128
    res = cv2.cvtColor(np.clip(lab, 0, 255).round().astype(np.uint8), cv2.COLOR_LAB2BGR)
    return res, "màu phai: " + ", ".join(notes)


def fade_restore_v1(img: np.ndarray, amount: float = 1.0, sat: float = 0.4) -> tuple[np.ndarray, str]:
    """Phục hồi ảnh phai/ám màu. amount 0..1: mức sửa; sat: tăng độ rực (ảnh màu). Trả (ảnh, ghi chú)."""
    amount = float(np.clip(amount, 0, 1))
    if amount <= 0:
        return img, ""
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255
    my, mx = int(h * 0.15), int(w * 0.15)
    core = f[my:h - my, mx:w - mx].reshape(-1, 3)
    if is_mono(img):
        g = f @ np.array([0.114, 0.587, 0.299], np.float32)  # BGR → độ sáng
        lo, hi = np.percentile(g[my:h - my, mx:w - mx], (0.5, 99.5))
        hi = max(hi, lo + 0.25)
        y = np.clip((g - lo * 0.9) / (hi - lo * 0.9), 0, 1)
        out = np.repeat(y[..., None], 3, -1)
        out = f * (1 - amount) + out * amount
        return (out * 255).round().clip(0, 255).astype(np.uint8), "ảnh đen trắng: chuẩn mức sáng, gỡ ám màu giấy"
    st = np.empty_like(f)
    gains = []
    for c in range(3):
        lo, hi = np.percentile(core[:, c], (0.5, 99.5))
        lo, hi = lo * 0.9, max(hi, lo * 0.9 + 0.3)
        st[..., c] = np.clip((f[..., c] - lo) / (hi - lo), 0, 1)
        gains.append(1.0 / (hi - lo))
    out = f + (st - f) * amount
    # cân bằng trắng mềm (gray-world trên điểm không quá sáng/tối)
    lum = out.mean(-1)
    sel = (lum > 0.15) & (lum < 0.85)
    mean = out[sel].mean(0) if sel.any() else out.reshape(-1, 3).mean(0)
    gw = np.clip(mean.mean() / np.maximum(mean, 1e-3), 0.8, 1.25)
    out = np.clip(out * (1 + (gw - 1) * 0.5 * amount), 0, 1)
    u8 = (out * 255).round().astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + sat * amount), 0, 255)
    res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return res, f"màu phai: kéo giãn kênh B/G/R ×{gains[0]:.2f}/{gains[1]:.2f}/{gains[2]:.2f}, độ rực +{sat * amount * 100:.0f} %"


# ---------- vùng hỏng nặng do người dùng tô ----------

ROT_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def load_damage_mask(path: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Đọc mặt nạ vùng hỏng nặng (PNG, trắng = hỏng) do người dùng tô, đổi về cỡ `shape`=(h, w). Lỗi → None."""
    if not path:
        return None
    try:
        m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    except OSError:
        return None
    if m is None:
        return None
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m > 127


def face_damage_weight(enh, helper, idx: int) -> np.ndarray | None:
    """Trọng số 0..1 (512×512, trong khung mặt đã căn) của vùng hỏng nặng trên mặt thứ idx; None nếu mặt không dính."""
    m = getattr(enh, "_repair_mask", None)
    if m is None or not len(getattr(helper, "affine_matrices", [])) > idx:
        return None
    w = cv2.warpAffine(m.astype(np.uint8) * 255, helper.affine_matrices[idx], (512, 512), flags=cv2.INTER_LINEAR)
    w = w.astype(np.float32) / 255
    if w.mean() < 0.0008:
        return None
    w = cv2.dilate(w, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return np.clip(cv2.GaussianBlur(w, (0, 0), 7) * 1.4, 0, 1)


# ---------- chạy cả chuỗi ----------

def run(enh, img: np.ndarray, repair: float, fade: float, rotate: bool, rep, mask_path: str = "",
        style: str = "0.10") -> np.ndarray:
    """Làm sạch ảnh cũ. Ghi các bước đã làm vào rep.notes. Vùng người dùng tô (mask_path) luôn được vá, kể cả khi
    repair = 0, và được nhớ lại ở enh._repair_mask để bước làm nét mặt dựng lại mặt bị hỏng nặng."""
    enh._repair_mask = None
    user = load_damage_mask(mask_path, img.shape[:2])
    if rotate:
        enh.say("Đang kiểm tra hướng ảnh…")
        img, deg = auto_rotate(enh, img)
        if deg:
            rep.notes.append(f"Đã xoay ảnh {deg}° cho thẳng đứng")
            if user is not None:
                user = cv2.rotate(user.astype(np.uint8), ROT_CODES[deg]) > 0
    enh._check()
    mask, note = (np.zeros(img.shape[:2], bool), "")
    if repair > 0:
        enh.say("Đang dò vết xước, bụi, vết ố…")
        mask, note = detect_defects(enh, img, repair)
        enh._check()
    if user is not None:
        grown = cv2.dilate(user.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        mask = mask | grown
        enh._repair_mask = grown
        note = (note + ", " if note else "") + f"vùng bạn tô {user.mean() * 100:.1f} %"
    if repair > 0 or user is not None:
        if mask.any():
            img, how = inpaint(enh, img, mask, enh.say)
            rep.notes.append(f"Vá vết hỏng ({how}): {note}")
        else:
            rep.notes.append("Không thấy vết xước hay bụi đáng kể")
    enh._check()
    if fade > 0:
        enh.say("Đang phục hồi màu phai…")
        faces = None
        if style == "giu_da":
            try:
                faces = [lm for _, lm in enh.detect_faces(img)]
            except Exception:
                faces = None
        img, note = (fade_restore(img, fade, faces=faces) if style == "giu_da" else fade_restore_v1(img, fade))
        if note:
            rep.notes.append(note.capitalize())
    return img
