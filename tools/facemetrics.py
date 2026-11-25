"""Đo mặt: căn mặt theo landmark của ảnh tham chiếu (GT hoặc ảnh vào), cắt cùng toạ độ trên ảnh ra, rồi đo
IDS (ArcFace), độ nét vùng mắt, lệch landmark 5 điểm (mắt, mũi, khoé miệng) theo % khoảng cách hai mắt."""
import cv2, numpy as np, torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from net_anh.identity import Identity

_helper = None; _idn = None
def helper():
    global _helper
    if _helper is None:
        _helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50',
                                    save_ext='png', use_parse=True, device=torch.device('cpu'),
                                    model_rootpath='/home/user/work/weights')
    return _helper
def idn():
    global _idn
    if _idn is None: _idn = Identity(torch.device('cpu'))
    return _idn

def landmarks(img, min_eye=8):
    h = helper(); h.clean_all(); h.read_image(img)
    h.get_face_landmarks_5(only_center_face=False, resize=None, eye_dist_threshold=min_eye)
    return [np.asarray(l, np.float32).reshape(5, 2) for l in h.all_landmarks_5], [np.asarray(b) for b in h.det_faces]

def align(img, lm):
    h = helper()
    M = cv2.estimateAffinePartial2D(lm, h.face_template, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, M, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)), M

def eye_mask(face512):
    h = helper()
    x = cv2.cvtColor(face512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(((x - 0.5) / 0.5).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        lab = h.face_parse(t)[0].argmax(1).squeeze().numpy()
    return np.isin(lab, (4, 5, 6, 7)), np.isin(lab, (1,)), lab

def sharp(face, m):
    g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(np.sqrt((lap[m] ** 2).mean())) if m.sum() > 30 else float('nan')

def match(lms_ref, lms_out):
    """ghép mặt ra với mặt tham chiếu theo tâm mắt gần nhất"""
    pairs = []
    for i, a in enumerate(lms_ref):
        best, bd = None, 1e9
        for j, b in enumerate(lms_out):
            d = np.linalg.norm(a[:2].mean(0) - b[:2].mean(0))
            if d < bd: best, bd = j, d
        ed = np.linalg.norm(a[0] - a[1])
        pairs.append((i, best if best is not None and bd < 0.5 * ed else None))
    return pairs

def compare(ref, out, ref_lms=None):
    """ref, out: cùng kích thước. Trả list dict cho từng mặt của ref."""
    if out.shape[:2] != ref.shape[:2]:
        out = cv2.resize(out, ref.shape[1::-1], interpolation=cv2.INTER_AREA)
    lr = ref_lms if ref_lms is not None else landmarks(ref)[0]
    lo = landmarks(out)[0]
    res = []
    for i, j in match(lr, lo):
        a = lr[i]; ed = float(np.linalg.norm(a[0] - a[1]))
        fr, M = align(ref, a); fo, _ = align(out, a)
        em, skin, _ = eye_mask(fr)
        d = dict(face=i, eye_px=round(ed, 1), ids=round(idn().similarity(fr, fo), 4),
                 eye_sharp_ref=round(sharp(fr, em), 2), eye_sharp_out=round(sharp(fo, em), 2),
                 skin_sharp_ref=round(sharp(fr, skin), 2), skin_sharp_out=round(sharp(fo, skin), 2))
        if j is not None:
            dev = np.linalg.norm(lo[j] - a, axis=1) / ed * 100
            d.update(lm_eye=round(float(dev[:2].mean()), 2), lm_mouth=round(float(dev[3:].mean()), 2))
        res.append(d)
    return res
