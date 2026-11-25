"""Da nhựa, viền sắc giả: giữ nguyên phần phục hồi mặt (cùng model, cùng mức trộn B: 50 % AI), chỉ đổi từng tham số
hậu xử lý. Đo so với ảnh sạch (đáp án): vân da (tỉ lệ độ nét da ra/đáp án, 1 = tự nhiên, <1 = da nhựa), sai số và
quầng sáng ở dải viền mặt/chân tóc (ngoài mặt 3–14 px), ΔE da."""
import sys, json, cv2, numpy as np, torch
sys.path.insert(0, '/home/user/work/tools')
torch.set_num_threads(4)
from net_anh import pipeline, facestyle
from net_anh.pipeline import Enhancer, load_image, save_image
from net_anh.analysis import recommend
from net_anh.levels import level_settings
from net_anh.settings import Settings
from skimage.color import deltaE_ciede2000
import facemetrics as fm
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
D = facestyle.DEFAULT_SET.styles
ORIG = {k: v for k, v in D.items()}
def styles_scaled(sm=1.0, gl=1.0):
    for k, v in ORIG.items():
        D[k] = facestyle.FaceStyle(**{**v.to_dict(), 'smooth': v.smooth * sm, 'glow': v.glow * gl}) if hasattr(v, 'to_dict') else v
VARS = {
 'P0_hien_tai': {},
 'P1_khong_bong': dict(s=dict(gloss=0.0)),
 'P2_min_da_nua': dict(s=dict(gloss=0.25), sm=0.5, gl=0.5),
 'P3_vien_mem': dict(contour=0.5, soft=9.0, grow=6),
 'P4_ket_hop': dict(s=dict(gloss=0.25), sm=0.5, gl=0.5, contour=0.6, soft=8.0, grow=8),
}
CASES = [('mo_chandung_trungnien', 'chandung_trungnien'), ('mo_9mat', 'sach_9mat'), ('chandung_nu_xanh', None)]
eng = Enhancer('cpu'); rows = []
def lab(x): return cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100 / 255, 1, 1]) - np.array([0, 128, 128])
for name, gt in CASES:
    img = load_image(f'/home/user/work/set/{name}.png')
    an = eng.analyze(img, with_faces=True)
    _, auto, _ = recommend(an, Settings())
    base = level_settings(5, auto.with_(mode='portrait'), an).with_(mode='portrait', fidelity=0.5, detail_fidelity=0.3)
    ref = cv2.imread(f'/home/user/work/set/{gt}.png') if gt else None
    for vk, v in VARS.items():
        styles_scaled(v.get('sm', 1.0), v.get('gl', 1.0))
        pipeline.CONTOUR_AI = v.get('contour', 0.85)
        Enhancer.PASTE_SOFT = v.get('soft', 6.0); Enhancer.PASTE_GROW = v.get('grow', 10)
        f = OUT / f'{name}__{vk}.png'
        if f.exists(): res = cv2.imread(str(f))
        else:
            res, _ = eng.process(img, base.with_(**v.get('s', {}))); save_image(res, f)
        r = ref if ref is not None else cv2.resize(img, res.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        o = cv2.resize(res, r.shape[1::-1], interpolation=cv2.INTER_AREA) if res.shape[:2] != r.shape[:2] else res
        for i, lm in enumerate(fm.landmarks(r)[0]):
            fr, M = fm.align(r, lm); fo, _ = fm.align(o, lm)
            em, skin, labp = fm.eye_mask(fr)
            face = np.isin(labp, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)).astype(np.uint8)
            ring = (cv2.dilate(face, np.ones((29, 29), np.uint8)) - cv2.dilate(face, np.ones((7, 7), np.uint8))).astype(bool)
            Lr, Lo = lab(fr)[..., 0], lab(fo)[..., 0]
            d = dict(image=name, variant=vk, face=i, ref='GT' if gt else 'ảnh vào',
                     skin_tex=round(fm.sharp(fo, skin) / max(fm.sharp(fr, skin), 1e-3), 3),
                     ring_err=round(float(np.abs(Lo - Lr)[ring].mean()), 2),
                     halo=round(float(np.clip(Lo - Lr, 0, None)[ring].mean()), 2),
                     skin_dE=round(float(deltaE_ciede2000(lab(fo), lab(fr))[skin].mean()), 2))
            rows.append(d); print(d, flush=True)
styles_scaled()
json.dump(rows, open(OUT / 'metrics_post.json', 'w'), ensure_ascii=False, indent=0)
