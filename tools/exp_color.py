"""So màu sau fade_restore với ảnh gốc sạch (đáp án): ΔE00 trên da, áo, nền; giữ sắc ấm của ảnh ánh ấm có chủ ý."""
import sys, cv2, numpy as np, json
sys.path.insert(0, '/home/user/work/tools')
from skimage.color import deltaE_ciede2000
import importlib
from net_anh import restore
import facemetrics as fm
def lab(x): return cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100 / 255, 1, 1]) - np.array([0, 128, 128])
def regions(img):
    lms, _ = fm.landmarks(img); h, w = img.shape[:2]
    full = np.zeros((h, w), np.int16) - 1
    for lm in lms:
        f, M = fm.align(img, lm); _, _, lb = fm.eye_mask(f)
        back = cv2.warpAffine(lb.astype(np.uint8) + 1, cv2.invertAffineTransform(M), (w, h), flags=cv2.INTER_NEAREST)
        full[back > 0] = back[back > 0].astype(np.int16) - 1
    return {'da': np.isin(full, (1,)), 'môi': np.isin(full, (11, 12)), 'áo': full == 18, 'nền': full == 0}
def stats(out, gt, R):
    d = deltaE_ciede2000(lab(out), lab(gt)); L = lab(out); G = lab(gt)
    r = {}
    for k, m in R.items():
        if m.sum() < 200: continue
        r[k] = dict(dE=round(float(d[m].mean()), 1), a=round(float(L[m][:, 1].mean() - G[m][:, 1].mean()), 1),
                    b=round(float(L[m][:, 2].mean() - G[m][:, 2].mean()), 1), C=round(float(np.hypot(L[m][:, 1], L[m][:, 2]).mean() - np.hypot(G[m][:, 1], G[m][:, 2]).mean()), 1))
    r['all_dE'] = round(float(d.mean()), 1)
    return r
fn = sys.argv[1] if len(sys.argv) > 1 else 'fade_restore'
rows = []
for base in ['chandung_trungnien', 'nhom_bongda']:
    gt = cv2.imread(f'/home/user/work/set/{base}.png'); R = regions(gt)
    for kind in ['phai', 'amvang', 'amam']:
        x = cv2.imread(f'/home/user/work/set/{kind}_{base}.png')
        out, note = (restore.fade_restore(x, 1.0, faces=fm.landmarks(x)[0]) if fn == 'fade_restore' else getattr(restore, fn)(x, 1.0))
        ref = gt if kind != 'amam' else x   # ánh ấm có chủ ý: đáp án là chính ảnh vào (giữ sắc ấm)
        rows.append(dict(img=f'{kind}_{base}', fn=fn, note=note, vao=stats(x, ref, R), ra=stats(out, ref, R)))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        cv2.imwrite(f'/home/user/work/out/color/{kind}_{base}__{fn}.jpg', out)
