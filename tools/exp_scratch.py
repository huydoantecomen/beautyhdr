"""Mặt nạ dò xước: đo nhận nhầm trên tóc/mi/kính/hoa văn (ảnh sạch) và độ phủ vết thật (ảnh có đáp án)."""
import sys, json, cv2, numpy as np, torch
sys.path.insert(0, '/home/user/work/tools')
torch.set_num_threads(2)
from net_anh.pipeline import Enhancer, load_image
from net_anh import restore
import facemetrics as fm
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
names = sys.argv[2].split(',')
eng = Enhancer('cpu')
REG = {'tóc': (13,), 'mắt+mày': (4, 5, 6, 7), 'kính': (3,), 'da': (1,), 'môi': (11, 12)}
def region_labels(img):
    """nhãn ParseNet đưa ngược về toạ độ ảnh (theo từng mặt)"""
    lms, _ = fm.landmarks(img)
    full = np.zeros(img.shape[:2], np.int16) - 1
    for lm in lms:
        f, M = fm.align(img, lm)
        _, _, lab = fm.eye_mask(f)
        Mi = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(lab.astype(np.uint8) + 1, Mi, img.shape[1::-1], flags=cv2.INTER_NEAREST)
        sel = back > 0; full[sel] = back[sel].astype(np.int16) - 1
    return full
rows = []
for n in names:
    img = load_image(f'/home/user/work/set/{n}.png')
    impl = sys.argv[3] if len(sys.argv) > 3 else 'goc'
    mask, note = (restore.detect_defects(eng, img, 1.0))
    lab = region_labels(img)
    d = dict(image=n, impl=impl, note=note, mask_pct=round(mask.mean() * 100, 2))
    for k, cl in REG.items():
        r = np.isin(lab, cl)
        if r.sum() > 100: d['nhầm_' + k + '_%'] = round(mask[r].mean() * 100, 2)
    gtp = Path(f'/home/user/work/set/{n}_gt.png')
    if gtp.exists():
        gt = cv2.imread(str(gtp), 0) > 127
        d['recall_%'] = round((mask & gt).sum() / gt.sum() * 100, 1)
        d['precision_%'] = round((mask & gt).sum() / max(mask.sum(), 1) * 100, 1)
    ov = img.copy(); ov[mask] = (0.4 * ov[mask] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(str(OUT / f'{n}__mask_{impl}.jpg'), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / f'{n}__mask_{impl}.png'), mask.astype(np.uint8) * 255)
    print(d, flush=True); rows.append(d)
json.dump(rows, open(OUT / f'scratch_{impl}_{names[0]}.json', 'w'), ensure_ascii=False, indent=0)
