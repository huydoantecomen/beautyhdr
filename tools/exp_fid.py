"""Thử 3 mức trộn AI (cơ chế fidelity/detail_fidelity hiện có) trên cùng ảnh, mọi thông số khác như mức 5."""
import sys, json, cv2, numpy as np, torch
sys.path.insert(0, '/home/user/work/tools')
torch.set_num_threads(4)
from net_anh.pipeline import Enhancer, load_image, save_image
from net_anh.analysis import recommend
from net_anh.levels import level_settings
from net_anh.settings import Settings
import facemetrics as fm
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
VARS = {'A_ai30': dict(fidelity=0.7, detail_fidelity=None), 'B_ai50': dict(fidelity=0.5, detail_fidelity=0.3),
        'C_ai70': dict(fidelity=0.3, detail_fidelity=0.0)}
CASES = [('mo_chandung_trungnien', 'chandung_trungnien'), ('mo_9mat', 'sach_9mat'), ('chandung_nu_xanh', None),
         ('nhom_mua', None), ('nhom_nu_dungnghieng', None), ('chandung_trungnien', None)]
if len(sys.argv) > 2: CASES = [c for c in CASES if c[0] in sys.argv[2].split(',')]
eng = Enhancer('cpu'); rows = []
for name, gt in CASES:
    img = load_image(f'/home/user/work/set/{name}.png')
    an = eng.analyze(img, with_faces=True)
    _, auto, why = recommend(an, Settings())
    base = level_settings(5, auto.with_(mode='portrait'), an).with_(mode='portrait')
    ref = cv2.imread(f'/home/user/work/set/{gt}.png') if gt else None
    for vk, kw in VARS.items():
        s = base.with_(**kw)
        f = OUT / f'{name}__{vk}.png'
        if f.exists(): res = cv2.imread(str(f))
        else:
            res, rep = eng.process(img, s); save_image(res, f)
        r = ref if ref is not None else cv2.resize(img, res.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        for d in fm.compare(r, res):
            d.update(image=name, variant=vk, ref='GT' if gt else 'ảnh vào'); rows.append(d); print(d, flush=True)
    if ref is not None:  # mốc: ảnh vào phóng bicubic so với GT
        up = cv2.resize(img, ref.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        for d in fm.compare(ref, up):
            d.update(image=name, variant='0_bicubic', ref='GT'); rows.append(d); print(d, flush=True)
json.dump(rows, open(OUT / ('metrics_' + (sys.argv[2].replace(',', '_') if len(sys.argv) > 2 else 'all') + '.json'), 'w'), ensure_ascii=False, indent=0)
