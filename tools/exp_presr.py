import sys, cv2, numpy as np, torch
sys.path.insert(0,'/home/user/work/tools'); torch.set_num_threads(4)
from net_anh.pipeline import Enhancer, load_image, save_image
from net_anh.analysis import recommend
from net_anh.levels import level_settings
from net_anh.settings import Settings
import facemetrics as fm
eng = Enhancer('cpu'); name = sys.argv[1]
img = load_image(f'/home/user/work/set/{name}.png'); an = eng.analyze(img, with_faces=True)
_, auto, _ = recommend(an, Settings())
base = level_settings(5, auto.with_(mode='portrait'), an).with_(mode='portrait')
for vk, kw in {'A_ai30': dict(fidelity=0.7, detail_fidelity=None), 'C_ai70': dict(fidelity=0.3, detail_fidelity=0.0)}.items():
    res, rep = eng.process(img, base.with_(**kw)); save_image(res, f'/home/user/work/out/presr/{name}__{vk}.png')
    r = cv2.resize(img, res.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    print(name, vk, [d['ids'] for d in fm.compare(r, res)], flush=True)
