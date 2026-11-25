import sys, cv2, numpy as np, torch
torch.set_num_threads(4)
from net_anh.pipeline import Enhancer, load_image, save_image
from net_anh.analysis import recommend
from net_anh.levels import level_settings
from net_anh.settings import Settings
eng = Enhancer('cpu')
for name in sys.argv[1].split(','):
    img = load_image(f'/home/user/work/set/{name}.png'); an = eng.analyze(img, with_faces=True)
    _, auto, _ = recommend(an, Settings())
    base = level_settings(5, auto.with_(mode='portrait'), an).with_(mode='portrait', gloss=0.5)
    for k, sr in (('goc', False), ('sr', True)):
        res, rep = eng.process(img, base.with_(small_face_sr=sr)); save_image(res, f'/home/user/work/out/presr/{name}__r2_{k}.png'); print(name, k, rep.faces_restored, flush=True)
