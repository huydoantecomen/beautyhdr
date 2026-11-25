import sys, cv2, numpy as np, torch
torch.set_num_threads(4)
from net_anh.pipeline import Enhancer, load_image, _to_tensor, _to_image
from net_anh.settings import Settings
eng = Enhancer('cpu')
img = load_image('/home/user/work/set/nhom_nu_dungnghieng.png')
helper = eng._load_helper(1); helper.read_image(img)
helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
lms = [np.asarray(l, np.float32) for l in helper.all_landmarks_5]
net = eng._load_face_net('gfpgan')
sr = eng._esrgan_run(img, 4, 400, net=eng._load_compact(0.5))
print(img.shape, sr.shape)
def aligned(src, lm):
    M = cv2.estimateAffinePartial2D(lm, helper.face_template, method=cv2.LMEDS)[0]
    return cv2.warpAffine(src, M, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
def gf(face):
    t = (_to_tensor(face, eng.dev, False) - 0.5) / 0.5
    with torch.no_grad(): return _to_image(net(t, return_rgb=False)[0], -1.0, 1.0)
rows = []
for i, lm in enumerate(lms):
    a = aligned(img, lm); b = aligned(sr, lm * 4)
    rows.append(np.hstack([a, gf(a), b, gf(b)]))
    print(i, np.linalg.norm(lm[0]-lm[1]))
cv2.imwrite('/home/user/work/peek/smallsr.jpg', cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 90])
