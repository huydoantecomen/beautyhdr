import sys, cv2, numpy as np, torch
torch.set_num_threads(1)
from net_anh.pipeline import Enhancer
from net_anh import restore
eng = Enhancer('cpu')
O = '/home/user/work/out/scratch/'
for n in sys.argv[1].split(','):
    img = cv2.imread(f'/home/user/work/set/{n}.png')
    for impl in ('goc', 'moi'):
        m = cv2.imread(O + f'{n}__mask_{impl}.png', 0) > 127
        out, how = restore.inpaint(eng, img, m)
        cv2.imwrite(O + f'{n}__va_{impl}.png', out)
        if n == 'xuoc_9mat':
            gt = cv2.imread('/home/user/work/set/sach_9mat.png').astype(np.float32)
            g = cv2.imread('/home/user/work/set/xuoc_9mat_gt.png', 0) > 127
            e = np.abs(out.astype(np.float32) - gt).mean(-1)
            print(n, impl, 'MAE cả ảnh %.2f, trong vết %.2f, ngoài vết %.2f' % (e.mean(), e[g].mean(), e[~g].mean()), flush=True)
        else:
            print(n, impl, 'done', flush=True)
