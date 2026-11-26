"""Audit: chạy từng thông số ở nhiều giá trị, đo khác biệt so với mặc định."""
import sys, os, json, time, csv
sys.path.insert(0, '/home/user/beautyhdr')
import cv2, numpy as np, torch
torch.set_num_threads(4)
from net_anh.pipeline import Enhancer
from net_anh.settings import Settings
from net_anh.levels import level_settings
from net_anh.look import LOOKS
from net_anh import classic

IMG = sys.argv[1]; OUT = sys.argv[2]; ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else ['subject','fidelity','detail_fidelity','keep_lips','skin_tone','gloss','deshine','exposure','brightness','contrast','tone','low_light','look','look_amount','phone','wb','color','sharpness','denoise','restore','scene_vivid','bg_color','strength','mode','upscale','face_model','sr_model','level']
os.makedirs(OUT, exist_ok=True)
eng = Enhancer(device='cpu')

P = dict(mode='portrait', upscale=2, sharpness=0.5, fidelity=0.7)   # mức 3 "Rõ" (như ảnh chụp bảng)
Q = dict(mode='quick', upscale=2, sharpness=0.4, strength=0.85)    # mức 1 "Nhẹ"
O = dict(mode='old', upscale=2, sharpness=0.5, fidelity=0.7)

def img(n): return cv2.imread(f'{IMG}/{n}.png')

# (nhóm, nhãn thông số, ảnh, nền, [(nhãn giá trị, ghi đè)]) — giá trị đầu là mặc định để so
T = []
def t(param, ims, base, vals, tag=''):
    for im in ims: T.append((param, tag, im, base, vals))
TL = LOOKS['trong_sang'].to_dict(); DL = LOOKS['da_dep_the'].to_dict()
t('subject', ['nu_mo', 'nam_mo'], P, [('auto', {}), ('off', dict(subject='off')), ('child', dict(subject='child')),
   ('man', dict(subject='man')), ('woman', dict(subject='woman')), ('elder', dict(subject='elder')), ('adult', dict(subject='adult'))])
t('subject', ['nu_mo'], Q, [('auto', {}), ('off', dict(subject='off')), ('woman', dict(subject='woman')), ('elder', dict(subject='elder'))], 'muc1')
t('fidelity', ['nu_mo', 'nam_mo', 'the_mo'], P, [('0.7', {}), ('0', dict(fidelity=0.0)), ('0.35', dict(fidelity=0.35)), ('1', dict(fidelity=1.0))])
t('detail_fidelity', ['nu_mo', 'nam_mo'], P, [('None', {}), ('0', dict(detail_fidelity=0.0)), ('0.35', dict(detail_fidelity=0.35)), ('1', dict(detail_fidelity=1.0))])
t('detail_fidelity', ['nu_mo'], P, [('None', dict(keep_lips=False)), ('0', dict(detail_fidelity=0.0, keep_lips=False)), ('1', dict(detail_fidelity=1.0, keep_lips=False))], 'khong_giu_moi')
t('keep_lips', ['nu_mo', 'nam_mo'], P, [('True', {}), ('False', dict(keep_lips=False))])
t('keep_lips', ['nu_mo'], P, [('True', dict(fidelity=0.0, detail_fidelity=0.0)), ('False', dict(fidelity=0.0, detail_fidelity=0.0, keep_lips=False))], 'fid0')
t('skin_tone', ['nu_am', 'nu_mo'], P, [('-1 (hang 0.5)', {}), ('0', dict(skin_tone=0.0)), ('0.5', dict(skin_tone=0.5)), ('1', dict(skin_tone=1.0))])
t('gloss', ['nu_mo', 'the_mo'], P, [('0', {}), ('0.5', dict(gloss=0.5)), ('1', dict(gloss=1.0))])
t('deshine', ['nu_mo', 'the_mo'], P, [('0', {}), ('0.3', dict(deshine=0.3)), ('0.6', dict(deshine=0.6)), ('1', dict(deshine=1.0))])
t('exposure', ['the_toi'], P, [('1', {}), ('0', dict(exposure=0.0)), ('0.5', dict(exposure=0.5)), ('0 + Retinex', dict(exposure=0.0, low_light=True)), ('1 + Retinex', dict(exposure=1.0, low_light=True))])
t('exposure', ['the_mo'], P, [('1', {}), ('0', dict(exposure=0.0))], 'anh_thuong')
t('brightness', ['the_mo'], P, [('0', {}), ('-0.5', dict(brightness=-0.5)), ('0.25', dict(brightness=0.25)), ('0.5', dict(brightness=0.5))])
t('brightness', ['the_mo'], Q, [('0', {}), ('-0.5', dict(brightness=-0.5)), ('0.5', dict(brightness=0.5))], 'muc1')
t('contrast', ['the_nhat'], P, [('0', {}), ('0.33', dict(contrast=0.33)), ('0.67', dict(contrast=0.67)), ('1', dict(contrast=1.0))])
t('contrast', ['the_nhat'], Q, [('0', {}), ('0.33', dict(contrast=0.33)), ('0.67', dict(contrast=0.67)), ('1', dict(contrast=1.0))], 'muc1')
t('contrast', ['the_nhat'], dict(mode='upscale', upscale=2, sharpness=0.5), [('0', {}), ('0.67', dict(contrast=0.67)), ('1', dict(contrast=1.0))], 'muc2')
t('contrast', ['the_nhat'], O, [('0', {}), ('0.67', dict(contrast=0.67)), ('1', dict(contrast=1.0))], 'anh_cu')
t('tone', ['the_mo', 'nam_mo'], P, [('0.6', {}), ('0', dict(tone=0.0)), ('0.3', dict(tone=0.3)), ('1', dict(tone=1.0))])
t('low_light', ['the_toi'], P, [('False', {}), ('True', dict(low_light=True))])
t('low_light', ['the_toi'], Q, [('False', {}), ('True', dict(low_light=True))], 'muc1')
t('look', ['the_mo', 'nam_mo'], P, [('None', {}), ('trong_sang', dict(look=TL)), ('da_dep_the', dict(look=DL))])
t('look_amount', ['the_mo'], P, [('1 (trong_sang)', dict(look=TL)), ('0', dict(look=TL, look_amount=0.0)), ('0.5', dict(look=TL, look_amount=0.5))])
t('look_amount', ['the_mo'], P, [('1 (khong tong)', {}), ('0', dict(look_amount=0.0))], 'khong_tong')
t('phone', ['nu_am'], P, [('generic', {}), ('apple', dict(phone='apple')), ('samsung', dict(phone='samsung')), ('oppo', dict(phone='oppo')), ('off', dict(phone='off'))])
t('wb', ['nu_am', 'nu_mo'], P, [('-1 (hang 0.6)', {}), ('0', dict(wb=0.0)), ('0.3', dict(wb=0.3)), ('1', dict(wb=1.0))])
t('color', ['nu_mo', 'the_mo'], P, [('0', {}), ('0.3', dict(color=0.3)), ('0.6', dict(color=0.6)), ('1', dict(color=1.0))])
t('sharpness', ['nu_mo', 'the_mo'], P, [('0.5', {}), ('0', dict(sharpness=0.0)), ('0.25', dict(sharpness=0.25)), ('1', dict(sharpness=1.0))])
t('sharpness', ['the_mo'], Q, [('0.4', {}), ('0', dict(sharpness=0.0)), ('1', dict(sharpness=1.0))], 'muc1')
t('denoise', ['nam_nhieu'], P, [('-1 (tu dong)', {}), ('0', dict(denoise=0.0)), ('4', dict(denoise=4.0)), ('10', dict(denoise=10.0))])
t('denoise', ['nam_nhieu'], Q, [('-1 (tu dong)', {}), ('0', dict(denoise=0.0)), ('4', dict(denoise=4.0)), ('10', dict(denoise=10.0))], 'muc1')
t('denoise', ['nam_nhieu'], dict(P, sr_model='plksr'), [('-1 (tu dong)', {}), ('0', dict(denoise=0.0)), ('10', dict(denoise=10.0))], 'plksr')
t('restore', ['anh_cu'], P, [('tat', {}), ('bat', dict(repair=1.0, fade=1.0, auto_rotate=True))])
t('scene_vivid', ['anh_cu'], P, [('tat (khoi phuc bat)', dict(repair=1.0, fade=1.0, auto_rotate=True)), ('bat', dict(repair=1.0, fade=1.0, auto_rotate=True, scene_vivid=1.0))])
t('bg_color', ['the_mo', 'nam_mo'], P, [('None', {}), ('trang', dict(bg_color=(255, 255, 255))), ('do', dict(bg_color=(40, 40, 200)))])
t('strength', ['nu_mo'], P, [('1', {}), ('0', dict(strength=0.0)), ('0.5', dict(strength=0.5))])
t('mode', ['nu_mo'], dict(upscale=2), [('portrait', dict(mode='portrait')), ('quick', dict(mode='quick')), ('manual', dict(mode='manual')),
   ('old', dict(mode='old')), ('upscale', dict(mode='upscale'))])
t('upscale', ['nu_mo'], P, [('2', {}), ('0 (tu dong)', dict(upscale=0)), ('1', dict(upscale=1)), ('4', dict(upscale=4))])
t('face_model', ['nu_mo', 'nam_mo'], dict(P, fidelity=0.3), [('gfpgan', {}), ('restoreformer', dict(face_model='restoreformer')), ('restoreformer_pp', dict(face_model='restoreformer_pp'))])
t('sr_model', ['the_mo'], dict(mode='upscale', upscale=2, sharpness=0.5), [('compact', {}), ('rrdb', dict(sr_model='rrdb')), ('plksr', dict(sr_model='plksr')), ('dat2', dict(sr_model='dat2'))])
lv = {n: level_settings(n, Settings(upscale=2)).to_dict() for n in range(1, 7)}
t('level', ['nu_mo'], {}, [('3', lv[3])] + [(str(n), lv[n]) for n in (1, 2, 4, 5, 6)])

t('skin_tone_x', ['nu_do', 'nu_vang'], dict(P, wb=0.0), [('0', dict(skin_tone=0.0)), ('0.5', dict(skin_tone=0.5)), ('1', dict(skin_tone=1.0))])
t('contrast_sau_sua', ['the_nhat'], P, [('0', {}), ('0.33', dict(contrast=0.33)), ('0.67', dict(contrast=0.67)), ('1', dict(contrast=1.0))])
t('contrast_sau_sua', ['the_nhat'], dict(mode='upscale', upscale=2, sharpness=0.5), [('0', {}), ('0.67', dict(contrast=0.67)), ('1', dict(contrast=1.0))], 'muc2')
t('denoise_sau_sua', ['nam_nhieu'], dict(P, sr_model='plksr'), [('-1 (tu dong)', {}), ('0', dict(denoise=0.0)), ('4', dict(denoise=4.0)), ('10', dict(denoise=10.0))], 'plksr')
t('scene_vivid', ['cu_nhom'], P, [('tat (khoi phuc bat)', dict(repair=1.0, fade=1.0, auto_rotate=True)), ('bat', dict(repair=1.0, fade=1.0, auto_rotate=True, scene_vivid=1.0))], 'canh')
t('scene_ai', ['cu_nhom'], P, [('tat (vivid bat)', dict(repair=1.0, fade=1.0, auto_rotate=True, scene_vivid=1.0)), ('bat', dict(repair=1.0, fade=1.0, auto_rotate=True, scene_vivid=1.0, scene_ai=1.0))])
t('determinism', ['nu_mo'], P, [('lan1', {}), ('lan2', {}), ('lan3', {})])
if ONLY: T = [x for x in T if x[0] in ONLY]
LAB = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)

def masks(out):
    """Mặt nạ vùng (da, mắt+mày, môi, cả mặt) từ ParseNet trên ảnh ra mặc định."""
    h, w = out.shape[:2]
    helper = eng._load_helper(1); helper.clean_all(); helper.read_image(out)
    helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    helper.align_warp_face(); helper.get_inverse_affine(None)
    M = {k: np.zeros((h, w), np.float32) for k in ('skin', 'eyes', 'lips', 'face')}
    groups = {'skin': (1,), 'eyes': (4, 5, 6, 7), 'lips': (10, 11, 12), 'face': (1, 2, 4, 5, 6, 7, 10, 11, 12)}
    for f, inv in zip(helper.cropped_faces, helper.inverse_affine_matrices):
        lab = eng._parse(helper, f)
        for k, cl in groups.items():
            m = cv2.warpAffine(np.isin(lab, cl).astype(np.float32), inv, (w, h))
            M[k] = np.maximum(M[k], m)
    for k in M: M[k] = M[k] > 0.5
    M['bg'] = ~cv2.dilate(M['face'].astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    b = np.zeros((h, w), bool); e = max(6, h // 40); b[:e] = b[-e:] = True; b[:, :e] = b[:, -e:] = True
    M['border'] = b
    return M

def metrics(x, M, ref=None):
    L = LAB(x); g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    ch = np.hypot(L[..., 1] - 128, L[..., 2] - 128)
    sk = M['skin']; Ls = L[..., 0][sk] * 100 / 255
    r = dict(L=float(L[..., 0].mean() * 100 / 255), Lstd=float(L[..., 0].std() * 100 / 255), chroma=float(ch.mean()),
             sharp=float(lap.var()), sharp_face=float(lap[M['face']].var()) if M['face'].any() else 0,
             noise_bg=float(classic.estimate_noise(x)),
             skinL=float(Ls.mean()) if sk.any() else 0, skin_a=float(L[..., 1][sk].mean() - 128) if sk.any() else 0,
             skin_b=float(L[..., 2][sk].mean() - 128) if sk.any() else 0,
             shine=float((Ls > np.percentile(Ls, 50) + 14).mean() * 100) if sk.any() else 0,
             skin_p99=float(np.percentile(Ls, 99)) if sk.any() else 0,
             skin_tex=float(np.abs(lap[sk]).mean()) if sk.any() else 0,
             border=[float(v) for v in x[M['border']].mean(0)])
    if ref is not None:
        d = np.abs(x.astype(np.float32) - ref.astype(np.float32)).mean(-1)
        for k in ('face', 'skin', 'eyes', 'lips', 'bg', 'border'):
            r['mad_' + k] = float(d[M[k]].mean()) if M[k].any() else 0
        r['mad'] = float(d.mean()); r['max'] = float(d.max())
    return r

rows = []
prev_rows = []
if os.path.exists(f'{OUT}/rows.json'):
    prev_rows = json.load(open(f'{OUT}/rows.json'))
for param, tag, im, base, vals in T:
    src = img(im); ref = None; M = None
    for i, (vlab, ov) in enumerate(vals):
        s = Settings(**{**base, **ov}) if base else Settings(**ov)
        t0 = time.perf_counter()
        out, rep = eng.process(src, s)
        dt = time.perf_counter() - t0
        fn = f'{param}{"_" + tag if tag else ""}__{im}__{i}.png'
        cv2.imwrite(f'{OUT}/{fn}', out)
        if i == 0:
            ref = out; M = masks(out)
        x = out if out.shape == ref.shape else cv2.resize(out, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        r = dict(param=param, tag=tag, image=im, value=vlab, default=i == 0, file=fn, seconds=round(dt, 1),
                 size=f'{out.shape[1]}x{out.shape[0]}', notes=rep.notes, subjects=rep.subjects, faces=rep.faces_restored,
                 face_model=rep.face_model, bg_model=rep.bg_model, mode_used=rep.mode, deshined=getattr(rep, 'deshined', 0),
                 **metrics(x, M, ref))
        rows.append(r)
        print(f"{param:16s} {tag:12s} {im:10s} {vlab:16s} mad={r['mad']:6.2f} face={r['mad_face']:6.2f} lips={r['mad_lips']:6.2f} "
              f"L={r['L']:5.1f} std={r['Lstd']:5.1f} C={r['chroma']:5.1f} sharp={r['sharp']:7.1f} nz={r['noise_bg']:5.2f} "
              f"skin=({r['skinL']:.1f},{r['skin_a']:.1f},{r['skin_b']:.1f}) shine={r['shine']:.2f} {dt:.0f}s", flush=True)
    json.dump(prev_rows + rows, open(f'{OUT}/rows.json', 'w'), ensure_ascii=False, indent=0)
