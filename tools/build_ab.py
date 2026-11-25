"""Cắt ảnh so sánh cùng toạ độ cho trang chấm mù A/B. Ghi ab/pairs.json (đáp án) + ab/img/*.jpg."""
import sys, json, random, cv2, numpy as np
sys.path.insert(0, '/home/user/work/tools')
import facemetrics as fm
from pathlib import Path
W = Path('/home/user/work'); OUT = W / 'ab/img'; OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(2511)
pairs = []
def save(img, name, maxw=720):
    if img.shape[1] > maxw:
        img = cv2.resize(img, (maxw, round(img.shape[0] * maxw / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90]); return 'img/' + name
def same_size(ims):
    H, Wd = max(i.shape[0] for i in ims), max(i.shape[1] for i in ims)
    return [cv2.resize(i, (Wd, H), interpolation=cv2.INTER_CUBIC if i.shape[0] < H else cv2.INTER_AREA) for i in ims]
def face_boxes(ref, k=1.9):
    out = []
    for lm in fm.landmarks(ref)[0]:
        ed = np.linalg.norm(lm[0] - lm[1]); c = lm[:2].mean(0) * 0.6 + lm[3:].mean(0) * 0.4
        r = k * ed; out.append((int(c[0] - r), int(c[1] - r * 1.05), int(c[0] + r), int(c[1] + r * 0.95), ed))
    return out
def crop(img, b):
    x0, y0, x1, y1 = [max(0, v) for v in b[:4]]; return img[y0:y1, x0:x1]
def add(group, q, pid, ref, a, b, la, lb, note=''):
    sides = [(a, la), (b, lb)]; rng.shuffle(sides)
    pairs.append(dict(group=group, q=q, id=pid, ref=ref, left=sides[0][0], right=sides[1][0],
                      left_is=sides[0][1], right_is=sides[1][1], note=note))
# G1 mức AI
fid = W / 'out/fid'; cap = W / 'out/fid_cap'
for name, gt, faces in [('mo_chandung_trungnien', 'chandung_trungnien', [0]), ('mo_9mat', 'sach_9mat', [2, 6]),
                        ('nhom_mua', None, [0, 2]), ('nhom_nu_dungnghieng', None, [0, 1])]:
    vs = {v: cv2.imread(str(fid / f'{name}__{v}.png')) for v in ('A_ai30', 'B_ai50', 'C_ai70')}
    src = cv2.imread(str(W / 'set' / f'{(gt or name)}.png'))
    ims = same_size([src] + list(vs.values())); ref, vs = ims[0], dict(zip(vs, ims[1:]))
    bx = face_boxes(ref)
    for fi in faces:
        if fi >= len(bx): continue
        b = bx[fi]; tag = f'{name}_f{fi}'
        r = save(crop(ref, b), f'{tag}_goc.jpg', 480)
        p = {v: save(crop(im, b), f'{tag}_{v}.jpg', 480) for v, im in vs.items()}
        q = 'Mặt nào giống người trong ảnh gốc hơn, mà vẫn rõ nét?'
        add('Mặt: mức trộn AI', q, tag + '_AB', r, p['A_ai30'], p['B_ai50'], '30 % AI (mức 3)', '50 % AI (mức 4)',
            'Gốc là ảnh sạch trước khi làm mờ' if gt else 'Gốc là ảnh vào phóng to')
        add('Mặt: mức trộn AI', q, tag + '_BC', r, p['B_ai50'], p['C_ai70'], '50 % AI (mức 4)', '70 % AI (mức 5, Rõ nét tối đa)',
            'Gốc là ảnh sạch trước khi làm mờ' if gt else 'Gốc là ảnh vào phóng to')
        cp = cap / f'{name}__C_ai70.png'
        if name == 'nhom_nu_dungnghieng' and cp.exists():
            c2 = same_size([ref, cv2.imread(str(cp))])[1]
            add('Mặt nhỏ: giới hạn AI', 'Mặt nào giống người trong ảnh gốc hơn?', tag + '_cap', r, p['C_ai70'],
                save(crop(c2, b), f'{tag}_cap.jpg', 480), '70 % AI như 0.10', 'Giới hạn AI theo cỡ mặt (0.11)')
# G2 hậu xử lý
post = W / 'out/post'
for name, gt in [('mo_chandung_trungnien', 'chandung_trungnien'), ('chandung_nu_xanh', None)]:
    vs = {v: cv2.imread(str(post / f'{name}__{v}.png')) for v in ('P0_hien_tai', 'P1_khong_bong', 'P4_ket_hop')}
    if any(v is None for v in vs.values()): continue
    src = cv2.imread(str(W / 'set' / f'{(gt or name)}.png'))
    ims = same_size([src] + list(vs.values())); ref, vs = ims[0], dict(zip(vs, ims[1:]))
    b = face_boxes(ref, 2.4)[0]
    r = save(crop(ref, b), f'post_{name}_goc.jpg', 560)
    p = {v: save(crop(im, b), f'post_{name}_{v}.jpg', 560) for v, im in vs.items()}
    q = 'Da mặt nào tự nhiên hơn? Xem má, trán, chân tóc, đường hàm.'
    add('Da và viền mặt', q, f'post_{name}_01', r, p['P0_hien_tai'], p['P1_khong_bong'], 'Hiện tại (độ bóng da 0,5)', 'Tắt độ bóng da')
    add('Da và viền mặt', q, f'post_{name}_04', r, p['P0_hien_tai'], p['P4_ket_hop'], 'Hiện tại', 'Bóng da 0,25, mịn da ½, viền mặt mềm hơn')
# G3 vá xước
sc = W / 'out/scratch'
for name, box in [('chandung_trungnien', (180, 330, 560, 470)), ('cu_giadinh', (300, 330, 1130, 620)), ('cu_lophoc', (150, 250, 700, 520))]:
    src = cv2.imread(str(W / 'set' / f'{name}.png'))
    a, bb = cv2.imread(str(sc / f'{name}__va_goc.png')), cv2.imread(str(sc / f'{name}__va_moi.png'))
    r = save(crop(src, box), f'va_{name}_goc.jpg')
    add('Vá xước, bụi', 'Bản vá nào giữ mắt, lông mày, môi đúng như ảnh gốc mà vẫn sạch bụi?', f'va_{name}', r,
        save(crop(a, box), f'va_{name}_v010.jpg'), save(crop(bb, box), f'va_{name}_v011.jpg'), 'Dò vết 0.10', 'Dò vết 0.11 (giữ ngũ quan)')
# G4 màu phai
col = W / 'out/color'
from net_anh import restore
for name in ['cu_phai_bocon', 'cu_phai_nhom', 'cu_giadinh', 'cu_mua_phai', 'cu_nhom_ngoai', 'cu_vochong_hong',
             'phai_chandung_trungnien', 'amvang_chandung_trungnien', 'amam_chandung_trungnien', 'amam_nhom_bongda']:
    src = cv2.imread(str(W / 'set' / f'{name}.png'))
    a = cv2.imread(str(col / f'{name}__v1.jpg')); bb = cv2.imread(str(col / f'{name}__v2.jpg'))
    if a is None or bb is None:
        a = restore.fade_restore_v1(src)[0]; bb = restore.fade_restore(src, faces=fm.landmarks(src)[0])[0]
    gt = name.split('_', 1)[1] if name.startswith(('phai_', 'amvang_', 'amam_')) else None
    refimg = src if (gt is None or name.startswith('amam_')) else cv2.imread(str(W / 'set' / f'{gt}.png'))
    note = {'amam': 'Ảnh chụp dưới nắng chiều có chủ ý: gốc là chính ảnh vào, nên giữ sắc ấm',
            'phai': 'Ảnh giả lập phai màu: gốc là ảnh sạch', 'amvang': 'Ảnh giả lập ám vàng: gốc là ảnh sạch'}.get(name.split('_')[0], 'Ảnh phai thật: gốc là ảnh vào')
    add('Màu ảnh phai', 'Bản nào màu da, áo, nền tự nhiên hơn? Môi không đỏ gắt, da không xám.', f'mau_{name}',
        save(refimg, f'mau_{name}_goc.jpg', 640), save(a, f'mau_{name}_v010.jpg', 640), save(bb, f'mau_{name}_v011.jpg', 640),
        'Phục hồi màu 0.10', 'Phục hồi màu 0.11 (giữ màu da)', note)
json.dump(pairs, open(W / 'ab/pairs.json', 'w'), ensure_ascii=False, indent=1)
print(len(pairs), 'cặp')
