import sys, json, math
sys.path.insert(0, '/home/user/beautyhdr')
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
OUT=sys.argv[1]; DST=sys.argv[2]
rows=json.load(open(f'{OUT}/rows.json'))
F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
FB=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
from net_anh.pipeline import Enhancer
eng=Enhancer(device='cpu')
def hue(r): return math.degrees(math.atan2(r['skin_b'], r['skin_a']))
M={  # khoá: (tiêu đề, hàm chỉ số)
 'L': lambda r: f"sáng L {r['L']:.1f}", 'Lstd': lambda r: f"tương phản {r['Lstd']:.1f}", 'chroma': lambda r: f"độ rực {r['chroma']:.1f}",
 'sharp': lambda r: f"độ nét {r['sharp']:.0f}", 'sharp_face': lambda r: f"nét mặt {r['sharp_face']:.0f}", 'noise': lambda r: f"nhiễu {r['noise_bg']:.2f}",
 'skin': lambda r: f"da L{r['skinL']:.0f} a{r['skin_a']:.0f} b{r['skin_b']:.0f}", 'hue': lambda r: f"góc màu da {hue(r):.0f}°",
 'shine': lambda r: f"điểm bóng {r['shine']:.2f} %", 'face': lambda r: f"lệch mặt {r['mad_face']:.2f}", 'lips': lambda r: f"lệch môi {r['mad_lips']:.2f}",
 'eyes': lambda r: f"lệch mắt {r['mad_eyes']:.2f}", 'mad': lambda r: f"lệch ảnh {r['mad']:.2f}", 'border': lambda r: "nền BGR " + ",".join(f"{v:.0f}" for v in r['border']),
 'size': lambda r: r['size'], 'tex': lambda r: f"vân da {r['skin_tex']:.2f}",
}
def crop_box(img, mode):
    h,w=img.shape[:2]
    if mode=='full': return 0,0,w,h
    f=eng.detect_faces(img)
    if not f: return 0,0,w,h
    b=max(f, key=lambda x:(x[0][2]-x[0][0]))[0]; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; s=(b[2]-b[0])*(1.0 if mode=='face' else 1.7)
    x0,y0=int(max(0,cx-s)),int(max(0,cy-s*1.1)); x1,y1=int(min(w,cx+s)),int(min(h,cy+s*0.95))
    return x0,y0,x1,y1
def sheet(name, title, param, tag, image, mode, keys, H=330):
    rs=[r for r in rows if r['param']==param and r['tag']==tag and r['image']==image]
    seen=set(); rs=[r for r in rs if not (r['value'] in seen or seen.add(r['value']))]
    if not rs: print('THIẾU', name); return
    base=cv2.imread(f"{OUT}/{rs[0]['file']}"); x0,y0,x1,y1=crop_box(base, mode); bh,bw=base.shape[:2]
    tiles=[]
    for r in rs:
        im=cv2.imread(f"{OUT}/{r['file']}")
        if im.shape[:2]!=(bh,bw): im=cv2.resize(im,(bw,bh),interpolation=cv2.INTER_AREA)
        c=im[y0:y1,x0:x1]; k=H/c.shape[0]; c=cv2.resize(c,(round(c.shape[1]*k),H),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_CUBIC)
        p=Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)); W=max(p.width,230)
        t=Image.new('RGB',(W,H+30+24*len(keys)),'white'); t.paste(p,((W-p.width)//2,30)); d=ImageDraw.Draw(t)
        d.text((6,4), ('★ ' if r['default'] else '')+r['value'], font=FB, fill=(180,0,0) if r['default'] else (0,0,0))
        for i,kk in enumerate(keys): d.text((6,H+34+24*i), M[kk](r), font=F, fill=(20,20,20))
        tiles.append(t)
    W=sum(t.width for t in tiles)+8*(len(tiles)+1); Hh=max(t.height for t in tiles)+46
    out=Image.new('RGB',(W,Hh),(235,235,235)); d=ImageDraw.Draw(out); d.text((8,8),title,font=FB,fill=(0,0,0))
    x=8
    for t in tiles: out.paste(t,(x,40)); x+=t.width+8
    out.save(f'{DST}/{name}.jpg', quality=88); print('ok', name, out.size)
J=[l.split('|') for l in '''
01-doi-tuong|Đối tượng (nữ, mức 3)|subject||nu_mo|face|skin,face
02-trung-thuc-ai|Trung thực AI (nam, mức 3)|fidelity||nam_mo|face|sharp_face,face
03-mat-moi|Trung thực mắt, môi khi tắt Giữ môi (nữ)|detail_fidelity|khong_giu_moi|nu_mo|face|eyes,lips
04-giu-moi|Giữ dáng, màu môi gốc (nữ, Trung thực 0)|keep_lips|fid0|nu_mo|face|lips
05-sua-mau-da|Sửa màu da trên da ngả đỏ (Cân bằng trắng tắt)|skin_tone_x||nu_do|face|hue,skin
06-sua-mau-da-vang|Sửa màu da trên da ngả vàng (Cân bằng trắng tắt)|skin_tone_x||nu_vang|face|hue,skin
07-do-bong-da|Độ bóng da (ảnh thẻ)|gloss||the_mo|face|skin,tex
08-khu-bong-da|Khử bóng da (ảnh thẻ)|deshine||the_mo|face|shine,face
09-tu-sua-sang|Tự sửa sáng và Retinex (ảnh tối −2 EV)|exposure||the_toi|full|L,skin
10-do-sang|Độ sáng (mức 3)|brightness||the_mo|full|L
11-tuong-phan-truoc|Tương phản ở mức 3, bản 0.15.1 (không tác dụng)|contrast||the_nhat|full|Lstd,mad
12-tuong-phan-muc1|Tương phản ở mức 1 (có tác dụng)|contrast|muc1|the_nhat|full|Lstd,mad
13-tuong-phan-sau-sua|Tương phản ở mức 3 sau khi sửa|contrast_sau_sua||the_nhat|full|Lstd,mad
14-tuoi-sang|Tươi sáng (nam)|tone||nam_mo|full|skin,L
15-tong-mau|Tông mẫu (ảnh thẻ)|look||the_mo|full|skin,chroma
16-do-manh-tong-mau|Độ mạnh tông mẫu (Trong sáng)|look_amount||the_mo|full|skin,mad
17-may-chup|Máy chụp (ảnh ám vàng)|phone||nu_am|full|chroma,skin
18-can-bang-trang|Cân bằng trắng (ảnh ám vàng)|wb||nu_am|full|chroma,skin
19-tang-mau|Tăng màu (ảnh thẻ)|color||the_mo|full|chroma,sharp
20-do-net|Độ nét (mức 3, ảnh thẻ)|sharpness||the_mo|half|sharp,mad
21-khu-nhieu|Khử nhiễu (mức 3, model compact)|denoise||nam_nhieu|half|noise,sharp
22-khu-nhieu-plksr-truoc|Khử nhiễu với RealPLKSR, bản 0.15.1 (không tác dụng)|denoise|plksr|nam_nhieu|half|noise,mad
23-khu-nhieu-plksr-sau|Khử nhiễu với RealPLKSR sau khi sửa|denoise_sau_sua|plksr|nam_nhieu|half|noise,mad
24-khoi-phuc|Khôi phục ảnh cũ|restore||anh_cu|full|chroma,sharp
25-mau-canh|Ảnh cảnh cũ: nâng màu cảnh|scene_vivid|canh|cu_nhom|full|chroma,L
26-to-mau-ai|Ảnh cảnh cũ: tô màu AI (DDColor)|scene_ai||cu_nhom|full|chroma,mad
27-doi-nen|Đổi nền (ảnh thẻ nền xanh)|bg_color||the_mo|full|border,face
28-cuong-do|Cường độ chung|strength||nu_mo|face|mad,sharp
29-che-do|Chế độ|mode||nu_mo|face|face,sharp
30-model-mat|Model mặt (Trung thực 0,3)|face_model||nu_mo|face|sharp_face,face
31-model-nen|Model nền (Phóng to AI)|sr_model||the_mo|half|sharp,noise
32-muc-net|Mức nét 1–6|level||nu_mo|face|sharp_face,face
33-phong-to|Phóng to|upscale||nu_mo|face|size,sharp
'''.strip().splitlines()]
only=sys.argv[3].split(',') if len(sys.argv)>3 else None
for n,title,p,tag,im,mode,keys in J:
    if only and not any(n.startswith(o) for o in only): continue
    sheet(n,title,p,tag,im,mode,keys.split(','))
