import cv2, numpy as np, sys
U='/mnt/project-files/uploads/hearth/'; O=sys.argv[1]
def load(f, L=640):
    im=cv2.imread(U+f); h,w=im.shape[:2]; k=L/max(h,w)
    return cv2.resize(im,(round(w*k),round(h*k)),interpolation=cv2.INTER_AREA)
rng=np.random.default_rng(0)
def degrade(im, d=2.0, noise=4):
    h,w=im.shape[:2]; s=cv2.resize(im,(round(w/d),round(h/d)),interpolation=cv2.INTER_AREA)
    s=cv2.resize(s,(w,h),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    s+=rng.normal(0,noise,s.shape); return np.clip(s,0,255).astype(np.uint8)
nu=load('64d373b5-c294-4885-9daa-a0b13090e61f')      # chân dung nữ
the=load('1a8bc253-0dbc-4685-9a94-f119906f2467')     # ảnh thẻ nữ trung niên, nền xanh
nam=load('f7806766-5879-4db8-a0ce-1ddccf64d162')     # ảnh thẻ nam nền trắng
cv2.imwrite(O+'/nu_mo.png', degrade(nu))
cv2.imwrite(O+'/the_mo.png', degrade(the))
cv2.imwrite(O+'/nam_mo.png', degrade(nam))
# ảnh tối: −2 EV gần đúng (tuyến tính hoá, nhân 0.25, gamma lại)
lin=(the_d:=degrade(the,1.6,3)).astype(np.float32)/255; lin=lin**2.2*0.22
cv2.imwrite(O+'/the_toi.png', np.clip((lin**(1/2.2))*255,0,255).astype(np.uint8))
# ám vàng đỏ (cho cân bằng trắng, sửa màu da)
c=degrade(nu).astype(np.float32); c*=np.array([0.80,0.97,1.12]); cv2.imwrite(O+'/nu_am.png', np.clip(c,0,255).astype(np.uint8))
# nhiễu mạnh
n=degrade(nam,1.5,0).astype(np.float32)+rng.normal(0,14,nam.shape); cv2.imwrite(O+'/nam_nhieu.png', np.clip(n,0,255).astype(np.uint8))
# tương phản thấp (sương)
the_f=degrade(the,1.6,2).astype(np.float32)*0.55+90; cv2.imwrite(O+'/the_nhat.png', np.clip(the_f,0,255).astype(np.uint8))
cv2.imwrite(O+'/anh_cu.png', load('021fc0be-7e1e-4787-8a5f-8391fe39dfa3'))
for f in ['nu_mo','the_mo','nam_mo','the_toi','nu_am','nam_nhieu','the_nhat','anh_cu']:
    print(f, cv2.imread(O+'/'+f+'.png').shape)
