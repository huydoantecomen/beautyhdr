import cv2, numpy as np, sys, math
O=sys.argv[1]; im=cv2.imread(O+'/nu_mo.png')
def rot(im, deg, k=1.0):
    lab=cv2.cvtColor(im.astype(np.float32)/255,cv2.COLOR_BGR2LAB); a,b=lab[...,1].copy(),lab[...,2].copy(); r=math.radians(deg)
    lab[...,1]=(a*math.cos(r)-b*math.sin(r))*k; lab[...,2]=(a*math.sin(r)+b*math.cos(r))*k
    return (np.clip(cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),0,1)*255+.5).astype(np.uint8)
cv2.imwrite(O+'/nu_do.png', rot(im,-22,1.3)); cv2.imwrite(O+'/nu_vang.png', rot(im,18,1.3))
