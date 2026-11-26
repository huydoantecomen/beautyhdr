import os, sys
os.environ["QT_QPA_PLATFORM"]="offscreen"; os.environ["HOME"]=sys.argv[1]
sys.path.insert(0,'/home/user/beautyhdr')
from PySide6.QtWidgets import QApplication
app=QApplication([])
from net_anh import gui
from net_anh.settings import Settings
from net_anh.levels import level_settings
from dataclasses import asdict
w=gui.MainWindow()
def diff(a,b):
    da,db=asdict(a),asdict(b); return {k:(da[k],db[k]) for k in da if da[k]!=db[k] and k!='face_styles'}
base=w.settings()
print("BASE", {k:v for k,v in asdict(base).items() if k!='face_styles'})
def probe(name, fn):
    w._apply_settings(Settings(mode='portrait', upscale=2)); b=w.settings(); fn(); a=w.settings()
    print(f"{name:28s}", diff(b,a) or "KHÔNG ĐỔI Settings")
probe("Trung thực AI", lambda: w.sl_fid.s.setValue(20))
probe("Trung thực mắt môi (khóa)", lambda: w.sl_detail.s.setValue(20))
probe("Mắt môi như cả mặt bỏ tick", lambda: (w.cb_detail_same.setChecked(False), w.sl_detail.s.setValue(20)))
probe("Giữ môi", lambda: w.cb_keep_lips.setChecked(False))
probe("Sửa màu da", lambda: w.sl_skin.s.setValue(90))
probe("Độ bóng", lambda: w.sl_gloss.s.setValue(50))
probe("Khử bóng tick", lambda: w.cb_deshine.setChecked(True))
probe("Tự sửa sáng", lambda: w.sl_expo.s.setValue(30))
probe("Độ sáng", lambda: w.sl_bright.s.setValue(30))
probe("Tương phản", lambda: w.sl_contrast.s.setValue(80))
probe("Tươi sáng", lambda: w.sl_tone.s.setValue(10))
probe("Retinex", lambda: w.cb_lowlight.setChecked(True))
probe("Tông mẫu", lambda: w.look_box.setCurrentIndex(1))
probe("Độ mạnh tông mẫu", lambda: w.sl_look.s.setValue(40))
probe("Máy chụp", lambda: w.phone.setCurrentIndex(3))
probe("Cân bằng trắng", lambda: w.sl_wb.s.setValue(10))
probe("Tăng màu", lambda: w.sl_color.s.setValue(50))
probe("Độ nét", lambda: w.sl_sharp.s.setValue(90))
probe("Khử nhiễu tự động bỏ", lambda: w.cb_denoise_auto.setChecked(False))
probe("Khử nhiễu (đang tự động)", lambda: w.sl_denoise.s.setValue(80))
probe("Khôi phục ảnh cũ", lambda: w.cb_restore.setChecked(True))
probe("Đổi nền", lambda: w.cb_bg.setChecked(True))
probe("Đối tượng", lambda: w.subject.setCurrentIndex(2))
probe("Cường độ chung", lambda: w.sl_strength.s.setValue(50))
probe("Chế độ", lambda: w.mode.setCurrentIndex(w.mode.findData('upscale')))
probe("Phóng to", lambda: w.scale.setCurrentIndex(w.scale.findData(4)))
probe("Model mặt", lambda: w.face_model.setCurrentIndex(w.face_model.findData('restoreformer_pp')))
probe("Model nền", lambda: w.bg_model.setCurrentIndex(w.bg_model.findData('plksr')))
probe("Thiết bị", lambda: w.device.setCurrentIndex(w.device.count()-1))
print("phone items", [w.phone.itemData(i) for i in range(w.phone.count())])
print("subject items", [w.subject.itemData(i) for i in range(w.subject.count())])
print("look items", [w.look_box.itemData(i) for i in range(w.look_box.count())])
print("mode items", [w.mode.itemData(i) for i in range(w.mode.count())])
# vòng: mức nét → giao diện → settings() có giữ đủ không
for n in range(1,7):
    for bm in ('portrait','old'):
        s=level_settings(n, Settings(upscale=2, mode=bm)).with_(**w._user_opts())
        w._apply_settings(s); g=w.settings(); d=diff(s,g)
        if d: print(f"Mức {n} (nền {bm}) mất qua giao diện:", d)
# preset
from net_anh.settings import PRESETS, apply_preset
for p in PRESETS:
    s=apply_preset(Settings(upscale=2), p); w._apply_settings(s); d=diff(s,w.settings())
    if d: print(f"Preset {p} mất:", d)
# enabled state
w._apply_settings(Settings(mode='portrait'))
print("sl_denoise enabled khi tự động:", w.sl_denoise.isEnabled(), "| sl_contrast enabled ở portrait:", w.sl_contrast.isEnabled())
w._apply_settings(Settings(mode='quick'))
print("ở quick: sl_fid enabled", w.sl_fid.isEnabled(), "face_model enabled", w.face_model.isEnabled())
