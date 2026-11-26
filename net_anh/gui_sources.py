"""Hộp thoại của Nguồn xử lý (sources.py): cài đặt Máy chủ GPU và khoá API ngoài, hỏi đồng ý gửi ảnh ra khỏi máy,
hỏi trước khi tải model DDColor 912 MB. Tách khỏi gui.py cho gọn; chỉ dùng PySide6 + sources/providers."""
from __future__ import annotations

import platform

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget)

from . import providers, sources, weights

APP_NAME = "Beauty HDR"

PRIVACY_API = (
    "Ảnh sẽ rời khỏi máy này và được gửi tới máy chủ của <b>{label}</b> ({provider}).<br>"
    "• App bỏ sạch EXIF (vị trí GPS, tên máy, thời gian) và thu nhỏ theo giới hạn của dịch vụ trước khi gửi.<br>"
    "• Dịch vụ có thể lưu ảnh một thời gian theo điều khoản riêng của họ (Replicate ghi xoá sau 1 giờ, Stability giữ "
    "kết quả 24 giờ). Đừng gửi ảnh giấy tờ, ảnh trẻ em hay ảnh riêng tư nếu không chắc.<br>"
    "• Có tính phí vào tài khoản của bạn: <b>{price}</b>.<br>"
    "• Kết quả được kiểm giữ khuôn mặt trên máy này; mặt bị đổi hoặc dịch vụ lỗi thì app tự xử lý lại bằng máy này.<br>"
    "• Dịch vụ ngoài chỉ làm nét/phóng to: khử bóng da, tông mẫu, đổi nền, khôi phục ảnh cũ của app không áp dụng."
)
PRIVACY_SERVER = (
    "Ảnh sẽ được gửi tới máy chủ Beauty HDR <b>{url}</b> ({name}) để xử lý rồi tải kết quả về.<br>"
    "• Ảnh chỉ giữ trong RAM máy chủ và tự xoá sau 15 phút.<br>"
    "• Trong mạng nhà, kết nối http không mã hoá; ở ngoài nên dùng Tailscale hoặc địa chỉ https.<br>"
    "• Chỉ dùng máy chủ của chính bạn hoặc người bạn tin."
)
DDCOLOR_ASK = ("Tô màu AI dùng model DDColor, cần tải khoảng 912 MB một lần (lưu trong thư mục model, xoá được ở Quản lý "
               "model). Mỗi ảnh chậm thêm vài chục giây, nên có GPU.\n\nTải và bật Tô màu AI?")


def ask_ddcolor(parent) -> bool:
    """Hỏi trước khi bật Tô màu AI lần đầu (model chưa tải). Đã có model → True, không hỏi."""
    if weights.status("ddcolor") == "ready":
        return True
    b = QMessageBox.question(parent, APP_NAME, DDCOLOR_ASK, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                             QMessageBox.StandardButton.No)
    return b == QMessageBox.StandardButton.Yes


class ConsentDialog(QDialog):
    """Hỏi đồng ý gửi ảnh ra khỏi máy. Nút mặc định là Không gửi."""

    def __init__(self, parent, html: str, remember_text: str):
        super().__init__(parent)
        self.setWindowTitle("Gửi ảnh ra khỏi máy?")
        lay = QVBoxLayout(self)
        lab = QLabel(html)
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setMinimumWidth(460)
        lay.addWidget(lab)
        self.remember = QCheckBox(remember_text)
        lay.addWidget(self.remember)
        bb = QDialogButtonBox()
        self.b_yes = bb.addButton("Đồng ý gửi", QDialogButtonBox.ButtonRole.AcceptRole)
        b_no = bb.addButton("Không gửi", QDialogButtonBox.ButtonRole.RejectRole)
        b_no.setDefault(True)
        b_no.setFocus()
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


def consent_api(parent, e: providers.Engine, session_ok: set) -> bool:
    """API ngoài: hỏi mỗi lần, trừ khi đã tích 'không hỏi lại' trong lần mở app này (không lưu xuống đĩa)."""
    if e.id in session_ok:
        return True
    d = ConsentDialog(parent, PRIVACY_API.format(label=e.label, provider=e.provider, price=e.price),
                      "Không hỏi lại cho dịch vụ này đến khi đóng app")
    if d.exec() != QDialog.DialogCode.Accepted:
        return False
    if d.remember.isChecked():
        session_ok.add(e.id)
    return True


def consent_server(parent, cfg: dict) -> bool:
    """Máy chủ GPU: hỏi lần đầu với mỗi địa chỉ; tích 'không hỏi lại' thì nhớ địa chỉ đó (remote.json)."""
    url = cfg.get("url", "")
    if url in (cfg.get("trusted") or []):
        return True
    d = ConsentDialog(parent, PRIVACY_SERVER.format(url=url, name=cfg.get("name") or "?"),
                      "Đây là máy chủ của tôi, không hỏi lại")
    if d.exec() != QDialog.DialogCode.Accepted:
        return False
    if d.remember.isChecked():
        cfg["trusted"] = sorted(set(cfg.get("trusted") or []) | {url})
        sources.save_remote(cfg)
    return True


class SourceDialog(QDialog):
    """Cài đặt nguồn xử lý: tab Máy chủ GPU (địa chỉ, ghép nối mã 6 số) và tab API ngoài (khoá do người dùng nhập)."""

    def __init__(self, parent, tab: str = "server"):
        super().__init__(parent)
        self.setWindowTitle("Nguồn xử lý")
        self.setMinimumWidth(540)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._server_tab(), "Máy chủ GPU")
        tabs.addTab(self._api_tab(), "API ngoài")
        tabs.setCurrentIndex(1 if tab == "api" else 0)
        lay.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)

    # --- máy chủ GPU
    def _server_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        intro = QLabel("Gửi ảnh tới một máy chủ Beauty HDR khác (PC có GPU mạnh trong nhà, hoặc GPU thuê). Trên máy chủ "
                       "mở <b>Beauty HDR - Máy chủ cho điện thoại</b>; cửa sổ đó hiện địa chỉ và mã ghép nối 6 số.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.cfg = sources.load_remote()
        f = QFormLayout()
        self.ed_url = QLineEdit(self.cfg.get("url", ""))
        self.ed_url.setPlaceholderText("vd 192.168.1.20 hoặc https://may-chu.ts.net")
        b_check = QPushButton("Kiểm tra")
        row = QHBoxLayout()
        row.addWidget(self.ed_url, 1)
        row.addWidget(b_check)
        f.addRow("Địa chỉ", row)
        self.ed_code = QLineEdit()
        self.ed_code.setPlaceholderText("6 số trên cửa sổ máy chủ")
        self.ed_code.setMaxLength(6)
        b_pair = QPushButton("Ghép nối")
        row2 = QHBoxLayout()
        row2.addWidget(self.ed_code, 1)
        row2.addWidget(b_pair)
        f.addRow("Mã ghép nối", row2)
        v.addLayout(f)
        self.srv_status = QLabel()
        self.srv_status.setWordWrap(True)
        v.addWidget(self.srv_status)
        b_forget = QPushButton("Bỏ ghép nối")
        v.addWidget(b_forget, 0, Qt.AlignmentFlag.AlignLeft)
        v.addStretch(1)
        b_check.clicked.connect(self._check)
        b_pair.clicked.connect(self._pair)
        b_forget.clicked.connect(self._forget)
        self._show_server()
        return w

    def _show_server(self, extra: str = ""):
        c = self.cfg
        if c.get("token"):
            t = f"Đã ghép nối với <b>{c.get('name') or c.get('url')}</b> ({c.get('device') or '?'})."
        else:
            t = "Chưa ghép nối máy chủ nào."
        self.srv_status.setText(t + ("<br>" + extra if extra else ""))

    def _url(self) -> str | None:
        try:
            return sources.normalize_url(self.ed_url.text())
        except sources.SourceError as e:
            self._show_server(f"⚠ {e}")
            return None

    def _check(self):
        url = self._url()
        if not url:
            return
        try:
            info = sources.server_info(url)
        except sources.SourceError as e:
            self._show_server(f"⚠ {e}")
            return
        feat = info.get("features") or []
        note = "" if "settings" in feat else ("<br>Máy chủ bản cũ (trước 0.16.0): chỉ nhận mức nét và vài tuỳ chọn; "
                                               "nên cập nhật máy chủ để ra ảnh giống chạy trên máy.")
        self._show_server(f"✓ Thấy máy chủ <b>{info.get('name')}</b>: {info.get('device')}, bản {info.get('version')}, "
                          f"{'GPU' if info.get('gpu') else 'chỉ CPU'}, đang chờ {info.get('queue', 0)} ảnh." + note)

    def _pair(self):
        url = self._url()
        if not url:
            return
        code = self.ed_code.text().strip()
        if not (len(code) == 6 and code.isdigit()):
            self._show_server("⚠ Mã ghép nối gồm 6 chữ số")
            return
        try:
            info = sources.server_info(url)
            token = sources.pair(url, code, "PC " + platform.node())
        except sources.SourceError as e:
            self._show_server(f"⚠ {e}")
            return
        self.cfg.update(url=url, token=token, name=info.get("name", ""), device=info.get("device", ""),
                        features=info.get("features") or [])
        sources.save_remote(self.cfg)
        self.ed_code.clear()
        self._show_server("✓ Ghép nối xong. Chọn Nguồn xử lý: Máy chủ GPU để dùng.")

    def _forget(self):
        self.cfg = {"trusted": []}
        sources.save_remote(self.cfg)
        self._show_server()

    # --- API ngoài
    def _api_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        warn = QLabel("⚠ Dùng dịch vụ ngoài là ảnh rời khỏi máy và có tính phí vào tài khoản của bạn. Khoá API do bạn tự "
                      "tạo trên trang của dịch vụ, chỉ lưu trên máy này (~/.net_anh/providers.json), không gửi cho ai "
                      "khác. Mỗi lần làm nét app sẽ hỏi lại trước khi gửi ảnh.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #e0a85a;")
        v.addWidget(warn)
        f = QFormLayout()
        self.api_prov = QComboBox()
        for key, label, _ in sources.API_PROVIDERS:
            self.api_prov.addItem(label, key)
        f.addRow("Dịch vụ", self.api_prov)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("dán khoá API vào đây")
        f.addRow("Khoá API", self.api_key)
        self.api_model = QLineEdit()
        self.api_model.setPlaceholderText("owner/name[:version], vd nightmareai/real-esrgan")
        f.addRow("Model Replicate", self.api_model)
        self.api_link = QLabel()
        self.api_link.setOpenExternalLinks(True)
        f.addRow("", self.api_link)
        v.addLayout(f)
        row = QHBoxLayout()
        b_save, b_del = QPushButton("Lưu khoá"), QPushButton("Xoá khoá")
        row.addWidget(b_save)
        row.addWidget(b_del)
        row.addStretch(1)
        v.addLayout(row)
        self.api_status = QLabel()
        self.api_status.setWordWrap(True)
        v.addWidget(self.api_status)
        v.addStretch(1)
        self.api_prov.currentIndexChanged.connect(self._api_show)
        b_save.clicked.connect(self._api_save)
        b_del.clicked.connect(self._api_delete)
        self._api_show()
        return w

    def _api_show(self, *_):
        prov = self.api_prov.currentData()
        cfg = providers.load_config()
        sec = cfg.get(prov) or {}
        self.api_key.clear()
        self.api_key.setPlaceholderText(f"đã lưu {sources.mask_key(sec['key'])}" if sec.get("key") else "dán khoá API vào đây")
        self.api_model.setText(sec.get("model", "") if prov == "replicate" else "")
        self.api_model.setEnabled(prov == "replicate")
        url = next(u for k, _, u in sources.API_PROVIDERS if k == prov)
        self.api_link.setText(f'<a href="{url}">Tạo khoá API trên trang của dịch vụ</a>')
        ready = [e.label for e in providers.engines(cfg).values()]
        self.api_status.setText("Sẵn dùng: " + (", ".join(ready) if ready else "chưa có dịch vụ nào có khoá"))

    def _api_save(self):
        prov = self.api_prov.currentData()
        key = self.api_key.text().strip()
        cfg = providers.load_config()
        if not key and not (cfg.get(prov) or {}).get("key"):
            self.api_status.setText("⚠ Chưa nhập khoá")
            return
        extra = {"model": self.api_model.text().strip()} if prov == "replicate" else None
        sources.save_api_key(prov, key or None, extra)
        self._api_show()

    def _api_delete(self):
        sources.save_api_key(self.api_prov.currentData(), "")
        self._api_show()


__all__ = ["SourceDialog", "ConsentDialog", "consent_api", "consent_server", "ask_ddcolor"]
