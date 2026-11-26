"""Phần giao diện của bản nhà phát triển: menu "Nhà phát triển" và cửa sổ Xuất prompt cho AI ngoài.

Gắn vào cửa sổ chính bằng móc setup của net_anh.gui.main (xem net_anh_dev/__main__.py); bản thường không gọi móc nào
nên không có menu, phím tắt hay chữ nào của chức năng này.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout)

from . import prompt as P

QS_ON = "dev/prompt_on"  # công tắc chế độ nhà phát triển (mặc định tắt)
QS_PRIVACY = "dev/privacy_ok"  # đã đọc và đồng ý cảnh báo riêng tư
QS_TARGET, QS_LANG = "dev/prompt_target", "dev/prompt_lang"


def context_of(w) -> P.Context:
    """Mức nét, cỡ ảnh, phân tích và tên bộ preset của cửa sổ chính `w`."""
    n = w.level_group.checkedId()
    size = None
    if getattr(w, "src_img", None) is not None:
        h, wd = w.src_img.shape[:2]
        size = (int(wd), int(h))
    name = ""
    try:
        name = w.preset_lib.get(w.preset_set.currentData()).name
    except Exception:  # noqa: BLE001 - bộ preset bị xoá giữa chừng: bỏ tên
        pass
    return P.Context(level=n if n > 0 else None, size=size, analysis=w.analysis, preset_name=name)


class PromptDialog(QDialog):
    def __init__(self, w):
        super().__init__(w)
        self.w = w
        self.setWindowTitle("Xuất prompt cho AI ngoài (nhà phát triển)")
        self.resize(760, 680)
        lay = QVBoxLayout(self)
        warn = QLabel("⚠ " + P.PRIVACY_VI)
        warn.setWordWrap(True)
        warn.setStyleSheet("background:#4a3b1c; color:#f3e2b3; padding:8px; border:1px solid #7a6330;")
        lay.addWidget(warn)
        row = QHBoxLayout()
        self.target = QComboBox()
        for k, t in P.TARGETS.items():
            self.target.addItem(t.name, k)
        self.lang = QComboBox()
        for k, v in P.LANGS.items():
            self.lang.addItem(v, k)
        qs = w.qs
        self.target.setCurrentIndex(max(0, self.target.findData(str(qs.value(QS_TARGET, "chatgpt")))))
        self.lang.setCurrentIndex(max(0, self.lang.findData(str(qs.value(QS_LANG, "en")))))
        self.diag = QCheckBox("Kèm chẩn đoán ảnh")
        self.diag.setChecked(True)
        self.diag.setToolTip("Thêm một dòng: cỡ ảnh, độ mờ, nhiễu, ánh sáng, ám màu, số khuôn mặt")
        row.addWidget(QLabel("Hệ đích"))
        row.addWidget(self.target)
        row.addWidget(QLabel("Ngôn ngữ"))
        row.addWidget(self.lang)
        row.addWidget(self.diag)
        row.addStretch()
        lay.addLayout(row)
        self.src = QLabel()
        self.src.setObjectName("muted")
        self.src.setWordWrap(True)
        lay.addWidget(self.src)
        self.text = QPlainTextEdit()
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.text.setFont(f)
        self.text.textChanged.connect(self._count)
        lay.addWidget(self.text, 1)
        self.count = QLabel()
        lay.addWidget(self.count)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        lay.addWidget(self.note)
        self.dropped = QLabel()
        self.dropped.setWordWrap(True)
        self.dropped.setObjectName("muted")
        lay.addWidget(self.dropped)
        btns = QHBoxLayout()
        b_re = QPushButton("Tạo lại")
        b_re.setToolTip("Tạo lại prompt từ thông số đang chỉnh (bỏ chỗ đã sửa tay)")
        b_copy = QPushButton("Sao chép")
        b_copy.setDefault(True)
        b_file = QPushButton("Xuất file…")
        b_file.setToolTip(".txt: prompt đang xem kèm ghi chú; .json: mọi hệ × ngôn ngữ kèm thông số")
        b_close = QPushButton("Đóng")
        b_re.clicked.connect(self.refresh)
        b_copy.clicked.connect(self.copy)
        b_file.clicked.connect(self.export)
        b_close.clicked.connect(self.accept)
        btns.addWidget(b_re)
        btns.addStretch()
        btns.addWidget(b_copy)
        btns.addWidget(b_file)
        btns.addWidget(b_close)
        lay.addLayout(btns)
        self.target.currentIndexChanged.connect(self.refresh)
        self.lang.currentIndexChanged.connect(self.refresh)
        self.diag.toggled.connect(self.refresh)
        self.pack: P.PromptPack | None = None
        self.refresh()

    def _ctx(self) -> P.Context:
        c = context_of(self.w)
        c.diagnosis = self.diag.isChecked()
        return c

    def refresh(self, *_):
        tk, lg = self.target.currentData(), self.lang.currentData()
        self.w.qs.setValue(QS_TARGET, tk)
        self.w.qs.setValue(QS_LANG, lg)
        c = self._ctx()
        self.pack = P.build(self.w.settings(), tk, lg, c)
        self.text.setPlainText(self.pack.text)
        lv = f"mức nét {c.level}" if c.level else "mức nét theo thông số tay"
        img = f"ảnh {c.size[0]}×{c.size[1]} px" if c.size else "chưa mở ảnh (không có cỡ ảnh, chẩn đoán)"
        self.src.setText(f"Từ thông số đang chỉnh: {lv}, bộ preset “{c.preset_name or '—'}”, {img}.")
        self.note.setText("Cách dùng: " + self.pack.note)
        self.dropped.setText(("Bỏ bớt vì quá dài: " + " | ".join(self.pack.dropped)) if self.pack.dropped else "")
        self.dropped.setVisible(bool(self.pack.dropped))

    def _count(self):
        if self.pack is None:
            return
        n = len(self.text.toPlainText())
        over = n > self.pack.limit
        self.count.setText(f"{n}/{self.pack.limit} ký tự" + (" · dài hơn giới hạn gợi ý" if over else ""))
        self.count.setStyleSheet("color:#e0a040;" if over else "")

    def copy(self):
        QApplication.clipboard().setText(self.text.toPlainText())
        self.w.say(f"Đã sao chép prompt {P.TARGETS[self.pack.target].name} ({len(self.text.toPlainText())} ký tự).")

    def export(self):
        t = P.TARGETS[self.pack.target]
        base = f"prompt-{t.key}-{self.pack.lang}"
        path, flt = QFileDialog.getSaveFileName(self, "Xuất prompt", str(Path.home() / f"{base}.txt"),
                                                "Văn bản (*.txt);;Mọi hệ, kèm thông số (*.json)")
        if not path:
            return
        p = Path(path)
        if "json" in flt and p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        try:
            if p.suffix.lower() == ".json":
                import net_anh
                p.write_text(P.export_json(self.w.settings(), self._ctx(), net_anh.__version__), encoding="utf-8")
            else:
                pk = P.PromptPack(self.pack.target, self.pack.lang, self.text.toPlainText(), self.pack.limit,
                                  self.pack.dropped, self.pack.note)
                p.write_text(P.export_text(pk), encoding="utf-8-sig")  # BOM cho Notepad cũ
        except OSError as e:
            QMessageBox.warning(self, "Xuất prompt", f"Không ghi được file:\n{e}")
            return
        self.w.say(f"Đã xuất prompt ra {p}")


def _privacy_ok(w) -> bool:
    if str(w.qs.value(QS_PRIVACY, "")) == "1":
        return True
    r = QMessageBox.question(w, "Chế độ nhà phát triển",
                             P.PRIVACY_VI + "\n\nBật chức năng xuất prompt?",
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                             QMessageBox.StandardButton.No)
    if r != QMessageBox.StandardButton.Yes:
        return False
    w.qs.setValue(QS_PRIVACY, "1")
    return True


def install(w) -> None:
    """Thêm menu "Nhà phát triển" vào cửa sổ chính. Xuất prompt chỉ dùng được ở chế độ Chuyên sâu, sau khi bật công
    tắc (mặc định tắt) và đồng ý cảnh báo riêng tư."""
    m = w.menuBar().addMenu("Nhà phát triển")
    a_on = QAction("Bật xuất prompt cho AI ngoài", w, checkable=True)
    a_on.setChecked(str(w.qs.value(QS_ON, "0")) == "1")
    a_prompt = QAction("Xuất prompt cho AI ngoài…", w)
    a_prompt.setShortcut("Ctrl+Shift+P")
    a_prompt.setToolTip("Tạo prompt theo thông số đang chỉnh để dán sang ChatGPT, Gemini, Grok (chỉ ở Chuyên sâu)")
    m.addAction(a_on)
    m.addSeparator()
    m.addAction(a_prompt)
    w.a_dev_on, w.a_dev_prompt = a_on, a_prompt

    def sync(*_):
        a_prompt.setVisible(a_on.isChecked())
        a_prompt.setEnabled(a_on.isChecked() and getattr(w, "ui_mode", "simple") == "pro")

    def toggled(on: bool):
        if on and not _privacy_ok(w):
            a_on.blockSignals(True)
            a_on.setChecked(False)
            a_on.blockSignals(False)
            on = False
        w.qs.setValue(QS_ON, "1" if on else "0")
        sync()
        w.say("Xuất prompt cho AI ngoài: " + ("bật (menu Nhà phát triển, Ctrl+Shift+P, chế độ Chuyên sâu)" if on
                                              else "tắt"))

    def open_dialog():
        if not a_on.isChecked():
            return
        if w.ui_mode != "pro":
            w.say("Xuất prompt chỉ có ở chế độ Chuyên sâu (Ctrl+2).")
            return
        PromptDialog(w).exec()

    a_on.toggled.connect(toggled)
    a_prompt.triggered.connect(lambda _=False: open_dialog())
    orig = w.set_ui_mode

    def set_ui_mode(mode: str):
        orig(mode)
        sync()

    w.set_ui_mode = set_ui_mode
    for b in ("a_simple", "a_pro"):  # các nút, phím tắt đã nối với hàm cũ: nối thêm sync
        getattr(w, b).triggered.connect(lambda _=False: sync())
    for b in w.mode_btns.values():
        b.clicked.connect(lambda _=False: sync())
    sync()
    w.setWindowTitle(w.windowTitle() + " · bản nhà phát triển")
