"""Giao diện desktop (PySide6) theo đặc tả: chọn ảnh → phân tích → chọn chế độ/preset → xử lý (có huỷ)
→ so sánh trước/sau (zoom, kéo) → chỉnh, hoàn tác, khôi phục → xuất (không ghi đè ảnh gốc) → lịch sử."""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QDockWidget, QFrame, QGridLayout, QStackedWidget, QToolBar, QToolButton,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget, QDoubleSpinBox,
                               QColorDialog, QInputDialog, QLineEdit, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from . import facestyle, weights
from .subject import SUBJECTS
from .analysis import Analysis, recommend
from .device import list_devices
from . import hardware
from .history import History
from .levels import BG_MODELS, FACE_MODELS, LEVELS, auto_level, auto_upscale, level_for, level_settings, models_needed
from .levels import describe as describe_level
from .analysis import MAX_SIDE
from .pipeline import (OUT_SIZES, SUPPORTED, BatchTally, resize_long, Cancelled, Enhancer, ImageError, load_image, save_image, unique_output_path)
from .phone import PROFILES
from .phone import detect as detect_phone
from .phone import profile as phone_profile
from .settings import MODE_HINTS, MODES, PRESETS, Settings, apply_preset

APP_NAME = "Beauty HDR"
ICON_PATH = Path(__file__).parent / "assets" / "icon.png"  # icon cửa sổ, thanh tác vụ
FILTER = "Ảnh (" + " ".join("*" + e for e in sorted(SUPPORTED)) + ")"

# Chọn model thủ công (kể cả model nặng). Số đo: tools/model_bench.py; thời gian trên máy này: hardware.estimate.
FACE_MODEL_ITEMS = [
    ("gfpgan", "GFPGAN 1.4 (tự nhiên, nhanh)"),
    ("restoreformer_pp", "RestoreFormer++ (mới nhất, giữ đúng người nhất, chậm gấp đôi)"),
    ("restoreformer", "RestoreFormer (vẽ chi tiết mạnh hơn)"),
]
BG_MODEL_ITEMS = [
    ("compact", "Real-ESRGAN compact (nhẹ 5 MB, nhanh)"),
    ("rrdb", "Real-ESRGAN x2plus/x4plus (67 MB, chậm hơn)"),
    ("plksr", "RealPLKSR (30 MB, nặng vừa)"),
    ("dat2", "DAT-2 (140 MB, chất lượng cao nhất, rất nặng: nên có GPU)"),
]
# model_id trong weights.MANIFEST → khoá hardware.COST (để hiện tốc độ ước tính trong Quản lý model)
COST_OF = {"esrgan_compact": "compact", "esrgan_x4": "rrdb", "esrgan_x2": "rrdb", "plksr_web": "plksr",
           "dat2_web": "dat2", "gfpgan": "gfpgan", "restoreformer": "restoreformer",
           "restoreformer_pp": "restoreformer_pp"}
BEST_MODELS = ("restoreformer_pp", "dat2_web")  # bộ "Chất lượng cao nhất"


def torch_threads() -> int:
    import torch

    return torch.get_num_threads()


def to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------- so sánh trước/sau

class CompareView(QWidget):
    """Trước/sau cùng tỉ lệ. Kéo chuột trái: thanh chia. Cuộn: zoom. Chuột phải/giữa: kéo ảnh. Nháy đúp: vừa khung."""

    def __init__(self):
        super().__init__()
        self.before: QImage | None = None
        self.after: QImage | None = None
        self.split = 0.5
        self.zoom: float | None = None  # None = vừa khung
        self.center = QPointF(0.5, 0.5)  # tâm nhìn, theo toạ độ chuẩn hoá của ảnh
        self._pan_from: QPointF | None = None
        self.setMinimumSize(320, 240)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    after_label = ""  # model đã dùng cho ảnh Sau, hiện ở góc phải trên
    show_mode = "split"  # "before" | "split" | "after" (thanh dưới ảnh)

    def set_images(self, before: QImage | None, after: QImage | None, keep_view: bool = False):
        self.before, self.after = before, after
        if not keep_view:
            self.zoom, self.center = None, QPointF(0.5, 0.5)
        self.update()

    def _ref(self) -> QImage | None:
        return self.after or self.before

    def _fit_scale(self) -> float:
        ref = self._ref()
        return min(self.width() / ref.width(), self.height() / ref.height())

    def _target(self) -> QRectF:
        ref = self._ref()
        if ref is None:
            return QRectF()
        k = self.zoom or self._fit_scale()
        w, h = ref.width() * k, ref.height() * k
        if self.zoom is None:
            return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        return QRectF(self.width() / 2 - self.center.x() * w, self.height() / 2 - self.center.y() * h, w, h)

    def set_zoom(self, z: float | None):
        self.zoom = z
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(26, 26, 26))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, (self.zoom or 0) < 2)
        if self.before is None:
            p.setPen(QColor(170, 170, 170))
            p.setFont(QFont(p.font().family(), 13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kéo thả ảnh vào đây\nhoặc Tệp → Mở ảnh (Ctrl+O)")
            return
        r = self._target()
        p.drawImage(r, self.before)  # ảnh gốc được vẽ cùng khung với ảnh ra: cùng tỉ lệ
        if self.after is not None and self.show_mode == "after":
            p.drawImage(r, self.after)
        elif self.after is not None and self.show_mode == "split":
            sx = r.left() + r.width() * self.split
            p.save()
            p.setClipRect(QRectF(sx, 0, self.width(), self.height()))
            p.drawImage(r, self.after)
            p.restore()
            p.setPen(QPen(QColor(255, 255, 255, 220), 1))
            p.drawLine(QPointF(sx, 0), QPointF(sx, self.height()))
            p.setPen(QPen(QColor(34, 34, 34), 1))
            p.setBrush(QColor(230, 230, 230))
            p.drawRect(QRectF(sx - 5, self.height() / 2 - 14, 10, 28))
        if self.after is not None:
            fm = p.fontMetrics()
            tags = {"split": ("Trước", "Sau"), "before": ("Trước", ""), "after": ("", "Sau")}[self.show_mode]
            for i, t in enumerate(tags):
                if not t:
                    continue
                if i == 1 and self.after_label:
                    t += f" · {self.after_label}"
                tw = fm.horizontalAdvance(t)
                x = 8 if i == 0 else self.width() - tw - 20
                p.fillRect(QRectF(x, 8, tw + 12, fm.height() + 4), QColor(0, 0, 0, 165))
                p.setPen(QColor(230, 230, 230))
                p.drawText(QPointF(x + 6, 10 + fm.ascent()), t)
        k = (self.zoom or self._fit_scale()) * 100
        p.setPen(QColor(160, 160, 160))
        p.drawText(QPointF(10, self.height() - 10), f"{k:.0f}%")

    def _split_to(self, x: float):
        r = self._target()
        if r.width() > 0:
            self.split = min(max((x - r.left()) / r.width(), 0.0), 1.0)
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self.after is not None and self.show_mode != "split":
                self.window()._set_view("split")
            self._split_to(e.position().x())
        else:
            self._pan_from = e.position()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._split_to(e.position().x())
        elif self._pan_from is not None and self._ref() is not None:
            r = self._target()
            if self.zoom is None:
                self.zoom = self._fit_scale()
            d = e.position() - self._pan_from
            self._pan_from = e.position()
            self.center = QPointF(self.center.x() - d.x() / r.width(), self.center.y() - d.y() / r.height())
            self.update()

    def mouseReleaseEvent(self, e):
        self._pan_from = None

    def mouseDoubleClickEvent(self, e):
        self.set_zoom(None)

    def wheelEvent(self, e):
        if self._ref() is None:
            return
        r = self._target()
        pos = e.position()
        # giữ điểm dưới con trỏ đứng yên khi zoom
        u = (pos.x() - r.left()) / r.width()
        v = (pos.y() - r.top()) / r.height()
        old = self.zoom or self._fit_scale()
        new = float(np.clip(old * (1.25 if e.angleDelta().y() > 0 else 0.8), 0.05, 16))
        ref = self._ref()
        w, h = ref.width() * new, ref.height() * new
        self.center = QPointF(u - (pos.x() - self.width() / 2) / w, v - (pos.y() - self.height() / 2) / h)
        self.set_zoom(new)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.window().open_path(urls[0].toLocalFile())


# ---------------------------------------------------------------- việc chạy nền

def fmt_clock(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


class ProgressTrack:
    """Đổi các dòng chữ pipeline báo ra (Đang nạp…, Real-ESRGAN: 40%, khuôn mặt 2/5…) thành % và tên bước, đếm giờ,
    và cảnh báo khi chạy lâu hơn ước tính nhiều hoặc một bước đứng yên quá lâu (để người dùng không phải chờ mãi)."""
    STALL_S = 90.0

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.start(0.0)

    def start(self, eta_s: float = 0.0):
        self.t0 = self.t_last = self.clock()
        self.eta = float(eta_s or 0.0)
        self.pct, self.step = 0, "Chuẩn bị"
        self.batch = None  # (i, n) khi xử lý cả thư mục

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def _set(self, pct: float, step: str):
        if self.batch is not None:
            i, n = self.batch
            pct = 100 * (i - 1 + min(pct, 99) / 100) / n
        self.pct = max(self.pct, min(99, int(pct)))
        self.step = step
        self.t_last = self.clock()

    def feed(self, msg: str):
        m = msg.strip()
        if not m:
            return
        if (r := re.match(r"\[(\d+)/(\d+)\]\s*(.*)", m)):
            i, n = int(r[1]), int(r[2])
            self.batch = (i, n)
            self.pct = max(self.pct, int(100 * (i - 1) / n))
            self.step, self.t_last = f"Ảnh {i}/{n}", self.clock()
            return
        pre = f"Ảnh {self.batch[0]}/{self.batch[1]} · " if self.batch else ""
        if (r := re.match(r"(.+?):\s*(\d+)%$", m)):  # Real-ESRGAN: 40%  (từng hàng ô của model nền)
            self._set(12 + 0.58 * int(r[2]), pre + f"Làm nét nền {r[2]}%")
        elif (r := re.search(r"khuôn mặt (\d+)/(\d+)", m)):
            i, n = int(r[1]), int(r[2])
            self._set(74 + 20 * (i - 1) / n, pre + f"Mặt {i}/{n}")
        elif m.startswith("Đang nạp") or m.startswith("Tải ") or "đang tải" in m:
            self._set(3, pre + m.rstrip("…."))
        elif m.startswith("Đang cải thiện ảnh"):
            self._set(6, pre + "Cải thiện màu, sáng")
        elif m.startswith("Đang làm sạch"):
            self._set(9, pre + "Làm sạch ảnh cũ")
        elif m.startswith("Đang làm nét toàn ảnh"):
            self._set(12, pre + "Làm nét nền")
        elif m.startswith("Đang tìm khuôn mặt") or m.startswith("Đang nạp model tìm"):
            self._set(72, pre + "Tìm khuôn mặt")
        elif m.startswith("Đang ghép"):
            self._set(95, pre + "Ghép mặt vào ảnh")
        elif m.startswith("Đang"):
            self._set(self.pct, pre + m.rstrip("…."))

    def bar_text(self) -> str:
        t = f"{self.step} · {self.pct}% · {fmt_clock(self.elapsed())}"
        return t + (f" / ~{fmt_clock(self.eta)}" if self.eta >= 5 else "")

    def warning(self) -> str:
        el, still = self.elapsed(), self.clock() - self.t_last
        if self.eta >= 5 and el > 2.5 * self.eta + 30:
            return (f"Đã chạy {fmt_clock(el)}, lâu hơn ước tính (~{fmt_clock(self.eta)}) nhiều, đang ở bước «{self.step}». "
                    "Máy có thể đang bận hoặc thiếu bộ nhớ; chờ thêm hoặc bấm Huỷ rồi chọn mức nét thấp hơn.")
        if still > max(self.STALL_S, self.eta):
            return (f"Bước «{self.step}» chưa có tiến triển {fmt_clock(still)}. Model nặng trên CPU có thể lâu; "
                    "bấm Huỷ nếu muốn dừng.")
        return ""


class Job(QThread):
    message = Signal(str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop = False

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except (Cancelled, InterruptedError):
            self.cancelled.emit()
        except ImageError as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc(limit=4))


class Slider(QWidget):
    """Thanh trượt có nhãn giá trị; giá trị thật = vị trí / scale."""
    changed = Signal()

    def __init__(self, lo: int, hi: int, scale: float = 100.0, fmt: str = "{:.0f}%", mult: float = 100.0):
        super().__init__()
        self.scale, self.fmt, self.mult = scale, fmt, mult
        self.s = QSlider(Qt.Orientation.Horizontal)
        self.s.setRange(lo, hi)
        self.lbl = QLabel()
        self.lbl.setMinimumWidth(44)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.s)
        lay.addWidget(self.lbl)
        self.s.valueChanged.connect(self._upd)
        self._upd()

    def _upd(self):
        self.lbl.setText(self.fmt.format(self.value() * self.mult))
        self.changed.emit()

    def value(self) -> float:
        return self.s.value() / self.scale

    def set(self, v: float):
        self.s.blockSignals(True)
        self.s.setValue(round(v * self.scale))
        self.s.blockSignals(False)
        self.lbl.setText(self.fmt.format(self.value() * self.mult))


class ModelsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Quản lý model")
        self.resize(1040, 440)
        m = hardware.machine()
        self.kind = parent._device_kind() if hasattr(parent, "_device_kind") else "cpu"
        hw = QLabel(f"<b>Máy này:</b> {m.summary()}<br>{machine_advice(m, self.kind)}")
        hw.setWordWrap(True)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Model", "Công dụng", "Phiên bản", "Dung lượng", "Giấy phép",
                                              "Ước tính trên máy này", "Trạng thái"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        b_dl = QPushButton("Tải các model đang dùng")
        b_best = QPushButton("Tải bộ chất lượng cao nhất (RestoreFormer++, DAT-2)")
        b_best.setToolTip("Model mới nhất, nặng nhất (~430 MB). Chọn chúng ở mục Model mặt / Model nền, hoặc bấm mức 6.")
        b_del = QPushButton("Xoá model đã chọn")
        b_dl.clicked.connect(self.download_all)
        b_best.clicked.connect(self.download_best)
        b_del.clicked.connect(self.delete_selected)
        note = QLabel(f"Thư mục: {weights.weights_dir()}  ·  Mỗi file được kiểm tra SHA-256 sau khi tải.")
        note.setStyleSheet("color: gray;")
        row = QHBoxLayout()
        row.addWidget(b_dl)
        row.addWidget(b_best)
        row.addWidget(b_del)
        row.addStretch()
        lay = QVBoxLayout(self)
        lay.addWidget(hw)
        lay.addWidget(self.table)
        lay.addWidget(note)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        names = {"ready": "Sẵn sàng", "missing": "Chưa tải", "corrupt": "Hỏng, cần tải lại"}
        self.table.setRowCount(len(weights.MANIFEST))
        for r, m in enumerate(weights.MANIFEST.values()):
            for c, v in enumerate([m.file, m.purpose, m.version, f"{m.size / 1e6:.0f} MB", m.license,
                                   self._speed(m.model_id), names[weights.status(m.model_id)]]):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def _speed(self, model_id: str) -> str:
        key = COST_OF.get(model_id)
        if key is None:
            return ""
        if key in hardware.FACE_COSTS:
            t = hardware.estimate("", key, 0, 0, 1, kind=self.kind, threads=torch_threads(), overhead=False)
            return f"~{hardware.fmt_seconds(t)} mỗi mặt"
        up = 2 if model_id == "esrgan_x2" else 4
        t = hardware.estimate(key, None, 1600, 1200, kind=self.kind, threads=torch_threads(), upscale=up, overhead=False)
        return f"~{hardware.fmt_seconds(t)} mỗi ảnh 2 MP"

    def download_all(self):
        self.parent().download_models()
        self.accept()

    def download_best(self):
        self.parent().download_models(extra=BEST_MODELS)
        self.accept()

    def delete_selected(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        ids = list(weights.MANIFEST)
        for r in rows:
            weights.delete(ids[r])
        self.parent().invalidate_engine()
        self.refresh()


def machine_advice(m: hardware.Machine, kind: str) -> str:
    """Một câu khuyên dùng model theo cấu hình máy (Quản lý model)."""
    vram = max((g.vram_gb for g in m.gpus if g.usable), default=0.0)
    if kind == "cuda" and vram >= 6:
        return "GPU đủ mạnh: dùng được cả model nặng (RestoreFormer++, DAT-2) với thời gian vài chục giây mỗi ảnh."
    if kind != "cpu":
        return ("Có GPU nhưng bộ nhớ ít hoặc không rõ: model nặng vẫn chạy (app tự chia ô nhỏ), chậm hơn. "
                "Model nhẹ (mặc định) nhanh.")
    g = m.unused_gpu
    if g is not None:
        return hardware.advice(g) + " Model nhẹ (mặc định) vẫn chạy tốt trên CPU."
    return ("Máy chỉ có CPU: model nhẹ (mặc định) vài giây đến vài chục giây mỗi ảnh; model nặng chọn được nhưng mất "
            "vài phút đến hơn chục phút mỗi ảnh tuỳ cỡ ảnh, app báo thời gian ước tính trước khi chạy.")


# ---------------------------------------------------------------- giao diện 0.12: phong cách, biểu tượng, khối gọn

# Kiểu app chỉnh ảnh desktop (Lightroom, Capture One): xám trung tính để mắt không lệch màu, góc gần vuông, không
# bóng đổ / gradient, một màu nhấn xanh thép cho nút Làm nét và thanh tiến độ.
ACCENT = "#4a78a8"
STYLE = """
QWidget { background: #2e2e2e; color: #d0d0d0; }
QMainWindow, QDialog { background: #262626; }
QMenuBar { background: #1f1f1f; color: #bdbdbd; border-bottom: 1px solid #151515; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: #3a3a3a; color: #eeeeee; }
QMenu { background: #2b2b2b; border: 1px solid #444; padding: 3px 0; }
QMenu::item { padding: 4px 26px 4px 22px; }
QMenu::item:selected { background: #4b4b4b; color: #ffffff; }
QMenu::item:disabled { color: #6f6f6f; }
QMenu::separator { height: 1px; background: #3e3e3e; margin: 3px 0; }
QToolBar { background: #2b2b2b; border: 0; border-bottom: 1px solid #1b1b1b; spacing: 2px; padding: 3px 6px; }
QToolBar QLabel { background: transparent; color: #9a9a9a; padding: 0 6px; }
QToolBar::separator { background: #444; width: 1px; margin: 4px 6px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 2px; padding: 3px 7px; color: #cfcfcf; }
QToolButton:hover { background: #3a3a3a; border-color: #484848; }
QToolButton:disabled { color: #6f6f6f; }
QPushButton { background: #3a3a3a; border: 1px solid #4a4a4a; border-radius: 2px; padding: 3px 10px; color: #dadada; min-height: 18px; }
QPushButton:hover { background: #444444; }
QPushButton:pressed { background: #505050; }
QPushButton:disabled { color: #727272; background: #333333; border-color: #3e3e3e; }
QPushButton:checked { background: #5a5a5a; color: #ffffff; border-color: #5a5a5a; }
QPushButton#go { background: %(accent)s; border-color: #5a88b8; color: #ffffff; font-weight: 600; min-height: 24px; }
QPushButton#go:hover { background: #5383b4; }
QPushButton#go:disabled { background: #3d4b59; border-color: #3d4b59; color: #8b97a3; }
QPushButton#seg { border-radius: 0; margin: 0; padding: 3px 6px; background: #333333; color: #bdbdbd; }
QPushButton#seg:checked { background: #5a5a5a; color: #ffffff; }
QPushButton#seg:hover:!checked { background: #3c3c3c; color: #e6e6e6; }
QPushButton#lst { text-align: left; border: 0; border-radius: 0; background: transparent; padding: 0 10px; min-height: 28px; }
QPushButton#lst:hover { background: #383838; }
QPushButton#lst:checked { background: #4b4b4b; }
QPushButton#lnk, QPushButton#mod { background: transparent; border: 0; padding: 0 2px; color: #9cc3e8; min-height: 0; }
QPushButton#lnk:hover { color: #c4dcf2; text-decoration: underline; }
QPushButton#lnk:disabled { color: #6f6f6f; }
QPushButton#mod { color: #8a8a8a; font-size: 10pt; padding: 0 4px; }
QPushButton#mod:hover { color: #cfcfcf; }
QPushButton#mod:checked { color: #f0f0f0; background: transparent; }
QWidget#hd { background: #353535; border-top: 1px solid #1d1d1d; border-bottom: 1px solid #1d1d1d; }
QWidget#hd QLabel, QWidget#hd QCheckBox { background: transparent; color: #e2e2e2; font-weight: 600; }
QToolButton#hdb { color: #e2e2e2; font-weight: 600; border: 0; padding: 0; background: transparent; }
QWidget#panel, QScrollArea, QScrollArea > QWidget > QWidget { background: #2e2e2e; }
QWidget#center { background: #1a1a1a; }
QWidget#viewbar, QStatusBar { background: #262626; border-top: 1px solid #1b1b1b; }
QStatusBar QLabel { background: transparent; color: #9a9a9a; }
QStatusBar::item { border: 0; }
QLabel { background: transparent; }
QLabel#muted { color: #9a9a9a; }
QLabel#warn { color: #e0a85a; }
QLabel#rowlbl { color: #c4c4c4; }
QComboBox, QSpinBox, QLineEdit { background: #262626; border: 1px solid #4a4a4a; border-radius: 2px; padding: 2px 6px; color: #dadada; min-height: 18px; selection-background-color: #4b4b4b; }
QComboBox:disabled, QSpinBox:disabled { color: #727272; }
QComboBox::drop-down { border: 0; width: 16px; }
QComboBox QAbstractItemView { background: #2b2b2b; border: 1px solid #444; selection-background-color: #4b4b4b; }
QLineEdit#search { padding-left: 6px; }
QSlider::groove:horizontal { height: 2px; background: #5a5a5a; }
QSlider::handle:horizontal { width: 9px; height: 9px; margin: -4px 0; border-radius: 4px; background: #d6d6d6; }
QSlider::handle:horizontal:disabled { background: #6a6a6a; }
QSlider::sub-page:horizontal { background: #7a7a7a; }
QLabel#num { background: #232323; border: 1px solid #444; padding: 0 4px; color: #d8d8d8; min-width: 34px; }
QCheckBox { spacing: 7px; background: transparent; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #6a6a6a; background: #262626; }
QCheckBox::indicator:checked { background: #5b9bd5; border-color: #5b9bd5; }
QCheckBox:disabled { color: #727272; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: #292929; color: #8a8a8a; padding: 5px 14px; border: 0; border-bottom: 1px solid #1d1d1d; }
QTabBar::tab:selected { color: #f0f0f0; background: #2e2e2e; }
QListWidget, QTableWidget, QPlainTextEdit { background: #262626; border: 1px solid #3a3a3a; alternate-background-color: #2a2a2a; }
QListWidget::item { padding: 3px 6px; }
QListWidget::item:selected { background: #4b4b4b; color: #ffffff; }
QListWidget::item:hover:!selected { background: #383838; }
QHeaderView::section { background: #353535; border: 0; border-right: 1px solid #1d1d1d; padding: 3px 6px; }
QProgressBar { background: #3c3c3c; border: 0; max-height: 4px; min-height: 4px; }
QProgressBar::chunk { background: #5b9bd5; }
QScrollBar:vertical { background: #2a2a2a; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar:horizontal { background: #2a2a2a; height: 10px; }
QScrollBar::handle:horizontal { background: #4a4a4a; min-width: 24px; }
QSplitter::handle { background: #1b1b1b; }
QToolTip { background: #ececec; color: #1c1c1c; border: 0; padding: 4px; }
QDockWidget::title { background: #353535; padding: 4px 8px; }
""" % {"accent": ACCENT}


UI_SIZES = (("Nhỏ", 9.0), ("Vừa", 10.0), ("Lớn", 11.5), ("Rất lớn", 13.0))


def auto_ui_size(app: QApplication) -> float:
    """Màn to mà Windows để tỉ lệ 100 % (2560×1440 chẳng hạn): chữ 9 pt rất nhỏ, chọn sẵn cỡ lớn hơn."""
    scr = app.primaryScreen()
    if scr is None:
        return 9.0
    w = scr.geometry().width()  # điểm logic: đã chia cho tỉ lệ Windows
    return 11.5 if w >= 2400 else 10.0 if w >= 1800 else 9.0


def set_ui_font(app: QApplication, pt: float):
    f = QFont("Segoe UI")
    f.setPointSizeF(pt)
    app.setFont(f)
    Section.LABEL_W = round(112 * pt / 9.0)


def apply_style(app: QApplication, pt: float | None = None):
    """Fusion + bảng màu tối trung tính + QSS ở trên. Chữ Segoe UI 9 pt (Windows tự phóng theo tỉ lệ màn hình)."""
    from PySide6.QtGui import QPalette
    app.setStyle("Fusion")
    pal = QPalette()
    for role, c in ((QPalette.ColorRole.Window, "#2e2e2e"), (QPalette.ColorRole.WindowText, "#d0d0d0"),
                    (QPalette.ColorRole.Base, "#262626"), (QPalette.ColorRole.AlternateBase, "#2a2a2a"),
                    (QPalette.ColorRole.Text, "#d6d6d6"), (QPalette.ColorRole.Button, "#3a3a3a"),
                    (QPalette.ColorRole.ButtonText, "#dadada"), (QPalette.ColorRole.Highlight, "#4b4b4b"),
                    (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, "#ececec"),
                    (QPalette.ColorRole.ToolTipText, "#1c1c1c"), (QPalette.ColorRole.PlaceholderText, "#8a8a8a"),
                    (QPalette.ColorRole.Link, "#9cc3e8")):
        pal.setColor(role, QColor(c))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#727272"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#727272"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#727272"))
    app.setPalette(pal)
    set_ui_font(app, pt or 9.0)
    app.setStyleSheet(STYLE)


def line_icon(name: str, color: str = "#bdbdbd") -> QIcon:
    """Biểu tượng nét mảnh một màu, vẽ bằng QPainter (không cần file ảnh), sắc nét cả ở màn 150–200 %."""
    from PySide6.QtGui import QPainterPath
    px = QPixmap(48, 48)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(3, 3)  # vẽ trên lưới 16×16
    pen = QPen(QColor(color), 1.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    def poly(*pts):
        path = QPainterPath(QPointF(*pts[0]))
        for q in pts[1:]:
            path.lineTo(QPointF(*q))
        p.drawPath(path)

    if name == "open":
        p.drawRect(QRectF(1.5, 3, 13, 10))
        poly((1.5, 11), (5, 7.5), (8, 10.5), (10, 8.5), (14.5, 12.5))
        p.drawEllipse(QPointF(11, 6), 1, 1)
    elif name == "folder":
        poly((1.5, 3.5), (6, 3.5), (7.5, 5), (14.5, 5), (14.5, 13), (1.5, 13), (1.5, 3.5))
    elif name == "undo":
        poly((5, 9), (2, 6), (5, 3))
        path = QPainterPath(QPointF(2, 6))
        path.lineTo(10, 6)
        path.arcTo(QRectF(6.5, 6, 7, 7), 90, -180)
        path.lineTo(7, 13)
        p.drawPath(path)
    elif name == "close":
        poly((4, 4), (12, 12))
        poly((12, 4), (4, 12))
    elif name == "export":
        poly((8, 2), (8, 10))
        poly((5, 7), (8, 10), (11, 7))
        poly((2.5, 13.5), (13.5, 13.5))
    elif name == "auto":
        for a, b in (((8, 1.5), (8, 4.5)), ((8, 11.5), (8, 14.5)), ((1.5, 8), (4.5, 8)), ((11.5, 8), (14.5, 8)),
                     ((3.5, 3.5), (5.5, 5.5)), ((10.5, 10.5), (12.5, 12.5)), ((3.5, 12.5), (5.5, 10.5)),
                     ((10.5, 5.5), (12.5, 3.5))):
            poly(a, b)
    elif name == "portrait":
        p.drawEllipse(QPointF(8, 5.3), 2.8, 2.8)
        p.drawArc(QRectF(2.5, 9.2, 11, 9.6), 0, 180 * 16)
    elif name == "old":
        p.drawRect(QRectF(2, 3, 12, 10))
        poly((7, 3), (8.5, 7), (6, 9), (7.5, 13))
    elif name == "id":
        p.drawRect(QRectF(3.5, 1.5, 9, 13))
        p.drawEllipse(QPointF(8, 5.7), 1.8, 1.8)
        p.drawArc(QRectF(5.5, 9.1, 5, 4.8), 0, 180 * 16)
    elif name == "scene":
        poly((1.5, 13), (5.5, 7.5), (8.5, 11), (10.5, 8.5), (14.5, 13), (1.5, 13))
        p.drawEllipse(QPointF(11, 4.5), 1.2, 1.2)
    elif name == "dark":
        path = QPainterPath()
        path.addEllipse(QRectF(2, 2.5, 11, 11))
        cut = QPainterPath()
        cut.addEllipse(QRectF(5.5, 0.5, 10, 10))
        p.drawPath(path.subtracted(cut))
    elif name == "search":
        p.drawEllipse(QPointF(7, 7), 4.5, 4.5)
        poly((10.5, 10.5), (14, 14))
    elif name == "down":
        poly((3, 6), (8, 11), (13, 6))
    elif name == "up":
        poly((3, 10), (8, 5), (13, 10))
    p.end()
    px.setDevicePixelRatio(3)
    return QIcon(px)


def norm_text(s: str) -> str:
    """Chữ thường, bỏ dấu (tìm "da" ra "Sửa màu da", "nen" ra "Đổi nền")."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def header_bar(title: str) -> QWidget:
    w = QWidget()
    w.setObjectName("hd")
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    w.setFixedHeight(26)
    lay = QHBoxLayout(w)
    lay.setContentsMargins(10, 0, 10, 0)
    lay.addWidget(QLabel(title))
    lay.addStretch()
    return w


def seg_row(buttons) -> QWidget:
    """Nhóm nút liền khối (như nút chuyển trong Lightroom)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    for b in buttons:
        b.setObjectName("seg")
        b.setCheckable(True)
        b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        lay.addWidget(b)
    return w


class ListChoice(QPushButton):
    """Một dòng trong danh sách chọn một: biểu tượng, tên, gợi ý mờ bên phải."""

    def __init__(self, icon: str, text: str, hint: str):
        super().__init__()
        self.setObjectName("lst")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)
        ic = QLabel()
        ic.setPixmap(line_icon(icon).pixmap(16, 16))
        t, h = QLabel(text), QLabel(hint)
        h.setObjectName("muted")
        for x in (ic, t, h):
            x.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(ic)
        lay.addWidget(t)
        lay.addStretch()
        lay.addWidget(h)
        self.setMinimumHeight(28)


class Section(QWidget):
    """Nhóm thông số thu gọn được: tiêu đề (ô bật/tắt tuỳ chọn), Đặt lại; mỗi dòng = nhãn | ô chỉnh. Có lọc theo ô tìm."""
    LABEL_W = 112

    def __init__(self, title: str, check: QCheckBox | None = None, on_reset=None, opened: bool = True):
        super().__init__()
        self.title = title
        self.head = QWidget()
        self.head.setObjectName("hd")
        self.head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.head.setFixedHeight(26)
        hl = QHBoxLayout(self.head)
        hl.setContentsMargins(8, 0, 10, 0)
        hl.setSpacing(6)
        if check is not None:
            check.setText("")
            check.setToolTip(f"Bật/tắt {title}")
            hl.addWidget(check)
        self.btn = QToolButton()
        self.btn.setObjectName("hdb")
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        self.btn.setChecked(opened)
        self.btn.toggled.connect(self.set_open)
        hl.addWidget(self.btn)
        hl.addStretch()
        if on_reset is not None:
            r = QPushButton("Đặt lại")
            r.setObjectName("lnk")
            r.setToolTip(f"Đặt lại nhóm {title} về mặc định")
            r.clicked.connect(on_reset)
            hl.addWidget(r)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(10, 8, 10, 10)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnMinimumWidth(0, self.LABEL_W)
        self.grid.setColumnStretch(1, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.head)
        lay.addWidget(self.body)
        self.rows: list[tuple[str, list[QWidget]]] = []
        self.body.setVisible(opened)
        self._user_open = opened

    def add(self, label: str, widget: QWidget, keywords: str = ""):
        r = self.grid.rowCount()
        ws = [widget]
        if label:
            lab = QLabel(label)
            lab.setObjectName("rowlbl")
            lab.setWordWrap(True)
            lab.setFixedWidth(self.LABEL_W)
            self.grid.addWidget(lab, r, 0, Qt.AlignmentFlag.AlignVCenter)
            self.grid.addWidget(widget, r, 1)
            ws.insert(0, lab)
        else:
            self.grid.addWidget(widget, r, 1)
        self.rows.append((norm_text(f"{label} {keywords} {getattr(widget, 'text', lambda: '')() if isinstance(widget, (QCheckBox, QPushButton)) else ''}"), ws))
        return widget

    def add_full(self, widget: QWidget, keywords: str = ""):
        """Một dòng chiếm cả hai cột (thanh mức nét)."""
        self.grid.addWidget(widget, self.grid.rowCount(), 0, 1, 2)
        self.rows.append((norm_text(keywords), [widget]))
        return widget

    def set_open(self, on: bool):
        self.btn.blockSignals(True)
        self.btn.setChecked(on)
        self.btn.blockSignals(False)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)
        if not getattr(self, "_filtering", False):
            self._user_open = on

    def filter(self, q: str) -> bool:
        """Ẩn dòng không khớp; nhóm khớp thì mở ra. q rỗng: hiện tất cả, trả về trạng thái mở như người dùng để."""
        if not q:
            for _, ws in self.rows:
                for w in ws:
                    w.setVisible(True)
            self._filtering = True
            self.set_open(self._user_open)
            self._filtering = False
            self.setVisible(True)
            return True
        hit_title = q in norm_text(self.title)
        any_hit = False
        for key, ws in self.rows:
            ok = hit_title or q in key
            any_hit |= ok
            for w in ws:
                w.setVisible(ok)
        self.setVisible(any_hit)
        if any_hit:
            self._filtering = True
            self.set_open(True)
            self._filtering = False
        return any_hit


class Histogram(QWidget):
    """Biểu đồ độ sáng ba kênh của ảnh đang mở (như Lightroom), giúp thấy ảnh tối, cháy sáng, ám màu."""

    def __init__(self):
        super().__init__()
        self.h: np.ndarray | None = None
        self.setFixedHeight(72)

    def set_image(self, img: np.ndarray | None):
        if img is None:
            self.h = None
        else:
            small = resize_long(img, 384) if max(img.shape[:2]) > 384 else img
            hs = [cv2.calcHist([small], [c], None, [64], [0, 256]).ravel() for c in range(3)]
            m = max(float(np.percentile(np.concatenate(hs), 99)), 1.0)
            self.h = np.clip(np.stack(hs) / m, 0, 1)
        self.update()

    def paintEvent(self, _):
        from PySide6.QtGui import QPolygonF
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#222222"))
        if self.h is None:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height() - 2
        p.setPen(Qt.PenStyle.NoPen)
        for c, col in ((2, QColor(194, 91, 91, 115)), (1, QColor(91, 163, 91, 105)), (0, QColor(91, 127, 194, 115))):
            pts = [QPointF(0, h + 1)]
            n = len(self.h[c])
            pts += [QPointF(i * w / (n - 1), h + 1 - v * h) for i, v in enumerate(self.h[c])]
            pts.append(QPointF(w, h + 1))
            p.setBrush(col)
            p.drawPolygon(QPolygonF(pts))



# ---------------------------------------------------------------- cửa sổ chính

LOOK_TIP = ("Tông mẫu: đưa màu da, môi, tóc, vùng trắng và nền về gần tông của một bộ ảnh mẫu. Trong sáng: da sáng, "
            "trong, ít vàng đỏ; môi tươi; tóc đen; nền phẳng sạch (đo từ ảnh thẻ mẫu). Bộ preset .netanh.json có kèm "
            "tông mẫu sẽ hiện thêm ở đây. Mặc định: Giữ tông ảnh.")
DESHINE_DEFAULT = 0.6  # cường độ khử bóng da khi vừa tích ô (đo trên ảnh chân dung da dầu của huy)
BG_SWATCHES = (("Trắng", (255, 255, 255)), ("Xanh", (219, 142, 67)), ("Xanh nhạt", (235, 206, 135)),
               ("Đỏ", (30, 30, 200)), ("Xám kem", (203, 212, 217)))
SUBJECT_ITEMS = [("auto", "Tự động theo tuổi, giới tính"), ("child", "Trẻ em"), ("man", "Nam"), ("woman", "Nữ"),
                 ("elder", "Người lớn tuổi"), ("adult", "Người lớn (trung tính)"), ("off", "Tắt preset đối tượng")]


class MaskCanvas(QWidget):
    """Tô vùng hỏng lên ảnh: kéo chuột để tô đỏ, chế độ tẩy để xoá. Mặt nạ giữ ở cỡ ảnh gốc."""

    def __init__(self, img: np.ndarray, mask: np.ndarray | None):
        super().__init__()
        self.img = to_qimage(img)
        h, w = img.shape[:2]
        self.mask = (mask.copy() if mask is not None and mask.shape[:2] == (h, w) else np.zeros((h, w), np.uint8))
        self.brush = 20
        self.erase = False
        self.setMinimumSize(560, 420)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._last: tuple[int, int] | None = None

    def _geom(self):
        w, h = self.img.width(), self.img.height()
        k = min(self.width() / w, self.height() / h)
        ox, oy = (self.width() - w * k) / 2, (self.height() - h * k) / 2
        return k, ox, oy

    def _to_img(self, p):
        k, ox, oy = self._geom()
        return int((p.x() - ox) / k), int((p.y() - oy) / k)

    def _stamp(self, a, b):
        k, _, _ = self._geom()
        r = max(1, int(self.brush / k / 2))
        cv2.line(self.mask, a, b, 0 if self.erase else 255, 2 * r, cv2.LINE_AA)
        self.update()

    def mousePressEvent(self, e):
        self._last = self._to_img(e.position())
        self._stamp(self._last, self._last)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and self._last is not None:
            p = self._to_img(e.position())
            self._stamp(self._last, p)
            self._last = p

    def mouseReleaseEvent(self, e):
        self._last = None

    def paintEvent(self, e):
        k, ox, oy = self._geom()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))
        p.translate(ox, oy)
        p.scale(k, k)
        p.drawImage(0, 0, self.img)
        red = np.zeros((*self.mask.shape, 4), np.uint8)
        red[..., 0] = 255
        red[..., 3] = (self.mask > 127) * 150
        qi = QImage(red.data, red.shape[1], red.shape[0], red.shape[1] * 4, QImage.Format_RGBA8888)
        p.drawImage(0, 0, qi)


class DamageDialog(QDialog):
    """Hộp thoại "Vẽ vùng hỏng": người dùng tô lên chỗ bị lá vàng, vết ố, rách che mất chi tiết."""

    def __init__(self, parent, img: np.ndarray, mask: np.ndarray | None):
        super().__init__(parent)
        self.setWindowTitle("Vẽ vùng hỏng nặng")
        self.canvas = MaskCanvas(img, mask)
        lay = QVBoxLayout(self)
        tip = QLabel("Tô lên chỗ bị lá vàng, vết ố, rách che mất chi tiết (nhất là trên mặt). Tô sát vết hỏng, đừng tô "
                     "cả mặt: chỗ nào còn thấy rõ mắt, mũi, miệng thì để nguyên. App vá rồi dựng lại phần bị mất.")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        lay.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cỡ cọ"))
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(4, 120)
        self.size.setValue(20)
        self.size.valueChanged.connect(lambda v: setattr(self.canvas, "brush", v))
        row.addWidget(self.size, 1)
        self.cb_erase = QCheckBox("Tẩy")
        self.cb_erase.toggled.connect(lambda on: setattr(self.canvas, "erase", on))
        row.addWidget(self.cb_erase)
        b_clear = QPushButton("Xoá hết")
        b_clear.clicked.connect(self._clear)
        row.addWidget(b_clear)
        b_ok, b_cancel = QPushButton("Xong"), QPushButton("Huỷ")
        b_ok.clicked.connect(self.accept)
        b_cancel.clicked.connect(self.reject)
        row.addWidget(b_ok)
        row.addWidget(b_cancel)
        lay.addLayout(row)
        self.resize(900, 700)

    def _clear(self):
        self.canvas.mask[:] = 0
        self.canvas.update()

    def mask(self) -> np.ndarray:
        return self.canvas.mask


class PresetDialog(QDialog):
    """Quản lý bộ preset theo đối tượng: chọn bộ, sửa thông số (bộ mặc định chỉ xem, sửa thì tạo bản sao), đổi tên,
    xoá, Xuất ra file .netanh.json để gửi người khác, Nạp file người khác gửi."""

    def __init__(self, parent, lib: "facestyle.Library"):
        super().__init__(parent)
        self.setWindowTitle("Bộ preset theo đối tượng")
        self.resize(900, 520)
        self.lib = lib
        self.sets = QListWidget()
        self.sets.currentRowChanged.connect(self._show)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.table = QTableWidget(len(facestyle.SUBJECT_KEYS), len(facestyle.RANGES))
        self.table.setVerticalHeaderLabels([SUBJECTS[k] for k in facestyle.SUBJECT_KEYS])
        self.table.setHorizontalHeaderLabels([facestyle.KNOB_NAMES[k] for k in facestyle.RANGES])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setToolTip("Giữ nét thật: giữ thêm mặt gốc (đúng người, đúng tuổi). Ngũ quan: + sắc nét mắt, mày, môi; "
                              "− bớt AI. Mịn da, Bóng da, Sáng da: 0..1. Hồng hào: + ấm hồng, − mát hơn. Khử bóng: cộng vào "
                              "thanh Khử bóng da chung, chỉ có tác dụng khi ô Khử bóng da được tích.")
        self.spins: dict[tuple[str, str], QDoubleSpinBox] = {}
        for r, subj in enumerate(facestyle.SUBJECT_KEYS):
            for c, knob in enumerate(facestyle.RANGES):
                sp = QDoubleSpinBox()
                lo, hi = facestyle.RANGES[knob]
                sp.setRange(lo, hi)
                sp.setSingleStep(0.05)
                sp.setDecimals(2)
                sp.valueChanged.connect(self._edited)
                self.table.setCellWidget(r, c, sp)
                self.spins[(subj, knob)] = sp
        from . import look as lookmod
        self.look_pick = QComboBox()  # tông mẫu kèm bộ (bộ người dùng): không có / tông có sẵn / tông của chính bộ
        self.look_pick.setToolTip(LOOK_TIP)
        self._looks = lookmod.LOOKS
        self.look_pick.currentIndexChanged.connect(self._look_edited)
        b_use = QPushButton("Dùng bộ này")
        b_copy = QPushButton("Tạo bản sao…")
        b_rename = QPushButton("Đổi tên…")
        b_del = QPushButton("Xoá")
        b_export = QPushButton("Xuất ra file…")
        b_import = QPushButton("Nạp từ file…")
        b_close = QPushButton("Đóng")
        b_use.clicked.connect(self._use)
        b_copy.clicked.connect(self._copy)
        b_rename.clicked.connect(self._rename)
        b_del.clicked.connect(self._delete)
        b_export.clicked.connect(self._export)
        b_import.clicked.connect(self._import)
        b_close.clicked.connect(self.accept)
        left = QVBoxLayout()
        left.addWidget(QLabel("Các bộ preset trên máy"))
        left.addWidget(self.sets, 1)
        for b in (b_use, b_copy, b_rename, b_del, b_export, b_import):
            left.addWidget(b)
        right = QVBoxLayout()
        right.addWidget(self.info)
        right.addWidget(self.table, 1)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Tông mẫu kèm bộ:"))
        lrow.addWidget(self.look_pick, 1)
        right.addLayout(lrow)
        hint = QLabel("Mỗi khuôn mặt được đoán tuổi, giới tính rồi chỉnh theo dòng của nhóm mình (ảnh gia đình: mỗi người "
                      "một kiểu). File xuất (.netanh.json) gửi qua Zalo, email; người nhận bấm Nạp từ file, dùng được cả "
                      "trên điện thoại.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        right.addWidget(hint)
        right.addWidget(b_close)
        root = QHBoxLayout(self)
        root.addLayout(left, 1)
        root.addLayout(right, 3)
        self._loading = False
        self._reload(lib.active_id())

    def _reload(self, select: str | None = None):
        self.all = self.lib.sets()
        active = self.lib.active_id()
        self.sets.blockSignals(True)
        self.sets.clear()
        for st in self.all:
            mark = " ✓ đang dùng" if st.id == active else ""
            self.sets.addItem(st.name + (" (có sẵn)" if st.builtin else "") + mark)
        self.sets.blockSignals(False)
        ids = [st.id for st in self.all]
        self.sets.setCurrentRow(ids.index(select) if select in ids else 0)
        self._show(self.sets.currentRow())

    def current(self):
        r = self.sets.currentRow()
        return self.all[r] if 0 <= r < len(self.all) else None

    def _show(self, _row: int):
        st = self.current()
        if st is None:
            return
        self._loading = True
        for (subj, knob), sp in self.spins.items():
            sp.setValue(getattr(st.style(subj), knob))
            sp.setEnabled(not st.builtin)
        self.look_pick.clear()
        self.look_pick.addItem("Không (giữ tông ảnh)", None)
        for k, lk in self._looks.items():
            self.look_pick.addItem(lk.name, k)
        idx = 0
        if st.look:
            match = [k for k, lk in self._looks.items() if lk.to_dict() == st.look]
            if match:
                idx = self.look_pick.findData(match[0])
            else:
                self.look_pick.addItem(f"{st.look.get('name', 'Tông riêng')} (của bộ này)", "__own__")
                idx = self.look_pick.count() - 1
        self.look_pick.setCurrentIndex(idx)
        self.look_pick.setEnabled(not st.builtin)
        self._loading = False
        meta = " · ".join(x for x in (st.author and f"Tác giả: {st.author}", st.created and f"Ngày: {st.created}") if x)
        self.info.setText(f"<b>{st.name}</b>" + (f"<br>{meta}" if meta else "") + (f"<br>{st.note}" if st.note else "")
                          + ("<br><i>Bộ có sẵn chỉ xem được; bấm Tạo bản sao để sửa.</i>" if st.builtin else ""))

    def _edited(self, *_):
        st = self.current()
        if self._loading or st is None or st.builtin:
            return
        for subj in facestyle.SUBJECT_KEYS:
            st.styles[subj] = facestyle.FaceStyle(**{k: self.spins[(subj, k)].value() for k in facestyle.RANGES})
        self.lib.save(st)

    def _look_edited(self, *_):
        st = self.current()
        if self._loading or st is None or st.builtin:
            return
        k = self.look_pick.currentData()
        if k == "__own__":
            return
        st.look = self._looks[k].to_dict() if k else None
        self.lib.save(st)

    def _use(self):
        st = self.current()
        if st is not None:
            self.lib.set_active(st.id)
            self._reload(st.id)

    def _copy(self):
        st = self.current()
        if st is None:
            return
        name, ok = QInputDialog.getText(self, "Tạo bản sao", "Tên bộ preset mới:", QLineEdit.EchoMode.Normal,
                                        f"{st.name} (bản của tôi)")
        if ok and name.strip():
            c = st.copy(name.strip()[: facestyle.MAX_NAME])
            c.author = ""
            self.lib.save(c)
            self.lib.set_active(c.id)
            self._reload(c.id)

    def _rename(self):
        st = self.current()
        if st is None or st.builtin:
            return
        name, ok = QInputDialog.getText(self, "Đổi tên", "Tên mới:", QLineEdit.EchoMode.Normal, st.name)
        if ok and name.strip():
            st.name = name.strip()[: facestyle.MAX_NAME]
            self.lib.save(st)
            self._reload(st.id)

    def _delete(self):
        st = self.current()
        if st is None or st.builtin:
            return
        if QMessageBox.question(self, APP_NAME, f"Xoá bộ preset “{st.name}”?") == QMessageBox.StandardButton.Yes:
            self.lib.delete(st.id)
            self._reload()

    def _export(self):
        st = self.current()
        if st is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Xuất bộ preset", facestyle.safe_filename(st.name),
                                           f"Bộ preset Beauty HDR (*{facestyle.EXT})")
        if not f:
            return
        if not f.endswith(facestyle.EXT):
            f = f.removesuffix(".json") + facestyle.EXT
        try:
            self.lib.export([st.id], f, app=f"Beauty HDR PC {__import__('net_anh').__version__}")
        except (OSError, facestyle.PresetError) as e:
            QMessageBox.warning(self, APP_NAME, f"Không xuất được: {e}")
            return
        QMessageBox.information(self, APP_NAME, f"Đã xuất {Path(f).name}. Gửi file này cho người khác, họ bấm "
                                                "“Nạp từ file…” để dùng.")

    def _import(self):
        f, _ = QFileDialog.getOpenFileName(self, "Nạp bộ preset", "", f"Bộ preset Beauty HDR (*{facestyle.EXT} *.json)")
        if not f:
            return
        try:
            got, warn = self.lib.import_file(f)
        except (OSError, UnicodeDecodeError, facestyle.PresetError) as e:
            QMessageBox.warning(self, APP_NAME, f"File không hợp lệ: {e}")
            return
        msg = "Đã nạp: " + ", ".join(st.name for st in got)
        if warn:
            msg += "\n\nĐã tự sửa:\n" + "\n".join(warn[:12])
        QMessageBox.information(self, APP_NAME, msg)
        self._reload(got[0].id if got else None)


KIND_ITEMS = [  # chế độ Đơn giản: loại ảnh → preset gốc
    ("auto", "auto", "Tự động", "app tự nhận"),
    ("portrait", "portrait", "Chân dung", "rõ mặt, giữ đúng người"),
    ("old", "old", "Ảnh cũ", "vá xước, hồi màu"),
    ("id", "id", "Ảnh thẻ, đổi nền", "nền đồng màu"),
    ("scene", "scene", "Phong cảnh, đồ vật", "phóng to cả ảnh"),
    ("dark", "dark", "Ảnh tối, ngược sáng", "sửa sáng, giảm nhiễu"),
]
KIND_PRESET = {"auto": "auto", "portrait": "max", "old": "restore", "id": "portrait", "scene": "auto",
               "dark": "low_light"}
UI_MODES = ("simple", "pro")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.setMinimumSize(980, 620)
        self._engine: Enhancer | None = None
        self._engine_dev = None
        self.history = History()
        self.src_path: Path | None = None
        self.src_img: np.ndarray | None = None
        self.analysis: Analysis | None = None
        self.results: list[tuple[np.ndarray, Settings, str, float]] = []  # ngăn xếp để hoàn tác
        self.job: Job | None = None
        self._models_used: dict[int, str] = {}  # id(ảnh kết quả) → model đã chạy, ghi vào lịch sử
        self.level_choice: int | None = None  # mức nét người dùng bấm chọn (None = theo preset / Tự động)
        self.simple_kind, self.simple_power = "auto", "auto"
        from PySide6.QtCore import QSettings
        self.qs = QSettings("BeautyHDR", "BeautyHDR")
        self._build()
        self._apply_settings(Settings())
        self._show_level(level_for("auto", Settings()), Settings())
        self.set_ui_mode(str(self.qs.value("ui_mode", "simple")))
        self._update_buttons()
        self.say("Sẵn sàng. Mở một ảnh (Ctrl+O) hoặc kéo thả vào giữa. Chế độ AI tải model ở lần đầu (khoảng 610 MB).")

    # ----- dựng giao diện -----

    def _act(self, text, fn, shortcut=None, icon=None, tip=None) -> QAction:
        a = QAction(text, self)
        if icon:
            a.setIcon(line_icon(icon))
        if shortcut:
            a.setShortcut(shortcut)
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(lambda _=False: fn())
        return a

    def _build(self):
        # --- lệnh dùng chung cho menu, thanh công cụ, phím tắt
        self.a_open = self._act("Mở ảnh…", self.choose_file, "Ctrl+O", "open", "Mở ảnh (Ctrl+O)")
        self.a_folder = self._act("Xử lý cả thư mục…", self.choose_folder, None, "folder", "Làm nét cả một thư mục ảnh")
        self.a_export = self._act("Xuất ảnh…", self.save_result, "Ctrl+S", "export",
                                  "Lưu ảnh đã làm nét ra file mới (Ctrl+S), không ghi đè ảnh gốc")
        self.a_close = self._act("Đóng ảnh", self.close_image, "Ctrl+W", "close",
                                 "Đóng ảnh đang mở (Ctrl+W); kết quả chưa xuất sẽ mất")
        self.a_undo = self._act("Hoàn tác", self.undo, "Ctrl+Z", "undo", "Quay về kết quả trước (Ctrl+Z)")
        self.a_restore = self._act("Khôi phục ảnh gốc", self.restore_original)
        self.a_reset = self._act("Đặt lại thông số", self._reset_adjust)
        self.a_run = self._act("Làm nét", self.run_single, "Ctrl+Return")
        self.a_cancel = self._act("Huỷ", self.cancel, "Esc")
        self.a_damage = self._act("Vẽ vùng hỏng nặng…", self._draw_damage)
        self.a_models = self._act("Quản lý model…", lambda: ModelsDialog(self).exec())
        self.a_presets = self._act("Bộ preset theo đối tượng…", self._manage_presets)
        self.a_simple = self._act("Chế độ Đơn giản", lambda: self.set_ui_mode("simple"), "Ctrl+1")
        self.a_pro = self._act("Chế độ Chuyên sâu", lambda: self.set_ui_mode("pro"), "Ctrl+2")
        self.a_hist = self._act("Lịch sử", self.show_history, "Ctrl+H")
        for a in (self.a_simple, self.a_pro):
            a.setCheckable(True)

        # --- thanh menu
        mb = self.menuBar()
        m = mb.addMenu("Tệp")
        m.addActions([self.a_open, self.a_folder])
        m.addSeparator()
        m.addActions([self.a_export, self.a_close])
        m.addSeparator()
        m.addAction(self._act("Thoát", self.close, "Ctrl+Q"))
        m = mb.addMenu("Sửa")
        m.addActions([self.a_undo, self.a_restore, self.a_reset])
        m = mb.addMenu("Ảnh")
        m.addActions([self.a_run, self.a_cancel])
        m.addSeparator()
        m.addAction(self.a_damage)
        m = mb.addMenu("Xem")
        m.addActions([self.a_simple, self.a_pro])
        m.addSeparator()
        for text, z, key in (("Vừa khung", None, "Ctrl+0"), ("Phóng 100%", 1.0, None), ("Phóng 200%", 2.0, None)):
            m.addAction(self._act(text, lambda z=z: self._zoom(z), key))
        m.addSeparator()
        fm = m.addMenu("Cỡ chữ giao diện")
        fg = QActionGroup(self)
        cur = float(self.qs.value("ui_pt", 0) or 0) or QApplication.font().pointSizeF()
        for name, pt in UI_SIZES:
            a = QAction(f"{name} ({pt:g} pt)", self, checkable=True)
            a.setChecked(abs(pt - cur) < 0.1)
            a.triggered.connect(lambda _=False, pt=pt: self.set_font_size(pt))
            fg.addAction(a)
            fm.addAction(a)
        m.addSeparator()
        m.addAction(self.a_hist)
        self.log_dock = QDockWidget("Nhật ký", self)
        self.log_dock.setObjectName("log")
        m.addAction(self.log_dock.toggleViewAction())
        m = mb.addMenu("Model")
        m.addActions([self.a_models, self.a_presets])
        m = mb.addMenu("Trợ giúp")
        m.addAction(self._act("Giới thiệu", lambda: QMessageBox.about(
            self, APP_NAME, f"{APP_NAME} PC {__import__('net_anh').__version__}\nLàm nét, khôi phục ảnh chân dung "
                            "bằng AI chạy trên máy.\n\nĐơn giản (Ctrl+1): chọn loại ảnh, mức nét, bấm Làm nét.\n"
                            "Chuyên sâu (Ctrl+2): đủ mọi thông số, preset, lịch sử.")))
        # đổi chế độ: hai nút liền khối ở đầu cột phải (0.12.2: chữ ở góc thanh menu dễ bị che bởi lớp phủ FPS/GPU
        # của MSI Afterburner, Steam… và khó thấy)
        self.mode_btns = {}
        grp = QButtonGroup(self)
        for key, text in (("simple", "Đơn giản"), ("pro", "Chuyên sâu")):
            b = QPushButton(text)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip("Chuyển sang chế độ " + text + (" (Ctrl+1)" if key == "simple" else " (Ctrl+2)"))
            b.clicked.connect(lambda _=False, k=key: self.set_ui_mode(k))
            grp.addButton(b)
            self.mode_btns[key] = b

        # --- thanh công cụ
        tb = QToolBar("Công cụ")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addAction(self.a_open)
        tb.addAction(self.a_folder)
        tb.addSeparator()
        for a in (self.a_undo, self.a_close):
            tb.addAction(a)
            tb.widgetForAction(a).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.addSeparator()
        self.file_lbl = QLabel("Chưa mở ảnh")
        tb.addWidget(self.file_lbl)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        tb.addWidget(spacer)
        self.using_lbl = QLabel()
        tb.addWidget(self.using_lbl)
        tb.addAction(self.a_export)
        self.addToolBar(tb)
        self.toolbar = tb

        # --- giữa: ảnh + thanh xem
        self.view = CompareView()
        self.view_btns = {}
        vbs = []
        for key, text in (("before", "Trước"), ("split", "Chia đôi"), ("after", "Sau")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self._set_view(k))
            self.view_btns[key] = b
            vbs.append(b)
        zbs = []
        for text, z in (("Vừa", None), ("100%", 1.0), ("200%", 2.0)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, z=z: self._zoom(z))
            zbs.append(b)
        self.zoom_btns = zbs
        vseg, zseg = seg_row(vbs), seg_row(zbs)
        fk = QApplication.font().pointSizeF() / 9.0  # cỡ chữ lớn (Xem → Cỡ chữ): nới các khối theo
        vseg.setFixedWidth(round(200 * fk))
        zseg.setFixedWidth(round(150 * fk))
        self._segs = (vseg, zseg)
        g = QButtonGroup(self)
        for b in vbs:
            g.addButton(b)
        g2 = QButtonGroup(self)
        for b in zbs:
            g2.addButton(b)
        self.view_btns["split"].setChecked(True)
        zbs[0].setChecked(True)
        hint = QLabel("Kéo trên ảnh để so sánh · cuộn chuột để phóng · chuột phải kéo ảnh · nháy đúp: vừa khung")
        hint.setObjectName("muted")
        hint.setMinimumWidth(10)
        vbar = QWidget()
        vbar.setObjectName("viewbar")
        vbar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vbar.setFixedHeight(32)
        vl = QHBoxLayout(vbar)
        vl.setContentsMargins(10, 0, 10, 0)
        vl.setSpacing(10)
        vl.addWidget(vseg)
        vl.addWidget(zseg)
        vl.addWidget(hint, 1)
        center = QWidget()
        center.setObjectName("center")
        center.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        cvl = QVBoxLayout(center)
        cvl.setContentsMargins(0, 0, 0, 0)
        cvl.setSpacing(0)
        cvl.addWidget(self.view, 1)
        cvl.addWidget(vbar)

        # --- các ô chỉnh (dùng chung cho cả hai chế độ: một bộ thông số)
        self.info = QLabel("Chưa có ảnh")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.pro_info = QLabel("")
        self.pro_info.setWordWrap(True)
        self.pro_info.setObjectName("muted")
        self.mode = QComboBox()
        for k, v in MODES.items():
            self.mode.addItem(v, k)
        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setObjectName("muted")
        self.preset = QComboBox()  # mô hình của danh sách Preset ở cột trái (không hiện trực tiếp)
        for k, (v, _) in PRESETS.items():
            self.preset.addItem(v, k)
        self.scale = QComboBox()
        for s, t in ((0, "Tự động (nhỏ 4×, còn lại 2×)"), (1, "1×"), (2, "2×"), (4, "4×")):
            self.scale.addItem(t, s)
        self.scale.setToolTip("Tự động: ảnh nhỏ (cạnh dài dưới 1000 px) phóng 4× để ảnh ra đủ lớn, còn lại phóng 2×")
        self.bg_color = (255, 255, 255)  # BGR, màu nền mới khi bật Đổi nền
        self.cb_bg = QCheckBox()
        self.cb_bg.setToolTip("Cho ảnh thẻ, phông trơn (xanh, trắng, đỏ…): tách người khỏi nền, viền tóc mềm, đổi sang màu "
                              "chọn. Nền có vật thể hoặc nhiều màu thì không đổi và báo lý do.")
        self.btn_bg_pick = QPushButton("Khác…")
        self.bg_swatches, self.simple_swatches = [], []
        for name, bgr in BG_SWATCHES:
            for lst in (self.bg_swatches, self.simple_swatches):
                b = QPushButton()
                b.setFixedSize(22, 22)
                b.setToolTip(name)
                b.setAccessibleName(name)
                b.setStyleSheet(f"background: rgb({bgr[2]},{bgr[1]},{bgr[0]}); border: 1px solid #555; padding: 0;")
                b.clicked.connect(lambda _=False, c=bgr, simple=lst is self.simple_swatches: self._swatch(c, simple))
                lst.append(b)
        self.preset_lib = facestyle.Library()
        self.subject = QComboBox()
        for k, t in SUBJECT_ITEMS:
            self.subject.addItem(t, k)
        self.subject.setToolTip("Preset theo đối tượng: Tự động đoán tuổi, giới tính từng khuôn mặt (trẻ em, nam, nữ, "
                                "người lớn tuổi) rồi chỉnh mặt theo kiểu của nhóm đó trong bộ preset.")
        self.preset_set = QComboBox()
        self.btn_presets = QPushButton("Quản lý…")
        self.btn_presets.setToolTip("Sửa, tạo bản sao, đổi tên, xoá bộ preset")
        self._fill_preset_sets()

        self.sl_sharp = Slider(0, 100)
        self.sl_denoise = Slider(0, 100, scale=10, fmt="{:.1f}", mult=1)
        self.cb_denoise_auto = QCheckBox("Khử nhiễu tự động")
        self.sl_contrast = Slider(0, 100)
        self.sl_bright = Slider(-50, 50)
        self.sl_expo = Slider(0, 100)
        self.sl_expo.setToolTip("Tự chỉnh ảnh quá tối, quá sáng, ngược sáng (mặt tối trên nền sáng): giữ màu, không làm "
                                "cháy vùng sáng. 0 = tắt.")
        self.sl_tone = Slider(0, 100)
        self.sl_tone.setToolTip("Tông tươi sáng, trong trẻo: nâng tông giữa, bỏ lớp sương xám ở vùng tối, màu tươi hơn, "
                                "da sáng hơn một chút. Không làm cháy vùng sáng. 0 = giữ tông ảnh gốc.")
        self.cb_restore = QCheckBox()
        self.cb_restore.setToolTip("Dò vết xước, nếp gãy, bụi rồi vá bằng LaMa; kéo lại màu phai; xoay ảnh nằm ngang. "
                                   "Tải thêm ~650 MB model ở lần đầu. Chậm hơn: nên có GPU NVIDIA.")
        self.cb_ai_color = QCheckBox()
        self.cb_ai_color.setToolTip("Dùng AI tô màu (DDColor) để đoán màu trang phục, cây cỏ: áo trắng ra trắng, jeans ra xanh. "
                                    "Cần bật 'Nâng màu sặc sỡ cho cảnh'. Lần đầu tải thêm khoảng 912 MB; chậm hơn vài chục giây.")
        self.cb_vivid = QCheckBox()
        self.cb_vivid.setToolTip("Ảnh cũ chụp cảnh (nền không đồng màu, không phải ảnh đen trắng): nâng tông sặc sỡ hơn, da ít đổi. "
                                 "Ảnh thẻ nền đồng màu và ảnh đen trắng được bỏ qua.")
        dmg_tip = ("Tô lên chỗ bị lá vàng, vết ố, rách che mất chi tiết. App vá chỗ đó (LaMa) rồi dựng lại mặt bị hỏng "
                   "(RestoreFormer++). Mặt nhỏ dưới ~60 px chỉ vá, không bịa thêm.")
        self.btn_damage = QPushButton("Vẽ vùng hỏng nặng…")
        self.btn_damage2 = QPushButton("Tô vùng hỏng nặng…")
        for b in (self.btn_damage, self.btn_damage2):
            b.setToolTip(dmg_tip)
        self.damage_path = ""
        self.cb_lowlight = QCheckBox("Ảnh tối: bật Retinex")
        self.cb_lowlight.setToolTip("Kéo sáng mạnh cả vùng tối cục bộ. Khi bật, Tự sửa sáng không chạy.")
        self.sl_fid = Slider(0, 100)
        self.sl_fid.setToolTip("Cao: giữ mặt gốc nhiều hơn (đúng người); thấp: AI vẽ lại nhiều hơn (nét hơn)")
        self.sl_detail = Slider(0, 100)
        self.cb_detail_same = QCheckBox("Mắt, môi như cả mặt")
        self.cb_keep_lips = QCheckBox("Giữ dáng, màu môi gốc")
        self.cb_keep_lips.setToolTip("Model mặt hay vẽ lại môi (mỏng hơn, đổi viền, thêm bóng). Bật: môi giữ như ảnh gốc, "
                                     "chỉ làm nét thêm.")
        self.sl_strength = Slider(0, 100)
        self.sl_strength.setToolTip("Trộn kết quả với ảnh gốc: 100 % = chỉ kết quả")
        self.phone = QComboBox()
        self.phone.addItem("Tự nhận theo EXIF", "auto")
        for k, pr in PROFILES.items():
            self.phone.addItem(pr.label, k)
        self.phone_info = QLabel()
        self.phone_info.setWordWrap(True)
        self.phone_info.setObjectName("muted")
        self.detected_phone = "generic"
        self.sl_wb = Slider(0, 100)
        self.sl_skin = Slider(0, 100)
        self.sl_color = Slider(0, 100)
        self.sl_gloss = Slider(0, 100)
        # Khử bóng loáng da: không bắt buộc, mặc định tắt. Một thông số chung: ô ở Đơn giản và ô ở Chuyên sâu luôn
        # cùng trạng thái; thanh cường độ chỉ ở Chuyên sâu (bật lần đầu: 60 %).
        ds_tip = ("Giảm vệt sáng chói do da dầu ở trán, mũi, gò má, cằm. Chỉ sửa vùng da mặt (không đụng mắt, môi, "
                  "tóc), giữ vân da và khối mặt. Không bắt buộc, mặc định tắt.")
        self.cb_deshine = QCheckBox("Bật")
        self.cb_deshine2 = QCheckBox("Khử bóng loáng da")
        self.sl_deshine = Slider(0, 100)
        self.sl_deshine.set(DESHINE_DEFAULT)
        self.sl_deshine.setEnabled(False)
        for w in (self.cb_deshine, self.cb_deshine2, self.sl_deshine):
            w.setToolTip(ds_tip)
        # Tông mẫu (look.py): tuỳ chọn của người dùng như Khử bóng da; hai ô chọn (Đơn giản, Chuyên sâu) luôn cùng giá trị,
        # thanh độ mạnh chỉ ở Chuyên sâu
        self.look_box = QComboBox()
        self.look_box2 = QComboBox()
        self.sl_look = Slider(0, 100)
        self.sl_look.set(1.0)
        self.sl_look.setEnabled(False)
        for w in (self.look_box, self.look_box2, self.sl_look):
            w.setToolTip(LOOK_TIP)
        self._fill_looks()
        self.face_model = QComboBox()
        for key, text in FACE_MODEL_ITEMS:
            self.face_model.addItem(text, key)
        self.bg_model = QComboBox()
        for key, text in BG_MODEL_ITEMS:
            self.bg_model.addItem(text, key)
        self.hw_warn = QLabel()  # ước tính thời gian / cảnh báo cấu hình máy khi chọn model nặng (không chặn)
        self.hw_warn.setWordWrap(True)
        self.hw_warn.setObjectName("muted")
        self.hw_warn.hide()
        self.level_group = QButtonGroup(self)
        self.level_group.setExclusive(True)
        for n, lv in LEVELS.items():  # mô hình của thanh mức nét (không hiện; hai thanh hiện ở hai chế độ)
            b = QPushButton(str(n))
            b.setCheckable(True)
            self.level_group.addButton(b, n)
        self._level_bars = []
        lvw = self._level_bar()
        self.level_desc = QLabel()
        self.level_desc.setWordWrap(True)
        self.level_desc.setObjectName("muted")
        self.btn_reset_adj = QPushButton("Đặt lại tất cả")
        self.device = QComboBox()
        self.dev_note = QLabel()
        self.dev_note.setObjectName("muted")
        self.dev_note.setWordWrap(True)
        self.fmt = QComboBox()
        for t, e in (("JPG", ".jpg"), ("PNG", ".png"), ("WebP", ".webp")):
            self.fmt.addItem(t, e)
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(94)
        self.out_size = QComboBox()
        for key, text in (("goc", "Giữ nguyên cỡ"), ("1k", "1K (1920 px)"), ("2k", "2K (2560 px)"), ("4k", "4K (3840 px)")):
            self.out_size.addItem(text, key)
        self.out_size.setToolTip("Cỡ ảnh xuất theo cạnh dài, giữ tỉ lệ. Lớn hơn ảnh kết quả thì phóng thêm (Lanczos), "
                                 "không thêm chi tiết; nên chọn Phóng to 4× nếu cần 4K từ ảnh nhỏ.")
        self.btn_run = QPushButton("Làm nét")
        self.btn_run.setObjectName("go")
        self.btn_run_simple = QPushButton("Làm nét")
        self.btn_run_simple.setObjectName("go")
        self.btn_run_simple.setMinimumHeight(32)
        self.btn_restore = QPushButton("Ảnh gốc")
        self.btn_restore.setToolTip("Bỏ kết quả, xem lại ảnh gốc (ảnh trên ổ đĩa không bao giờ bị sửa)")

        # --- cột phải, chế độ Đơn giản: Phân tích ảnh → Loại ảnh → Mức làm nét → Làm nét
        sp = QWidget()
        sp.setObjectName("panel")
        sl = QVBoxLayout(sp)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        sl.addWidget(header_bar("Phân tích ảnh"))
        box = QVBoxLayout()
        box.setContentsMargins(10, 8, 10, 10)
        box.addWidget(self.info)
        sl.addLayout(box)
        sl.addWidget(header_bar("Loại ảnh"))
        self.kind_group = QButtonGroup(self)
        self.kind_btns = {}
        kl = QVBoxLayout()
        kl.setContentsMargins(0, 4, 0, 4)
        kl.setSpacing(0)
        for key, icon, text, hint_ in KIND_ITEMS:
            b = ListChoice(icon, text, hint_)
            b.clicked.connect(lambda _=False, k=key: self._simple_pick(kind=k))
            self.kind_group.addButton(b)
            self.kind_btns[key] = b
            kl.addWidget(b)
        sl.addLayout(kl)
        self.id_box = QWidget()
        il = QHBoxLayout(self.id_box)
        il.setContentsMargins(10, 4, 10, 10)
        il.setSpacing(6)
        lab = QLabel("Nền mới")
        lab.setObjectName("muted")
        lab.setFixedWidth(60)
        il.addWidget(lab)
        for b in self.simple_swatches:
            il.addWidget(b)
        self.btn_bg_pick2 = QPushButton("Khác…")
        il.addWidget(self.btn_bg_pick2)
        il.addStretch()
        self.old_box = QWidget()
        ol = QVBoxLayout(self.old_box)
        ol.setContentsMargins(10, 4, 10, 10)
        ol.setSpacing(6)
        ol.addWidget(self.btn_damage2, 0, Qt.AlignmentFlag.AlignLeft)
        sl.addWidget(self.id_box)
        sl.addWidget(self.old_box)
        sl.addWidget(header_bar("Tuỳ chọn thêm"))
        ow = QVBoxLayout()
        ow.setContentsMargins(10, 8, 10, 8)
        ow.addWidget(self.cb_deshine2)
        lr = QHBoxLayout()
        lr.setSpacing(6)
        lab_look = QLabel("Tông màu")
        lab_look.setObjectName("muted")
        lr.addWidget(lab_look)
        lr.addWidget(self.look_box2, 1)
        ow.addLayout(lr)
        sl.addLayout(ow)
        sl.addWidget(header_bar("Mức làm nét"))
        pw = QVBoxLayout()
        pw.setContentsMargins(10, 10, 10, 10)
        pw.setSpacing(8)
        pw.addWidget(self._level_bar())
        sl.addLayout(pw)
        sl.addStretch(1)
        s_scroll = QScrollArea()
        s_scroll.setWidget(sp)
        s_scroll.setWidgetResizable(True)
        s_scroll.setFrameShape(QFrame.Shape.NoFrame)
        s_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        s_foot = QWidget()
        s_foot.setObjectName("viewbar")
        s_foot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sfl = QVBoxLayout(s_foot)
        sfl.setContentsMargins(10, 10, 10, 10)
        sfl.setSpacing(6)
        sfl.addWidget(self.btn_run_simple)
        row = QHBoxLayout()
        self.simple_eta = QLabel("")
        self.simple_eta.setObjectName("muted")
        self.simple_eta.setWordWrap(True)
        to_pro = QPushButton("Mở trong Chuyên sâu")
        to_pro.setObjectName("lnk")
        to_pro.clicked.connect(lambda: self.set_ui_mode("pro"))
        row.addWidget(self.simple_eta, 1)
        row.addWidget(to_pro, 0, Qt.AlignmentFlag.AlignTop)
        sfl.addLayout(row)
        simple_panel = QWidget()
        spl = QVBoxLayout(simple_panel)
        spl.setContentsMargins(0, 0, 0, 0)
        spl.setSpacing(0)
        spl.addWidget(s_scroll, 1)
        spl.addWidget(s_foot)

        # --- cột phải, chế độ Chuyên sâu: histogram, ô tìm, 8 nhóm thu gọn
        self.histo = Histogram()
        top = QWidget()
        top.setObjectName("viewbar")
        top.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tl = QVBoxLayout(top)
        tl.setContentsMargins(10, 8, 10, 6)
        tl.setSpacing(4)
        tl.addWidget(self.histo)
        tl.addWidget(self.pro_info)
        self.search = QLineEdit()
        self.search.setObjectName("search")
        self.search.setPlaceholderText("Tìm thông số (vd: da, nền, nhiễu)")
        self.search.setClearButtonEnabled(True)
        self.search.addAction(line_icon("search", "#8f8f8f"), QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self._filter_sections)
        b_open_all, b_close_all = QToolButton(), QToolButton()
        b_open_all.setIcon(line_icon("down"))
        b_open_all.setToolTip("Mở hết các nhóm")
        b_close_all.setIcon(line_icon("up"))
        b_close_all.setToolTip("Thu gọn các nhóm")
        b_open_all.clicked.connect(lambda: [s.set_open(True) for s in self.sections])
        b_close_all.clicked.connect(lambda: [s.set_open(False) for s in self.sections])
        srow = QHBoxLayout()
        srow.setContentsMargins(10, 6, 10, 6)
        srow.addWidget(self.search, 1)
        srow.addWidget(b_open_all)
        srow.addWidget(b_close_all)

        dtw = QWidget()
        dt = QVBoxLayout(dtw)
        dt.setContentsMargins(0, 0, 0, 0)
        dt.addWidget(self.sl_detail)
        sec_level = Section("Mức nét và model", on_reset=lambda: self._reset_fields(
            "mode", "upscale", "face_model", "sr_model", "strength"))
        sec_level.add_full(lvw, "muc net 1 2 3 4 5 6 nhe vua ro toi da cao nhat")
        sec_level.add("Chế độ", self.mode, "che do")
        sec_level.add("Phóng to", self.scale, "phong to upscale")
        sec_level.add("Model mặt", self.face_model, "gfpgan restoreformer")
        sec_level.add("Model nền", self.bg_model, "esrgan dat plksr")
        sec_level.add("", self.hw_warn)
        sec_level.add("Cường độ chung", self.sl_strength, "tron")
        b_models = QPushButton("Quản lý model…")
        b_models.clicked.connect(self.a_models.trigger)
        self.btn_models = b_models
        sec_level.add("", b_models, "quan ly model tai")
        sec_face = Section("Khuôn mặt", on_reset=lambda: self._reset_fields(
            "fidelity", "detail_fidelity", "keep_lips", "skin_tone", "gloss", "deshine", "subject"))
        sec_face.add("Đối tượng", self.subject, "tuoi gioi tinh tre em nam nu")
        sec_face.add("Trung thực AI", self.sl_fid, "do trung thuc giu mat goc")
        sec_face.add("Trung thực mắt, môi", dtw, "mat moi ngu quan")
        sec_face.add("", self.cb_detail_same, "mat moi")
        sec_face.add("", self.cb_keep_lips, "moi")
        sec_face.add("Sửa màu da", self.sl_skin, "da")
        sec_face.add("Độ bóng da", self.sl_gloss, "da bong")
        dsw = QWidget()
        dsl = QHBoxLayout(dsw)
        dsl.setContentsMargins(0, 0, 0, 0)
        dsl.setSpacing(6)
        dsl.addWidget(self.cb_deshine)
        dsl.addWidget(self.sl_deshine, 1)
        sec_face.add("Khử bóng da", dsw, "da bong dau loang chop mui tran khu bong deshine")
        sec_light = Section("Ánh sáng và tông", on_reset=lambda: self._reset_fields(
            "exposure", "brightness", "contrast", "tone", "low_light", "look", "look_amount"))
        sec_light.add("Tự sửa sáng", self.sl_expo, "phoi sang toi chay nguoc sang")
        sec_light.add("Độ sáng", self.sl_bright)
        sec_light.add("Tương phản", self.sl_contrast)
        sec_light.add("Tươi sáng", self.sl_tone, "tong")
        sec_light.add("", self.cb_lowlight, "retinex toi")
        sec_light.add("Tông mẫu", self.look_box, "tong mau trong sang look anh the da sang")
        sec_light.add("Độ mạnh tông mẫu", self.sl_look, "tong mau trong sang look")
        sec_color = Section("Màu sắc", on_reset=lambda: self._reset_fields("wb", "color"))
        sec_color.add("Máy chụp", self.phone, "dien thoai exif hang")
        sec_color.add("", self.phone_info)
        sec_color.add("Cân bằng trắng", self.sl_wb, "am mau vang")
        sec_color.add("Tăng màu", self.sl_color, "mau ruc")
        sec_detail = Section("Độ nét và nhiễu", on_reset=lambda: self._reset_fields("sharpness", "denoise"))
        sec_detail.add("Độ nét", self.sl_sharp, "net")
        sec_detail.add("Khử nhiễu", self.sl_denoise, "nhieu hat")
        sec_detail.add("", self.cb_denoise_auto, "nhieu")
        sec_restore = Section("Khôi phục ảnh cũ", check=self.cb_restore, opened=False)
        sec_restore.add("Nâng màu sặc sỡ cho cảnh", self.cb_vivid, "mau sac so canh tone ruc ro")
        sec_restore.add("Tô màu AI cho cảnh (DDColor)", self.cb_ai_color, "to mau ai ddcolor trang phuc")
        sec_restore.add("", self.btn_damage, "vung hong la vang rach xuoc bui phai xoay cu")
        sec_bg = Section("Đổi nền", check=self.cb_bg, opened=False)
        swr = QWidget()
        swl = QHBoxLayout(swr)
        swl.setContentsMargins(0, 0, 0, 0)
        swl.setSpacing(6)
        for b in self.bg_swatches + [self.btn_bg_pick]:
            swl.addWidget(b)
        swl.addStretch()
        sec_bg.add("Màu nền", swr, "nen anh the xanh trang do")
        sec_out = Section("Xuất và thiết bị", opened=False)
        sec_out.add("Thiết bị", self.device, "gpu cpu card thiết bị render")
        sec_out.add("", self.dev_note, "gpu cpu card thiết bị render")
        sec_out.add("Định dạng", self.fmt, "jpg png webp")
        sec_out.add("Chất lượng", self.quality, "jpg")
        sec_out.add("Cỡ ảnh ra", self.out_size, "1k 2k 4k")
        self.sections = [sec_level, sec_face, sec_light, sec_color, sec_detail, sec_restore, sec_bg, sec_out]
        sw = QWidget()
        sw.setObjectName("panel")
        swv = QVBoxLayout(sw)
        swv.setContentsMargins(0, 0, 0, 0)
        swv.setSpacing(0)
        for s in self.sections:
            swv.addWidget(s)
        self.no_match = QLabel("Không có thông số nào khớp.")
        self.no_match.setObjectName("muted")
        self.no_match.setContentsMargins(10, 12, 10, 12)
        self.no_match.hide()
        swv.addWidget(self.no_match)
        swv.addStretch(1)
        p_scroll = QScrollArea()
        p_scroll.setWidget(sw)
        p_scroll.setWidgetResizable(True)
        p_scroll.setFrameShape(QFrame.Shape.NoFrame)
        p_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        p_foot = QWidget()
        p_foot.setObjectName("viewbar")
        p_foot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pfl = QHBoxLayout(p_foot)
        pfl.setContentsMargins(10, 8, 10, 8)
        pfl.setSpacing(6)
        pfl.addWidget(self.btn_reset_adj)
        pfl.addWidget(self.btn_restore)
        pfl.addStretch()
        self.btn_run.setMinimumWidth(110)
        self.btn_run.setMinimumHeight(28)
        pfl.addWidget(self.btn_run)
        pro_panel = QWidget()
        ppl = QVBoxLayout(pro_panel)
        ppl.setContentsMargins(0, 0, 0, 0)
        ppl.setSpacing(0)
        ppl.addWidget(top)
        ppl.addLayout(srow)
        ppl.addWidget(p_scroll, 1)
        ppl.addWidget(p_foot)

        self.right = QStackedWidget()
        self.right.addWidget(simple_panel)
        self.right.addWidget(pro_panel)
        self.right.setMinimumWidth(round(360 * QApplication.font().pointSizeF() / 9.0))
        right_box = QWidget()
        rbl = QVBoxLayout(right_box)
        rbl.setContentsMargins(0, 0, 0, 0)
        rbl.setSpacing(0)
        mode_bar = QWidget()
        mode_bar.setObjectName("viewbar")
        mode_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        mbl = QHBoxLayout(mode_bar)
        mbl.setContentsMargins(10, 8, 10, 8)
        mbl.addWidget(seg_row(list(self.mode_btns.values())))
        for b in self.mode_btns.values():
            b.setMinimumHeight(28)
        rbl.addWidget(mode_bar)
        rbl.addWidget(self.right, 1)

        # --- cột trái (Chuyên sâu): Preset / Lịch sử
        self.preset_list = QListWidget()
        for k, (v, _) in PRESETS.items():
            it = QListWidgetItem(v)
            it.setData(Qt.ItemDataRole.UserRole, k)
            self.preset_list.addItem(it)
        self.preset_list.itemClicked.connect(self._preset_list_clicked)
        b_imp, b_exp = QPushButton("Nạp…"), QPushButton("Xuất…")
        b_imp.setToolTip("Nạp bộ preset người khác gửi (.netanh.json)")
        b_exp.setToolTip("Xuất bộ đang dùng ra file .netanh.json để gửi người khác (dùng được cả trên điện thoại)")
        b_imp.clicked.connect(self._import_preset_file)
        b_exp.clicked.connect(self._export_preset_file)
        pt = QWidget()
        pl = QVBoxLayout(pt)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        pl.addWidget(header_bar("Kiểu làm nét"))
        self.preset_list.setFrameShape(QFrame.Shape.NoFrame)
        pl.addWidget(self.preset_list, 1)
        pl.addWidget(header_bar("Preset theo đối tượng"))
        pb = QVBoxLayout()
        pb.setContentsMargins(10, 8, 10, 10)
        pb.setSpacing(6)
        pb.addWidget(self.preset_set)
        prow = QHBoxLayout()
        prow.setSpacing(6)
        for b in (b_imp, b_exp, self.btn_presets):
            prow.addWidget(b)
        pb.addLayout(prow)
        pl.addLayout(pb)

        self.hist_list = QListWidget()
        self.hist_list.setIconSize(QSize(56, 56))
        self.hist_list.setFrameShape(QFrame.Shape.NoFrame)
        self.hist_list.itemDoubleClicked.connect(self._open_history_item)
        hb = [(QPushButton("Mở"), lambda: self._open_history_item(self.hist_list.currentItem())),
              (QPushButton("Thư mục chứa"), self._reveal_history_item),
              (QPushButton("Xoá khỏi lịch sử"), lambda: self._delete_history(False)),
              (QPushButton("Xoá cả file ra"), lambda: self._delete_history(True))]
        ht = QWidget()
        hl = QVBoxLayout(ht)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        hl.addWidget(self.hist_list, 1)
        hg = QGridLayout()
        hg.setContentsMargins(10, 8, 10, 6)
        hg.setSpacing(6)
        for i, (b, fn) in enumerate(hb):
            b.clicked.connect(lambda _=False, fn=fn: fn())
            hg.addWidget(b, i // 2, i % 2)
        hl.addLayout(hg)
        b_clear = QPushButton("Xoá toàn bộ lịch sử…")
        b_clear.setObjectName("lnk")
        b_clear.clicked.connect(self._clear_history)
        hl.addWidget(b_clear, 0, Qt.AlignmentFlag.AlignLeft)
        hl.addSpacing(8)
        self.left = QTabWidget()
        self.left.addTab(pt, "Preset")
        self.left.addTab(ht, "Lịch sử")
        self.left.currentChanged.connect(lambda i: i == 1 and self._refresh_history())
        self.left.setMinimumWidth(round(220 * QApplication.font().pointSizeF() / 9.0))

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.addWidget(self.left)
        self.splitter.addWidget(center)
        self.splitter.addWidget(right_box)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        for i in range(3):
            self.splitter.setCollapsible(i, i == 0)
        self.splitter.setSizes([round(260 * fk), 820, round(380 * fk)])
        self.setCentralWidget(self.splitter)

        # --- nhật ký (Xem → Nhật ký), thanh trạng thái
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        devices = list_devices()
        self._devices = devices
        self._gpu_kind = next((d.kind for d in devices if d.kind != "cpu"), "cpu")
        self.hw_text = hardware.machine().summary()
        self._fill_devices()
        lw = QWidget()
        lwl = QVBoxLayout(lw)
        lwl.setContentsMargins(6, 4, 6, 6)
        hwl = QLabel("Máy: " + self.hw_text + " · Chạy model trên: " + ", ".join(d.name for d in devices))
        hwl.setWordWrap(True)
        hwl.setObjectName("muted")
        lwl.addWidget(hwl)
        lwl.addWidget(self.log, 1)
        self.log_dock.setWidget(lw)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        self.status_lbl = QLabel()
        self.status_lbl.setMinimumWidth(10)
        sb.addWidget(self.status_lbl, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(round(300 * QApplication.font().pointSizeF() / 9.0))
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(round(20 * QApplication.font().pointSizeF() / 9.0))
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.hide()
        self.track = ProgressTrack()
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._show_progress)
        self.btn_cancel = QPushButton("Huỷ")
        self.btn_cancel.setObjectName("lnk")
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_cancel.hide()
        self.dev_lbl = QLabel()
        self.dev_lbl.setToolTip("Máy: " + self.hw_text)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.btn_cancel)
        sb.addPermanentWidget(self.dev_lbl)
        self._lock_widgets = [sp, sw, pt, ht, self.btn_reset_adj, self.btn_restore]
        self._lock_actions = [self.a_reset, self.a_damage, self.a_presets]
        self._device_changed()

        # combo co theo bề ngang, chữ dài xem ở tooltip: không cắt mép, không cuộn ngang
        for c in self.findChildren(QComboBox) + [self.preset_set]:
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)
            c.setMinimumWidth(80)  # minimumSize thắng minimumSizeHint: combo co theo cột, không đẩy cột rộng ra
            c.currentIndexChanged.connect(lambda _=0, c=c: c.setToolTip(c.currentText()))

        # --- nối tín hiệu
        self.btn_run.clicked.connect(self._run_clicked)
        self.btn_run_simple.clicked.connect(self._run_clicked)
        self.btn_restore.clicked.connect(self.restore_original)
        self.btn_presets.clicked.connect(self._manage_presets)
        self.btn_bg_pick.clicked.connect(self._pick_bg_color)
        self.btn_bg_pick2.clicked.connect(lambda: self._pick_bg_color(simple=True))
        self.cb_bg.toggled.connect(lambda on: self._set_bg_color(self.bg_color))
        self._set_bg_color(self.bg_color)
        self.preset_set.currentIndexChanged.connect(
            lambda i: i >= 0 and self.preset_lib.set_active(self.preset_set.itemData(i)))
        self.btn_reset_adj.clicked.connect(self._reset_adjust)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.cb_denoise_auto.toggled.connect(lambda on: (self.sl_denoise.setEnabled(not on), self._to_custom()))
        for w in (self.sl_sharp, self.sl_denoise, self.sl_contrast, self.sl_bright, self.sl_fid, self.sl_strength,
                  self.sl_color, self.sl_gloss, self.sl_wb, self.sl_skin, self.sl_expo, self.sl_tone):
            w.changed.connect(self._to_custom)
        self.phone.currentIndexChanged.connect(self._phone_changed)
        self.level_group.idClicked.connect(self._level_clicked)
        self.face_model.currentIndexChanged.connect(self._models_changed)
        self.bg_model.currentIndexChanged.connect(self._models_changed)
        self.device.currentIndexChanged.connect(self._device_changed)
        self.mode.currentIndexChanged.connect(self._update_hw_warn)
        self.scale.currentIndexChanged.connect(self._update_hw_warn)
        self.sl_detail.changed.connect(self._to_custom)
        self.cb_detail_same.toggled.connect(lambda on: (self.sl_detail.setEnabled(not on), self._to_custom()))
        self.cb_lowlight.toggled.connect(self._to_custom)
        self.cb_restore.toggled.connect(self._to_custom)
        self.cb_vivid.toggled.connect(self._to_custom)
        self.cb_ai_color.toggled.connect(self._to_custom)
        self.btn_damage.clicked.connect(self._draw_damage)
        self.btn_damage2.clicked.connect(self._draw_damage)
        self.cb_keep_lips.toggled.connect(self._to_custom)
        self.cb_deshine.toggled.connect(self._deshine_toggled)
        self.cb_deshine2.toggled.connect(self._deshine_toggled)
        self.look_box.currentIndexChanged.connect(lambda i: self._look_picked(self.look_box, i))
        self.look_box2.currentIndexChanged.connect(lambda i: self._look_picked(self.look_box2, i))
        self.cb_lowlight.toggled.connect(lambda on: self.sl_expo.setEnabled(not on))
        self._sync_simple()

    def set_font_size(self, pt: float):
        """Đổi cỡ chữ ngay, nhớ cho lần sau (Xem → Cỡ chữ giao diện)."""
        app = QApplication.instance()
        set_ui_font(app, pt)
        app.setStyleSheet(app.styleSheet())  # áp lại QSS cho chữ mới
        for sec in self.sections:
            sec.grid.setColumnMinimumWidth(0, Section.LABEL_W)
            for _, ws in sec.rows:
                if len(ws) == 2 and isinstance(ws[0], QLabel):
                    ws[0].setFixedWidth(Section.LABEL_W)
        k = pt / 9.0
        self.right.setMinimumWidth(round(360 * k))
        self.left.setMinimumWidth(round(220 * k))
        self._segs[0].setFixedWidth(round(200 * k))
        self._segs[1].setFixedWidth(round(150 * k))
        sz = self.splitter.sizes()
        self.splitter.setSizes([max(sz[0], round(260 * k)) if sz[0] else 0, sz[1], max(sz[2], round(360 * k))])
        self.qs.setValue("ui_pt", pt)
        self.say(f"Cỡ chữ giao diện: {pt:g} pt")

    # ----- hai chế độ giao diện -----

    def set_ui_mode(self, mode: str):
        mode = mode if mode in UI_MODES else "simple"
        self.ui_mode = mode
        pro = mode == "pro"
        self.right.setCurrentIndex(1 if pro else 0)
        self.left.setVisible(pro)
        self.mode_btns[mode].setChecked(True)
        self.a_simple.setChecked(not pro)
        self.a_pro.setChecked(pro)
        if not pro:
            self._sync_simple()
        self.qs.setValue("ui_mode", mode)
        self._sync_preset_list()

    def show_history(self):
        self.set_ui_mode("pro")
        self.left.setCurrentIndex(1)
        sizes = self.splitter.sizes()
        if sizes[0] < 50:
            self.splitter.setSizes([260, max(400, sizes[1] - 260), sizes[2]])

    def _set_view(self, key: str):
        self.view.show_mode = key
        self.view_btns[key].setChecked(True)
        self.view.update()

    def _zoom(self, z):
        self.view.set_zoom(z)
        idx = {None: 0, 1.0: 1, 2.0: 2}.get(z)
        if idx is not None:
            self.zoom_btns[idx].setChecked(True)

    def _filter_sections(self, text: str):
        q = norm_text(text.strip())
        hits = [s.filter(q) for s in self.sections]
        self.no_match.setVisible(bool(q) and not any(hits))
        self._update_hw_warn()  # dòng cảnh báo chỉ hiện khi có cảnh báo

    def _level_bar(self) -> QWidget:
        """Thanh mức nét dùng chung cho hai chế độ: Tự động, 1 Nhẹ … 6 Cao nhất (cùng một thang, cùng trạng thái)."""
        auto = QPushButton("Tự\nđộng")
        auto.setToolTip("Tự động: app chọn mức theo loại ảnh, độ mờ và cỡ mặt đo được")
        auto.clicked.connect(lambda: self._level_pick(None))
        btns = {}
        for n, lv in LEVELS.items():
            b = QPushButton(f"{n}\n{lv.name}")
            b.setToolTip(f"Mức {n} · {lv.name}: {lv.short}")
            b.clicked.connect(lambda _=False, n=n: self._level_pick(n))
            btns[n] = b
        w = seg_row([auto] + list(btns.values()))
        for b in [auto] + list(btns.values()):
            b.setMinimumHeight(round(36 * QApplication.font().pointSizeF() / 9.0))
            b.setStyleSheet(f"padding: 2px 0; margin: 0; font-size: {QApplication.font().pointSizeF() * 0.9:.1f}pt;")
        self._level_bars.append((auto, btns))
        return w

    def _level_pick(self, n: int | None):
        if self.ui_mode == "simple":
            self._simple_pick(power="auto" if n is None else n)
        elif n is None:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(self.preset.findData("auto"))
            self.preset.blockSignals(False)
            self._preset_changed()
        else:
            self._level_clicked(n)
        self._sync_level_bars()

    def _sync_level_bars(self):
        n = self.level_group.checkedId()
        auto = self.level_choice is None and self.preset.currentData() != "custom"
        for a, btns in getattr(self, "_level_bars", []):
            a.setChecked(auto)
            for k, b in btns.items():
                b.setChecked(k == n)

    def _simple_pick(self, kind: str | None = None, power=None):
        """Đơn giản: loại ảnh → preset gốc; mức làm nét → thang mức 1–6. Ghi thẳng vào bộ thông số chung."""
        if kind:
            self.simple_kind = kind
        if power:
            self.simple_power = power
        self._simple_apply()
        self._sync_simple()

    def _simple_apply(self):
        kind, power = self.simple_kind, self.simple_power
        n = None if power == "auto" else int(power)
        if n is None:
            key = KIND_PRESET[kind]
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(self.preset.findData(key))
            self.preset.blockSignals(False)
            self._preset_changed()
        else:
            self._level_clicked(n)
        # thêm phần riêng của loại ảnh (không đẩy bộ thông số sang Tuỳ chỉnh)
        boxes = (self.cb_restore, self.cb_bg, self.cb_lowlight, self.mode)
        for w in boxes:
            w.blockSignals(True)
        self.cb_restore.setChecked(kind == "old")
        self.cb_bg.setChecked(kind == "id")
        if kind == "dark":
            self.cb_lowlight.setChecked(True)
            self.sl_expo.setEnabled(False)
        if kind == "scene":
            self.mode.setCurrentIndex(self.mode.findData("upscale"))
        for w in boxes:
            w.blockSignals(False)
        self._set_bg_color(self.bg_color)
        self._mode_changed(update_custom=False)
        self._update_hw_warn()
        self._sync_preset_list()

    def _sync_simple(self):
        self.kind_btns[self.simple_kind].setChecked(True)
        self.id_box.setVisible(self.simple_kind == "id")
        self.old_box.setVisible(self.simple_kind == "old")
        self._sync_level_bars()

    def _sync_preset_list(self):
        key = self.preset.currentData()
        for i in range(self.preset_list.count()):
            it = self.preset_list.item(i)
            it.setSelected(it.data(Qt.ItemDataRole.UserRole) == key)
        n = self.level_group.checkedId()
        lv = f"mức {n} · {LEVELS[n].name}" if n > 0 else "chỉnh tay"
        if getattr(self, "ui_mode", "simple") == "simple":
            name = next(t for k, _, t, _ in KIND_ITEMS if k == self.simple_kind)
        else:
            name = PRESETS[key][0] if key in PRESETS else ""
        self.using_lbl.setText(f"{name} · {lv}")

    def _preset_list_clicked(self, it: QListWidgetItem):
        key = it.data(Qt.ItemDataRole.UserRole)
        if key == "custom":
            self._to_custom()
            return
        idx = self.preset.findData(key)
        if idx == self.preset.currentIndex():
            self._preset_changed()
        else:
            self.preset.setCurrentIndex(idx)

    def _reset_fields(self, *fields: str):
        """Đặt lại một nhóm thông số về mặc định, giữ các nhóm khác."""
        d = Settings()
        s = self.settings().with_(**{f: getattr(d, f) for f in fields})
        bg_on = self.cb_bg.isChecked()
        self._apply_settings(s)
        self.cb_bg.setChecked(bg_on)
        self._to_custom()
        self.say("Đã đặt lại: " + ", ".join(fields))

    def _swatch(self, bgr, simple: bool):
        if simple or not self.cb_bg.isChecked():
            self.cb_bg.setChecked(True)
        self._set_bg_color(bgr)

    def _set_info(self, html: str, short: str = ""):
        self.info.setText(html)
        self.pro_info.setText(short)

    def _import_preset_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Nạp bộ preset", "", f"Bộ preset Beauty HDR (*{facestyle.EXT} *.json)")
        if not f:
            return
        try:
            got, warn = self.preset_lib.import_file(f)
        except (OSError, UnicodeDecodeError, facestyle.PresetError) as e:
            QMessageBox.warning(self, APP_NAME, f"File không hợp lệ: {e}")
            return
        if got:
            self.preset_lib.set_active(got[0].id)
        self._fill_preset_sets()
        self._fill_looks()
        msg = "Đã nạp và chọn: " + ", ".join(st.name for st in got)
        if warn:
            msg += "\n\nĐã tự sửa:\n" + "\n".join(warn[:12])
        QMessageBox.information(self, APP_NAME, msg)

    def _export_preset_file(self):
        st = self.preset_lib.get(self.preset_set.currentData())
        if st is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Xuất bộ preset", facestyle.safe_filename(st.name),
                                           f"Bộ preset Beauty HDR (*{facestyle.EXT})")
        if not f:
            return
        if not f.endswith(facestyle.EXT):
            f = f.removesuffix(".json") + facestyle.EXT
        try:
            self.preset_lib.export([st.id], f, app=f"Beauty HDR PC {__import__('net_anh').__version__}")
        except (OSError, facestyle.PresetError) as e:
            QMessageBox.warning(self, APP_NAME, f"Không xuất được: {e}")
            return
        self.say(f"Đã xuất {Path(f).name}. Người nhận bấm Nạp… để dùng.")



    # ----- đổi màu nền -----

    def _set_bg_color(self, bgr):
        self.bg_color = tuple(int(v) for v in bgr)
        b, g, r = self.bg_color
        txt = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
        for btn in (self.btn_bg_pick, self.btn_bg_pick2):
            btn.setStyleSheet(f"background: rgb({r},{g},{b}); color: {txt}; border: 1px solid #555;")
        on = self.cb_bg.isChecked()
        for w in self.bg_swatches + [self.btn_bg_pick]:
            w.setEnabled(on)
        for w, (_, c) in zip(self.bg_swatches + self.simple_swatches, BG_SWATCHES + BG_SWATCHES):
            sel = on and tuple(c) == self.bg_color
            w.setStyleSheet(f"background: rgb({c[2]},{c[1]},{c[0]}); padding: 0; border: "
                            + ("2px solid #e6e6e6;" if sel else "1px solid #555;"))

    def _pick_bg_color(self, simple: bool = False):
        b, g, r = self.bg_color
        c = QColorDialog.getColor(QColor(r, g, b), self, "Chọn màu nền")
        if c.isValid():
            if simple:
                self.cb_bg.setChecked(True)
            self._set_bg_color((c.blue(), c.green(), c.red()))

    # ----- bộ preset theo đối tượng -----

    def _fill_preset_sets(self):
        self.preset_set.blockSignals(True)
        self.preset_set.clear()
        active = self.preset_lib.active_id()
        for st in self.preset_lib.sets():
            self.preset_set.addItem(st.name, st.id)
        self.preset_set.setCurrentIndex(max(0, self.preset_set.findData(active)))
        self.preset_set.blockSignals(False)

    def _manage_presets(self):
        PresetDialog(self, self.preset_lib).exec()
        self._fill_preset_sets()
        self._fill_looks()

    # ----- tông mẫu -----

    def _fill_looks(self, keep: str | None = None):
        """Ô chọn tông mẫu: Giữ tông ảnh + các bộ preset có kèm tông mẫu (có sẵn hoặc nạp từ file)."""
        if keep is None:
            keep = self.look_box.currentData() or ""
        for box in (self.look_box, self.look_box2):
            box.blockSignals(True)
            box.clear()
            box.addItem("Giữ tông ảnh", "")
            for st in self.preset_lib.looks():
                name = (st.look or {}).get("name") or st.name
                box.addItem(name if st.builtin else f"{name} ({st.name})", st.id)
            box.setCurrentIndex(max(0, box.findData(keep)))
            box.blockSignals(False)
        self.sl_look.setEnabled(bool(self.look_box.currentData()))

    def _look_picked(self, box, i: int):
        sid = box.itemData(i) or ""
        for b in (self.look_box, self.look_box2):
            if b is not box:
                b.blockSignals(True)
                b.setCurrentIndex(max(0, b.findData(sid)))
                b.blockSignals(False)
        self.sl_look.setEnabled(bool(sid))
        self.say("Tông mẫu: " + (box.itemText(i) if sid else "tắt (giữ tông ảnh)"))

    def _look_value(self) -> dict | None:
        sid = self.look_box.currentData()
        if not sid:
            return None
        st = self.preset_lib.get(sid)
        return dict(st.look) if st.look else None

    def _set_look(self, look: dict | None, amount: float):
        """Đặt ô Tông mẫu theo Settings (lịch sử, đặt lại): tìm bộ có đúng tông đó; không thấy thì giữ lựa chọn."""
        sid = ""
        if look:
            sid = next((st.id for st in self.preset_lib.looks() if st.look == look), self.look_box.currentData() or "")
        for b in (self.look_box, self.look_box2):
            b.blockSignals(True)
            b.setCurrentIndex(max(0, b.findData(sid)))
            b.blockSignals(False)
        self.sl_look.set(amount)
        self.sl_look.setEnabled(bool(sid))

    def _user_opts(self) -> dict:
        """Tuỳ chọn người dùng giữ qua mọi đổi loại ảnh, mức nét, preset, mở ảnh mới: khử bóng da, tông mẫu."""
        return dict(deshine=self._deshine_value(), look=self._look_value(), look_amount=self.sl_look.value())

    # ----- thông số <-> giao diện -----

    def _deshine_value(self) -> float:
        return self.sl_deshine.value() if self.cb_deshine.isChecked() else 0.0

    def _set_deshine(self, v: float):
        """Đặt ô Khử bóng da (cả hai chế độ) và thanh cường độ theo Settings.deshine. Tắt thì thanh giữ giá trị cũ."""
        on = v > 1e-3
        for cb in (self.cb_deshine, self.cb_deshine2):
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        if on:
            self.sl_deshine.set(v)
        self.sl_deshine.setEnabled(on)

    def _deshine_toggled(self, on: bool):
        if on and self.sl_deshine.value() <= 1e-3:
            self.sl_deshine.set(DESHINE_DEFAULT)
        self._set_deshine(self.sl_deshine.value() if on else 0.0)
        self.say("Khử bóng loáng da: " + (f"bật ({self.sl_deshine.value() * 100:.0f} %)" if on else "tắt"))

    def _apply_settings(self, s: Settings):
        widgets = (self.mode, self.preset, self.scale, self.cb_denoise_auto, self.cb_lowlight, self.face_model,
                   self.bg_model, self.cb_detail_same, self.cb_keep_lips, self.subject, self.cb_restore, self.cb_vivid, self.cb_ai_color)
        for w in widgets:
            w.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(s.mode))
        self.scale.setCurrentIndex(self.scale.findData(s.upscale))
        self.sl_sharp.set(s.sharpness)
        self.cb_denoise_auto.setChecked(s.denoise < 0)
        self.sl_denoise.set(4 if s.denoise < 0 else s.denoise)
        self.sl_denoise.setEnabled(s.denoise >= 0)
        self.sl_contrast.set(s.contrast)
        self.sl_bright.set(s.brightness)
        self.sl_expo.set(s.exposure)
        self.sl_expo.setEnabled(not s.low_light)
        self.sl_tone.set(s.tone)
        self.cb_lowlight.setChecked(s.low_light)
        self.cb_restore.setChecked(s.repair > 0 or s.fade > 0 or s.auto_rotate)
        self.cb_vivid.setChecked(s.scene_vivid > 0)
        self.cb_ai_color.setChecked(s.scene_ai > 0)
        self.sl_fid.set(s.fidelity)
        self.cb_detail_same.setChecked(s.detail_fidelity is None)
        self.sl_detail.set(s.fidelity if s.detail_fidelity is None else s.detail_fidelity)
        self.sl_detail.setEnabled(s.detail_fidelity is not None)
        self.sl_strength.set(s.strength)
        self.cb_keep_lips.setChecked(s.keep_lips)
        self.subject.setCurrentIndex(max(0, self.subject.findData(s.subject)))
        prof = phone_profile(self._phone_key())
        self.sl_wb.set(prof.wb if s.wb < 0 else s.wb)
        self.sl_skin.set(prof.skin if s.skin_tone < 0 else s.skin_tone)
        self.sl_color.set(s.color)
        self.sl_gloss.set(s.gloss)
        self._set_deshine(s.deshine)
        self._set_look(s.look, s.look_amount)
        self.face_model.setCurrentIndex(max(0, self.face_model.findData(s.face_model)))
        self.bg_model.setCurrentIndex(max(0, self.bg_model.findData(s.sr_model)))
        for w in widgets:
            w.blockSignals(False)
        self._mode_changed(update_custom=False)
        self._update_hw_warn()

    def settings(self) -> Settings:
        return Settings(mode=self.mode.currentData(), upscale=self.scale.currentData(),
                        strength=self.sl_strength.value(), sharpness=self.sl_sharp.value(),
                        denoise=-1.0 if self.cb_denoise_auto.isChecked() else self.sl_denoise.value(),
                        contrast=self.sl_contrast.value(), brightness=self.sl_bright.value(),
                        low_light=self.cb_lowlight.isChecked(), exposure=self.sl_expo.value(),
                        tone=self.sl_tone.value(), fidelity=self.sl_fid.value(),
                        color=self.sl_color.value(), gloss=self.sl_gloss.value(), **self._user_opts(),
                        face_model=self.face_model.currentData(), phone=self._phone_key(),
                        wb=self.sl_wb.value(), skin_tone=self.sl_skin.value(), sr_model=self.bg_model.currentData(),
                        detail_fidelity=None if self.cb_detail_same.isChecked() else self.sl_detail.value(),
                        keep_lips=self.cb_keep_lips.isChecked(), subject=self.subject.currentData(),
                        bg_color=self.bg_color if self.cb_bg.isChecked() else None,
                        face_styles=facestyle.styles_to_settings(self.preset_lib.get(self.preset_set.currentData())),
                        repair=1.0 if self.cb_restore.isChecked() else 0.0,
                        fade=1.0 if self.cb_restore.isChecked() else 0.0, auto_rotate=self.cb_restore.isChecked(),
                        damage_mask=self.damage_path, scene_vivid=1.0 if self.cb_vivid.isChecked() else 0.0,
                        scene_ai=1.0 if self.cb_ai_color.isChecked() else 0.0)

    def _draw_damage(self):
        if self.src_img is None:
            QMessageBox.information(self, APP_NAME, "Mở một ảnh trước, rồi tô vùng hỏng.")
            return
        from . import restore
        old = restore.load_damage_mask(self.damage_path, self.src_img.shape[:2]) if self.damage_path else None
        dlg = DamageDialog(self, self.src_img, None if old is None else old.astype(np.uint8) * 255)
        if dlg.exec() != QDialog.Accepted:
            return
        m = dlg.mask()
        if not (m > 127).any():
            self.damage_path = ""
            self._damage_text("")
            return
        import tempfile
        d = Path(tempfile.gettempdir()) / "net_anh_mask"
        d.mkdir(exist_ok=True)
        path = d / f"{self.src_path.stem}_{int(time.time())}.png"
        ok, buf = cv2.imencode(".png", m)
        if ok:
            buf.tofile(str(path))
            self.damage_path = str(path)
            self._damage_text(f"Vùng hỏng: {(m > 127).mean() * 100:.1f} % ảnh (bấm để sửa)")
            if not self.cb_restore.isChecked():
                self.cb_restore.setChecked(True)
            self._to_custom()

    def _damage_text(self, t: str):
        self.btn_damage.setText(t or "Vẽ vùng hỏng nặng…")
        self.btn_damage2.setText(t or "Tô vùng hỏng nặng…")

    def _to_custom(self):
        if self.preset.currentData() != "custom":
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(self.preset.findData("custom"))
            self.preset.blockSignals(False)
        if self.level_choice is not None or self.level_group.checkedId() > 0:
            self.level_choice = None
            self._check_level(None)
            self.level_desc.setText("Đã chỉnh tay từng thông số (Tuỳ chỉnh). Bấm một mức để quay lại thang mức nét.")
        if hasattr(self, "using_lbl"):
            self._sync_preset_list()

    # ----- thang mức nét -----

    def _check_level(self, n: int | None):
        self.level_group.setExclusive(False)
        for b in self.level_group.buttons():
            b.setChecked(self.level_group.id(b) == n)
        self.level_group.setExclusive(True)
        self._sync_level_bars()

    def _show_level(self, n: int, s: Settings):
        self._check_level(n)
        self.level_desc.setText(describe_level(n, s))
        if hasattr(self, "kind_btns"):
            self._sync_simple()
            self._sync_preset_list()

    def _auto_base(self) -> Settings:
        """Thông số Tự động của ảnh đang mở (giữ hệ số phóng, hãng máy, model đã chọn) làm nền cho mức nét."""
        base = Settings(upscale=self.scale.currentData(), face_model=self.face_model.currentData(),
                        sr_model=self.bg_model.currentData(), phone=self._phone_key(), wb=self.sl_wb.value(),
                        skin_tone=self.sl_skin.value(), exposure=self.sl_expo.value(), tone=self.sl_tone.value(),
                        subject=self.subject.currentData(), keep_lips=self.cb_keep_lips.isChecked(),
                        bg_color=self.bg_color if self.cb_bg.isChecked() else None)
        if self.analysis is None:
            return base
        s = recommend(self.analysis, base)[1]
        return s.with_(upscale=base.upscale, face_model=base.face_model, sr_model=base.sr_model)

    def _level_clicked(self, n: int):
        if n < 6 and self.level_choice == 6:  # rời mức 6: model về mặc định (nhẹ), trừ khi người dùng chọn tay sau đó
            for box, default in ((self.face_model, Settings.face_model), (self.bg_model, Settings.sr_model)):
                box.blockSignals(True)
                box.setCurrentIndex(max(0, box.findData(default)))
                box.blockSignals(False)
        s = level_settings(n, self._auto_base(), self.analysis).with_(**self._user_opts())
        self._apply_settings(s)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData("custom"))
        self.preset.blockSignals(False)
        self.level_choice = n
        self._show_level(n, s)
        self.say(f"Mức nét {n} · {LEVELS[n].name}: {MODES[s.mode]}, {self._scale_text(s)}")

    def _models_changed(self, *_):
        """Đổi model mặt / model nền: không đổi mức nét, chỉ cập nhật mô tả (model, dung lượng cần tải) và cảnh báo
        theo cấu hình máy."""
        n = self.level_group.checkedId()
        if n > 0:
            self.level_desc.setText(describe_level(n, self.settings()))
        self._update_hw_warn()

    def _device_kind(self) -> str:
        if self.device.currentData() == "cpu":
            return "cpu"
        return self._engine.dev.kind if self._engine is not None else self._gpu_kind

    def _fill_devices(self):
        """Tự động / GPU / CPU. GPU không dùng được (không có, hoặc bản cài chỉ có CPU) vẫn hiện nhưng xám, kèm lý do."""
        gpu = next((d for d in self._devices if d.kind != "cpu"), None)
        cpu = self._devices[-1]
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem("Tự động (có GPU thì dùng GPU)", "auto")
        if gpu is not None:
            self.device.addItem(gpu.name + ("" if gpu.half else ", fp32"), "gpu")
            self._gpu_reason = ""
        else:
            unused = hardware.machine().unused_gpu
            if unused is not None:
                self._gpu_reason = hardware.advice(unused)
                label = f"GPU {unused.name} (chưa dùng được)"
            else:
                self._gpu_reason = "Không thấy card đồ hoạ nào dùng được cho AI trên máy này."
                label = "GPU (không có)"
            self.device.addItem(label, "gpu")
            item = self.device.model().item(1)
            item.setEnabled(False)
            item.setToolTip(self._gpu_reason)
        self.device.addItem(cpu.name, "cpu")
        self.device.blockSignals(False)

    def _device_changed(self, *_):
        want = self.device.currentData()
        gpu = next((d for d in self._devices if d.kind != "cpu"), None)
        use = gpu if (want != "cpu" and gpu is not None) else self._devices[-1]
        txt = "Đang dùng: " + use.name
        if use.kind == "cpu" and want == "cpu" and gpu is not None:
            txt += " (bạn chọn CPU)"
        elif use.kind == "cpu" and self._gpu_reason:
            txt += ". " + self._gpu_reason
        self.dev_note.setText(txt)
        self.dev_note.setToolTip(self._gpu_reason)
        self.dev_lbl.setText(("GPU" if use.kind != "cpu" else "CPU") + ": " + use.name.split(" (")[0].replace("GPU ", ""))
        self.dev_lbl.setToolTip(txt + "\nMáy: " + self.hw_text)
        self._update_hw_warn()

    def _update_hw_warn(self, *_):
        """Ước tính thời gian của model đang chọn trên máy này, cảnh báo khi chậm (CPU), thiếu VRAM, thiếu RAM hoặc có
        GPU NVIDIA mà bản cài chỉ chạy CPU. Chỉ cảnh báo: vẫn bấm Làm nét được."""
        if not hasattr(self, "hw_warn"):
            return
        s = self.settings()
        if s.mode in ("quick", "manual"):
            self.hw_warn.hide()
            self.simple_eta.setText("Vài giây")
            self.simple_eta.setStyleSheet("color: #9a9a9a;")
            return
        w, h = (self.src_img.shape[1], self.src_img.shape[0]) if self.src_img is not None else (1600, 1200)
        k = min(1.0, MAX_SIDE / max(w, h))  # ảnh quá lớn được thu nhỏ trước khi xử lý
        w, h = round(w * k), round(h * k)
        a = self.analysis
        faces = 0
        if s.mode != "upscale":
            faces = min(len(a.good_faces), s.max_faces) if a is not None and a.faces_checked else 1
        blur_d = (a.blur_d if a is not None else 1.0) if s.deblur < 0 else max(1.0, s.deblur)
        up = s.upscale or auto_upscale(w, h)
        kind = self._device_kind()
        msgs = hardware.warnings(s.sr_model, s.face_model if faces else None, w, h, faces, blur_d, kind,
                                 torch_threads(), upscale=up)
        t = hardware.estimate(s.sr_model, s.face_model if faces else None, w, h, faces, blur_d, kind, torch_threads(), up)
        self._last_eta = t
        where = "GPU" if kind != "cpu" else "CPU"
        head = f"Ước tính trên máy này ({where}): khoảng {hardware.fmt_seconds(t)}" + \
            ("" if self.src_img is not None else " cho ảnh 2 MP") + "."
        self.hw_warn.setText("⚠ " + "<br>⚠ ".join(msgs) if msgs else head)
        self.hw_warn.setStyleSheet("color: #e0a85a;")
        self.hw_warn.setVisible(bool(msgs))
        self.simple_eta.setText(("⚠ " + msgs[0]) if msgs else f"Khoảng {hardware.fmt_seconds(t)}")
        self.simple_eta.setStyleSheet("color: #e0a85a;" if msgs else "color: #9a9a9a;")

    def _phone_key(self) -> str:
        k = self.phone.currentData()
        return self.detected_phone if k == "auto" else k

    def _phone_changed(self, *_):
        prof = phone_profile(self._phone_key())
        self.sl_wb.set(prof.wb)
        self.sl_skin.set(prof.skin)
        self._to_custom()

    def _mode_changed(self, *_, update_custom: bool = True):
        m = self.mode.currentData()
        self.mode_hint.setText(MODE_HINTS[m])
        self.sl_fid.setEnabled(m in ("portrait", "old"))
        self.cb_detail_same.setEnabled(m in ("portrait", "old"))
        self.cb_keep_lips.setEnabled(m in ("portrait", "old"))
        self.sl_detail.setEnabled(m in ("portrait", "old") and not self.cb_detail_same.isChecked())
        self.face_model.setEnabled(m in ("portrait", "old"))
        self.bg_model.setEnabled(m in ("portrait", "old", "upscale"))
        if update_custom and m == "manual":
            self._to_custom()

    def _preset_changed(self):
        key = self.preset.currentData()
        if key == "custom":
            return
        base = Settings(upscale=self.scale.currentData())
        if key == "auto":
            if self.analysis is None:
                self.say("Chưa có kết quả phân tích ảnh; preset Tự động dùng thông số mặc định.")
                s = base
            else:
                _, s, why = recommend(self.analysis, base)
                self.say("Tự động: " + "; ".join(why))
        else:
            s = apply_preset(base, key)
        s = s.with_(face_model=self.face_model.currentData(), sr_model=self.bg_model.currentData(),
                    subject=self.subject.currentData(), keep_lips=self.cb_keep_lips.isChecked(),
                    bg_color=self.bg_color if self.cb_bg.isChecked() else None, **self._user_opts())
        self._apply_settings(s)
        self.level_choice = None
        # Tự động: mức theo độ mờ đo được (auto_level), không suy ngược từ thông số
        auto = key == "auto" and self.analysis is not None
        self._show_level(auto_level(self.analysis)[0] if auto else level_for(key, s), s)

    def _reset_adjust(self):
        if self.level_choice is not None:
            self._level_clicked(self.level_choice)
        elif self.preset.currentData() != "custom":
            self._preset_changed()
        else:
            self._apply_settings(Settings(mode=self.mode.currentData(), upscale=self.scale.currentData(),
                                          face_model=self.face_model.currentData(), sr_model=self.bg_model.currentData(),
                                          **self._user_opts()))

    # ----- trạng thái -----

    def say(self, msg: str):
        self.log.appendPlainText(msg)
        self.status_lbl.setText(msg.splitlines()[0] if msg else "")

    def _busy(self) -> bool:
        return self.job is not None and self.job.isRunning()

    def _run_clicked(self):
        """Nút Làm nét thành nút Huỷ khi đang chạy."""
        if self._busy():
            self.cancel()
        else:
            self.run_single()

    def _update_buttons(self):
        """Đang làm nét: khoá mọi chỉnh sửa (thông số, preset, lịch sử, mở/đóng ảnh, thiết bị), chỉ còn Huỷ. Thông số
        đọc một lần lúc bấm Làm nét, nên sửa giữa chừng không có tác dụng, chỉ gây hiểu nhầm."""
        busy = self._busy()
        has_src, has_res = self.src_img is not None, bool(self.results)
        for w in (self.a_open, self.a_folder, self.device, self.btn_models, self.a_models):
            w.setEnabled(not busy)
        for w in getattr(self, "_lock_widgets", []) + getattr(self, "_lock_actions", []):
            w.setEnabled(not busy)
        lock_tip = "Đang làm nét: chỉnh sửa bị khoá. Bấm Huỷ (Esc) để dừng."
        self.device.setToolTip(lock_tip if busy else "")
        self.a_run.setEnabled(not busy and has_src)
        for w in (self.btn_run, self.btn_run_simple):
            w.setEnabled(busy or has_src)
            w.setText("Huỷ" if busy else "Làm nét")
            w.setToolTip(lock_tip if busy else "Làm nét (Ctrl+Enter)")
        self.a_cancel.setEnabled(busy)
        self.btn_cancel.setVisible(busy)
        self.a_export.setEnabled(not busy and has_res)
        self.a_close.setEnabled(not busy and has_src)
        self.a_undo.setEnabled(not busy and len(self.results) > 1)
        for w in (self.btn_restore, self.a_restore):
            w.setEnabled(not busy and has_res)
        self.progress.setVisible(busy)
        if not busy and getattr(self, "status_lbl", None) is not None and self.status_lbl.text().startswith("⚠"):
            self.status_lbl.setStyleSheet("")

    def engine(self, progress) -> Enhancer:
        dev = self.device.currentData()
        if self._engine is None or self._engine_dev != dev:
            self._engine = Enhancer(dev, progress=progress)
            self._engine_dev = dev
        self._engine.say = progress
        return self._engine

    def invalidate_engine(self):
        self._engine = None

    def _start(self, fn, on_done, label: str):
        job = Job(fn)
        self.job = job
        self._job_label = label
        self.track.start(getattr(self, "_eta_s", 0.0) if label == "xử lý ảnh" else 0.0)
        job.message.connect(self.say)
        job.message.connect(self._job_msg)
        job.finished.connect(self._tick.stop)
        job.done.connect(on_done)
        job.failed.connect(self._failed)
        job.cancelled.connect(lambda: self.say(f"Đã huỷ: {label}. Ảnh gốc và kết quả trước vẫn giữ nguyên."))
        job.finished.connect(self._update_buttons)
        job.start()
        self._tick.start()
        self._show_progress()
        self._update_buttons()

    def _engine_for(self, job: Job) -> Enhancer:
        eng = self.engine(job.message.emit)
        eng.cancel_requested = lambda: job.stop
        return eng

    def cancel(self):
        if self._busy():
            self.job.stop = True
            self.say("Đang huỷ…")

    def _job_msg(self, msg: str):
        self.track.feed(msg)
        self._show_progress()

    def _show_progress(self):
        if not self._busy():
            return
        t = self.track
        self.progress.setValue(t.pct)
        self.progress.setFormat(t.bar_text())
        for b in (self.btn_run, self.btn_run_simple):
            b.setText(f"Huỷ · {t.pct}%")
        warn = t.warning()
        if warn:
            self.status_lbl.setText("⚠ " + warn)
            self.status_lbl.setStyleSheet("color: #e0a85a;")
        else:
            self.status_lbl.setStyleSheet("")

    def _failed(self, err: str):
        step = self.track.step or "bắt đầu"
        last = [ln for ln in err.strip().splitlines() if ln.strip()]
        reason = last[-1].strip() if last else err
        self.say(f"Lỗi ở bước «{step}» sau {fmt_clock(self.track.elapsed())}:\n" + err)
        self.status_lbl.setStyleSheet("color: #e06c5a;")
        self.status_lbl.setText(f"✖ Lỗi ở bước «{step}»: {reason}")
        QMessageBox.warning(self, APP_NAME, f"Làm nét KHÔNG xong: lỗi ở bước «{step}».\n\nLý do: {reason[:300]}\n\n"
                            "Ảnh gốc và kết quả trước vẫn giữ nguyên. Chi tiết ở Xem → Nhật ký. Có thể thử lại, chọn "
                            "mức nét thấp hơn, hoặc đổi Thiết bị sang CPU.")

    # ----- mở ảnh + phân tích -----

    def choose_file(self):
        if self._busy():
            return
        f, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", FILTER)
        if f:
            self.open_path(f)

    def open_path(self, path: str, analyze: bool = True):
        if self._busy():
            return
        p = Path(path)
        try:
            img = load_image(p)
        except ImageError as e:
            QMessageBox.warning(self, APP_NAME, str(e))
            return
        self.src_path, self.src_img, self.results, self.analysis = p, img, [], None
        self.damage_path = ""  # mặt nạ vùng hỏng thuộc về ảnh cũ
        self._damage_text("")
        self.detected_phone, desc = detect_phone(p)
        self.phone_info.setText(f"Máy chụp: {desc} → bộ lọc {PROFILES[self.detected_phone].label}")
        if self.phone.currentData() == "auto":
            prof = phone_profile(self.detected_phone)
            self.sl_wb.set(prof.wb)
            self.sl_skin.set(prof.skin)
        h, w = img.shape[:2]
        self.view.set_images(to_qimage(img), None)
        self.histo.set_image(img)
        size = f"{w} × {h} · {p.stat().st_size / 1e6:.1f} MB"
        self.file_lbl.setText(f"{p.name} · {w} × {h}")
        self._file_short = f"{size} · {desc}"
        self._set_info(f"<b>{p.name}</b><br>{size} · đang phân tích…", self._file_short)
        self.say(f"Đã mở {p.name} ({w}×{h})")
        self._update_buttons()
        if analyze:
            self._analyze()

    def _analyze(self):
        def work(job: Job):
            eng = self._engine_for(job)
            with_faces = eng.models_ready()
            return eng.analyze(self.src_img, with_faces=with_faces)

        self._start(work, self._analyzed, "phân tích ảnh")

    def _analyzed(self, a: Analysis):
        self.analysis = a
        preset, s, why = recommend(a, Settings(upscale=self.scale.currentData()))
        lines = [f"<b>{self.src_path.name}</b>", a.summary()]
        self.say(f"Gợi ý: {MODES[s.mode]}, preset {PRESETS[preset][0]} ({'; '.join(why)})")
        lines += [f"<span style='color:#e0a85a'>⚠ {w}</span>" for w in a.warnings]
        self._set_info("<br>".join(lines), getattr(self, "_file_short", "") + "<br>" + a.summary())
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData("auto"))
        self.preset.blockSignals(False)
        s = s.with_(face_model=self.face_model.currentData(), sr_model=self.bg_model.currentData(),
                    subject=self.subject.currentData(), keep_lips=self.cb_keep_lips.isChecked(),
                    **self._user_opts())  # khử bóng, tông mẫu là tuỳ chọn của người dùng: mở ảnh mới vẫn giữ
        self._apply_settings(s)
        self.level_choice = None
        self._show_level(auto_level(a)[0], s)
        if self.ui_mode == "simple" and (self.simple_kind, self.simple_power) != ("auto", "auto"):
            self._simple_apply()  # ảnh mới giữ lựa chọn loại ảnh, mức nét của chế độ Đơn giản
        self._update_hw_warn()
        self.say("Phân tích: " + a.summary())

    # ----- xử lý -----

    def _scale_text(self, s: Settings) -> str:
        if s.upscale > 0:
            return f"phóng {s.upscale}×"
        if self.src_img is None:
            return "phóng tự động"
        return f"phóng tự động ({auto_upscale(self.src_img.shape[1], self.src_img.shape[0])}×)"

    def run_single(self):
        if self.src_img is None or self._busy():
            return
        s = self.settings()
        missing = [m for m in models_needed(s) if weights.status(m) != "ready"]
        if missing:
            mb = sum(weights.MANIFEST[m].size for m in missing) / 1e6
            self.say(f"Lần đầu dùng model này: đang tải {len(missing)} model (~{mb:.0f} MB), chỉ một lần.")
        if self.hw_warn.isVisible() and self.hw_warn.text().startswith("⚠"):
            self.say(self.hw_warn.text().replace("<br>", "\n"))
        img = self.src_img
        self._eta_s = getattr(self, "_last_eta", 0.0) if s.mode not in ("quick", "manual") else 5.0
        self.say(f"Bắt đầu: {MODES[s.mode]}, {self._scale_text(s)}")

        def work(job: Job):
            res, rep = self._engine_for(job).process(img, s)
            return res, rep, s

        self._start(work, self._finished, "xử lý ảnh")

    def _finished(self, payload):
        res, rep, s = payload
        self.say(rep.summary())
        self.results.append((res, s, rep.mode, rep.seconds))
        self._models_used[id(res)] = rep.models()
        self.view.after_label = rep.models()
        self.view.set_images(to_qimage(self.src_img), to_qimage(res), keep_view=False)

    def undo(self):
        if len(self.results) > 1 and not self._busy():
            self.results.pop()
            res, s, _, _ = self.results[-1]
            self._apply_settings(s)
            self._to_custom()
            self.view.set_images(to_qimage(self.src_img), to_qimage(res), keep_view=True)
            self.say("Đã hoàn tác về kết quả trước.")
            self._update_buttons()

    def close_image(self):
        if self.src_img is None or self._busy():
            return
        unsaved = self.results and self.results[-1][0] is not getattr(self, "_exported", None)
        if unsaved and QMessageBox.question(
                self, APP_NAME, "Kết quả chưa xuất sẽ mất. Đóng ảnh?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        name = self.src_path.name if self.src_path else "ảnh"
        self.src_path, self.src_img, self.results, self.analysis = None, None, [], None
        self.view.set_images(None, None)
        self.histo.set_image(None)
        self.file_lbl.setText("Chưa mở ảnh")
        self._set_info("Chưa có ảnh")
        self.say(f"Đã đóng {name}")
        self._update_buttons()

    def restore_original(self):
        if self.src_img is None or self._busy():
            return
        self.results = []
        self.view.set_images(to_qimage(self.src_img), None, keep_view=True)
        self.say("Đã khôi phục ảnh gốc (ảnh gốc trên ổ đĩa không bao giờ bị sửa).")
        self._update_buttons()

    def save_result(self):
        if not self.results or self._busy():
            return
        res, s, mode, secs = self.results[-1]
        ext = self.fmt.currentData()
        default = unique_output_path(self.src_path, None, ext)
        f, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", str(default), f"{self.fmt.currentText()} (*{ext})")
        if not f:
            return
        out = Path(f)
        if not out.suffix:
            out = out.with_suffix(ext)
        if out.exists() and out.resolve() == self.src_path.resolve():
            QMessageBox.warning(self, APP_NAME, "Không ghi đè ảnh gốc. Hãy chọn tên khác.")
            return
        long_side = OUT_SIZES[self.out_size.currentData()]
        res_out = resize_long(res, long_side)
        if long_side and long_side > max(res.shape[:2]) * 1.05:
            self.say(f"Ảnh kết quả {res.shape[1]}×{res.shape[0]} nhỏ hơn cỡ xuất: phóng thêm bằng Lanczos")
        try:
            save_image(res_out, out, self.quality.value())
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Lưu thất bại: {e}\nKết quả vẫn còn trong bộ nhớ, bạn có thể thử lại.")
            return
        self.history.add(self.src_path, out, res, mode, {**s.to_dict(), "models_used": self._models_used.get(id(res), "")},
                         secs)
        self._exported = res
        self.say(f"Đã lưu {out}")

    def choose_folder(self):
        if self._busy():
            return
        src = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh cần làm nét")
        if not src:
            return
        files = sorted(f for f in Path(src).iterdir() if f.suffix.lower() in SUPPORTED)
        if not files:
            QMessageBox.information(self, APP_NAME, "Thư mục không có ảnh.")
            return
        dst = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if not dst:
            return
        s, ext, q = self.settings(), self.fmt.currentData(), self.quality.value()
        long_side = OUT_SIZES[self.out_size.currentData()]
        auto = self.preset.currentData() == "auto"
        level = self.level_choice
        self.say(f"Xử lý {len(files)} ảnh" + (f" ở mức nét {level} (chế độ theo từng ảnh)" if level else
                                           " (tự chọn chế độ từng ảnh)" if auto else "") + f", lưu vào {dst}")

        def work(job: Job):
            eng = self._engine_for(job)
            ok = 0
            tally = BatchTally(len(files))
            for i, f in enumerate(files, 1):
                eng._check()
                job.message.emit(f"[{i}/{len(files)}] {f.name}")
                try:
                    img = load_image(f)
                    if level or auto:
                        an = eng.analyze(img, with_faces=eng.models_ready())
                        base = Settings(upscale=s.upscale, face_model=s.face_model, sr_model=s.sr_model,
                                        phone=s.phone, wb=s.wb, skin_tone=s.skin_tone, subject=s.subject,
                                        face_styles=s.face_styles, keep_lips=s.keep_lips, bg_color=s.bg_color,
                                        deshine=s.deshine, look=s.look, look_amount=s.look_amount)
                        si = recommend(an, base)[1]
                        if level:
                            si = level_settings(level, si.with_(upscale=s.upscale), an)
                    else:
                        si = s
                    if self.phone.currentData() == "auto":  # mỗi ảnh một hãng, thông số màu theo hãng đó
                        si = si.with_(phone=detect_phone(f)[0], wb=-1.0, skin_tone=-1.0)
                    res, rep = eng.process(img, si)
                    job.message.emit(f"   {rep.models()}")
                    out = save_image(resize_long(res, long_side), unique_output_path(f, Path(dst), ext), q)
                    self.history.add(f, out, res, rep.mode, {**si.to_dict(), "models_used": rep.models()}, rep.seconds)
                    job.message.emit(f"   → {out.name} · {rep.seconds:.1f} s")
                    ok += 1
                    tally.add(rep, f.name)
                    if rep.ai_fallback:
                        job.message.emit("   ⚠ AI lỗi, ảnh này chỉ qua xử lý nhanh")
                except ImageError as e:
                    tally.fail()
                    job.message.emit(f"   bỏ qua: {e}")
                except (Cancelled, InterruptedError):
                    raise
                except Exception as e:  # một ảnh lỗi (hết RAM, ổ đầy, không ghi được…) không dừng cả thư mục
                    tally.fail()
                    job.message.emit(f"   LỖI: {e.__class__.__name__}: {e}")
            return ok, len(files), tally

        self._start(work, lambda r: self._batch_done(r), "xử lý thư mục")

    def _batch_done(self, r):
        self.say("Xong thư mục: " + r[2].summary())
        if r[2].counts["fallback"]:
            QMessageBox.warning(self, APP_NAME, f"{r[2].counts['fallback']} ảnh chỉ được xử lý nhanh vì model AI lỗi, "
                                "không phải bằng AI. Xem chi tiết trong nhật ký.\n\n" + r[2].summary())

    def download_models(self, extra: tuple[str, ...] = ()):
        def work(job: Job):
            # chỉ model app cần + model đang chọn (không tải ArcFace chỉ dùng để đo, RRDB/RestoreFormer/DAT-2… khi chưa
            # chọn: tải hết là hơn 1,5 GB)
            s = self.settings().with_(mode="portrait")
            need = list(weights.RUNTIME)
            for mid in models_needed(s) + list(extra):
                if mid not in need:
                    need.append(mid)
            for mid in need:
                weights.ensure(mid, job.message.emit, cancel=lambda: job.stop)
            return True

        self._start(work, lambda _: (self.say("Đã tải đủ model."), self.src_img is not None and self._analyze()),
                    "tải model")

    # ----- lịch sử -----

    def _refresh_history(self):
        self.hist_list.clear()
        for e in self.history.list():
            when = time.strftime("%d/%m/%Y %H:%M", time.localtime(e.created))
            used = (e.settings or {}).get("models_used", "")
            it = QListWidgetItem(QIcon(QPixmap(e.thumb)), f"{Path(e.output).name}\n{when} · {MODES.get(e.mode, e.mode)}"
                                 + (f" · {used}" if used else "")
                                 + f" · {e.width}×{e.height} · {e.size_bytes / 1e6:.1f} MB"
                                 + ("" if Path(e.output).exists() else " · (file đã bị xoá)"))
            it.setData(Qt.ItemDataRole.UserRole, e)
            self.hist_list.addItem(it)

    def _open_history_item(self, item):
        if item is None:
            return
        e = item.data(Qt.ItemDataRole.UserRole)
        if not Path(e.output).exists():
            QMessageBox.information(self, APP_NAME, "File ảnh ra không còn trên máy.")
            return
        src = Path(e.source)
        if src.exists():
            self.open_path(str(src), analyze=False)
            try:
                res = load_image(Path(e.output))
                s = Settings(**{k: v for k, v in e.settings.items() if k in Settings.__dataclass_fields__})
                self.results = [(res, s, e.mode, e.seconds)]
                self._apply_settings(s)
                self.view.set_images(to_qimage(self.src_img), to_qimage(res))
                self._update_buttons()
            except ImageError:
                pass
        else:
            self.open_path(e.output)

    def _reveal_history_item(self):
        it = self.hist_list.currentItem()
        if it:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(it.data(Qt.ItemDataRole.UserRole).output).parent)))

    def _delete_history(self, with_file: bool):
        it = self.hist_list.currentItem()
        if not it:
            return
        e = it.data(Qt.ItemDataRole.UserRole)
        if with_file and QMessageBox.question(self, APP_NAME, f"Xoá file {Path(e.output).name}? (ảnh gốc không bị xoá)") \
                != QMessageBox.StandardButton.Yes:
            return
        self.history.delete(e.id, delete_output=with_file)
        self._refresh_history()

    def _clear_history(self):
        if QMessageBox.question(self, APP_NAME, "Xoá toàn bộ lịch sử? File ảnh ra vẫn giữ trên máy.") \
                == QMessageBox.StandardButton.Yes:
            self.history.clear()
            self._refresh_history()

    def closeEvent(self, e):
        if self._busy():
            self.job.stop = True
            self.job.wait(5000)
        self.history.close()
        super().closeEvent(e)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    from PySide6.QtCore import QSettings
    pt = float(QSettings("BeautyHDR", "BeautyHDR").value("ui_pt", 0) or 0) or auto_ui_size(app)
    apply_style(app, pt)
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.open_path(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
