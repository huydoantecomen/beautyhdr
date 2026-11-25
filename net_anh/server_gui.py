"""Cửa sổ máy chủ cho điện thoại (NetAnh.exe --may-chu, hoặc may_chu.bat): hiện địa chỉ, mã ghép nối, nhật ký.

Tách riêng khỏi gui.py: chỉ dùng server.Server, chạy được cả khi cửa sổ chính không mở.
"""
from __future__ import annotations

import sys
from pathlib import Path
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from . import __version__
from . import server as srvmod


class _Bridge(QObject):
    line = Signal(str)


class ServerWindow(QMainWindow):
    def __init__(self, port: int = srvmod.PORT, device: str = "auto"):
        super().__init__()
        self.setWindowTitle(f"Beauty HDR {__version__} · Máy chủ cho điện thoại")
        self.resize(640, 520)
        self.bridge = _Bridge()
        self.bridge.line.connect(self._append)
        say = self.bridge.line.emit

        w = QWidget()
        lay = QVBoxLayout(w)
        self.status = QLabel("Đang khởi động…")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.addr = QLabel()
        self.addr.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.addr.setWordWrap(True)
        lay.addWidget(self.addr)
        lay.addWidget(QLabel("Mã ghép nối (nhập vào app trên điện thoại, mỗi mã dùng một lần):"))
        self.code = QLabel()
        f = QFont()
        f.setPointSize(30)
        f.setBold(True)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 6)
        self.code.setFont(f)
        self.code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.code)
        row = QHBoxLayout()
        b_code = QPushButton("Đổi mã")
        b_code.clicked.connect(self._new_code)
        b_unpair = QPushButton("Huỷ mọi điện thoại đã ghép")
        b_unpair.clicked.connect(self._unpair)
        self.b_prep = QPushButton("Tải sẵn model Cao nhất (~450 MB)")
        self.b_prep.clicked.connect(self._prepare)
        for b in (b_code, b_unpair, self.b_prep):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        hint = QLabel("Điện thoại và máy tính cần cùng wifi. Ở ngoài: cài Tailscale trên cả hai máy rồi nhập địa chỉ "
                      "100.x.x.x. Windows hỏi quyền tường lửa thì chọn cho phép mạng Riêng tư (Private). "
                      "Đóng cửa sổ này là dừng máy chủ.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.setCentralWidget(w)

        self.srv: srvmod.Server | None = None
        try:
            self.srv = srvmod.Server(port=port, device=device, say=say).start()
        except OSError as e:
            self.status.setText(f"Không mở được cổng {port}: {e}\nCó thể máy chủ Beauty HDR đang chạy ở cửa sổ khác.")
            return
        self.status.setText(f"Máy chủ đang chạy trên {self.srv.worker.eng.dev.name}. Điện thoại tự tìm thấy máy này "
                            f"khi cùng wifi (nút \"Tìm máy tính\").")
        addrs = srvmod.describe_addresses(self.srv.port) or [f"<IP máy này>:{self.srv.port}"]
        self.addr.setText("Hoặc nhập địa chỉ tay:\n" + "\n".join("   " + a for a in addrs))
        self._last_code = ""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(700)
        self._refresh()

    def _append(self, line: str):
        self.log.appendPlainText(line)

    def _refresh(self):
        if not self.srv:
            return
        p = self.srv.pairing
        if p.code != self._last_code:
            self._last_code = p.code
            self.code.setText(f"{p.code[:3]} {p.code[3:]}")
        running, waiting = self.srv.worker.busy()
        job = self.srv.worker.current
        state = f"Đang xử lý: {job.progress}" if running and job else "Đang chờ ảnh"
        self.statusBar().showMessage(f"{state} · chờ {waiting} · đã ghép {len(p.devices)} điện thoại")

    def _new_code(self):
        if self.srv:
            self.srv.pairing.new_code()
            self._refresh()

    def _unpair(self):
        if self.srv and QMessageBox.question(self, "Huỷ ghép nối",
                                             "Mọi điện thoại phải nhập mã lại. Tiếp tục?") == QMessageBox.StandardButton.Yes:
            self.srv.pairing.unpair_all()
            self._append("Đã huỷ mọi điện thoại đã ghép nối.")

    def _prepare(self):
        self.b_prep.setEnabled(False)
        say = self.bridge.line.emit

        def run():
            try:
                srvmod.prepare_models(say)
                say("Đã có đủ model mức 6 Cao nhất.")
            except Exception as e:
                say(f"Tải model lỗi: {e}")

        threading.Thread(target=run, daemon=True).start()

    def closeEvent(self, e):
        if self.srv:
            self.srv.stop()
        super().closeEvent(e)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    port = srvmod.PORT
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "icon.png")))
    win = ServerWindow(port)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
