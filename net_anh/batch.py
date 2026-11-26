"""Xử lý cả thư mục nhanh hơn: model nạp một lần (mỗi Enhancer giữ model suốt lô), đọc ảnh kế tiếp trước trong lúc ảnh
hiện tại đang chạy, ghi file ra ở luồng riêng, và trên GPU nhiều bộ nhớ thì chạy 2 ảnh song song (mỗi luồng một bộ
model, xem gpu_profile.batch_workers).

Dùng chung cho giao diện (gui.py) và dòng lệnh (cli.py). Không phụ thuộc Qt để kiểm được bằng pytest.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Sequence

_END = object()


class Loaded:
    """Ảnh đã đọc sẵn (hoặc lỗi khi đọc, để báo đúng ảnh đó)."""

    __slots__ = ("index", "item", "value", "error")

    def __init__(self, index: int, item: Any, value: Any = None, error: BaseException | None = None):
        self.index, self.item, self.value, self.error = index, item, value, error

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value


class BatchRunner:
    """`engines`: danh sách bộ xử lý (mỗi cái chỉ một luồng dùng). `load(item)` đọc ảnh, chạy ở luồng đọc trước.
    `work(engine, loaded)` xử lý một ảnh; được gọi lần lượt theo thứ tự với 1 engine, song song với nhiều engine
    (engine thứ hai chỉ bắt đầu sau khi engine đầu xong ảnh đầu tiên: model đã tải đủ về máy, không tải trùng).
    Lỗi Cancelled / InterruptedError trong `work` dừng cả lô và được ném lại cho người gọi; lỗi khác `work` tự xử lý."""

    def __init__(self, engines: Sequence[Any], load: Callable[[Any], Any], prefetch: int = 2,
                 stop_errors: tuple[type, ...] = (InterruptedError,)):
        if not engines:
            raise ValueError("cần ít nhất một engine")
        self.engines = list(engines)
        self.load = load
        self.prefetch = max(1, prefetch)
        self.stop_errors = stop_errors
        self.lock = threading.Lock()  # cho người gọi: lịch sử, bảng tổng kết dùng chung giữa các luồng
        self._saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-save")
        self._saves: list[Future] = []
        self._stop = threading.Event()

    # ghi file ở luồng riêng: GPU làm ảnh sau trong lúc ảnh trước đang nén JPG / ghi đĩa
    def later(self, fn: Callable[[], Any]) -> Future:
        f = self._saver.submit(fn)
        with self.lock:
            self._saves.append(f)
        return f

    def stop(self) -> None:
        self._stop.set()

    def run(self, items: Sequence[Any], work: Callable[[Any, Loaded], Any]) -> None:
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)

        def reader():
            for i, it in enumerate(items):
                if self._stop.is_set():
                    break
                try:
                    lo = Loaded(i, it, self.load(it))
                except BaseException as e:  # noqa: BLE001 - báo lại ở đúng ảnh
                    lo = Loaded(i, it, error=e)
                while not self._stop.is_set():
                    try:
                        q.put(lo, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            for _ in self.engines:
                while True:
                    try:
                        q.put(_END, timeout=0.2)
                        break
                    except queue.Full:
                        if self._stop.is_set():
                            _drain(q)

        rt = threading.Thread(target=reader, name="batch-read", daemon=True)
        rt.start()
        first_done = threading.Event()
        failure: list[BaseException] = []

        def loop(eng, idx: int):
            if idx > 0:
                while not first_done.wait(0.2):
                    if self._stop.is_set():
                        return
            while not self._stop.is_set():
                lo = q.get()
                if lo is _END:
                    return
                try:
                    work(eng, lo)
                except self.stop_errors as e:
                    failure.append(e)
                    self._stop.set()
                    return
                except BaseException as e:  # noqa: BLE001 - lỗi lạ ở một luồng: dừng lô, báo ra ngoài
                    failure.append(e)
                    self._stop.set()
                    return
                finally:
                    if idx == 0:
                        first_done.set()
            return

        try:
            if len(self.engines) == 1:
                loop(self.engines[0], 0)
            else:
                ts = [threading.Thread(target=loop, args=(e, i), name=f"batch-{i}", daemon=True)
                      for i, e in enumerate(self.engines)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
        finally:
            if failure:
                self._stop.set()
            _drain(q)
            rt.join(timeout=5)
            for f in list(self._saves):
                try:
                    f.result()
                except BaseException:  # noqa: BLE001 - hàm ghi tự báo lỗi của nó
                    pass
            self._saver.shutdown(wait=True)
        if failure:
            raise failure[0]


def _drain(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def workers_for(eng) -> int:
    """Số ảnh song song cho engine này (theo dòng card, VRAM, RAM, số luồng CPU)."""
    from . import gpu_profile, hardware

    m = hardware.machine()
    return gpu_profile.batch_workers(eng.dev.kind, getattr(eng.dev, "vram_gb", 0.0), m.ram_gb, m.threads)


def extra_engines(first, n: int, make: Callable[[], Any]) -> list:
    """[first] + (n − 1) engine mới cùng thiết bị. Mỗi engine dùng 1/n bộ nhớ GPU khi tự chọn cỡ ô."""
    engines = [first] + [make() for _ in range(max(0, n - 1))]
    for e in engines:
        if hasattr(e, "vram_share"):
            e.vram_share = 1.0 / len(engines)
    return engines


__all__ = ["BatchRunner", "Loaded", "extra_engines", "workers_for"]
