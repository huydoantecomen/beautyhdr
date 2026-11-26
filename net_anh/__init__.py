"""Beauty HDR: phần mềm làm nét ảnh chạy cục bộ bằng GPU/CPU của PC."""
__version__ = "0.14.1"


def _utf8_streams() -> None:
    """Console Windows tiếng Anh dùng cp1252: print chữ Việt có dấu làm app sập (UnicodeEncodeError, lỗi 0.13.0 trên máy
    huy khi chạy python app.py --may-chu). Ép stdout/stderr sang UTF-8, ký tự không hiện được thì thay bằng ?; chạy ngay
    khi nạp gói nên phủ mọi lối vào (app.py, python -m net_anh.*, máy chủ, dòng lệnh, exe). stdout là None (pythonw,
    exe không console) thì bỏ qua."""
    import sys

    for name in ("stdout", "stderr"):
        st = getattr(sys, name, None)
        try:
            if st is not None and (getattr(st, "encoding", "") or "").lower().replace("-", "") != "utf8":
                st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_utf8_streams()
