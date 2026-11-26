"""Điểm vào cho PyInstaller và cho lệnh `python app.py`.

    NetAnh.exe                    mở giao diện (có thể kèm đường dẫn ảnh)
    NetAnh.exe --tu-kiem          tự kiểm tra, ghi tu_kiem.log
    NetAnh.exe --tai-model DIR    tải sẵn model
    NetAnh.exe --cli ...          chạy dòng lệnh (như python -m net_anh.cli)
    NetAnh.exe --may-chu          máy chủ cho app điện thoại (cửa sổ hiện mã ghép nối)
    NetAnh.exe --do-gpu           đo tốc độ card (fp32/fp16/bf16, bộ nhớ, cả quy trình), ghi file kết quả
"""
import sys

if __name__ == "__main__":
    import multiprocessing
    import os

    # bản .exe không có cửa sổ dòng lệnh: stdout/stderr là None, thư viện ghi log vào đó sẽ lỗi
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))

    multiprocessing.freeze_support()
    import net_anh  # noqa: F401  ép stdout/stderr sang UTF-8 (xem net_anh/__init__.py)
    from net_anh.selftest import dispatch

    code = dispatch(sys.argv[1:])
    if code is None:
        from net_anh.gui import main

        code = main()
    raise SystemExit(code)
