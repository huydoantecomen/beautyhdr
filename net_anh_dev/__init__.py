"""Beauty HDR bản nhà phát triển (chỉ có trong gói PC Full bản dev, KHÔNG có trong bản phát hành thường).

Gói này nằm NGOÀI net_anh nên mọi cách đóng gói thường (zip PC Full / Cập nhật, exe PyInstaller, bộ cài, máy chủ,
Android) đều không chứa nó; tests/test_dev_exclusion.py chặn việc vô tình đưa vào. Chạy bản dev:

    python -m net_anh_dev            mở app như thường, có thêm menu "Nhà phát triển"
    python -m net_anh_dev prompt ... in prompt từ dòng lệnh (xem net_anh_dev/cli.py)
    python net_anh_dev/dong_goi.py   tạo BeautyHDR-<phiên bản>-dev-TronBo.zip (PC Full bản dev)

Chức năng hiện có: xuất prompt theo thông số tinh chỉnh đang dùng để dán sang hệ render ngoài (ChatGPT, Gemini, Grok).
"""
DEV_BUILD = True
