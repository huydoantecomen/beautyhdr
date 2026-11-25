"""Kiểm torch và torchvision cài trong môi trường build có khớp nhau không (build_windows.ps1 gọi trước PyInstaller).

Lỗi 0.8.1 trên máy huy: exe báo "operator torchvision::nms does not exist" vì torchvision không nạp được phần C++
(_C) của nó, thường do torchvision được build cho một bản torch khác bản đang cài (môi trường build cũ dùng lại, hoặc
pip lấy torchvision từ PyPI thay vì từ kho của PyTorch). Mã thoát 0 = khớp, 1 = phải cài lại.
"""
import sys

try:
    import torch
    import torchvision
    from torchvision.ops import nms

    keep = nms(torch.tensor([[0.0, 0, 10, 10], [1, 1, 10, 10]]), torch.tensor([0.9, 0.8]), 0.5)
    assert keep.tolist() == [0], keep
except Exception as e:  # noqa: BLE001
    print(f"torch/torchvision KHÔNG khớp: {e.__class__.__name__}: {e}")
    sys.exit(1)
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}: khớp, nms chạy được")
