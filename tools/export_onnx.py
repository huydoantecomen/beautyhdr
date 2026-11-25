"""Xuất Real-ESRGAN compact (realesr-general-x4v3) sang ONNX cho Android.

Mức khử nhiễu được trộn sẵn vào trọng số (DNI), mỗi mức là một file 4,7 MB.
    python tools/export_onnx.py --dn 0.5 -o realesr_compact_dn50.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh.pipeline import Enhancer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dn", type=float, default=0.5)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    net = Enhancer("cpu")._load_compact(a.dn).float().eval()
    x = torch.rand(1, 3, 64, 64)
    torch.onnx.export(net, x, a.out, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h4", 3: "w4"}}, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    x = torch.rand(1, 3, 37, 53)
    ref = net(x).detach().numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    print(f"{a.out}: {Path(a.out).stat().st_size / 1e6:.2f} MB, sai lệch tối đa so với PyTorch {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
