"""Xuất RetinaFace MobileNet-0.25 (facexlib, MIT, 1,7 MB) sang ONNX để Android dò mặt + 5 điểm mốc.

Vào: 1×3×H×W ảnh BGR float 0..255 (đồ thị tự trừ trung bình 104/117/123). Ra: loc N×4, conf N×2 (softmax), landm N×10,
theo thứ tự anchor của PriorBox (min_sizes [[16,32],[64,128],[256,512]], steps [8,16,32]); giải mã ở phía Kotlin.
    python tools/export_face_onnx.py -o retinaface_mnet025.onnx --ref anh.png ref.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net_anh import weights  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net(x - self.mean)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--ref", nargs=2, metavar=("ANH", "JSON"), help="lưu kết quả dò của facexlib để test phía Android")
    a = p.parse_args()
    from facexlib.detection import init_detection_model
    net = init_detection_model("retinaface_mobile0.25", device="cpu", model_rootpath=str(weights.weights_dir()))
    net.eval()
    w = Wrap(net).eval()
    x = torch.rand(1, 3, 320, 480) * 255
    torch.onnx.export(w, x, a.out, input_names=["input"], output_names=["loc", "conf", "landm"], opset_version=17,
                      dynamic_axes={"input": {2: "h", 3: "w"}, "loc": {1: "n"}, "conf": {1: "n"}, "landm": {1: "n"}},
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    ref = [t.detach().numpy() for t in w(x)]
    got = sess.run(None, {"input": x.numpy()})
    print(f"{a.out}: {Path(a.out).stat().st_size / 1e6:.2f} MB, sai lệch tối đa {max(np.abs(r - g).max() for r, g in zip(ref, got)):.2e}")
    if a.ref:
        img = cv2.imread(a.ref[0])
        with torch.no_grad():
            det = net.detect_faces(img, conf_threshold=0.8, nms_threshold=0.4)
        json.dump([[float(v) for v in d] for d in det], open(a.ref[1], "w"))
        print(f"{len(det)} mặt trong {a.ref[0]}")


if __name__ == "__main__":
    main()
