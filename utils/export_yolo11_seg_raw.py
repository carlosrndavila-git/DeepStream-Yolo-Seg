"""
Export YOLO11-Seg to a two-output ONNX compatible with TRT 8.5.x (Jetson JetPack 5.x).

Outputs:
  0 - detections:  float32[1, 8400, 38]
                   layout per anchor: [x1, y1, x2, y2, conf, class_id, mask_coeff_0..31]
  1 - prototypes:  float32[1, 32, 160, 160]

No TRT-specific plugins (EfficientNMSX_TRT / ROIAlignX_TRT) are used.
The custom C++ parser (NvDsInferParseYoloSeg) receives both tensors, filters by
confidence threshold, computes the full 160x160 mask per detection via
  sigmoid(mask_coeffs @ protos.reshape(32, 25600))
and stores it in NvDsInferInstanceMaskInfo.  DeepStream cluster-mode=4 handles
IoU-based NMS after parsing.

Usage (run from the ultralytics repo root):
    python3 export_yolo11_seg_raw.py -w yolo11s-seg.pt --simplify
"""

import os
import sys
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _m

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat([x1y1, x2y2], dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class RawSegOutput(nn.Module):
    """
    Minimal output wrapper: transpose detections, pass prototypes through.

    Input  x[0]: (1, 38, 8400)      -> detections: (1, 8400, 38)
    Input  x[1]: (1, 32, 160, 160)  -> prototypes: (1, 32, 160, 160)  (unchanged)
    """

    def forward(self, x):
        detections = x[0].transpose(1, 2)   # (1, 38, 8400) -> (1, 8400, 38)
        prototypes = x[1]                    # (1, 32, 160, 160)
        return detections, prototypes


def load_model(weights, device, fuse=True):
    model = YOLO(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    if fuse:
        model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args):
    suppress_warnings()
    print(f"\nStarting: {args.weights}")

    device = torch.device("cpu")
    model = load_model(args.weights, device, fuse=False)

    if model.names:
        print("Creating labels.txt")
        with open("labels.txt", "w", encoding="utf-8") as f:
            for name in model.names.values():
                f.write(f"{name}\n")

    model = nn.Sequential(model, RawSegOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size
    dummy = torch.zeros(args.batch, 3, *img_size).to(device)
    out_file = args.weights.rsplit(".", 1)[0] + "_raw.onnx"

    print("Exporting to ONNX")
    torch.onnx.export(
        model,
        dummy,
        out_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["detections", "prototypes"],
        dynamo=False,
    )

    if args.simplify:
        print("Simplifying")
        import onnxslim
        m = onnx.load(out_file)
        m = onnxslim.slim(m)
        onnx.save(m, out_file)

    print(f"Done: {out_file}\n")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="DeepStream YOLO11-Seg raw export (TRT 8.5.x compatible)")
    p.add_argument("-w", "--weights", required=True, type=str)
    p.add_argument("-s", "--size", nargs="+", type=int, default=[640])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    return args


if __name__ == "__main__":
    main(parse_args())
