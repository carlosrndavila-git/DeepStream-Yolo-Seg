# DeepStream-Yolo-Seg

NVIDIA DeepStream SDK 8.0 / 7.1 / 7.0 / 6.4 / 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 application for YOLO-Seg models

--------------------------------------------------------------------------------------------------
### YOLO object detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------
### Important: Please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------

### Getting started

* [Supported models](#supported-models)
* [Instructions](#basic-usage)
* [YOLOv5-Seg usage](docs/YOLOv5_Seg.md)
* [YOLOv7-Seg usage](docs/YOLOv7_Seg.md)
* [YOLOv7-Mask usage](docs/YOLOv7_Mask.md)
* [YOLOv8-Seg usage](docs/YOLOv8_Seg.md)
* [YOLO11-Seg usage](docs/YOLO11_Seg.md)
* [RF-DETR-Seg usage](docs/RFDETR_Seg.md)
* [NMS configuration](#nms-configuration)
* [Detection threshold configuration](#detection-threshold-configuration)

##

### Supported models

* [RF-DETR-Seg](https://github.com/roboflow/rf-detr)
* [YOLO11-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv8-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv7-Mask](https://github.com/WongKinYiu/yolov7/tree/mask)
* [YOLOv7-Seg](https://github.com/WongKinYiu/yolov7/tree/u7/seg)
* [YOLOv5-Seg](https://github.com/ultralytics/yolov5)

##

### Instructions

#### 1. Download the DeepStream-Yolo-Seg repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git
cd DeepStream-Yolo-Seg
```

#### 2. Compile the libs

2.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 = 10.2
  ```

2.2. Make the libs

```
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg
```

#### 3. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

##

### NMS configuration

For now, the NMS is configured in the ONNX exporter file.

**NOTE**: Make sure to set `cluster-mode=4` in the config_infer file.

##

### Detection threshold configuration

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
[class-attrs-all]
pre-cluster-threshold=0.25
```

##

My projects: https://www.youtube.com/MarcosLucianoTV

## Standard YOLO11 entry point

`NvDsInferParseYoloSegStandard` is an additional instance-mask callback for the
standard FP32 YOLO11 output contract: `[4 + classes + 32, anchors]` center-xywh
boxes/class scores/mask coefficients and `[32, proto_height, proto_width]`
prototypes, per batch member. It rejects ambiguous channel dimensions and the
legacy anchor-first raw layout. The original `NvDsInferParseYoloSeg` is retained.

Build with the deployment DeepStream/CUDA SDK and OpenCV 4 core/imgproc development
packages (`pkg-config opencv4`). Select `network-type=3`,
`parse-bbox-instance-mask-func-name=NvDsInferParseYoloSegStandard`,
`output-instance-mask=1`, and `cluster-mode=4` (no SDK clustering).

This callback fixes class-aware IoU NMS at 0.45 on unclipped boxes, globally caps
survivors at 300 by confidence with input-order ties, and constructs masks only
for survivors. Masks use sigmoid, strict `>0.5`, uint8 bilinear resize to network
resolution, and a truncating input-space crop. Confidence filtering uses the
SDK-provided per-class pre-cluster thresholds. Return buffers use `new[]`; ownership
passes to DeepStream only after successful parsing. SDK mask rendering threshold
0.5 preserves these binary float masks. `nms-iou-threshold` does not configure the
callback; retain `topk=300` for compatibility with legacy consumers.

Qualification is 640×640, FP32 I/O, batch-one execution on DeepStream 6.2 / TensorRT
8.5.2.2. The callback does not restrict the enclosing batch size because nvinfer
calls it per member. Use an engine built from the standard model on its deployment
host; do not reuse the legacy raw-model engine.
