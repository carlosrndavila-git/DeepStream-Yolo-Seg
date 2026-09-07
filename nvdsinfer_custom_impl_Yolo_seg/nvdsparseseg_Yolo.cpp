/*
 * NvDsInferParseYoloSeg — custom instance-segmentation parser for YOLO11-Seg.
 *
 * Compatible with the two-output ONNX produced by export_yolo11_seg_raw.py:
 *
 *   outputLayersInfo[0]  detections  (numAnchors, numChannels)
 *                        For COCO (nc=80): numChannels = 4 + 80 + 32 = 116
 *                        Layout per anchor: [x1, y1, x2, y2,
 *                                            cls_score_0 .. cls_score_{nc-1},
 *                                            mask_coeff_0 .. mask_coeff_31]
 *                        Boxes  — xyxy pixel coords (decoded by dist2bbox in the model)
 *                        Scores — sigmoid already applied by the Ultralytics export head
 *
 *   outputLayersInfo[1]  prototypes  (numProtos, protoH, protoW)
 *                        e.g. (32, 160, 160)
 *
 * NOTE: The export_yolo11_seg_raw.py comment states "(1, 38, 8400)" but that was
 * written for a 2-class model.  For COCO (nc=80) the actual shape is (1, 116, 8400).
 * numChannels is read directly from inferDims so the parser works for any nc.
 *
 * For each anchor above the per-class confidence threshold the parser:
 *   1. Takes argmax over the nc pre-sigmoid'd class scores.
 *   2. Clips the decoded xyxy bounding box to the network input dimensions.
 *   3. Computes the full protoH×protoW instance mask via
 *        mask[p] = sigmoid( sum_k( coeff[k] * proto[k, p] ) )
 *      and binarizes at 0.5 so nvdsosd renders shape-fitting masks (not rectangles).
 *   4. Crops the mask to the bbox region in proto space.
 *   5. Applies greedy IoU-NMS per class before returning objectList, because
 *      cluster-mode=4 in DeepStream 6.x performs no clustering or deduplication.
 *
 * Config keys required in nvinfer config:
 *   network-type=3
 *   parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg
 *   output-instance-mask=1
 *   cluster-mode=4
 *
 * No EfficientNMSX_TRT / ROIAlignX_TRT plugins required.
 * Compatible with TRT 8.5.x (Jetson JetPack 5.x / DeepStream 6.x).
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

/* Number of mask prototype channels (fixed by the segmentation head). */
static constexpr size_t kNumMaskCoeffs = 32;

/* NMS IoU threshold used inside the parser; the SDK does not pass its config
 * nms-iou-threshold into the instance-mask parser callback. */
static constexpr float kNmsIouThreshold = 0.45f;

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                      NvDsInferNetworkInfo const& networkInfo,
                      NvDsInferParseDetectionParams const& detectionParams,
                      std::vector<NvDsInferInstanceMaskInfo>& objectList);

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static inline float
clamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float
sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float
iou(const NvDsInferInstanceMaskInfo& a, const NvDsInferInstanceMaskInfo& b)
{
    float ix1 = std::max(a.left, b.left);
    float iy1 = std::max(a.top,  b.top);
    float ix2 = std::min(a.left + a.width,  b.left + b.width);
    float iy2 = std::min(a.top  + a.height, b.top  + b.height);
    float iw  = std::max(0.0f, ix2 - ix1);
    float ih  = std::max(0.0f, iy2 - iy1);
    float intersection = iw * ih;
    if (intersection == 0.0f) return 0.0f;
    float aArea = a.width * a.height;
    float bArea = b.width * b.height;
    return intersection / (aArea + bArea - intersection);
}

/* Greedy per-class NMS.  Frees mask memory for suppressed detections. */
static std::vector<NvDsInferInstanceMaskInfo>
applyNMS(std::vector<NvDsInferInstanceMaskInfo>& dets, float iouThresh)
{
    std::sort(dets.begin(), dets.end(),
              [](const NvDsInferInstanceMaskInfo& a, const NvDsInferInstanceMaskInfo& b) {
                  return a.detectionConfidence > b.detectionConfidence;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<NvDsInferInstanceMaskInfo> kept;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) {
            delete[] dets[i].mask;
            continue;
        }
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].classId == dets[j].classId) {
                if (iou(dets[i], dets[j]) > iouThresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return kept;
}

/* ------------------------------------------------------------------ */
/* Core decode                                                          */
/* ------------------------------------------------------------------ */

static std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloSeg(const float* dets,
                    size_t numAnchors,
                    size_t numChannels,   /* e.g. 116 for COCO: [x1,y1,x2,y2, score_0..79, coeff_0..31] */
                    const float* protos,
                    size_t numProtos,     /* 32  */
                    size_t protoH,        /* 160 */
                    size_t protoW,        /* 160 */
                    uint   netW,
                    uint   netH,
                    const std::vector<float>& confThresholds)
{
    std::vector<NvDsInferInstanceMaskInfo> objects;

    /* nc = number of classes; mask coefficients start after boxes + class scores */
    const size_t nc        = numChannels - 4 - kNumMaskCoeffs;  /* e.g. 80 for COCO */
    const size_t maskStart = 4 + nc;                            /* e.g. 84 for COCO */
    const size_t maskPixels = protoH * protoW;

    /* Scale from image pixel space to proto space */
    const float scaleX = static_cast<float>(protoW) / static_cast<float>(netW);
    const float scaleY = static_cast<float>(protoH) / static_cast<float>(netH);

    /* Fallback threshold when perClassPreclusterThreshold is smaller than nc */
    const float fallbackThresh = confThresholds.empty() ? 0.25f : confThresholds[0];

    for (size_t n = 0; n < numAnchors; ++n) {
        const float* row = dets + n * numChannels;

        /* Class scores are pre-sigmoid'd by the Ultralytics export head — take argmax directly. */
        float maxScore = -1.0f;
        int   classId  = -1;
        for (size_t c = 0; c < nc; ++c) {
            float score = row[4 + c];
            if (score > maxScore) {
                maxScore = score;
                classId  = static_cast<int>(c);
            }
        }

        if (classId < 0) continue;

        /* Per-class confidence threshold — fall back to class-0 threshold when
         * perClassPreclusterThreshold has fewer entries than nc. */
        float thresh = (classId < static_cast<int>(confThresholds.size()))
                       ? confThresholds[classId]
                       : fallbackThresh;
        if (maxScore < thresh) continue;

        /* Bounding box — xyxy pixel coordinates, already decoded by dist2bbox in the model */
        float x1 = clamp(row[0], 0.0f, static_cast<float>(netW));
        float y1 = clamp(row[1], 0.0f, static_cast<float>(netH));
        float x2 = clamp(row[2], 0.0f, static_cast<float>(netW));
        float y2 = clamp(row[3], 0.0f, static_cast<float>(netH));

        if ((x2 - x1) < 1.0f || (y2 - y1) < 1.0f) continue;

        /* Compute full protoH×protoW instance mask:
         *   mask[p] = sigmoid( sum_k( coeff[k] * proto[k * maskPixels + p] ) )
         * Protos layout: (numProtos, protoH, protoW) row-major.
         *
         * Binarize at 0.5: nvdsosd renders any non-zero pixel as "inside mask",
         * so continuous sigmoid values would fill the entire bbox.
         */
        std::vector<float> fullMask(maskPixels);
        const float* coeffs = row + maskStart;
        for (size_t p = 0; p < maskPixels; ++p) {
            float acc = 0.0f;
            for (size_t k = 0; k < numProtos; ++k)
                acc += coeffs[k] * protos[k * maskPixels + p];
            fullMask[p] = (sigmoid(acc) >= 0.5f) ? 1.0f : 0.0f;
        }

        /* Crop mask to the bbox region in proto space.
         * nvdsosd scales the stored mask to fill the bounding box — passing the
         * full proto mask would produce a rectangular fill even after binarization. */
        int px1 = std::max(0, static_cast<int>(std::floor(x1 * scaleX)));
        int py1 = std::max(0, static_cast<int>(std::floor(y1 * scaleY)));
        int px2 = std::min(static_cast<int>(protoW), static_cast<int>(std::ceil(x2 * scaleX)));
        int py2 = std::min(static_cast<int>(protoH), static_cast<int>(std::ceil(y2 * scaleY)));

        int cropW = px2 - px1;
        int cropH = py2 - py1;
        if (cropW <= 0 || cropH <= 0) continue;

        float* croppedMask = new float[cropW * cropH];
        for (int cy = 0; cy < cropH; ++cy)
            for (int cx = 0; cx < cropW; ++cx)
                croppedMask[cy * cropW + cx] = fullMask[(py1 + cy) * protoW + (px1 + cx)];

        NvDsInferInstanceMaskInfo b{};
        b.left                = x1;
        b.top                 = y1;
        b.width               = x2 - x1;
        b.height              = y2 - y1;
        b.detectionConfidence = maxScore;
        b.classId             = classId;
        b.mask                = croppedMask;
        b.mask_width          = static_cast<uint>(cropW);
        b.mask_height         = static_cast<uint>(cropH);
        b.mask_size           = sizeof(float) * cropW * cropH;

        objects.push_back(b);
    }

    return applyNMS(objects, kNmsIouThreshold);
}

/* ------------------------------------------------------------------ */
/* Parser entry point                                                   */
/* ------------------------------------------------------------------ */

static bool
NvDsInferParseCustomYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                            NvDsInferNetworkInfo const& networkInfo,
                            NvDsInferParseDetectionParams const& detectionParams,
                            std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    if (outputLayersInfo.size() < 2) {
        std::cerr << "NvDsInferParseYoloSeg: expected 2 output layers "
                     "(detections + prototypes), got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Find detection and prototype layers by number of dims */
    const NvDsInferLayerInfo* detLayer   = nullptr;
    const NvDsInferLayerInfo* protoLayer = nullptr;

    for (const auto& layer : outputLayersInfo) {
        if (layer.inferDims.numDims == 3)
            protoLayer = &layer;   /* (numProtos, protoH, protoW) */
        else if (layer.inferDims.numDims == 2)
            detLayer = &layer;     /* (numAnchors, numChannels) */
    }

    if (!detLayer || !protoLayer) {
        std::cerr << "NvDsInferParseYoloSeg: could not identify detection and "
                     "prototype output layers from dims" << std::endl;
        return false;
    }

    const size_t numAnchors  = static_cast<size_t>(detLayer->inferDims.d[0]);
    const size_t numChannels = static_cast<size_t>(detLayer->inferDims.d[1]);
    const size_t numProtos   = static_cast<size_t>(protoLayer->inferDims.d[0]);
    const size_t protoH      = static_cast<size_t>(protoLayer->inferDims.d[1]);
    const size_t protoW      = static_cast<size_t>(protoLayer->inferDims.d[2]);

    /* Minimum: 4 box + 1 class + 32 mask = 37 channels */
    if (numChannels < 4 + 1 + kNumMaskCoeffs) {
        std::cerr << "NvDsInferParseYoloSeg: detection tensor has only "
                  << numChannels << " channels (need >= "
                  << (4 + 1 + kNumMaskCoeffs) << ")" << std::endl;
        return false;
    }

    objectList = decodeTensorYoloSeg(
        static_cast<const float*>(detLayer->buffer),
        numAnchors, numChannels,
        static_cast<const float*>(protoLayer->buffer),
        numProtos, protoH, protoW,
        networkInfo.width, networkInfo.height,
        detectionParams.perClassPreclusterThreshold);

    return true;
}

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                      NvDsInferNetworkInfo const& networkInfo,
                      NvDsInferParseDetectionParams const& detectionParams,
                      std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
    return NvDsInferParseCustomYoloSeg(outputLayersInfo, networkInfo,
                                       detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloSeg);
