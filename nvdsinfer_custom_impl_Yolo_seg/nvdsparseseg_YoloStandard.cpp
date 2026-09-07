/* Standard YOLO11 segmentation: [4+classes+32, anchors] center-xywh detections
 * and [32, protoH, protoW] prototypes, per DeepStream batch member.
 *
 * Fixed contract: class-aware IoU NMS 0.45, strict mask threshold >0.5,
 * global top 300, stable input-order ties. Confidence thresholds come from
 * NvDsInferParseDetectionParams. cluster-mode=4 disables SDK clustering;
 * config nms-iou-threshold does not configure this function.
 *
 * Masks follow eager: sigmoid -> binary uint8 -> OpenCV INTER_LINEAR resize
 * to network dimensions -> truncating input-space crop. The SDK owns the
 * returned new[] mask buffers. This entry point never accepts the legacy
 * raw export's anchor-first/corner-box contract.
 */
#include "nvdsinfer_custom_impl.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {
constexpr unsigned kCoefficients = 32;
constexpr size_t kMaxDetections = 300;
constexpr double kNmsThreshold = 0.45;

struct Candidate {
    double x1, y1, x2, y2;
    float score;
    unsigned classId;
    size_t anchor;
};

bool validLayer(const NvDsInferLayerInfo& layer, unsigned rank)
{
    if (!layer.buffer || layer.dataType != FLOAT || layer.inferDims.numDims != rank)
        return false;
    size_t size = 1;
    for (unsigned i = 0; i < rank; ++i) {
        if (layer.inferDims.d[i] <= 0 ||
            size > std::numeric_limits<unsigned>::max() / layer.inferDims.d[i])
            return false;
        size *= layer.inferDims.d[i];
    }
    return size == layer.inferDims.numElements;
}

double overlap(const Candidate& a, const Candidate& b)
{
    double area = std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                  std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    double total = (a.x2 - a.x1) * (a.y2 - a.y1) +
                   (b.x2 - b.x1) * (b.y2 - b.y1) - area;
    return total > 0 ? area / total : 0;
}

double clip(double value, unsigned limit)
{
    return std::max(0.0, std::min(value, static_cast<double>(limit)));
}
} // namespace

extern "C" bool NvDsInferParseYoloSegStandard(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferInstanceMaskInfo>& objects)
{
    // All validation precedes allocating or publishing any SDK-owned masks.
    const NvDsInferLayerInfo *det = nullptr, *proto = nullptr;
    if (layers.size() != 2 || !network.width || !network.height ||
        network.width > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        network.height > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
        !params.numClassesConfigured || params.numClassesConfigured > 100000 ||
        params.perClassPreclusterThreshold.size() < params.numClassesConfigured)
        return false;
    for (const auto& layer : layers) {
        if (layer.inferDims.numDims == 2 && !det) det = &layer;
        else if (layer.inferDims.numDims == 3 && !proto) proto = &layer;
        else return false;
    }
    if (!det || !proto || !validLayer(*det, 2) || !validLayer(*proto, 3)) return false;
    const unsigned channels = 4 + params.numClassesConfigured + kCoefficients;
    const bool first = det->inferDims.d[0] == channels;
    const bool second = det->inferDims.d[1] == channels;
    if (first == second || !first || proto->inferDims.d[0] != kCoefficients)
        return false;
    const size_t anchors = det->inferDims.d[1];
    const auto* values = static_cast<const float*>(det->buffer);
    const auto* prototypes = static_cast<const float*>(proto->buffer);
    const int ph = proto->inferDims.d[1], pw = proto->inferDims.d[2];
    if (static_cast<size_t>(ph) * pw > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    for (float threshold : params.perClassPreclusterThreshold)
        if (!std::isfinite(threshold)) return false;

    std::vector<Candidate> candidates;
    for (size_t n = 0; n < anchors; ++n) {
        unsigned cls = 0;
        float score = values[4 * anchors + n];
        for (unsigned c = 1; c < params.numClassesConfigured; ++c) {
            float value = values[(4 + c) * anchors + n];
            if (value > score) { score = value; cls = c; }
        }
        if (!std::isfinite(score) || score < params.perClassPreclusterThreshold[cls]) continue;
        double cx = values[n], cy = values[anchors + n];
        double w = values[2 * anchors + n], h = values[3 * anchors + n];
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
            !std::isfinite(h) || w <= 0 || h <= 0) continue;
        candidates.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, cls, n});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<Candidate> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& prior : kept) {
            if (candidate.classId == prior.classId && overlap(candidate, prior) > kNmsThreshold) {
                suppressed = true; break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
        if (kept.size() == kMaxDetections) break;
    }

    std::vector<NvDsInferInstanceMaskInfo> decoded;
    std::vector<std::unique_ptr<float[]>> ownedMasks;
    try {
        decoded.reserve(kept.size()); ownedMasks.reserve(kept.size());
        cv::Mat basis(kCoefficients, ph * pw, CV_32F, const_cast<float*>(prototypes));
        for (const auto& candidate : kept) {
            cv::Mat coefficients(1, kCoefficients, CV_32F);
            for (unsigned k = 0; k < kCoefficients; ++k) {
                float value = values[(4 + params.numClassesConfigured + k) * anchors + candidate.anchor];
                if (!std::isfinite(value)) return false;
                coefficients.at<float>(0, k) = value;
            }
            cv::Mat logits = coefficients * basis;
            cv::Mat binary(ph, pw, CV_8U);
            for (int p = 0; p < ph * pw; ++p) {
                const float value = logits.at<float>(0, p);
                if (!std::isfinite(value)) return false;
                binary.data[p] = (1.0f / (1.0f + std::exp(-value))) > 0.5f ? 1 : 0;
            }
            cv::Mat full;
            cv::resize(binary, full, cv::Size(network.width, network.height), 0, 0, cv::INTER_LINEAR);
            double x1 = clip(candidate.x1, network.width), y1 = clip(candidate.y1, network.height);
            double x2 = clip(candidate.x2, network.width), y2 = clip(candidate.y2, network.height);
            int ix1 = static_cast<int>(x1), iy1 = static_cast<int>(y1);
            int cropW = static_cast<int>(x2) - ix1, cropH = static_cast<int>(y2) - iy1;
            // Preserve the selected instance even when its visible mask is empty.
            unsigned mw = std::max(1, cropW), mh = std::max(1, cropH);
            auto mask = std::unique_ptr<float[]>(new float[static_cast<size_t>(mw) * mh]());
            for (int y = 0; y < cropH; ++y)
                for (int x = 0; x < cropW; ++x)
                    mask[y * mw + x] = full.at<unsigned char>(iy1 + y, ix1 + x);
            NvDsInferInstanceMaskInfo object{};
            object.left = x1; object.top = y1; object.width = x2 - x1; object.height = y2 - y1;
            object.detectionConfidence = candidate.score; object.classId = candidate.classId;
            object.mask = mask.get(); object.mask_width = mw; object.mask_height = mh;
            object.mask_size = sizeof(float) * static_cast<size_t>(mw) * mh;
            decoded.push_back(object); ownedMasks.push_back(std::move(mask));
        }
        objects.insert(objects.end(), decoded.begin(), decoded.end());
        for (auto& mask : ownedMasks) mask.release();
    } catch (const std::exception& error) {
        std::cerr << "NvDsInferParseYoloSegStandard: " << error.what() << std::endl;
        return false;
    }
    return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloSegStandard);
