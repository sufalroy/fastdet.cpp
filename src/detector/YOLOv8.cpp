#include "detector/YOLOv8.h"

#include "common/Logger.h"
#include "common/Assert.h"

#include <filesystem>
#include <algorithm>
#include <ranges>
#include <fmt/format.h>
#include <span>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn.hpp>

namespace fastdet::detector {

    const cv::Scalar YOLOv8::BBOX_COLOR{0, 0, 255};
    const cv::Scalar YOLOv8::LABEL_COLOR{255, 255, 255};

    YOLOv8::YOLOv8(std::string &onnxPath,
                   std::string &enginePath,
                   std::span<const std::string> labels,
                   float probabilityThreshold,
                   float nmsThreshold,
                   int topK)
        : mLabels(labels.begin(), labels.end())
        , mProbabilityThreshold(probabilityThreshold)
        , mNmsThreshold(nmsThreshold)
        , mTopK(topK)
    {
        FASTDET_ASSERT_MSG(!labels.empty(), "Labels cannot be empty");
        
        FASTDET_ASSERT_MSG(probabilityThreshold > 0.0f && probabilityThreshold < 1.0f, 
                          "Probability threshold must be in range (0, 1), got: {}", probabilityThreshold);

        FASTDET_ASSERT_MSG(nmsThreshold > 0.0f && nmsThreshold < 1.0f, 
                          "NMS threshold must be in range (0, 1), got: {}", nmsThreshold);

        FASTDET_ASSERT_MSG(topK > 0, "TopK must be positive, got: {}", topK);

        mInferenceEngine = fastdet::inference::EngineFactory::create(fastdet::inference::EngineType::TensorRT);

        FASTDET_ASSERT_MSG(mInferenceEngine != nullptr, "Failed to create TensorRT inference engine");

        if (std::filesystem::exists(enginePath)) {
            FASTDET_LOG_INFO("Loading pre-built TensorRT engine from: {}", enginePath);

            const bool success = mInferenceEngine->load(enginePath, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, true);
            FASTDET_ASSERT_MSG(success, "Failed to load TensorRT engine from: {}", enginePath);
            
            FASTDET_LOG_INFO("Successfully loaded TensorRT engine");
        } else {
            FASTDET_LOG_INFO("Building TensorRT engine from ONNX model: {}", onnxPath);
            
            const fastdet::inference::Options options{
                .optBatchSize = 1,
                .maxBatchSize = 1,
            };

            const bool success = mInferenceEngine->build(onnxPath, options);
            FASTDET_ASSERT_MSG(success, "Failed to build TensorRT engine from ONNX model: {}", onnxPath);
            
            FASTDET_LOG_INFO("Successfully built TensorRT engine");
        }
    }

    std::vector<Detection> YOLOv8::detect(const cv::Mat& image) {
        FASTDET_ASSERT_MSG(!image.empty(), "Input image cannot be empty");
        FASTDET_ASSERT_MSG(image.type() == CV_8UC3, "Input image must be 3-channel BGR format");

        const auto input = preprocess(image);
        
        std::vector<std::vector<std::vector<float>>> output;
        const bool success = mInferenceEngine->infer(input, output);
        FASTDET_ASSERT_MSG(success, "Inference failed");

        const auto& outputDims = mInferenceEngine->getOutputDims();
        FASTDET_ASSERT_MSG(outputDims.size() == 1, "Expected single output, got: {}", outputDims.size());
        FASTDET_ASSERT_MSG(output.size() == 1 && output[0].size() == 1, "Invalid output dimensions structure");

        return postprocess(output[0][0]);
    }

    void YOLOv8::draw(cv::Mat& image, const std::vector<Detection>& detections, unsigned int scale) {
        FASTDET_ASSERT_MSG(!image.empty(), "Image cannot be empty for drawing");
        FASTDET_ASSERT_MSG(scale > 0, "Scale must be positive, got: {}", scale);

        if (detections.empty()) {
            FASTDET_LOG_WARNING("No detections to draw");
            return;
        }

        const int thickness = static_cast<int>(scale) + 1;
        const double labelScale = LABEL_SCALE_FACTOR * scale;

        for (const auto& detection : detections) {
            FASTDET_ASSERT_MSG(detection.label >= 0 && 
                              detection.label < static_cast<int>(mLabels.size()),
                              "Invalid label index: {} (max: {})", 
                              detection.label, mLabels.size() - 1);

            cv::rectangle(image, detection.rect, BBOX_COLOR * 255, thickness);

            const auto labelText = fmt::format("{} {:.1f}%", 
                                             mLabels[detection.label], 
                                             detection.probability * 100.0f);

            int baseLine = 0;
            const cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 
                                                      labelScale, static_cast<int>(scale), &baseLine);

            const cv::Point labelOrigin{
                static_cast<int>(detection.rect.x),
                static_cast<int>(detection.rect.y) + 1
            };

            const cv::Scalar backgroundColorWithAlpha = BBOX_COLOR * BACKGROUND_ALPHA * 255;
            const cv::Rect labelBackground{
                labelOrigin, 
                cv::Size{labelSize.width, labelSize.height + baseLine}
            };
            cv::rectangle(image, labelBackground, backgroundColorWithAlpha, -1);

            const cv::Point textPosition{
                labelOrigin.x, 
                labelOrigin.y + labelSize.height
            };
            cv::putText(image, labelText, textPosition, cv::FONT_HERSHEY_SIMPLEX, 
                       labelScale, LABEL_COLOR, static_cast<int>(scale));
        }

        FASTDET_LOG_VERBOSE("Drew {} detections on image", detections.size());
    }

    std::vector<std::vector<cv::cuda::GpuMat>> YOLOv8::preprocess(const cv::Mat& image) {
        const auto& inputDims = mInferenceEngine->getInputDims();
        FASTDET_ASSERT_MSG(!inputDims.empty(), "No input dimensions available");

        const auto& inputDim = inputDims[0];
        FASTDET_ASSERT_MSG(inputDim.nbDims >= 3, "Invalid input dimensions");

        cv::cuda::GpuMat gpuImage;
        gpuImage.upload(image);

        cv::cuda::GpuMat rgbImage;
        cv::cuda::cvtColor(gpuImage, rgbImage, cv::COLOR_BGR2RGB);

        mImageMetrics.width = static_cast<float>(rgbImage.cols);
        mImageMetrics.height = static_cast<float>(rgbImage.rows);

        cv::cuda::GpuMat processedImage = rgbImage;

        if (rgbImage.rows != inputDim.d[1] || rgbImage.cols != inputDim.d[2]) {
            const float scaleRatio = std::min(
                static_cast<float>(inputDim.d[2]) / static_cast<float>(rgbImage.cols),
                static_cast<float>(inputDim.d[1]) / static_cast<float>(rgbImage.rows)
            );

            const int scaledWidth = static_cast<int>(scaleRatio * rgbImage.cols);
            const int scaledHeight = static_cast<int>(scaleRatio * rgbImage.rows);

            cv::cuda::GpuMat resized(scaledHeight, scaledWidth, CV_8UC3);
            cv::cuda::resize(rgbImage, resized, resized.size());

            processedImage = cv::cuda::GpuMat(inputDim.d[1], inputDim.d[2], CV_8UC3, cv::Scalar::all(0));
            
            resized.copyTo(processedImage(cv::Rect(0, 0, resized.cols, resized.rows)));
        }

        mImageMetrics.ratio = 1.0f / std::min(
            static_cast<float>(inputDim.d[2]) / mImageMetrics.width,
            static_cast<float>(inputDim.d[1]) / mImageMetrics.height
        );

        return {{std::move(processedImage)}};
    }

    std::vector<Detection> YOLOv8::postprocess(std::span<float> features) const {
        const auto& outputDims = mInferenceEngine->getOutputDims();
        FASTDET_ASSERT_MSG(!outputDims.empty(), "No output dimensions available");

        const auto& outputDim = outputDims[0];
        const int numChannels = outputDim.d[1];
        const int numAnchors = outputDim.d[2];
        const int numClasses = static_cast<int>(mLabels.size());

        FASTDET_ASSERT_MSG(features.size() == static_cast<size_t>(numAnchors * numChannels),
                          "Feature size mismatch: expected {}, got {}", 
                          numAnchors * numChannels, features.size());

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> labels;
        
        bboxes.reserve(numAnchors);
        scores.reserve(numAnchors);
        labels.reserve(numAnchors);

        cv::Mat output(numChannels, numAnchors, CV_32F, features.data());
        output = output.t();

        for (int i = 0; i < numAnchors; ++i) {
            const float* rowPtr = output.row(i).ptr<float>();
            const float* bboxPtr = rowPtr;
            const float* scoresPtr = rowPtr + 4;

            const auto maxScoreIter = std::max_element(scoresPtr, scoresPtr + numClasses);
            const float maxScore = *maxScoreIter;

            if (maxScore <= mProbabilityThreshold) {
                continue;
            }

            const float centerX = bboxPtr[0];
            const float centerY = bboxPtr[1];
            const float width = bboxPtr[2];
            const float height = bboxPtr[3];

            const float x0 = std::clamp((centerX - 0.5f * width) * mImageMetrics.ratio, 
                                       0.0f, mImageMetrics.width);
            const float y0 = std::clamp((centerY - 0.5f * height) * mImageMetrics.ratio, 
                                       0.0f, mImageMetrics.height);
            const float x1 = std::clamp((centerX + 0.5f * width) * mImageMetrics.ratio, 
                                       0.0f, mImageMetrics.width);
            const float y1 = std::clamp((centerY + 0.5f * height) * mImageMetrics.ratio, 
                                       0.0f, mImageMetrics.height);

            const int classLabel = static_cast<int>(maxScoreIter - scoresPtr);

            bboxes.emplace_back(x0, y0, x1 - x0, y1 - y0);
            scores.emplace_back(maxScore);
            labels.emplace_back(classLabel);
        }

        if (bboxes.empty()) {
            FASTDET_LOG_WARNING("No detections above threshold {}", mProbabilityThreshold);
            return {};
        }

        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxesBatched(bboxes, scores, labels, 
                                mProbabilityThreshold, mNmsThreshold, nmsIndices);

        std::vector<Detection> detections;
        detections.reserve(std::min(static_cast<int>(nmsIndices.size()), mTopK));

        const auto selectedIndices = nmsIndices 
            | std::views::take(mTopK)
            | std::views::transform([&](int idx) -> Detection {
                return {
                    .label = labels[idx],
                    .probability = scores[idx],
                    .rect = bboxes[idx]
                };
            });

        detections.assign(selectedIndices.begin(), selectedIndices.end());

        FASTDET_LOG_VERBOSE("Post-processing complete: {} detections after NMS", detections.size());
        return detections;
    }

}