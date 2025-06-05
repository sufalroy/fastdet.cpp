#pragma once

#include "IDetector.h"
#include "inference/EngineFactory.h"

#include <memory>
#include <vector>
#include <string>
#include <span>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

namespace fastdet::detector {
    
    class YOLOv8 final : public IDetector {
    public:
        explicit YOLOv8(std::string &onnxPath,
                       std::string &enginePath,
                       std::span<const std::string> labels,
                       float probabilityThreshold = 0.25f,
                       float nmsThreshold = 0.45f,
                       int topK = 100);

        ~YOLOv8() override = default;

        YOLOv8(const YOLOv8&) = delete;

        YOLOv8& operator=(const YOLOv8&) = delete;

        YOLOv8(YOLOv8&&) noexcept = default;

        YOLOv8& operator=(YOLOv8&&) noexcept = default;

        [[nodiscard]] std::vector<Detection> detect(const cv::Mat& image) override;

        void draw(cv::Mat& image, const std::vector<Detection>& detections, unsigned int scale = 1) override;
        
    private:
        struct ImageMetrics {
            float ratio{1.0f};
            float width{0.0f};
            float height{0.0f};
        };

        [[nodiscard]] std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::Mat& image);

        [[nodiscard]] std::vector<Detection> postprocess(std::span<float> features) const;
        
        std::unique_ptr<fastdet::inference::IEngine> mInferenceEngine;
        mutable ImageMetrics mImageMetrics;
        
        const std::vector<std::string> mLabels;
        const float mProbabilityThreshold;
        const float mNmsThreshold;
        const int mTopK;
        
        static const cv::Scalar BBOX_COLOR;
        static const cv::Scalar LABEL_COLOR;
        static constexpr double LABEL_SCALE_FACTOR = 0.35;
        static constexpr double BACKGROUND_ALPHA = 0.7;
    };

}