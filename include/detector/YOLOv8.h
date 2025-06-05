#pragma once

#include "IDetector.h"
#include "inference/EngineFactory.h"
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

namespace fastdet::detector {

    class YOLOv8 : public IDetector {
    public:
        YOLOv8(const std::string &onnxPath,
            const std::string &enginePath,
            const std::vector<std::string> &labels,
            float probabilityThreshold,
            float nmsThreshold);

        ~YOLOv8() override = default;

        YOLOv8(const YOLOv8 &) = delete;

        YOLOv8 &operator=(const YOLOv8 &) = delete;

        YOLOv8(YOLOv8 &&other) noexcept = default;

        YOLOv8 &operator=(YOLOv8 &&other) noexcept = default;

        std::vector<Detection> detect(const cv::Mat &image) override;

        void draw(cv::Mat &image, const std::vector<Detection> &detections, unsigned int scale) override;
        
    private:
        std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::Mat &image);

        std::vector<Detection> postprocess(std::vector<float> &features);
        
        std::unique_ptr<fastdet::inference::IEngine> mInferenceEngine;
        
        float mRatio = 1.0f;
        float mImgWidth = 0.0f;
        float mImgHeight = 0.0f;

        const float mProbabilityThreshold;
        const float mNmsThreshold;
        const int mTopK;

        std::vector<std::string> mLabels;
    };
}