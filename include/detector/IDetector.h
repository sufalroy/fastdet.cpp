#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace fastdet::detector {

    struct Detection {
        int label{};

        float probability{};

        cv::Rect_<float> rect;
    };

    class IDetector {
    public:
        virtual ~IDetector() = default;

        virtual std::vector<Detection> detect(const cv::Mat& image) = 0;

        virtual void draw(cv::Mat& image, const std::vector<Detection>& detections, unsigned int scale) = 0;
    };
}