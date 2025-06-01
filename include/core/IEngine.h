#pragma once

#include <string>
#include <array>
#include <vector>
#include <cstdint>
#include <opencv2/core/cuda.hpp>

namespace fastdet::core {
    enum class Precision : uint8_t {
        FP32 = 0,
        FP16 = 1
    };

    struct Options {
        Precision precision = Precision::FP16;
        int32_t batchSize = 1;
        int32_t inputWidth = 640;
        int32_t inputHeight = 640;
        std::string engineDir = "./engines";
    };

    class IEngine {
    public:
        virtual ~IEngine() = default;

        virtual bool build(const std::string &onnxPath, const Options &options) = 0;

        virtual bool load(const std::string &enginePath, const std::array<float, 3> &subVals,
                          const std::array<float, 3> &divVals, bool normalize) = 0;

        virtual bool infer(const std::vector<std::vector<cv::cuda::GpuMat> > &inputs,
                           std::vector<std::vector<std::vector<float> > > &outputs) = 0;
    };
}
