#pragma once

#include <string>
#include <array>
#include <vector>
#include <cstdint>
#include <opencv2/core/cuda.hpp>
#include "NvInfer.h"

namespace fastdet::inference {
    enum class Precision : uint8_t {
        FP32 = 0,
        FP16 = 1
    };

    struct Options {
        Precision precision = Precision::FP16;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 1;
        int32_t maxInputWidth = -1;
        int32_t minInputWidth = -1;
        int32_t optInputWidth = -1;
        std::string engineDir = "./engines";
    };

    class IEngine {
    public:
        virtual ~IEngine() = default;

        virtual bool build(const std::string &onnxPath, const Options &options) = 0;

        virtual bool load(const std::string &enginePath, const std::array<float, 3> &subVals,
                          const std::array<float, 3> &divVals, bool normalize) = 0;

        virtual bool infer(const std::vector<std::vector<cv::cuda::GpuMat> > &input,
                           std::vector<std::vector<std::vector<float> > > &output) = 0;

        virtual const std::vector<nvinfer1::Dims> &getInputDims() const = 0;
        
        virtual const std::vector<nvinfer1::Dims> &getOutputDims() const = 0;
    };
}
