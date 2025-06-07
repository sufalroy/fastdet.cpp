#pragma once

#include "IEngine.h"

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

namespace fastdet::inference {
    class TensorRTEngine : public IEngine {
    public:
        TensorRTEngine();

        ~TensorRTEngine() override;

        TensorRTEngine(const TensorRTEngine &) = delete;

        TensorRTEngine &operator=(const TensorRTEngine &) = delete;

        TensorRTEngine(TensorRTEngine &&other) noexcept;

        TensorRTEngine &operator=(TensorRTEngine &&other) noexcept;

        bool build(const std::string &onnxPath, const Options &options) override;

        bool load(const std::string &enginePath, const std::array<float, 3> &subVals,
                  const std::array<float, 3> &divVals, bool normalize) override;

        bool infer(const std::vector<std::vector<cv::cuda::GpuMat> > &input,
                   std::vector<std::vector<std::vector<float> > > &output) override;

        [[nodiscard]] const std::vector<nvinfer1::Dims> &getInputDims() const override { return mInputDims; };

        [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const override { return mOutputDims; };

    private:
        [[nodiscard]] std::string generateEnginePath(const std::string &onnxPath) const;

        void clearGpuBuffers();

        [[nodiscard]] cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput,
                                                       const std::array<float, 3> &subVals,
                                                       const std::array<float, 3> &divVals,
                                                       bool normalize, bool swapRB = false) const;

        Options mOptions;

        std::vector<void *> mBuffers;
        std::vector<std::string> mIOTensorNames;
        std::vector<uint32_t> mOutputLengths;
        std::vector<nvinfer1::Dims> mInputDims;
        std::vector<nvinfer1::Dims> mOutputDims;
        int32_t mInputBatchSize{1};

        std::array<float, 3> mSubVals{};
        std::array<float, 3> mDivVals{};
        bool mNormalize;

        std::unique_ptr<nvinfer1::IRuntime> mRuntime;
        std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
        std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    };
}
