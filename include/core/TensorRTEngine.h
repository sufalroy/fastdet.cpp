#pragma once

#include "IEngine.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace fastdet::core {

    struct TensorSpec {
        std::string name;
        nvinfer1::Dims shape;
        nvinfer1::DataType dataType;
        nvinfer1::TensorIOMode ioMode;
        size_t byteSize;

        [[nodiscard]] constexpr size_t getElementSize() const noexcept {
            switch (dataType) {
                case nvinfer1::DataType::kFLOAT: return sizeof(float);
                case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
                case nvinfer1::DataType::kINT32: return sizeof(int32_t);
                case nvinfer1::DataType::kBOOL: return sizeof(bool);
                case nvinfer1::DataType::kUINT8: return sizeof(uint8_t);
                default: return 0;
            }
        }

        [[nodiscard]] constexpr size_t getElementCount() const noexcept {
            size_t count = 1;
            for (int32_t i = 0; i < shape.nbDims; ++i) {
                count *= shape.d[i];
            }
            return count;
        }
    };

    class TensorRTEngine : public IEngine {
    public:
        TensorRTEngine();
        ~TensorRTEngine() override;

        TensorRTEngine(const TensorRTEngine &) = delete;
        TensorRTEngine &operator=(const TensorRTEngine &) = delete;
        TensorRTEngine(TensorRTEngine &&other) noexcept;
        TensorRTEngine &operator=(TensorRTEngine &&other) noexcept;

        bool build(const std::string &onnxPath, const Options &options) override;
        
        bool load(const std::string &enginePath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize) override;
        
        [[nodiscard]] const std::vector<TensorSpec> &getTensorSpecs() const noexcept { return mTensorSpecs; }
        
        [[nodiscard]] const Options &getOptions() const noexcept { return mOptions; }

    private:
        [[nodiscard]] std::string generateEnginePath(const std::string &onnxPath) const;
        void clearGpuBuffers();
        
        Options mOptions;

        std::vector<TensorSpec> mTensorSpecs;
        std::vector<void *> mBuffers;
                
        std::array<float, 3> mSubVals{};
        std::array<float, 3> mDivVals{};
        bool mNormalize;

        std::unique_ptr<nvinfer1::IRuntime> mRuntime;
        std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
        std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    };
}