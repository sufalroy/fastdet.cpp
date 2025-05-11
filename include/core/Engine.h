#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <concepts>
#include <type_traits>

#include "common/Logger.h"
#include "common/Assert.h"

namespace fastdet::core {

    enum class Precision : uint8_t {
        FP32 = 0,
        FP16 = 1
    };

    struct Options {
        Precision precision = Precision::FP16;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 4;
        int32_t minInputWidth = 640;
        int32_t optInputWidth = 640;
        int32_t maxInputWidth = 640;
        std::string engineFileDir = "./engines";
    };

    class IEngine {
    public:
        virtual ~IEngine() = default;
        virtual bool build(const std::string& onnxPath, const Options& options) = 0;
    };

    class Engine : public IEngine {
    public:
        Engine();
        ~Engine() override;

        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;

        Engine(Engine&& other) noexcept;
        Engine& operator=(Engine&& other) noexcept;

        bool build(const std::string& onnxPath, const Options& options) override;

    private:
        [[nodiscard]] std::string generateEnginePath(const std::string& onnxPath) const;

        std::unique_ptr<nvinfer1::IRuntime> mRuntime;
        std::unique_ptr<nvinfer1::ICudaEngine> mCudaEngine;
        std::unique_ptr<nvinfer1::IExecutionContext> mExecutionContext;
        Options mOptions;
    };
}