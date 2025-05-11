#include "core/Engine.h"

#include <fstream>
#include <format>
#include <filesystem>

namespace fastdet::core {

    Engine::Engine()
        : mRuntime(nullptr), mCudaEngine(nullptr), mExecutionContext(nullptr), mOptions{} {
    }

    Engine::~Engine() = default;

    Engine::Engine(Engine&& other) noexcept
        : mRuntime(std::move(other.mRuntime)),
          mCudaEngine(std::move(other.mCudaEngine)),
          mExecutionContext(std::move(other.mExecutionContext)),
          mOptions(other.mOptions) {
    }

    Engine& Engine::operator=(Engine&& other) noexcept {
        if (this != &other) {
            mRuntime = std::move(other.mRuntime);
            mCudaEngine = std::move(other.mCudaEngine);
            mExecutionContext = std::move(other.mExecutionContext);
            const_cast<Options&>(mOptions) = other.mOptions;
        }
        return *this;
    }

    std::string Engine::generateEnginePath(const std::string& onnxPath) const {
        std::filesystem::path p(onnxPath);
        std::string baseName = p.stem().string();
        std::string precStr = (mOptions.precision == Precision::FP16) ? "fp16" : "fp32";

        std::string batchStr = (mOptions.optBatchSize == mOptions.maxBatchSize)
            ? std::to_string(mOptions.optBatchSize)
            : std::format("{}-{}", mOptions.optBatchSize, mOptions.maxBatchSize);

        std::string widthStr = (mOptions.minInputWidth == mOptions.maxInputWidth)
            ? std::to_string(mOptions.optInputWidth)
            : std::format("{}-{}-{}", mOptions.minInputWidth, mOptions.optInputWidth, mOptions.maxInputWidth);

        std::string filename = std::format("{}_{}_b{}_w{}.engine", baseName, precStr, batchStr, widthStr);
        return (std::filesystem::path(mOptions.engineFileDir) / filename).string();
    }

    bool Engine::build(const std::string& onnxPath, const Options& options) {
        const_cast<Options&>(mOptions) = options;

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        FASTDET_ASSERT_MSG(builder != nullptr, "Failed to create TensorRT builder");

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        FASTDET_ASSERT_MSG(network != nullptr, "Failed to create network definition");

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        FASTDET_ASSERT_MSG(parser != nullptr, "Failed to create ONNX parser");

        std::ifstream file(onnxPath, std::ios::binary | std::ios::ate);
        FASTDET_ASSERT_MSG(file.is_open(), "Failed to open ONNX file: {}", onnxPath);

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        FASTDET_ASSERT_MSG(file.read(buffer.data(), size).good(), "Failed to read ONNX file");
        file.close();

        FASTDET_ASSERT_MSG(parser->parse(buffer.data(), buffer.size()), "Failed to parse ONNX model");

        const auto numInputs = network->getNbInputs();
        FASTDET_ASSERT_MSG(numInputs > 0, "Model needs at least 1 input");

        const auto input0Batch = network->getInput(0)->getDimensions().d[0];
        for (int32_t i = 1; i < numInputs; ++i) {
            FASTDET_ASSERT_MSG(network->getInput(i)->getDimensions().d[0] == input0Batch,
                "Model has multiple inputs with inconsistent batch sizes");
        }

        bool supportsDynamicBatch = (input0Batch == -1);
        FASTDET_LOG_INFO("{}", supportsDynamicBatch ?
            "Model supports dynamic batch size" :
            std::string("Model only supports fixed batch size of ") + std::to_string(input0Batch));

        if (!supportsDynamicBatch) {
            FASTDET_ASSERT_MSG(mOptions.optBatchSize == input0Batch && mOptions.maxBatchSize == input0Batch,
                "Model only supports fixed batch size of {}. Must set optBatchSize and maxBatchSize accordingly",
                input0Batch);
        }

        const auto inputWidth = network->getInput(0)->getDimensions().d[3];
        bool supportsDynamicWidth = (inputWidth == -1);

        FASTDET_LOG_INFO("{}", supportsDynamicWidth ?
            "Model supports dynamic width" :
            std::string("Model only supports fixed width of ") + std::to_string(inputWidth));

        if (supportsDynamicWidth) {
            FASTDET_ASSERT_MSG(mOptions.maxInputWidth >= mOptions.minInputWidth &&
                mOptions.maxInputWidth >= mOptions.optInputWidth &&
                mOptions.minInputWidth <= mOptions.optInputWidth &&
                mOptions.minInputWidth >= 1,
                "Invalid values for min/opt/max input width");
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        FASTDET_ASSERT_MSG(config != nullptr, "Failed to create builder config");

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

        if (mOptions.precision == Precision::FP16) {
            FASTDET_ASSERT_MSG(builder->platformHasFastFp16(), "FP16 precision not supported on this platform");
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            FASTDET_LOG_INFO("Building network with FP16 precision");
        } else {
            FASTDET_LOG_INFO("Building network with FP32 precision");
        }

        auto optProfile = builder->createOptimizationProfile();
        FASTDET_ASSERT_MSG(optProfile != nullptr, "Failed to create optimization profile");

        for (int32_t i = 0; i < numInputs; ++i) {
            auto input = network->getInput(i);
            auto inputName = input->getName();
            auto inputDims = input->getDimensions();
            int32_t inputC = inputDims.d[1];
            int32_t inputH = inputDims.d[2];
            int32_t minWidth = supportsDynamicWidth ? mOptions.minInputWidth : inputDims.d[3];

            nvinfer1::Dims4 minDims;
            if (supportsDynamicBatch) {
                minDims = nvinfer1::Dims4(1, inputC, inputH, minWidth);
            } else {
                minDims = nvinfer1::Dims4(input0Batch, inputC, inputH, minWidth);
            }

            nvinfer1::Dims4 optDims;
            if (supportsDynamicWidth) {
                optDims = nvinfer1::Dims4(mOptions.optBatchSize, inputC, inputH, mOptions.optInputWidth);
            } else {
                optDims = nvinfer1::Dims4(mOptions.optBatchSize, inputC, inputH, inputDims.d[3]);
            }

            nvinfer1::Dims4 maxDims;
            if (supportsDynamicWidth) {
                maxDims = nvinfer1::Dims4(mOptions.maxBatchSize, inputC, inputH, mOptions.maxInputWidth);
            } else {
                maxDims = nvinfer1::Dims4(mOptions.maxBatchSize, inputC, inputH, inputDims.d[3]);
            }

            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);

            FASTDET_LOG_INFO("Input '{}' dimensions set to: min={},{}x{}x{} opt={},{}x{}x{} max={},{}x{}x{}",
                        inputName,
                        minDims.d[0], minDims.d[1], minDims.d[2], minDims.d[3],
                        optDims.d[0], optDims.d[1], optDims.d[2], optDims.d[3],
                        maxDims.d[0], maxDims.d[1], maxDims.d[2], maxDims.d[3]);
        }

        config->addOptimizationProfile(optProfile);

        cudaStream_t profileStream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&profileStream) == cudaSuccess, "Failed to create CUDA stream for profiling");
        config->setProfileStream(profileStream);

        FASTDET_LOG_INFO("Building TensorRT engine. This may take a while...");
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        FASTDET_ASSERT_MSG(plan != nullptr, "Failed to build serialized network");
        FASTDET_LOG_INFO("TensorRT engine built successfully");

        std::string enginePath = generateEnginePath(onnxPath);
        std::filesystem::path dirPath = std::filesystem::path(mOptions.engineFileDir);

        try {
            if (!std::filesystem::exists(dirPath)) {
                std::filesystem::create_directories(dirPath);
                FASTDET_LOG_INFO("Created engine directory: {}", mOptions.engineFileDir);
            }
        } catch (const std::filesystem::filesystem_error& e) {
            FASTDET_LOG_ERROR("Failed to create engine directory: {}", e.what());
            cudaStreamDestroy(profileStream);
            return false;
        }

        try {
            std::ofstream engineFile(enginePath, std::ios::binary);
            FASTDET_ASSERT_MSG(engineFile.is_open(), "Failed to open engine file for writing: {}", enginePath);

            engineFile.write(static_cast<const char*>(plan->data()), plan->size());
            engineFile.close();

            FASTDET_LOG_INFO("TensorRT engine serialized to {}", enginePath);
        } catch (const std::exception& e) {
            FASTDET_LOG_ERROR("Failed to write engine file: {}", e.what());
            cudaStreamDestroy(profileStream);
            return false;
        }
        cudaStreamDestroy(profileStream);

        return true;
    }
}