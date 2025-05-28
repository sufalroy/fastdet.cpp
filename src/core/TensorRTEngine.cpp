#include "core/TensorRTEngine.h"
#include "common/logging.h"
#include "common/assertion.h"

#include <fstream>
#include <fmt/format.h>
#include <filesystem>

namespace fastdet::core {
    TensorRTEngine::TensorRTEngine()
        : mRuntime(nullptr), mCudaEngine(nullptr), mExecutionContext(nullptr), mOptions{} {
    }

    TensorRTEngine::~TensorRTEngine() {
        clearGpuBuffers();
        if (mCudaStream != nullptr) {
            cudaStreamDestroy(mCudaStream);
            mCudaStream = nullptr;
        }
    }

    TensorRTEngine::TensorRTEngine(TensorRTEngine &&other) noexcept
        : mRuntime(std::move(other.mRuntime)),
          mCudaEngine(std::move(other.mCudaEngine)),
          mExecutionContext(std::move(other.mExecutionContext)),
          mTensorInfos(std::move(other.mTensorInfos)),
          mBuffers(std::move(other.mBuffers)),
          mCudaStream(other.mCudaStream),
          mOptions(other.mOptions) {
        other.mCudaStream = nullptr;
    }

    TensorRTEngine &TensorRTEngine::operator=(TensorRTEngine &&other) noexcept {
        if (this != &other) {
            clearGpuBuffers();
            if (mCudaStream != nullptr) {
                cudaStreamDestroy(mCudaStream);
            }

            mRuntime = std::move(other.mRuntime);
            mCudaEngine = std::move(other.mCudaEngine);
            mExecutionContext = std::move(other.mExecutionContext);
            mTensorInfos = std::move(other.mTensorInfos);
            mBuffers = std::move(other.mBuffers);
            mCudaStream = other.mCudaStream;
            mOptions = other.mOptions;

            other.mCudaStream = nullptr;
        }

        return *this;
    }

    std::string TensorRTEngine::generateEnginePath(const std::string &onnxPath) const {
        std::filesystem::path p(onnxPath);
        std::string baseName = p.stem().string();
        std::string precStr = (mOptions.precision == Precision::FP16) ? "fp16" : "fp32";

        std::string batchStr = (mOptions.optBatchSize == mOptions.maxBatchSize)
                                   ? std::to_string(mOptions.optBatchSize)
                                   : fmt::format("{}-{}", mOptions.optBatchSize, mOptions.maxBatchSize);

        std::string widthStr = (mOptions.minInputWidth == mOptions.maxInputWidth)
                                   ? std::to_string(mOptions.optInputWidth)
                                   : fmt::format("{}-{}-{}", mOptions.minInputWidth, mOptions.optInputWidth,
                                                 mOptions.maxInputWidth);

        std::string filename = fmt::format("{}_{}_b{}_w{}.engine", baseName, precStr, batchStr, widthStr);
        return (std::filesystem::path(mOptions.engineFileDir) / filename).string();
    }

    void TensorRTEngine::clearGpuBuffers() {
        if (mCudaStream != nullptr) {
            for (auto *buffer: mBuffers) {
                if (buffer != nullptr) { cudaFreeAsync(buffer, mCudaStream); }
            }
            if (!mBuffers.empty()) { cudaStreamSynchronize(mCudaStream); }
        }

        mBuffers.clear();
        mTensorInfos.clear();
    }

    bool TensorRTEngine::build(const std::string &onnxPath, const Options &options) {
        mOptions = options;

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        FASTDET_ASSERT_MSG(builder != nullptr, "Failed to create TensorRT builder");

        const auto explicitBatch = 1U << static_cast<uint32_t>(
                                       nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
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
                         fmt::format("Model only supports fixed batch size of {}", input0Batch));

        if (!supportsDynamicBatch) {
            FASTDET_ASSERT_MSG(mOptions.optBatchSize == input0Batch && mOptions.maxBatchSize == input0Batch,
                               "Model only supports fixed batch size of {}. Must set optBatchSize and maxBatchSize accordingly",
                               input0Batch);
        }

        const auto inputWidth = network->getInput(0)->getDimensions().d[3];
        bool supportsDynamicWidth = (inputWidth == -1);

        FASTDET_LOG_INFO("{}", supportsDynamicWidth ?
                         "Model supports dynamic width" :
                         fmt::format("Model only supports fixed width of {}", inputWidth));

        if (supportsDynamicWidth) {
            FASTDET_ASSERT_MSG(mOptions.maxInputWidth >= mOptions.minInputWidth &&
                               mOptions.maxInputWidth >= mOptions.optInputWidth &&
                               mOptions.minInputWidth <= mOptions.optInputWidth &&
                               mOptions.minInputWidth >= 1,
                               "Invalid values for min/opt/max input width");
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        FASTDET_ASSERT_MSG(config != nullptr, "Failed to create builder config");

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

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
        FASTDET_ASSERT_MSG(cudaStreamCreate(&profileStream) == cudaSuccess,
                           "Failed to create CUDA stream for profiling");
        config->setProfileStream(profileStream);

        FASTDET_LOG_INFO("Building TensorRT Engine. This may take a while...");
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        FASTDET_ASSERT_MSG(plan != nullptr, "Failed to build serialized network");
        FASTDET_LOG_INFO("TensorRT Engine built successfully");

        std::string EnginePath = generateEnginePath(onnxPath);
        std::filesystem::path dirPath = std::filesystem::path(mOptions.engineFileDir);

        try {
            if (!std::filesystem::exists(dirPath)) {
                std::filesystem::create_directories(dirPath);
                FASTDET_LOG_INFO("Created Engine directory: {}", mOptions.engineFileDir);
            }
        } catch (const std::filesystem::filesystem_error &e) {
            FASTDET_LOG_ERROR("Failed to create Engine directory: {}", e.what());
            cudaStreamDestroy(profileStream);
            return false;
        }

        try {
            std::ofstream EngineFile(EnginePath, std::ios::binary);
            FASTDET_ASSERT_MSG(EngineFile.is_open(), "Failed to open Engine file for writing: {}", EnginePath);

            EngineFile.write(static_cast<const char *>(plan->data()), plan->size());
            EngineFile.close();

            FASTDET_LOG_INFO("TensorRT Engine serialized to {}", EnginePath);
        } catch (const std::exception &e) {
            FASTDET_LOG_ERROR("Failed to write Engine file: {}", e.what());
            cudaStreamDestroy(profileStream);
            return false;
        }

        cudaStreamDestroy(profileStream);

        return true;
    }

    bool TensorRTEngine::load(const std::string &enginePath) {
        FASTDET_LOG_INFO("Loading TensorRT engine from {}", enginePath);

        FASTDET_ASSERT_MSG(std::filesystem::exists(enginePath), "Engine file not found: {}", enginePath);

        std::ifstream EngineFile(enginePath, std::ios::binary | std::ios::ate);
        FASTDET_ASSERT_MSG(EngineFile.is_open(), "Failed to open Engine file: {}", enginePath);

        const auto size = EngineFile.tellg();
        EngineFile.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        FASTDET_ASSERT_MSG(EngineFile.read(buffer.data(), size).good(), "Failed to read engine file");
        EngineFile.close();

        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        FASTDET_ASSERT_MSG(mRuntime != nullptr, "Failed to create TensorRT runtime");

        mCudaEngine = std::unique_ptr<
            nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
        FASTDET_ASSERT_MSG(mCudaEngine != nullptr, "Failed to deserialize CUDA engine");

        mExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mCudaEngine->createExecutionContext());
        FASTDET_ASSERT_MSG(mExecutionContext != nullptr, "Failed to create execution context");

        clearGpuBuffers();

        FASTDET_ASSERT_MSG(cudaStreamCreate(&mCudaStream) == cudaSuccess, "Failed to create CUDA stream");

        const int32_t numTensors = mCudaEngine->getNbIOTensors();
        mTensorInfos.reserve(numTensors);
        mBuffers.resize(numTensors, nullptr);
        FASTDET_LOG_INFO("Engine loaded with {} I/O tensors", numTensors);

        for (int32_t i = 0; i < numTensors; ++i) {
            const auto *name = mCudaEngine->getIOTensorName(i);

            TensorInfo tensorInfo{
                .name = name,
                .dims = mCudaEngine->getTensorShape(name),
                .dataType = mCudaEngine->getTensorDataType(name),
                .mode = mCudaEngine->getTensorIOMode(name)
            };

            tensorInfo.size = tensorInfo.getTotalElements() * tensorInfo.getElementSize();
            FASTDET_ASSERT_MSG(tensorInfo.getElementSize() > 0, "Unsupported data type: {}", tensorInfo.name);

            if (tensorInfo.mode == nvinfer1::TensorIOMode::kINPUT) {
                FASTDET_ASSERT_MSG(tensorInfo.dataType == nvinfer1::DataType::kFLOAT, "Input must be float32: {}",
                                   tensorInfo.name);

                FASTDET_LOG_INFO("Input '{}': {}x{}x{}x{} ({} bytes)",
                                 tensorInfo.name, tensorInfo.dims.d[0], tensorInfo.dims.d[1],
                                 tensorInfo.dims.d[2], tensorInfo.dims.d[3], tensorInfo.size);
            } else if (tensorInfo.mode == nvinfer1::TensorIOMode::kOUTPUT) {
                FASTDET_ASSERT_MSG(cudaMallocAsync(&mBuffers[i], tensorInfo.size, mCudaStream) == cudaSuccess,
                                   "GPU allocation failed: {}", tensorInfo.name);
                FASTDET_LOG_INFO("Output '{}': {} elements ({} bytes)", tensorInfo.name, tensorInfo.getTotalElements(),
                                 tensorInfo.size);
            }

            mTensorInfos.emplace_back(std::move(tensorInfo));
        }

        FASTDET_ASSERT_MSG(cudaStreamSynchronize(mCudaStream) == cudaSuccess, "Stream synchronization failed");
        FASTDET_LOG_INFO("Engine loaded: {} tensors configured", mTensorInfos.size());

        return true;
    }
}