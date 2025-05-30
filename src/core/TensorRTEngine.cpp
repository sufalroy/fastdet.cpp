#include "core/TensorRTEngine.h"
#include "common/logging.h"
#include "common/assertion.h"

#include <fstream>
#include <fmt/format.h>
#include <filesystem>

namespace fastdet::core {

    TensorRTEngine::TensorRTEngine()
        : mRuntime(nullptr), mEngine(nullptr), mContext(nullptr), mOptions{} {
    }

    TensorRTEngine::~TensorRTEngine() {
        clearGpuBuffers();
    }

    TensorRTEngine::TensorRTEngine(TensorRTEngine &&other) noexcept
        : mRuntime(std::move(other.mRuntime)),
          mEngine(std::move(other.mEngine)),
          mContext(std::move(other.mContext)),
          mTensorSpecs(std::move(other.mTensorSpecs)),
          mBuffers(std::move(other.mBuffers)),
          mOptions(other.mOptions) {
    }

    TensorRTEngine &TensorRTEngine::operator=(TensorRTEngine &&other) noexcept {
        if (this != &other) {
            clearGpuBuffers();

            mRuntime = std::move(other.mRuntime);
            mEngine = std::move(other.mEngine);
            mContext = std::move(other.mContext);
            mTensorSpecs = std::move(other.mTensorSpecs);
            mBuffers = std::move(other.mBuffers);
            mOptions = other.mOptions;
        }

        return *this;
    }

    std::string TensorRTEngine::generateEnginePath(const std::string &onnxPath) const {
        std::filesystem::path p(onnxPath);
        std::string baseName = p.stem().string();
        std::string precStr = (mOptions.precision == Precision::FP16) ? "fp16" : "fp32";
        std::string filename = fmt::format("{}_{}_b{}_{}x{}.engine", 
                                         baseName, precStr, mOptions.batchSize, 
                                         mOptions.inputWidth, mOptions.inputHeight);

        return (std::filesystem::path(mOptions.engineDir) / filename).string();
    }

    void TensorRTEngine::clearGpuBuffers() {
        if (!mBuffers.empty()) {
             for (size_t i = 0; i < mTensorSpecs.size(); ++i) {
                if (mTensorSpecs[i].ioMode == nvinfer1::TensorIOMode::kOUTPUT && 
                    mBuffers[i] != nullptr) {
                    cudaFree(mBuffers[i]);
                    mBuffers[i] = nullptr;
                }
            }
            mBuffers.clear();
        }
        mTensorSpecs.clear();
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

        for (int32_t i = 0; i < numInputs; ++i) {
            auto input = network->getInput(i);
            auto inputDims = input->getDimensions();
            
            nvinfer1::Dims4 dims(mOptions.batchSize, inputDims.d[1], mOptions.inputHeight, mOptions.inputWidth);
            input->setDimensions(dims);
            
            FASTDET_LOG_INFO("Input '{}' set to fixed dimensions: {}x{}x{}x{}", 
                           input->getName(), dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        }

        cudaStream_t profileStream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&profileStream) == cudaSuccess, "Failed to create CUDA stream for profiling");
        config->setProfileStream(profileStream);

        FASTDET_LOG_INFO("Building TensorRT Engine. This may take a while...");
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        FASTDET_ASSERT_MSG(plan != nullptr, "Failed to build serialized network");
        FASTDET_LOG_INFO("TensorRT Engine built successfully");

        std::string enginePath = generateEnginePath(onnxPath);
        std::filesystem::path dirPath = std::filesystem::path(mOptions.engineDir);

        try {
            if (!std::filesystem::exists(dirPath)) {
                std::filesystem::create_directories(dirPath);
                FASTDET_LOG_INFO("Created Engine directory: {}", mOptions.engineDir);
            }
        } catch (const std::filesystem::filesystem_error &e) {
            FASTDET_LOG_ERROR("Failed to create Engine directory: {}", e.what());
            cudaStreamDestroy(profileStream);
            return false;
        }

        try {
            std::ofstream engineFile(enginePath, std::ios::binary);
            FASTDET_ASSERT_MSG(engineFile.is_open(), "Failed to open Engine file for writing: {}", enginePath);

            engineFile.write(static_cast<const char *>(plan->data()), plan->size());
            engineFile.close();

            FASTDET_LOG_INFO("TensorRT Engine serialized to {}", enginePath);
        } catch (const std::exception &e) {
            FASTDET_LOG_ERROR("Failed to write Engine file: {}", e.what());
            cudaStreamDestroy(profileStream);
            return false;
        }

        cudaStreamDestroy(profileStream);
        return true;
    }

    bool TensorRTEngine::load(const std::string &enginePath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize) {
        mSubVals = subVals;
        mDivVals = divVals;
        mNormalize = normalize;  

        FASTDET_LOG_INFO("Loading TensorRT engine from {}", enginePath);
        
        FASTDET_ASSERT_MSG(std::filesystem::exists(enginePath), "Engine file not found: {}", enginePath);
        
        std::ifstream engineFile(enginePath, std::ios::binary | std::ios::ate);
        FASTDET_ASSERT_MSG(engineFile.is_open(), "Failed to open Engine file: {}", enginePath);
        
        const auto size = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        FASTDET_ASSERT_MSG(engineFile.read(buffer.data(), size).good(), "Failed to read engine file");
        engineFile.close();
        
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        FASTDET_ASSERT_MSG(mRuntime != nullptr, "Failed to create TensorRT runtime");
        
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
        FASTDET_ASSERT_MSG(mEngine != nullptr, "Failed to deserialize CUDA engine");
        
        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        FASTDET_ASSERT_MSG(mContext != nullptr, "Failed to create execution context");
        
        clearGpuBuffers();
        const int32_t numTensors = mEngine->getNbIOTensors();
        mTensorSpecs.reserve(numTensors);
        mBuffers.resize(numTensors, nullptr);
        FASTDET_LOG_INFO("Engine loaded with {} I/O tensors", numTensors);

        cudaStream_t stream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&stream) == cudaSuccess, "Failed to create CUDA stream for execution");

        for (int32_t i = 0; i < numTensors; ++i) {
            const auto *name = mEngine->getIOTensorName(i);
            
            TensorSpec tensorSpec{
                .name = name,
                .shape = mEngine->getTensorShape(name),
                .dataType = mEngine->getTensorDataType(name),
                .ioMode = mEngine->getTensorIOMode(name)
            };
            
            tensorSpec.byteSize = tensorSpec.getElementCount() * tensorSpec.getElementSize();
            FASTDET_ASSERT_MSG(tensorSpec.getElementSize() > 0, "Unsupported data type: {}", tensorSpec.name);
            
            if (tensorSpec.ioMode == nvinfer1::TensorIOMode::kINPUT) {
                FASTDET_ASSERT_MSG(tensorSpec.dataType == nvinfer1::DataType::kFLOAT, "Input tensor '{}' must be float32, got unsupported type", tensorSpec.name);
                FASTDET_LOG_INFO("Input '{}': {}x{}x{}x{} ({} bytes) - No GPU allocation (external buffer expected)",
                    tensorSpec.name, tensorSpec.shape.d[0], tensorSpec.shape.d[1], tensorSpec.shape.d[2], tensorSpec.shape.d[3], tensorSpec.byteSize);         
                
            } else if (tensorSpec.ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
                if (tensorSpec.dataType == nvinfer1::DataType::kINT8) {
                    FASTDET_ASSERT_MSG(false, "Output tensor '{}' has unsupported INT8 precision", tensorSpec.name);
                }

                if (tensorSpec.dataType == nvinfer1::DataType::kFP8) {
                    FASTDET_ASSERT_MSG(false, "Output tensor '{}' has unsupported FP8 precision", tensorSpec.name);
                }
                
                FASTDET_ASSERT_MSG(cudaMallocAsync(&mBuffers[i], tensorSpec.byteSize, stream) == cudaSuccess, "GPU allocation failed for tensor: {}", tensorSpec.name);
                
                FASTDET_LOG_INFO("Output '{}': {} elements ({} bytes) of shape {}x{}x{} - GPU memory allocated", 
                    tensorSpec.name, tensorSpec.getElementCount(), tensorSpec.byteSize,
                    tensorSpec.shape.d[0], tensorSpec.shape.d[1], tensorSpec.shape.d[2], tensorSpec.shape.d[3]);

            } else {
                FASTDET_ASSERT_MSG(false, "Tensor '{}' is neither input nor output", tensorSpec.name);
            }
            
            mTensorSpecs.emplace_back(std::move(tensorSpec));
        }
        
        FASTDET_ASSERT_MSG(cudaStreamSynchronize(stream) == cudaSuccess, "Failed to synchronize CUDA stream after loading engine");
        cudaStreamDestroy(stream);
        
        FASTDET_LOG_INFO("Engine loaded: {} tensors configured", mTensorSpecs.size());

        return true;
    }
}