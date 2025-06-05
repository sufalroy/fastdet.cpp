#include "inference/TensorRTEngine.h"
#include "common/Logger.h"
#include "common/Assert.h"

#include <fstream>
#include <fmt/format.h>
#include <filesystem>

namespace fastdet::inference {
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
          mBuffers(std::move(other.mBuffers)),
          mIOTensorNames(std::move(other.mIOTensorNames)),
          mOutputLengths(std::move(other.mOutputLengths)),
          mInputDims(std::move(other.mInputDims)),
          mOutputDims(std::move(other.mOutputDims)),
          mInputBatchSize(other.mInputBatchSize),
          mOptions(other.mOptions) {
    }

    TensorRTEngine &TensorRTEngine::operator=(TensorRTEngine &&other) noexcept {
        if (this != &other) {
            clearGpuBuffers();

            mRuntime = std::move(other.mRuntime);
            mEngine = std::move(other.mEngine);
            mContext = std::move(other.mContext);
            mBuffers = std::move(other.mBuffers);
            mIOTensorNames = std::move(other.mIOTensorNames);
            mOutputLengths = std::move(other.mOutputLengths);
            mInputDims = std::move(other.mInputDims);
            mOutputDims = std::move(other.mOutputDims);
            mInputBatchSize = other.mInputBatchSize;
            mOptions = other.mOptions;
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
        return (std::filesystem::path(mOptions.engineDir) / filename).string();
    }

    void TensorRTEngine::clearGpuBuffers() {
        if (mBuffers.empty()) return;
        
        const auto numInputs = mInputDims.size();
        const auto totalTensors = mEngine ? mEngine->getNbIOTensors() : static_cast<int32_t>(mBuffers.size());
        
        for (int32_t i = numInputs; i < totalTensors; ++i) {
            if (mBuffers[i] != nullptr) {
                FASTDET_ASSERT_MSG(cudaFree(mBuffers[i]) == cudaSuccess, "Failed to free GPU buffer at index {}", i);
                mBuffers[i] = nullptr;
            }
        }

        mBuffers.clear();
        mIOTensorNames.clear();
        mOutputLengths.clear();
        mInputDims.clear();
        mOutputDims.clear();
    }

    cv::cuda::GpuMat TensorRTEngine::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput,
                                                     const std::array<float, 3> &subVals,
                                                     const std::array<float, 3> &divVals,
                                                     bool normalize,
                                                     bool swapRB) const {
        FASTDET_ASSERT_MSG(!batchInput.empty(), "Batch input cannot be empty");
        FASTDET_ASSERT_MSG(batchInput[0].channels() == 3, "Input must have 3 channels");
        
        cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);
        
        size_t width = batchInput[0].cols * batchInput[0].rows;
        
        if (swapRB) {
            for (size_t img = 0; img < batchInput.size(); ++img) {
                std::vector<cv::cuda::GpuMat> input_channels{
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img]))
                };
                cv::cuda::split(batchInput[img], input_channels);
            }
        } else {
            for (size_t img = 0; img < batchInput.size(); ++img) {
                std::vector<cv::cuda::GpuMat> input_channels{
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))
                };
                cv::cuda::split(batchInput[img], input_channels);
            }
        }
        
        cv::cuda::GpuMat blob;
        if (normalize) {
            gpu_dst.convertTo(blob, CV_32FC3, 1.f / 255.f);
        } else {
            gpu_dst.convertTo(blob, CV_32FC3);
        }
        
        cv::cuda::subtract(blob, cv::Scalar(subVals[0], subVals[1], subVals[2]), blob, cv::noArray(), -1);
        cv::cuda::divide(blob, cv::Scalar(divVals[0], divVals[1], divVals[2]), blob, 1, -1);
        
        return blob;
    }

    bool TensorRTEngine::build(const std::string &onnxPath, const Options &options) {
        mOptions = options;
        
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        FASTDET_ASSERT_MSG(builder != nullptr, "Failed to create TensorRT builder");
        
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
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
            FASTDET_ASSERT_MSG(network->getInput(i)->getDimensions().d[0] == input0Batch, "Model has multiple inputs with differing batch sizes");
        }
        
        const bool supportsDynamicBatch = (input0Batch == -1);
        const bool supportsDynamicWidth = (network->getInput(0)->getDimensions().d[3] == -1);
        
        if (!supportsDynamicBatch) {
            FASTDET_ASSERT_MSG(mOptions.optBatchSize == input0Batch && mOptions.maxBatchSize == input0Batch,
                              "Fixed batch model requires optBatchSize and maxBatchSize to be {}", input0Batch);
        }
        
        if (supportsDynamicWidth) {
            FASTDET_ASSERT_MSG(mOptions.maxInputWidth >= mOptions.minInputWidth &&
                               mOptions.maxInputWidth >= mOptions.optInputWidth &&
                               mOptions.minInputWidth <= mOptions.optInputWidth &&
                               mOptions.maxInputWidth > 0 && mOptions.minInputWidth > 0 && mOptions.optInputWidth > 0,
                               "Invalid dynamic width configuration");
        }
        
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        FASTDET_ASSERT_MSG(config != nullptr, "Failed to create builder config");
        
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        
        auto optProfile = builder->createOptimizationProfile();
        FASTDET_ASSERT_MSG(optProfile != nullptr, "Failed to create optimization profile");
        
        for (int32_t i = 0; i < numInputs; ++i) {
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
    
            const int32_t inputC = static_cast<int32_t>(inputDims.d[1]);
            const int32_t inputH = static_cast<int32_t>(inputDims.d[2]);
            const int32_t inputW = static_cast<int32_t>(inputDims.d[3]);
            
            const auto minBatch = supportsDynamicBatch ? 1 : mOptions.optBatchSize;
            const auto optBatch = mOptions.optBatchSize;
            const auto maxBatch = mOptions.maxBatchSize;
            
            const auto minWidth = supportsDynamicWidth ? std::max(mOptions.minInputWidth, inputW) : inputW;
            const auto optWidth = supportsDynamicWidth ? mOptions.optInputWidth : inputW;
            const auto maxWidth = supportsDynamicWidth ? mOptions.maxInputWidth : inputW;

            FASTDET_ASSERT_MSG(optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                                     nvinfer1::Dims4(minBatch, inputC, inputH, minWidth)),
                                "Failed to set MIN dimensions for input '{}'", inputName);
                           
            FASTDET_ASSERT_MSG(optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                                        nvinfer1::Dims4(optBatch, inputC, inputH, optWidth)),
                                "Failed to set OPT dimensions for input '{}'", inputName);
                           
            FASTDET_ASSERT_MSG(optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                                        nvinfer1::Dims4(maxBatch, inputC, inputH, maxWidth)),
                                "Failed to set MAX dimensions for input '{}'", inputName);
        }
        
        config->addOptimizationProfile(optProfile);
        
        if (mOptions.precision == Precision::FP16) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            FASTDET_LOG_INFO("Building engine with FP16 precision");
        } else {
            FASTDET_LOG_INFO("Building engine with FP32 precision");
        }

        cudaStream_t profileStream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&profileStream) == cudaSuccess, "Failed to create CUDA stream for profiling");
        config->setProfileStream(profileStream);
        
        FASTDET_LOG_INFO("Building TensorRT Engine. This may take a while...");
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        FASTDET_ASSERT_MSG(plan != nullptr, "Failed to build serialized network");
        FASTDET_LOG_INFO("TensorRT Engine built successfully");

        const std::string enginePath = generateEnginePath(onnxPath);
        const std::filesystem::path dirPath = std::filesystem::path(mOptions.engineDir);
        
        try {
            if (!std::filesystem::exists(dirPath)) {
                std::filesystem::create_directories(dirPath);
                FASTDET_LOG_INFO("Created Engine directory: {}", mOptions.engineDir);
            }
        } catch (const std::filesystem::filesystem_error &e) {
            FASTDET_LOG_ERROR("Failed to create Engine directory: {}", e.what());
            FASTDET_ASSERT_MSG(cudaStreamDestroy(profileStream) == cudaSuccess, "Failed to destroy CUDA stream");
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
            FASTDET_ASSERT_MSG(cudaStreamDestroy(profileStream) == cudaSuccess, "Failed to destroy CUDA stream");
            return false;
        }
        
        FASTDET_ASSERT_MSG(cudaStreamDestroy(profileStream) == cudaSuccess, "Failed to destroy CUDA stream");
        return true;
    }

    bool TensorRTEngine::load(const std::string &enginePath, const std::array<float, 3> &subVals,
                              const std::array<float, 3> &divVals, bool normalize) {
        mSubVals = subVals;
        mDivVals = divVals;
        mNormalize = normalize;
        
        FASTDET_ASSERT_MSG(std::filesystem::exists(enginePath), "Engine file not found: {}", enginePath);
        FASTDET_LOG_INFO("Loading TensorRT engine file at path: {}", enginePath);
        
        std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
        FASTDET_ASSERT_MSG(file.is_open(), "Unable to read engine file");
        
        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        FASTDET_ASSERT_MSG(file.read(buffer.data(), size).good(), "Unable to read engine file");
        file.close();
        
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        FASTDET_ASSERT_MSG(mRuntime != nullptr, "Failed to create TensorRT runtime");
        
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
        FASTDET_ASSERT_MSG(mEngine != nullptr, "Failed to deserialize CUDA engine");
        
        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        FASTDET_ASSERT_MSG(mContext != nullptr, "Failed to create execution context");
        
        clearGpuBuffers();
        mBuffers.resize(mEngine->getNbIOTensors());
        
        cudaStream_t stream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&stream) == cudaSuccess, "Failed to create CUDA stream");
        
        for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i) {
            const auto tensorName = mEngine->getIOTensorName(i);
            mIOTensorNames.emplace_back(tensorName);
            const auto tensorType = mEngine->getTensorIOMode(tensorName);
            const auto tensorShape = mEngine->getTensorShape(tensorName);
            const auto tensorDataType = mEngine->getTensorDataType(tensorName);
            
            if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
                FASTDET_ASSERT_MSG(tensorDataType == nvinfer1::DataType::kFLOAT, "Input tensor '{}' must be float32", tensorName);
                
                mInputDims.emplace_back(nvinfer1::Dims{3, {tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]}});
                mInputBatchSize = tensorShape.d[0];
            } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
                FASTDET_ASSERT_MSG(tensorDataType != nvinfer1::DataType::kINT8, "Output tensor '{}' has unsupported INT8 precision", tensorName);
                FASTDET_ASSERT_MSG(tensorDataType != nvinfer1::DataType::kFP8, "Output tensor '{}' has unsupported FP8 precision", tensorName);
                
                uint32_t outputLength = 1;
                mOutputDims.push_back(tensorShape);
                
                for (int32_t j = 1; j < tensorShape.nbDims; ++j) {
                    outputLength *= tensorShape.d[j];
                }
                
                mOutputLengths.push_back(outputLength);
                FASTDET_ASSERT_MSG(cudaMallocAsync(&mBuffers[i], outputLength * mOptions.maxBatchSize * sizeof(float), stream) == cudaSuccess,
                                "GPU allocation failed for tensor: {}", tensorName);
                
                FASTDET_LOG_INFO("Output '{}': {} elements ({} bytes) allocated", tensorName, outputLength, outputLength * sizeof(float));
            } else {
                FASTDET_ASSERT_MSG(false, "Tensor '{}' is neither input nor output", tensorName);
            }
        }
        
        FASTDET_ASSERT_MSG(cudaStreamSynchronize(stream) == cudaSuccess, "Failed to synchronize CUDA stream");
        FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");
        FASTDET_LOG_INFO("Engine loaded: {} I/O tensors configured", mIOTensorNames.size());
        
        return true;
    }

    bool TensorRTEngine::infer(const std::vector<std::vector<cv::cuda::GpuMat> > &input,
                               std::vector<std::vector<std::vector<float> > > &output) {
                                
        if (input.empty() || input[0].empty()) {
            FASTDET_LOG_ERROR("Input batch is empty");
            return false;
        }
        
        FASTDET_ASSERT_MSG(mEngine != nullptr && mContext != nullptr, "Engine and context must be initialized before inference");
        const auto numInputs = mInputDims.size();
        
        if (input.size() != numInputs) {
            FASTDET_LOG_ERROR("Incorrect number of inputs provided! Expected: {}, Got: {}", numInputs, input.size());
            return false;
        }
        
        const auto batchSize = static_cast<int32_t>(input[0].size());
        
        if (batchSize > mOptions.maxBatchSize) {
            FASTDET_LOG_ERROR("Batch size {} exceeds model max batch size {}", batchSize, mOptions.maxBatchSize);
            return false;
        }
        
        if (mInputBatchSize != -1 && batchSize != mInputBatchSize) {
            FASTDET_LOG_ERROR("Model expects fixed batch size {}, but got {}", mInputBatchSize, batchSize);
            return false;
        }
        
        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i].size() != static_cast<size_t>(batchSize)) {
                FASTDET_LOG_ERROR("Inconsistent batch size: input[0] has {}, input[{}] has {}", batchSize, i, input[i].size());
                return false;
            }
        }
        
        FASTDET_LOG_INFO("Starting inference with batch size: {}", batchSize);
        
        cudaStream_t stream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&stream) == cudaSuccess, "Failed to create CUDA stream");
        
        std::vector<cv::cuda::GpuMat> preprocessedInputs;
        preprocessedInputs.reserve(numInputs);
        
        for (size_t i = 0; i < numInputs; ++i) {
            const auto &batchInput = input[i];
            const auto &dims = mInputDims[i];
            
            const auto &input = batchInput[0];
            
            if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
                FASTDET_LOG_ERROR("Input {} dimension mismatch. Expected: ({}, {}, {}), Got: ({}, {}, {})",
                                i, dims.d[0], dims.d[1], dims.d[2], 
                                input.channels(), input.rows, input.cols);
                                
                FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");
                return false;
            }
            
            const nvinfer1::Dims4 inputDims{batchSize, dims.d[0], dims.d[1], dims.d[2]};
            FASTDET_ASSERT_MSG(mContext->setInputShape(mIOTensorNames[i].c_str(), inputDims),
                            "Failed to set input shape for tensor '{}'", mIOTensorNames[i]);            
            
            auto blob = blobFromGpuMats(batchInput, mSubVals, mDivVals, mNormalize);
            preprocessedInputs.emplace_back(std::move(blob));
            
            mBuffers[i] = preprocessedInputs.back().ptr<void>();
            FASTDET_LOG_INFO("Input '{}' configured: batch_size={}, shape={}x{}x{}", mIOTensorNames[i], batchSize, dims.d[0], dims.d[1], dims.d[2]);
        }
        
        if (!mContext->allInputDimensionsSpecified()) {
            FASTDET_LOG_ERROR("Not all required input dimensions have been specified");
            FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");
            return false;
        }
        
        for (size_t i = 0; i < mBuffers.size(); ++i) {
            if (!mContext->setTensorAddress(mIOTensorNames[i].c_str(), mBuffers[i])) {
                FASTDET_LOG_ERROR("Failed to set tensor address for '{}'", mIOTensorNames[i]);
                FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");
                
                return false;
            }
        }
        
        FASTDET_LOG_INFO("Executing inference...");
        if (!mContext->enqueueV3(stream)) {
            FASTDET_LOG_ERROR("Inference execution failed");
            FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");
            return false;
        }
        
        output.clear();
        output.reserve(batchSize);
        
        const auto numOutputs = mOutputLengths.size();
        
        for (int32_t batch = 0; batch < batchSize; ++batch) { 
            std::vector<std::vector<float>> batchOutputs;
            batchOutputs.reserve(numOutputs);
            
            for (size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
                const auto outputBinding = numInputs + outputIdx;
                const auto outputLength = mOutputLengths[outputIdx];
                
                std::vector<float> outputVector(outputLength);
                const auto* srcPtr = static_cast<const char*>(mBuffers[outputBinding]) + (batch * sizeof(float) * outputLength);
                
                FASTDET_ASSERT_MSG(
                    cudaMemcpyAsync(outputVector.data(), srcPtr, outputLength * sizeof(float),
                     cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                     "Failed to copy output data for tensor '{}', batch {}", mIOTensorNames[outputBinding], batch);
                     
                batchOutputs.emplace_back(std::move(outputVector));
            }
            
            output.emplace_back(std::move(batchOutputs));
        }
        
        FASTDET_ASSERT_MSG(cudaStreamSynchronize(stream) == cudaSuccess, "Failed to synchronize CUDA stream");
        FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");
        FASTDET_LOG_INFO("Inference completed successfully. Processed {} batches with {} output tensors each", batchSize, numOutputs);
        
        return true;
    }
}
