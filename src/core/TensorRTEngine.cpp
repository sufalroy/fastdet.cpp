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
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                                     &(gpu_dst.ptr()[width * 2 + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                                     &(gpu_dst.ptr()[width + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                                     &(gpu_dst.ptr()[0 + width * 3 * img]))
                };

                cv::cuda::split(batchInput[img], input_channels);
            }
        } else {
            for (size_t img = 0; img < batchInput.size(); ++img) {
                std::vector<cv::cuda::GpuMat> input_channels{
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                                     &(gpu_dst.ptr()[0 + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                                     &(gpu_dst.ptr()[width + width * 3 * img])),
                    cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U,
                                     &(gpu_dst.ptr()[width * 2 + width * 3 * img]))
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

            FASTDET_LOG_INFO("Input '{}' set to fixed dimensions: {}x{}x{}x{}", input->getName(), dims.d[0], dims.d[1],
                             dims.d[2], dims.d[3]);
        }

        cudaStream_t profileStream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&profileStream) == cudaSuccess,
                           "Failed to create CUDA stream for profiling");
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
                FASTDET_ASSERT_MSG(tensorSpec.dataType == nvinfer1::DataType::kFLOAT,
                                   "Input tensor '{}' must be float32, got unsupported type", tensorSpec.name);

                FASTDET_LOG_INFO("Input '{}': {}x{}x{}x{} ({} bytes)",
                                 tensorSpec.name, tensorSpec.shape.d[0], tensorSpec.shape.d[1], tensorSpec.shape.d[2],
                                 tensorSpec.shape.d[3], tensorSpec.byteSize);
            } else if (tensorSpec.ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
                if (tensorSpec.dataType == nvinfer1::DataType::kINT8) {
                    FASTDET_ASSERT_MSG(false, "Output tensor '{}' has unsupported INT8 precision", tensorSpec.name);
                }

                if (tensorSpec.dataType == nvinfer1::DataType::kFP8) {
                    FASTDET_ASSERT_MSG(false, "Output tensor '{}' has unsupported FP8 precision", tensorSpec.name);
                }

                FASTDET_ASSERT_MSG(cudaMallocAsync(&mBuffers[i], tensorSpec.byteSize, stream) == cudaSuccess,
                                   "GPU allocation failed for tensor: {}", tensorSpec.name);

                FASTDET_LOG_INFO("Output '{}': {} elements ({} bytes) of shape {}x{}x{}",
                                 tensorSpec.name, tensorSpec.getElementCount(), tensorSpec.byteSize,
                                 tensorSpec.shape.d[0], tensorSpec.shape.d[1], tensorSpec.shape.d[2],
                                 tensorSpec.shape.d[3]);
            } else {
                FASTDET_ASSERT_MSG(false, "Tensor '{}' is neither input nor output", tensorSpec.name);
            }

            mTensorSpecs.emplace_back(std::move(tensorSpec));
        }

        FASTDET_ASSERT_MSG(cudaStreamSynchronize(stream) == cudaSuccess,
                           "Failed to synchronize CUDA stream after loading engine");
        FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");

        FASTDET_LOG_INFO("Engine loaded: {} tensors configured", mTensorSpecs.size());

        return true;
    }

    bool TensorRTEngine::infer(const std::vector<std::vector<cv::cuda::GpuMat> > &inputs,
                               std::vector<std::vector<std::vector<float> > > &outputs) {
        FASTDET_ASSERT_MSG(!inputs.empty() && !inputs[0].empty(), "Input vector cannot be empty");
        FASTDET_ASSERT_MSG(mEngine != nullptr && mContext != nullptr,
                           "Engine and context must be initialized before inference");

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        const auto inputCount = std::ranges::count_if(mTensorSpecs, [](const auto &spec) {
            return spec.ioMode == nvinfer1::TensorIOMode::kINPUT;
        });

        FASTDET_ASSERT_MSG(inputs.size() == static_cast<size_t>(inputCount), "Expected {} input tensors, got {}",
                           inputCount, inputs.size());

        for (size_t i = 1; i < inputs.size(); ++i) {
            FASTDET_ASSERT_MSG(inputs[i].size() == static_cast<size_t>(batchSize),
                               "Inconsistent batch size: input[0] has {}, input[{}] has {}", batchSize, i,
                               inputs[i].size());
        }

        FASTDET_LOG_INFO("Starting inference with batch size: {}", batchSize);

        cudaStream_t stream;
        FASTDET_ASSERT_MSG(cudaStreamCreate(&stream) == cudaSuccess, "Failed to create CUDA stream");

        std::vector<cv::cuda::GpuMat> preprocessedInputs;
        preprocessedInputs.reserve(inputs.size());

        size_t inputIndex = 0;

        for (size_t i = 0; i < mTensorSpecs.size(); ++i) {
            const auto &spec = mTensorSpecs[i];

            if (spec.ioMode == nvinfer1::TensorIOMode::kINPUT) {
                FASTDET_ASSERT_MSG(inputIndex < inputs.size(), "Input index out of bounds");

                const auto &batchInput = inputs[inputIndex];
                const auto &first = batchInput[0];

                const auto [exp_c, exp_h, exp_w] = std::make_tuple(spec.shape.d[1], spec.shape.d[2], spec.shape.d[3]);

                FASTDET_ASSERT_MSG(first.channels() == exp_c && first.rows == exp_h && first.cols == exp_w,
                                   "Input '{}' dimension mismatch. Expected: {}x{}x{}, Got: {}x{}x{}",
                                   spec.name, exp_c, exp_h, exp_w, first.channels(), first.rows, first.cols);

                nvinfer1::Dims4 dims{batchSize, exp_c, exp_h, exp_w};
                FASTDET_ASSERT_MSG(mContext->setInputShape(spec.name.c_str(), dims),
                                   "Failed to set input shape for tensor '{}'", spec.name);

                auto preprocessedInput = blobFromGpuMats(batchInput, mSubVals, mDivVals, mNormalize);
                preprocessedInputs.emplace_back(std::move(preprocessedInput));

                FASTDET_ASSERT_MSG(mContext->setTensorAddress(spec.name.c_str(), preprocessedInputs.back().ptr<void>()),
                                   "Failed to set tensor address for input '{}'", spec.name);
                FASTDET_LOG_INFO("Input '{}' configured: batch_size={}, shape={}x{}x{}", spec.name, batchSize, exp_c,
                                 exp_h, exp_w);

                ++inputIndex;
            } else {
                FASTDET_ASSERT_MSG(mContext->setTensorAddress(spec.name.c_str(), mBuffers[i]),
                                   "Failed to set tensor address for output '{}'", spec.name);
            }
        }

        FASTDET_ASSERT_MSG(mContext->allInputDimensionsSpecified(),
                           "Not all required input dimensions have been specified");

        FASTDET_LOG_INFO("Executing inference...");
        FASTDET_ASSERT_MSG(mContext->enqueueV3(stream), "Inference execution failed");

        const auto outputCount = std::ranges::count_if(mTensorSpecs, [](const auto &spec) {
            return spec.ioMode == nvinfer1::TensorIOMode::kOUTPUT;
        });

        outputs.clear();
        outputs.reserve(batchSize);

        for (int32_t b = 0; b < batchSize; ++b) {
            std::vector<std::vector<float> > batchOutput;
            batchOutput.reserve(outputCount);

            for (size_t i = 0; i < mTensorSpecs.size(); ++i) {
                const auto &spec = mTensorSpecs[i];

                if (spec.ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
                    const size_t elemPerBatch = spec.getElementCount() / batchSize;
                    const size_t bytesPerBatch = elemPerBatch * spec.getElementSize();

                    std::vector<float> output(elemPerBatch);

                    const auto *src = static_cast<const char *>(mBuffers[i]) + (b * bytesPerBatch);

                    FASTDET_ASSERT_MSG(
                        cudaMemcpyAsync(output.data(), src, bytesPerBatch, cudaMemcpyDeviceToHost, stream) ==
                        cudaSuccess,
                        "Failed to copy output data for tensor '{}', batch {}", spec.name, b);

                    batchOutput.emplace_back(std::move(output));
                }
            }

            outputs.emplace_back(std::move(batchOutput));
        }

        FASTDET_ASSERT_MSG(cudaStreamSynchronize(stream) == cudaSuccess, "Failed to synchronize CUDA stream");
        FASTDET_ASSERT_MSG(cudaStreamDestroy(stream) == cudaSuccess, "Failed to destroy CUDA stream");

        FASTDET_LOG_INFO("Inference completed successfully. Processed {} batches with {} output tensors each",
                         batchSize, outputCount);

        return true;
    }
}
