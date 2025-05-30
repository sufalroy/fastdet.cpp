#include <memory>
#include <stdexcept>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "common/logging.h"
#include "common/assertion.h"
#include "core/TensorRTEngine.h"

auto main(int argc, char *argv[]) -> int {
    try {
        FASTDET_LOG_INFO("Starting TensorRT sanity check");
        FASTDET_LOG_INFO("TensorRT version: {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        FASTDET_ASSERT_MSG(error == cudaSuccess, "CUDA error: {}", cudaGetErrorString(error));
        FASTDET_ASSERT_MSG(deviceCount > 0, "No CUDA devices found");
        FASTDET_LOG_INFO("Found {} CUDA device(s)", deviceCount);

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp properties{};
            cudaGetDeviceProperties(&properties, i);
            FASTDET_LOG_INFO("Device {}: {}, Compute: {}.{}, Memory: {:.2f} GB, Multiprocessors: {}", 
                i, properties.name, properties.major, properties.minor, 
                static_cast<float>(properties.totalGlobalMem) / (1024 * 1024 * 1024), 
                properties.multiProcessorCount);
        }

        std::string onnxPath = (argc > 1) ? argv[1] : "/home/dev/Laboratory/fastdet.cpp/models/yolo11s.onnx";
        FASTDET_LOG_INFO("Using ONNX model: {}", onnxPath);

        fastdet::core::Options options;
        options.precision = fastdet::core::Precision::FP16;
        options.batchSize = 1;
        options.inputWidth = 640;
        options.inputHeight = 640;
        options.engineDir = "./engines";

        FASTDET_LOG_INFO("Creating and building TensorRT engine");
        std::unique_ptr<fastdet::core::IEngine> engine = std::make_unique<fastdet::core::TensorRTEngine>();
        if (!engine->build(onnxPath, options)) {
            FASTDET_LOG_ERROR("Failed to build engine");
            return EXIT_FAILURE;
        }
        FASTDET_LOG_INFO("Engine built successfully");

        const std::string enginePath = "./engines/yolo11s_fp16_b1_640x640.engine";
        std::array<float, 3> subVals{0.f, 0.f, 0.f};
        std::array<float, 3> divVals{1.f, 1.f, 1.f};
        bool normalize = true;
        FASTDET_LOG_INFO("Loading engine from: {}", enginePath);
        if (!engine->load(enginePath, subVals, divVals, normalize)) {
            FASTDET_LOG_ERROR("Failed to load engine");
            return EXIT_FAILURE;
        }
        FASTDET_LOG_INFO("Engine loaded successfully");

        const std::string imagePath = "/home/dev/Laboratory/fastdet.cpp/inputs/sample.jpg";
        FASTDET_LOG_INFO("Loading image from: {}", imagePath);
        cv::Mat cpuImage = cv::imread(imagePath, cv::IMREAD_COLOR);
        FASTDET_ASSERT_MSG(!cpuImage.empty(), "Failed to load image: {}", imagePath);
        FASTDET_LOG_INFO("Original image size: {}x{}", cpuImage.cols, cpuImage.rows);

        cv::Mat rgbImage;
        cv::cvtColor(cpuImage, rgbImage, cv::COLOR_BGR2RGB);
        FASTDET_LOG_INFO("Converted image to RGB");

        cv::cuda::GpuMat gpuImage, inputImage;
        gpuImage.upload(rgbImage);
        cv::cuda::resize(gpuImage, inputImage, cv::Size(640, 640));
        FASTDET_LOG_INFO("Resized image to: {}x{}", inputImage.cols, inputImage.rows);

        FASTDET_LOG_INFO("Running inference");
        std::vector<std::vector<cv::cuda::GpuMat>> inputs{{{inputImage}}};
        std::vector<std::vector<std::vector<float>>> outputs;
        if (!engine->infer(inputs, outputs) || outputs.empty()) {
            FASTDET_LOG_ERROR("Inference failed or returned empty output");
            return EXIT_FAILURE;
        }

        FASTDET_LOG_INFO("Inference successful! Batches: {}", outputs.size());
        if (!outputs[0].empty()) {
            FASTDET_LOG_INFO("Output tensors per batch: {}", outputs[0].size());
            for (size_t i = 0; i < outputs[0].size(); ++i) {
                FASTDET_LOG_INFO("Tensor {}: {} elements", i, outputs[0][i].size());
                if (!outputs[0][i].empty()) {
                    FASTDET_LOG_INFO("First 5 elements: [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]", 
                        outputs[0][i].size() > 0 ? outputs[0][i][0] : 0.0f,
                        outputs[0][i].size() > 1 ? outputs[0][i][1] : 0.0f,
                        outputs[0][i].size() > 2 ? outputs[0][i][2] : 0.0f,
                        outputs[0][i].size() > 3 ? outputs[0][i][3] : 0.0f,
                        outputs[0][i].size() > 4 ? outputs[0][i][4] : 0.0f);
                }
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        FASTDET_LOG_FATAL("Sanity check failed: {}", e.what());
        return EXIT_FAILURE;
    }
}