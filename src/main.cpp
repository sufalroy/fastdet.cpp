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
        cudaError_t const error = cudaGetDeviceCount(&deviceCount);
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

        std::string onnxPath = (argc > 1)
                                   ? argv[1]
                                   : "/home/dev/Laboratory/fastdet.cpp/models/yolo11s.onnx";
        FASTDET_LOG_INFO("Using ONNX model: {}", onnxPath);

        fastdet::core::Options options;
        options.precision = fastdet::core::Precision::FP16;
        options.optBatchSize = 1;
        options.optInputWidth = 640;
        options.engineDir = "./engines";

        FASTDET_LOG_INFO("Creating and building TensorRT engine");
        std::unique_ptr<fastdet::core::IEngine> const engine = std::make_unique<fastdet::core::TensorRTEngine>();
        if (!engine->build(onnxPath, options)) {
            FASTDET_LOG_ERROR("Failed to build engine");
            return EXIT_FAILURE;
        }
        FASTDET_LOG_INFO("Engine built successfully");

        const std::string enginePath = "./engines/yolo11s_fp16_b1_w640.engine";
        std::array<float, 3> const subVals{0.F, 0.F, 0.F};
        std::array<float, 3> const divVals{1.F, 1.F, 1.F};
        bool const normalize = true;

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

        cv::cuda::GpuMat gpuImage;
        cv::cuda::GpuMat inputImage;
        gpuImage.upload(rgbImage);
        cv::cuda::resize(gpuImage, inputImage, cv::Size(640, 640));
        FASTDET_LOG_INFO("Resized image to: {}x{}", inputImage.cols, inputImage.rows);

        FASTDET_LOG_INFO("Running inference");
        std::vector<std::vector<cv::cuda::GpuMat> > const inputs{{{inputImage}}};
        std::vector<std::vector<std::vector<float> > > outputs;
        if (!engine->infer(inputs, outputs)) {
            FASTDET_LOG_ERROR("Inference failed or returned empty output");
            return EXIT_FAILURE;
        }
        
        for (std::size_t batch = 0; batch < outputs.size(); ++batch) {
            const auto& featureVectors = outputs[batch];
            
            for (std::size_t outputNum = 0; outputNum < featureVectors.size(); ++outputNum) {
                const auto& output = featureVectors[outputNum];
                FASTDET_LOG_INFO("Batch {}, output {}: [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, ...] (size: {})", 
                    batch, outputNum, 
                    output.empty() ? 0.0f : output[0],
                    output.size() > 1 ? output[1] : 0.0f,
                    output.size() > 2 ? output[2] : 0.0f,
                    output.size() > 3 ? output[3] : 0.0f,
                    output.size() > 4 ? output[4] : 0.0f,
                    output.size() > 5 ? output[5] : 0.0f,
                    output.size() > 6 ? output[6] : 0.0f,
                    output.size() > 7 ? output[7] : 0.0f,
                    output.size() > 8 ? output[8] : 0.0f,
                    output.size() > 9 ? output[9] : 0.0f,
                    output.size());
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        FASTDET_LOG_FATAL("Sanity check failed: {}", e.what());
        return EXIT_FAILURE;
    }
}
