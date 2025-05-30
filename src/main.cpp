#include <memory>
#include <stdexcept>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "common/logging.h"
#include "common/assertion.h"
#include "core/TensorRTEngine.h"


void printDeviceInfo() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    FASTDET_ASSERT_MSG(error == cudaSuccess, "CUDA error: {}", cudaGetErrorString(error));
    FASTDET_ASSERT_MSG(deviceCount > 0, "No CUDA devices found, expected at least one");

    FASTDET_LOG_INFO("Found {} CUDA device(s)", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp properties{};
        error = cudaGetDeviceProperties(&properties, i);

        if (error == cudaSuccess) {
            FASTDET_LOG_INFO("Device {}: {}", i, properties.name);
            FASTDET_LOG_INFO("  Compute capability: {}.{}", properties.major, properties.minor);
            FASTDET_LOG_INFO("  Total global memory: {:.2f} GB", static_cast<float>(properties.totalGlobalMem) / (1024 * 1024 * 1024));
            FASTDET_LOG_INFO("  Multiprocessors: {}", properties.multiProcessorCount);
        }
    }
}

auto main(int argc, char *argv[]) -> int {
    try {
        FASTDET_LOG_INFO("Starting TensorRT sanity check");
        FASTDET_LOG_INFO("TensorRT version: {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

        printDeviceInfo();

        std::string onnxPath = (argc > 1)
                                   ? argv[1]
                                   : "/home/dev/Laboratory/fastdet.cpp/models/yolo11s.onnx";

        FASTDET_LOG_INFO("Using ONNX model: {}", onnxPath);

        fastdet::core::Options options;
        options.precision = fastdet::core::Precision::FP16;
        options.batchSize = 1;
        options.inputWidth = 640;
        options.inputHeight = 640;
        options.engineDir = "./engines";

        FASTDET_LOG_INFO("Creating engine instance");
        std::unique_ptr<fastdet::core::IEngine> engine = std::make_unique<fastdet::core::TensorRTEngine>();

        FASTDET_LOG_INFO("Building TensorRT engine from ONNX model");
        bool success = engine->build(onnxPath, options);

        if (success) {
            FASTDET_LOG_INFO("Engine built successfully!");
            
            const std::string enginePath = "/home/dev/Laboratory/fastdet.cpp/build/src/engines/yolo11s_fp16_b1_640x640.engine";
            
            std::array<float, 3> subVals{0.f, 0.f, 0.f};
            std::array<float, 3> divVals{1.f, 1.f, 1.f};
            bool normalize = true;

            FASTDET_LOG_INFO("Loading built engine from: {}", enginePath);
            bool loadSuccess = engine->load(enginePath, subVals, divVals, normalize);

            if (loadSuccess) {
                FASTDET_LOG_INFO("Engine loaded successfully!");
                return EXIT_SUCCESS;
            } else {
                FASTDET_LOG_ERROR("Failed to load engine");
                return EXIT_FAILURE;
            }
        } else {
            FASTDET_LOG_ERROR("Failed to build engine");
            return EXIT_FAILURE;
        }
    } catch (const std::exception &e) {
        FASTDET_LOG_FATAL("Sanity check failed: {}", e.what());
        return EXIT_FAILURE;
    }
}
