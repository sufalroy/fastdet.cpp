#include <memory>
#include <stdexcept>
#include <string>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "common/logging.h"
#include "common/assertion.h"
#include "core/engine.h"


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
            FASTDET_LOG_INFO("  Total global memory: {:.2f} GB",
                           static_cast<float>(properties.totalGlobalMem) / (1024 * 1024 * 1024));
            FASTDET_LOG_INFO("  Multiprocessors: {}", properties.multiProcessorCount);
        }
    }
}

auto main(int argc, char* argv[]) -> int {
    try {
        FASTDET_LOG_INFO("Starting TensorRT sanity check");
        FASTDET_LOG_INFO("TensorRT version: {}.{}.{}",
                        NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

        printDeviceInfo();

        std::string onnxPath = (argc > 1) ? argv[1] : "C:/Laboratory/cpp-workspace/fastdet.cpp/models/version-RFB-640.onnx";
        FASTDET_LOG_INFO("Using ONNX model: {}", onnxPath);

        fastdet::core::Options options;
        options.precision = fastdet::core::Precision::FP16;
        options.optBatchSize = 1;
        options.maxBatchSize = 1;
        options.minInputWidth = 640;
        options.optInputWidth = 640;
        options.maxInputWidth = 640;
        options.engineFileDir = "./engines";

        FASTDET_LOG_INFO("Creating engine instance");
        fastdet::core::Engine engine;

        FASTDET_LOG_INFO("Building TensorRT engine from ONNX model");
        bool success = engine.build(onnxPath, options);

        if (success) {
            FASTDET_LOG_INFO("Engine built successfully!");
            return EXIT_SUCCESS;
        } else {
            FASTDET_LOG_ERROR("Failed to build engine");
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        FASTDET_LOG_FATAL("Sanity check failed: {}", e.what());
        return EXIT_FAILURE;
    }
}