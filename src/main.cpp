#include <memory>
#include <stdexcept>
#include <string_view>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "common/Logger.h"

auto main() -> int {
    try {
        FASTDET_LOG_INFO("Starting TensorRT application");

        FASTDET_LOG_INFO("TensorRT version: {}.{}.{}",
                         NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess) {
            FASTDET_LOG_FATAL("CUDA error: {}", cudaGetErrorString(error));
            return EXIT_FAILURE;
        }

        if (deviceCount == 0) {
            FASTDET_LOG_ERROR("No CUDA devices found");
            return EXIT_FAILURE;
        }

        FASTDET_LOG_INFO("Found {} CUDA device(s)", deviceCount);

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp properties{};
            error = cudaGetDeviceProperties(&properties, i);
            if (error != cudaSuccess) {
                FASTDET_LOG_ERROR("Failed to get properties for device {}: {}",
                                 i, cudaGetErrorString(error));
                continue;
            }

            FASTDET_LOG_INFO("Device {}: {}", i, properties.name);
            FASTDET_LOG_INFO("  Compute capability: {}.{}", properties.major, properties.minor);
            FASTDET_LOG_INFO("  Total global memory: {:.2f} GB",
                           static_cast<float>(properties.totalGlobalMem) / (1024 * 1024 * 1024));
            FASTDET_LOG_INFO("  Multiprocessors: {}", properties.multiProcessorCount);
        }

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(gLogger));

        if (!builder) {
            FASTDET_LOG_FATAL("Failed to create TensorRT builder");
            return EXIT_FAILURE;
        }

        FASTDET_LOG_INFO("Successfully created TensorRT builder");

        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));

        if (!network) {
            FASTDET_LOG_FATAL("Failed to create TensorRT network definition");
            return EXIT_FAILURE;
        }

        FASTDET_LOG_INFO("Successfully created TensorRT network definition");

        FASTDET_LOG_INFO("All TensorRT API tests passed successfully");
        FASTDET_LOG_INFO("TensorRT test completed successfully");

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        FASTDET_LOG_FATAL("Application failed: {}", e.what());
        return EXIT_FAILURE;
    }
}