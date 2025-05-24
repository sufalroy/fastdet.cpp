#pragma once

#include <string>
#include <cstdint>

namespace fastdet::core {
    enum class Precision : uint8_t {
        FP32 = 0,
        FP16 = 1
    };

    struct Options {
        Precision precision = Precision::FP16;
        int32_t optBatchSize = 1;
        int32_t maxBatchSize = 4;
        int32_t minInputWidth = 640;
        int32_t optInputWidth = 640;
        int32_t maxInputWidth = 640;
        std::string engineFileDir = "./engines";
    };

    class IEngine {
    public:
        virtual ~IEngine() = default;

        virtual bool build(const std::string &onnxPath, const Options &options) = 0;

        virtual bool load(const std::string &enginePath) = 0;
    };
}
