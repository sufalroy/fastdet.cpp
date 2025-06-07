#pragma once

#include "IEngine.h"
#include <memory>

namespace fastdet::inference {
    enum class EngineType : uint8_t {
        TensorRT = 0,
    };

    class EngineFactory {
    public:
        [[nodiscard]] static std::unique_ptr<IEngine> create(EngineType type);

    private:
        EngineFactory() = delete;
    };
}
