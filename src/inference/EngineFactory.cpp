#include "inference/EngineFactory.h"
#include "inference/TensorRTEngine.h"

#include <stdexcept>

namespace fastdet::inference {
    std::unique_ptr<IEngine> EngineFactory::create(EngineType type) {
        switch (type) {
            case EngineType::TensorRT:
                return std::make_unique<TensorRTEngine>();
            default:
                throw std::invalid_argument("Unsupported engine type");
        }
    }
}
