#pragma once

#include "Logger.h"

#include <source_location>
#include <string_view>
#include <utility>

#if defined(_MSC_VER)
#define FASTDET_LIKELY(x) (x)
#define FASTDET_UNLIKELY(x) (x)
#else
#define FASTDET_LIKELY(x) __builtin_expect(!!(x), 1)
#define FASTDET_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

namespace fastdet::common {
    class Assert {
    public:
        template<typename... Args>
        static void check(
            bool condition,
            const std::source_location &loc,
            std::string_view message,
            Args &&... args) {
            if (FASTDET_UNLIKELY(!condition)) {
                gLogger.logFormatted(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                                     message,
                                     loc,
                                     std::forward<Args>(args)...);
                throw std::runtime_error(fmt::format(fmt::runtime(message), std::forward<Args>(args)...));
            }
        }

        template<typename... Args>
        static void checkDebug(
            bool condition,
            const std::source_location &loc,
            std::string_view message,
            Args &&... args) {
#ifndef NDEBUG
            check(condition, loc, message, std::forward<Args>(args)...);
#endif
        }
    };
}

#define FASTDET_ASSERT(condition) \
    fastdet::common::Assert::check((condition), std::source_location::current(), "Assertion failed: {}", #condition)

#define FASTDET_ASSERT_MSG(condition, message, ...) \
    fastdet::common::Assert::check((condition), std::source_location::current(), "Assertion failed: " message, ##__VA_ARGS__)

#define FASTDET_ASSERT_DEBUG(condition) \
    fastdet::common::Assert::checkDebug((condition), std::source_location::current(), "Debug assertion failed: {}", #condition)

#define FASTDET_ASSERT_DEBUG_MSG(condition, message, ...) \
    fastdet::common::Assert::checkDebug((condition), std::source_location::current(), "Debug assertion failed: " message, ##__VA_ARGS__)

#define FASTDET_THROW(message, ...) \
    do { \
        gLogger.logFormatted(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, message, std::source_location::current(), ##__VA_ARGS__); \
        throw std::runtime_error(fmt::format(fmt::runtime(message), ##__VA_ARGS__)); \
    } while (0)
