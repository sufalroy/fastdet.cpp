#pragma once

#include <source_location>
#include <string_view>
#include <utility>

#include "Logger.h"

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
        template <typename... Args>
        static void check(
            bool condition,
            std::string_view message,
            const std::source_location& loc = std::source_location::current(),
            Args&&... args) {
            if (FASTDET_UNLIKELY(!condition)) {
                FASTDET_LOG_FATAL(message, std::forward<Args>(args)...);
                throw std::runtime_error(fmt::format(fmt::runtime(message), std::forward<Args>(args)...));
            }
        }

        template <typename... Args>
        static void checkDebug(
            bool condition,
            std::string_view message,
            const std::source_location& loc = std::source_location::current(),
            Args&&... args) {
#ifndef NDEBUG
            check(condition, message, loc, std::forward<Args>(args)...);
#endif
        }
    };
}

#define FASTDET_ASSERT(condition) \
    fastdet::common::Assert::check((condition), "Assertion failed: {}", std::source_location::current(), #condition)

#define FASTDET_ASSERT_MSG(condition, message, ...) \
    fastdet::common::Assert::check((condition), "Assertion failed: " message, std::source_location::current(), ##__VA_ARGS__)

#define FASTDET_ASSERT_DEBUG(condition) \
    fastdet::common::Assert::checkDebug((condition), "Debug assertion failed: {}", std::source_location::current(), #condition)

#define FASTDET_ASSERT_DEBUG_MSG(condition, message, ...) \
    fastdet::common::Assert::checkDebug((condition), "Debug assertion failed: " message, std::source_location::current(), ##__VA_ARGS__)

#define FASTDET_THROW(message, ...) \
    do { \
        FASTDET_LOG_FATAL(message, ##__VA_ARGS__); \
        throw std::runtime_error(fmt::format(message, ##__VA_ARGS__)); \
    } while (0)