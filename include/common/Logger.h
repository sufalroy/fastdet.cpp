#pragma once

#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <fmt/format.h>
#include <source_location>
#include <string_view>

using namespace nvinfer1;
using Severity = nvinfer1::ILogger::Severity;

namespace fastdet::common {

    class Logger : public ILogger {
    public:
        explicit Logger(Severity severity = Severity::kINFO)
        : mLogger{spdlog::stdout_color_mt("FASTDET")} {
            mLogger->set_level(toSpdlogLevel(severity));
            mLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        }

        void log(Severity severity, const char* msg) noexcept override {
            if (mLogger->should_log(toSpdlogLevel(severity))) {
                mLogger->log(toSpdlogLevel(severity), msg);
            }
        }

        template <typename... Args>
        void logFormatted(Severity severity,
            std::string_view msg,
            const std::source_location& loc = std::source_location::current(),
            Args&&... args) {

            auto level = toSpdlogLevel(severity);

            if (mLogger->should_log(level)) {
            mLogger->log(spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
                         level,
                         fmt::format(fmt::runtime(msg), std::forward<Args>(args)...));
            }
        }

        void setLevel(Severity severity) noexcept {
            mLogger->set_level(toSpdlogLevel(severity));
        }

        [[nodiscard]] bool isEnabled(Severity severity) const noexcept {
            return mLogger->should_log(toSpdlogLevel(severity));
        }

    private:

        static constexpr spdlog::level::level_enum toSpdlogLevel(Severity severity) noexcept {
            switch (severity) {
                case Severity::kINTERNAL_ERROR: return spdlog::level::critical;
                case Severity::kERROR:         return spdlog::level::err;
                case Severity::kWARNING:       return spdlog::level::warn;
                case Severity::kINFO:          return spdlog::level::info;
                case Severity::kVERBOSE:       return spdlog::level::debug;
                default:                       return spdlog::level::info;
            }
        }

        std::shared_ptr<spdlog::logger> mLogger;
    };
}

extern fastdet::common::Logger gLogger;

#define FASTDET_LOG_FATAL(format, ...)   gLogger.logFormatted(Severity::kINTERNAL_ERROR, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_ERROR(format, ...)   gLogger.logFormatted(Severity::kERROR, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_WARNING(format, ...) gLogger.logFormatted(Severity::kWARNING, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_INFO(format, ...)    gLogger.logFormatted(Severity::kINFO, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_VERBOSE(format, ...) gLogger.logFormatted(Severity::kVERBOSE, format, std::source_location::current(), ##__VA_ARGS__)