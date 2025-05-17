#pragma once

#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <fmt/format.h>
#include <source_location>
#include <string_view>
#include <mutex>

namespace fastdet::common {

    class Logger : public nvinfer1::ILogger {
    public:
        explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
        : mLogger{spdlog::stdout_color_mt("FASTDET")} {
            mLogger->set_level(toSpdlogLevel(severity));
            mLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        }

        virtual ~Logger() = default;

        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
            std::lock_guard<std::mutex> lock(mLogMutex);
            if (mLogger->should_log(toSpdlogLevel(severity))) {
                mLogger->log(toSpdlogLevel(severity), msg);
            }
        }

        template <typename... Args>
        void logFormatted(nvinfer1::ILogger::Severity severity,
            std::string_view msg,
            const std::source_location& loc = std::source_location::current(),
            Args&&... args) noexcept {

            auto level = toSpdlogLevel(severity);

            if (mLogger->should_log(level)) {
                std::lock_guard<std::mutex> lock(mLogMutex);
                mLogger->log(spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
                             level,
                             fmt::format(fmt::runtime(msg), std::forward<Args>(args)...));
            }
        }

        void setLevel(nvinfer1::ILogger::Severity severity) noexcept {
            std::lock_guard<std::mutex> lock(mLogMutex);
            mLogger->set_level(toSpdlogLevel(severity));
        }

        [[nodiscard]] bool isEnabled(nvinfer1::ILogger::Severity severity) const noexcept {
            return mLogger->should_log(toSpdlogLevel(severity));
        }

    private:
        static constexpr spdlog::level::level_enum toSpdlogLevel(nvinfer1::ILogger::Severity severity) noexcept {
            switch (severity) {
                case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return spdlog::level::critical;
                case nvinfer1::ILogger::Severity::kERROR:         return spdlog::level::err;
                case nvinfer1::ILogger::Severity::kWARNING:       return spdlog::level::warn;
                case nvinfer1::ILogger::Severity::kINFO:          return spdlog::level::info;
                case nvinfer1::ILogger::Severity::kVERBOSE:       return spdlog::level::debug;
                default:                                          return spdlog::level::info;
            }
        }

        std::shared_ptr<spdlog::logger> mLogger;
        std::mutex mLogMutex;
    };
}

extern fastdet::common::Logger gLogger;

#define FASTDET_LOG_FATAL(format, ...)   gLogger.logFormatted(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_ERROR(format, ...)   gLogger.logFormatted(nvinfer1::ILogger::Severity::kERROR, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_WARNING(format, ...) gLogger.logFormatted(nvinfer1::ILogger::Severity::kWARNING, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_INFO(format, ...)    gLogger.logFormatted(nvinfer1::ILogger::Severity::kINFO, format, std::source_location::current(), ##__VA_ARGS__)
#define FASTDET_LOG_VERBOSE(format, ...) gLogger.logFormatted(nvinfer1::ILogger::Severity::kVERBOSE, format, std::source_location::current(), ##__VA_ARGS__)