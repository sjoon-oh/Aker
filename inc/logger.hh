// 
// Author: Sukjoon Oh

#ifndef TOPKACHE_LOGGER_H_
#define TOPKACHE_LOGGER_H_

#include <string>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

namespace topkache 
{

    class Logger {
    private:
        std::string                         logger_name;
        spdlog::logger*                     shared_logger;

    public:
        // Private constructor to prevent instantiation
        Logger(const char* logger_name) noexcept
            : logger_name(logger_name)
        {
            shared_logger = spdlog::stdout_color_mt(std::string(logger_name)).get();
            spdlog::set_pattern("[%n:%^%l%$] %v");
        }

        // Delete copy constructor and assignment operator
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        static Logger& getInstance() noexcept
        {
            static Logger global_logger("log");
            return global_logger;
        }

        spdlog::logger* getLogger() noexcept
        {
            return shared_logger;
        }
    };

}

#endif