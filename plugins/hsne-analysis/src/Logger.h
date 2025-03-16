#ifndef LOGGER_H
#define LOGGER_H

#include <memory>
#include <filesystem>
#include <string>
#include <streambuf>
#include <functional>
#include <array>
#include <iostream>

#include "spdlog/spdlog.h"                      // spdlog, main, header-only

// Use like:
//      #include "Logger.h"
//      Log::info("Important message");
//      Log::debug(fmt::format("Very {0} {1} messages", "helpful", 2));
namespace Log
{

// Singleton logger class
// Logs to both console and file
// 
// spdlog source: https://github.com/gabime/spdlog
// spdlog docs:   https://spdlog.docsforge.com/
class Logger {
public:
    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    static Logger& getInstance()
    {
        static Logger logger;
        return logger;
    }

    // lowest level log (0)
    static void trace(const std::string& message)
    {
        getInstance()._logger->trace(message);
    }

    // debug level log (1)
    static void debug(const std::string& message)
    {
        getInstance()._logger->debug(message);
    }

    // standard level log (2)
    static void info(const std::string& message)
    {
        getInstance()._logger->info(message);
    }

    // warning level log (3)
    static void warn(const std::string& message)
    {
        getInstance()._logger->warn(message);
    }

    // error level log (4)
    static void error(const std::string& message)
    {
        getInstance()._logger->error(message);
    }

    // critical level log (5)
    static void critical(const std::string& message)
    {
        getInstance()._logger->critical(message);
    }

    // set to "off" (6) to not log
    static void set_level(spdlog::level::level_enum log_level) {
        getInstance()._logger->set_level(log_level);
    }

    // redirect std cout, clog and cerr to this logger
    static void redirect_std_io_to_logger();

    // reset cout, clog and cerr to previous output (by default, console)
    static void reset_std_io(bool verbose = true);

    // Return path of log file
    std::string getLogFilePath() const
    {
        return getInstance()._log_file_path.string();
    }

    void static flush()
    {
        getInstance()._logger->flush();
    }

private:

    // Helper class for redirecting std io to this Logger, see https://en.cppreference.com/w/cpp/io/basic_streambuf
    class RedirectLog: public std::streambuf
    {
    public:
        RedirectLog(std::ostream& o, const spdlog::level::level_enum& sdplevel);
        ~RedirectLog();

    public:
        int overflow(int c) override;
        int sync(void) override;

    private:
        std::ostream& _std_out;
        std::streambuf* const _std_out_buf;
        std::string _sbuffer;
        std::function<void(const std::string& msg)> _log_fn;
    };

    // Private constructor
    Logger();

    // Logger singleton instance
    std::unique_ptr<spdlog::logger> _logger;

    // Log file path
    std::filesystem::path _log_file_path;

    // Pointer to std io redirection handler
    std::array<std::unique_ptr<RedirectLog>, 3> _redirectLogs;
};


// lowest level log (0)
inline void trace(const std::string& message){ Logger::trace(message); }

// debug level log (1)
inline void debug(const std::string& message){ Logger::debug(message); }

// standard level log (2)
inline void info(const std::string& message){ Logger::info(message); }

// warning level log (3)
inline void warn(const std::string& message){ Logger::warn(message); }

// error level log (4)
inline void error(const std::string& message){ Logger::error(message); }

// critical level log (5)
inline void critical(const std::string& message){ Logger::critical(message); }

// set to "off" (6) to not log
inline void set_level(spdlog::level::level_enum log_level) { Logger::set_level(log_level); }

// redirect std cout, clog and cerr to this logger
inline void redirect_std_io_to_logger() { Logger::redirect_std_io_to_logger();  }

// reset cout, clog and cerr to previous output (by default, console)
inline void reset_std_io(bool verbose = true) { Logger::reset_std_io(verbose); }

}


#endif LOGGER_H