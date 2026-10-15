#pragma once
#include "dms/engine.h"

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>

namespace dms {

// Rotating file logger: <dir>/dms.log, rotated to dms.1.log .. dms.<keep>.log at max_bytes.
class Logger {
public:
    Logger(std::filesystem::path dir, int level, size_t max_bytes = 2u * 1024u * 1024u, int keep = 5);
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void write(int level, const std::string& msg) noexcept;
    void set_level(int level) noexcept { level_ = level; }
    int level() const noexcept { return level_; }
    bool enabled(int level) const noexcept { return level <= level_; }
    std::filesystem::path path() const { return dir_ / "dms.log"; }

private:
    void rotate_locked() noexcept;
    void open_locked() noexcept;

    std::filesystem::path dir_;
    int level_;
    size_t max_bytes_;
    int keep_;
    std::mutex mu_;
    std::FILE* fp_ = nullptr;
    size_t bytes_ = 0;
};

// Process-wide logger slot (owned by the engine singleton; may be null).
void set_global_logger(Logger* logger) noexcept;
void log(int level, const std::string& msg) noexcept;
bool log_enabled(int level) noexcept;
void logf(int level, const char* fmt, ...) noexcept
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

}  // namespace dms
