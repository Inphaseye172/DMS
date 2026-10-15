#include "logging.h"

#include "common.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <ctime>
#include <system_error>
#include <thread>

namespace dms {

namespace {
std::atomic<Logger*> g_logger{nullptr};
thread_local char t_last_error[DMS_MAX_MSG] = {0};
}  // namespace

void set_last_error(const char* msg) noexcept {
    if (!msg) msg = "";
    std::snprintf(t_last_error, sizeof(t_last_error), "%s", msg);
}
void clear_last_error() noexcept { t_last_error[0] = '\0'; }
const char* last_error() noexcept { return t_last_error; }

void log_error_noexcept(const char* where, const char* what) noexcept {
    logf(DMS_LOG_ERROR, "%s: %s", where ? where : "?", what ? what : "?");
}

Logger::Logger(std::filesystem::path dir, int level, size_t max_bytes, int keep)
    : dir_(std::move(dir)), level_(level), max_bytes_(max_bytes), keep_(keep) {
    std::lock_guard<std::mutex> lk(mu_);
    open_locked();
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lk(mu_);
    if (fp_) std::fclose(fp_);
    fp_ = nullptr;
}

void Logger::open_locked() noexcept {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    const auto p = dir_ / "dms.log";
#if defined(_WIN32)
    fp_ = _wfopen(p.c_str(), L"ab");
#else
    fp_ = std::fopen(p.c_str(), "ab");
#endif
    bytes_ = 0;
    if (fp_) {
        const auto sz = std::filesystem::file_size(p, ec);
        if (!ec) bytes_ = static_cast<size_t>(sz);
    }
}

void Logger::rotate_locked() noexcept {
    if (fp_) std::fclose(fp_);
    fp_ = nullptr;
    std::error_code ec;
    std::filesystem::remove(dir_ / ("dms." + std::to_string(keep_) + ".log"), ec);
    for (int i = keep_ - 1; i >= 1; --i) {
        std::filesystem::rename(dir_ / ("dms." + std::to_string(i) + ".log"), dir_ / ("dms." + std::to_string(i + 1) + ".log"), ec);
    }
    std::filesystem::rename(dir_ / "dms.log", dir_ / "dms.1.log", ec);
    open_locked();
}

void Logger::write(int level, const std::string& msg) noexcept {
    if (!enabled(level)) return;
    try {
        static const char* kNames[] = {"ERROR", "WARN ", "INFO ", "DEBUG"};
        const char* name = (level >= 0 && level <= 3) ? kNames[level] : "?????";
        const auto now = std::chrono::system_clock::now();
        const auto t = std::chrono::system_clock::to_time_t(now);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
        const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 100000;
        char head[96];
        std::snprintf(head, sizeof(head), "%s.%03d [%s] [t%05zu] ", ts, static_cast<int>(ms), name, static_cast<size_t>(tid));

        std::lock_guard<std::mutex> lk(mu_);
        if (!fp_) return;
        const int n = std::fprintf(fp_, "%s%s\n", head, msg.c_str());
        std::fflush(fp_);
        if (n > 0) bytes_ += static_cast<size_t>(n);
        if (bytes_ >= max_bytes_) rotate_locked();
    } catch (...) {  // NOLINT(bugprone-empty-catch): logging must never throw and has nowhere to report
    }
}

void set_global_logger(Logger* logger) noexcept { g_logger.store(logger); }

void log(int level, const std::string& msg) noexcept {
    Logger* l = g_logger.load();
    if (l) l->write(level, msg);
}

bool log_enabled(int level) noexcept {
    Logger* l = g_logger.load();
    return l && l->enabled(level);
}

void logf(int level, const char* fmt, ...) noexcept {
    Logger* l = g_logger.load();
    if (!l || !l->enabled(level) || !fmt) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    l->write(level, buf);
}

}  // namespace dms
