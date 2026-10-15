// Internal engine utilities: error type, ABI guard, small helpers.
#pragma once

#include "dms/engine.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace dms {

// Engine-internal exception. Never crosses the C ABI: every exported function goes through guard().
class Error : public std::runtime_error {
public:
    Error(dms_status s, const std::string& msg) : std::runtime_error(msg), status_(s) {}
    dms_status status() const noexcept { return status_; }

private:
    dms_status status_;
};

[[noreturn]] inline void fail(dms_status s, const std::string& msg) { throw Error(s, msg); }

// Thread-local detailed error message (exposed via dms_last_error()).
void set_last_error(const char* msg) noexcept;
void clear_last_error() noexcept;
const char* last_error() noexcept;

// Logs through the global logger if one exists (implemented in logging.cpp).
void log_error_noexcept(const char* where, const char* what) noexcept;

// Runs fn() and converts any exception into a dms_status. This is the ONLY way exported functions run code.
template <typename Fn>
dms_status guard(const char* where, Fn&& fn) noexcept {
    try {
        clear_last_error();
        return std::forward<Fn>(fn)();
    } catch (const Error& e) {
        set_last_error(e.what());
        log_error_noexcept(where, e.what());
        return e.status();
    } catch (const std::bad_alloc&) {
        set_last_error("out of memory");
        log_error_noexcept(where, "std::bad_alloc");
        return DMS_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        log_error_noexcept(where, e.what());
        return DMS_ERR_INTERNAL;
    } catch (...) {
        set_last_error("unknown exception");
        log_error_noexcept(where, "unknown exception");
        return DMS_ERR_INTERNAL;
    }
}

inline bool finite(float v) noexcept { return std::isfinite(v); }
inline float clampf(float v, float lo, float hi) noexcept {
    if (!std::isfinite(v)) return lo;
    return v < lo ? lo : (v > hi ? hi : v);
}
inline int clampi(int v, int lo, int hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }

// Copies a std::string into a fixed C buffer, always NUL-terminated.
inline void copy_str(char* dst, size_t cap, const std::string& s) noexcept {
    if (!dst || cap == 0) return;
    const size_t n = s.size() < cap - 1 ? s.size() : cap - 1;
    std::memcpy(dst, s.data(), n);
    dst[n] = '\0';
}

// Validates image dimensions (positive, bounded to avoid overflow: max 32768 per side as AE does).
inline void check_dims(int32_t w, int32_t h, const char* what) {
    if (w <= 0 || h <= 0 || w > 32768 || h > 32768)
        fail(DMS_ERR_INVALID_ARG, std::string(what) + ": invalid dimensions " + std::to_string(w) + "x" + std::to_string(h));
}

inline size_t pixel_count(int32_t w, int32_t h) noexcept { return static_cast<size_t>(w) * static_cast<size_t>(h); }

template <typename T>
inline void check_struct(const T* p, const char* what) {
    if (!p) fail(DMS_ERR_INVALID_ARG, std::string(what) + " is NULL");
    if (p->struct_size < sizeof(uint32_t) || p->struct_size > 1u << 20)
        fail(DMS_ERR_INVALID_ARG, std::string(what) + ": struct_size not set");
}

}  // namespace dms
