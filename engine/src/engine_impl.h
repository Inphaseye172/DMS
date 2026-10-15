// Engine singleton internals (not part of the C ABI).
#pragma once

#include "common.h"
#include "logging.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

struct dms_engine {
    // Marker so that a garbage pointer is rejected instead of dereferenced blindly.
    uint64_t magic = 0x444D5345'4E47494EULL;  // "DMSENGIN"
};

namespace dms {

struct EngineConfig {
    std::filesystem::path models_dir;
    std::filesystem::path log_dir;
    std::filesystem::path deps_dir;
    int log_level = DMS_LOG_INFO;
    uint64_t vram_reserve_bytes = 1ull << 30;
    uint64_t cache_budget_bytes = 2ull << 30;
    uint32_t inference_timeout_ms = 120000;
    uint32_t max_neighbor_frames = 32;
    uint32_t offload_idle_ms = 10000;
};

EngineConfig resolve_config(const dms_engine_config* c);

class Engine : public dms_engine {
public:
    explicit Engine(EngineConfig cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const EngineConfig& config() const noexcept { return cfg_; }
    Logger& logger() noexcept { return *logger_; }

    // Implemented in engine_compute.cpp (grows per milestone).
    dms_status compute(const dms_depth_request& req, dms_depth_result& res);

private:
    EngineConfig cfg_;
    std::unique_ptr<Logger> logger_;
};

// Returns a strong reference to the live engine iff `h` is the current engine handle; throws otherwise.
std::shared_ptr<Engine> lock_engine(dms_engine* h);

}  // namespace dms
