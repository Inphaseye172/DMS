// Engine singleton internals (not part of the C ABI).
#pragma once

#include "common.h"
#include "backend_plan.h"
#include "downloader.h"
#include "frame_cache.h"
#include "instance_registry.h"
#include "gpu_probe.h"
#include "inference_service.h"
#include "logging.h"
#include "manifest.h"

#include <atomic>
#include <chrono>
#include <map>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
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

// Per-stage timings (debug log).
struct StageTimes {
    float preprocess = 0, session = 0, inference = 0, postprocess = 0, total = 0;
};

class Engine : public dms_engine {
public:
    explicit Engine(EngineConfig cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const EngineConfig& config() const noexcept { return cfg_; }
    Logger& logger() noexcept { return *logger_; }
    Manifest& manifest() noexcept { return manifest_; }
    Downloader& downloader() noexcept { return downloader_; }
    InferenceService& inference() noexcept { return *inference_; }
    FrameCache& cache() noexcept { return cache_; }
    InstanceRegistry& instances() noexcept { return instances_; }
    // Builds the cache key for a request (normalizes every parameter that affects the depth result).
    CacheKey cache_key(const dms_cache_key& k, const std::string& model_id, int base_res, const dms_backend_settings* b, float edges,
                       float smoothing, int w, int h) const;

    dms_status compute(const dms_depth_request& req, dms_depth_result& res);
    GpuMonitor& gpu();
    BufferPool& pool() noexcept { return *pool_; }
    std::shared_ptr<BufferPool> pool_ptr() noexcept { return pool_; }

    // Test hooks.
    void set_gpu_monitor(std::unique_ptr<GpuMonitor> m);
    void fill_model_info(const ModelSpec& m, dms_model_info& out);
    dms_status verify_model(const std::string& id);
    void mark_verified(const std::string& id);

    struct ComputeCtx;  // defined in engine_compute.cpp

private:
    std::unique_ptr<OrtRuntime> make_runtime();
    dms_status run_attempt(ComputeCtx& ctx, const Attempt& a, std::string& err);
    dms_status run_pass(ComputeCtx& ctx, const Attempt& a, int crop_x0, int crop_y0, int crop_x1, int crop_y1, int iw, int ih,
                        std::vector<float>& out_small, int& ow, int& oh, std::string& err);
    dms_status run_tiled(ComputeCtx& ctx, const Attempt& a, std::string& err);
    dms_status run_temporal(ComputeCtx& ctx, const Attempt& a, std::string& err);
    bool oom_recent(const std::string& key);
    void note_oom(const std::string& key);

    EngineConfig cfg_;
    std::unique_ptr<Logger> logger_;
    Manifest manifest_;
    Downloader downloader_;
    std::unique_ptr<InferenceService> inference_;
    FrameCache cache_;
    InstanceRegistry instances_;
    std::mutex verified_mu_;
    std::set<std::string> verified_;

    std::mutex gpu_mu_;
    std::unique_ptr<GpuMonitor> gpu_;  // created lazily (NVML/DXGI probe)
    std::shared_ptr<BufferPool> pool_ = std::make_shared<BufferPool>(16);
    std::atomic<bool> cuda_sticky_unavailable_{false};
    std::atomic<bool> dml_sticky_unavailable_{false};
    std::atomic<bool> memshare_warned_{false};
    std::mutex oom_mu_;
    std::map<std::string, std::chrono::steady_clock::time_point> oom_memo_;
};

// Returns a strong reference to the live engine iff `h` is the current engine handle; throws otherwise.
std::shared_ptr<Engine> lock_engine(dms_engine* h);

}  // namespace dms
