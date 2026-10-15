// Serialized inference: one worker thread owns the ONNX Runtime, all sessions and all GPU work.
// Render threads submit jobs into a bounded queue and wait on a condition variable with a timeout,
// polling the host abort callback every 50 ms (on their own thread). An abandoned job is cancelled;
// if it is already running, ORT's RunOptions terminate flag stops it. Jobs own their buffers
// (shared_ptr), so a caller that gives up never leaves the worker with dangling pointers.
#pragma once

#include "dms/engine.h"
#include "ort_runtime.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dms {

// Reuses large float buffers across frames so the hot path does not allocate per frame.
class BufferPool {
public:
    explicit BufferPool(size_t max_cached = 8) : max_cached_(max_cached) {}
    std::vector<float> take(size_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < free_.size(); ++i) {
            if (free_[i].capacity() >= n) {
                std::vector<float> v = std::move(free_[i]);
                free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
                v.resize(n);
                ++hits_;
                return v;
            }
        }
        ++misses_;
        return std::vector<float>(n);
    }
    void give(std::vector<float>&& v) {
        if (v.capacity() == 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (free_.size() >= max_cached_) free_.erase(free_.begin());  // drop the oldest
        free_.push_back(std::move(v));
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        free_.clear();
    }
    uint64_t hits() const { std::lock_guard<std::mutex> lk(mu_); return hits_; }
    uint64_t misses() const { std::lock_guard<std::mutex> lk(mu_); return misses_; }

private:
    mutable std::mutex mu_;
    std::vector<std::vector<float>> free_;
    size_t max_cached_;
    uint64_t hits_ = 0, misses_ = 0;
};

struct InferenceJob {
    InferenceJob() = default;
    explicit InferenceJob(std::shared_ptr<BufferPool> p) : pool(std::move(p)) {}
    ~InferenceJob() {
        if (pool) {
            pool->give(std::move(input));
            pool->give(std::move(output));
        }
    }
    InferenceJob(const InferenceJob&) = delete;
    InferenceJob& operator=(const InferenceJob&) = delete;

    std::shared_ptr<BufferPool> pool;
    SessionConfig cfg;
    int offload_mode = DMS_OFFLOAD_KEEP_VRAM;
    std::vector<float> input;
    std::vector<int64_t> shape;
    // results
    std::vector<float> output;
    std::vector<int64_t> out_shape;
    dms_status status = DMS_OK;
    std::string error;
    float ms_run = 0.f;
    float ms_session = 0.f;
    // state (guarded by the service mutex)
    bool done = false;
    bool started = false;
    std::atomic<bool> cancelled{false};
};

class InferenceService {
public:
    using RuntimeFactory = std::function<std::unique_ptr<OrtRuntime>()>;
    // Injected session factory for tests (fault injection). Default creates a real ModelSession.
    using SessionFactory = std::function<std::unique_ptr<ModelSession>(OrtRuntime&, const SessionConfig&)>;

    InferenceService(RuntimeFactory runtime_factory, size_t queue_capacity, uint32_t idle_ms);
    ~InferenceService();
    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    // Runs a job and waits for it. Returns the job status, DMS_ERR_TIMEOUT or DMS_ERR_ABORTED.
    dms_status execute(const std::shared_ptr<InferenceJob>& job, uint32_t timeout_ms, dms_abort_fn abort_fn, void* user);

    // Loads the runtime if needed (on the calling thread, serialized). Throws dms::Error.
    void ensure_runtime();
    std::string runtime_version() const;
    bool runtime_has_provider(const std::string& name) const;
    int last_backend() const noexcept { return last_backend_.load(); }

    // Drops all cached sessions (and RAM copies) on the worker thread; waits up to timeout_ms.
    void release_sessions(uint32_t timeout_ms = 5000);
    size_t session_count() const;
    bool has_session(const SessionConfig& cfg) const;

    void set_session_factory(SessionFactory f);
    // Test hook: pretend the runtime exposes these providers.
    void set_provider_override(std::vector<std::string> providers);

private:
    struct CachedSession {
        std::unique_ptr<ModelSession> session;
        std::chrono::steady_clock::time_point last_used;
        int offload_mode = DMS_OFFLOAD_KEEP_VRAM;
    };
    void worker_loop();
    void run_job(InferenceJob& job);
    ModelSession& session_for(const InferenceJob& job, float& ms_create);
    void apply_offload_policy_locked_out();
    void evict_all_sessions();
    void publish_sessions();

    RuntimeFactory runtime_factory_;
    SessionFactory session_factory_;
    const size_t capacity_;
    const uint32_t idle_ms_;

    mutable std::mutex mu_;
    std::condition_variable cv_work_;   // worker waits for jobs
    std::condition_variable cv_done_;   // callers wait for completion / queue space
    std::deque<std::shared_ptr<InferenceJob>> queue_;
    InferenceJob* running_ = nullptr;
    Ort::RunOptions* running_ro_ = nullptr;
    bool stop_ = false;
    bool release_requested_ = false;
    uint64_t release_generation_ = 0;

    std::mutex runtime_mu_;
    std::unique_ptr<OrtRuntime> runtime_;  // created under runtime_mu_; destroyed after the worker stops
    std::string runtime_version_;          // guarded by mu_
    std::vector<std::string> providers_;   // guarded by mu_

    // Worker-thread-only state (except session_count which takes mu_ for the size snapshot).
    std::vector<std::pair<SessionConfig, CachedSession>> sessions_;
    std::map<std::string, std::shared_ptr<const std::vector<uint8_t>>> ram_models_;
    size_t session_count_snapshot_ = 0;
    std::vector<SessionConfig> session_keys_snapshot_;  // guarded by mu_
    std::atomic<int> last_backend_{DMS_BACKEND_NONE};

    std::thread worker_;
};

}  // namespace dms
