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

struct InferenceJob {
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

    void set_session_factory(SessionFactory f);

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
    std::atomic<int> last_backend_{DMS_BACKEND_NONE};

    std::thread worker_;
};

}  // namespace dms
