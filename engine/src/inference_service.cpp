#include "inference_service.h"

#include "common.h"
#include "logging.h"
#include "platform.h"

#include <fstream>

namespace dms {

InferenceService::InferenceService(RuntimeFactory runtime_factory, size_t queue_capacity, uint32_t idle_ms)
    : runtime_factory_(std::move(runtime_factory)),
      session_factory_([](OrtRuntime& rt, const SessionConfig& c) { return std::make_unique<ModelSession>(rt, c); }),
      capacity_(queue_capacity ? queue_capacity : 8),
      idle_ms_(idle_ms ? idle_ms : 10000) {
    worker_ = std::thread([this] { worker_loop(); });
}

InferenceService::~InferenceService() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        for (auto& j : queue_) {
            j->cancelled = true;
            j->status = DMS_ERR_ABORTED;
            j->done = true;
        }
        queue_.clear();
        if (running_ro_) running_ro_->SetTerminate();
    }
    cv_work_.notify_all();
    cv_done_.notify_all();
    if (worker_.joinable()) worker_.join();
    // Sessions were destroyed by the worker on exit; the runtime goes last.
    std::lock_guard<std::mutex> lk(runtime_mu_);
    runtime_.reset();
}

void InferenceService::set_session_factory(SessionFactory f) {
    std::lock_guard<std::mutex> lk(mu_);
    session_factory_ = std::move(f);
}

void InferenceService::set_provider_override(std::vector<std::string> providers) {
    ensure_runtime();
    std::lock_guard<std::mutex> lk(mu_);
    providers_ = std::move(providers);
}

void InferenceService::ensure_runtime() {
    std::lock_guard<std::mutex> lk(runtime_mu_);
    if (runtime_) return;
    runtime_ = runtime_factory_();
    if (!runtime_) fail(DMS_ERR_BACKEND_UNAVAILABLE, "no ONNX Runtime available");
    std::lock_guard<std::mutex> lk2(mu_);
    runtime_version_ = runtime_->version();
    providers_ = runtime_->providers();
}

std::string InferenceService::runtime_version() const {
    std::lock_guard<std::mutex> lk(mu_);
    return runtime_version_;
}

bool InferenceService::runtime_has_provider(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& p : providers_)
        if (p == name) return true;
    return false;
}

size_t InferenceService::session_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return session_count_snapshot_;
}

bool InferenceService::has_session(const SessionConfig& cfg) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& k : session_keys_snapshot_)
        if (k == cfg) return true;
    return false;
}

dms_status InferenceService::execute(const std::shared_ptr<InferenceJob>& job, uint32_t timeout_ms, dms_abort_fn abort_fn, void* user) {
    if (!job) return DMS_ERR_INVALID_ARG;
    ensure_runtime();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms ? timeout_ms : 120000);
    auto check_abort = [&]() -> dms_status {
        if (abort_fn && abort_fn(user)) return DMS_ERR_ABORTED;
        if (std::chrono::steady_clock::now() >= deadline) return DMS_ERR_TIMEOUT;
        return DMS_OK;
    };
    auto give_up = [&](dms_status why, std::unique_lock<std::mutex>& lk) {
        job->cancelled = true;
        if (running_ == job.get() && running_ro_) running_ro_->SetTerminate();
        // Remove from the queue if it has not started.
        for (auto it = queue_.begin(); it != queue_.end(); ++it)
            if (it->get() == job.get()) {
                queue_.erase(it);
                break;
            }
        lk.unlock();
        cv_done_.notify_all();
        logf(DMS_LOG_WARN, "inference job abandoned: %s", why == DMS_ERR_TIMEOUT ? "timeout" : "aborted by host");
        return why;
    };

    std::unique_lock<std::mutex> lk(mu_);
    // Bounded queue: wait for space (with abort/timeout), never spin.
    while (queue_.size() >= capacity_ && !stop_) {
        lk.unlock();
        const dms_status a = check_abort();
        lk.lock();
        if (a != DMS_OK) return give_up(a, lk);
        cv_done_.wait_for(lk, std::chrono::milliseconds(50));
    }
    if (stop_) return DMS_ERR_NOT_INITIALIZED;
    queue_.push_back(job);
    cv_work_.notify_one();
    while (!job->done) {
        cv_done_.wait_for(lk, std::chrono::milliseconds(50));
        if (job->done) break;
        lk.unlock();
        const dms_status a = check_abort();  // host callback invoked WITHOUT holding our lock
        lk.lock();
        if (a != DMS_OK && !job->done) return give_up(a, lk);
    }
    return job->status;
}

void InferenceService::release_sessions(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    const uint64_t gen = ++release_generation_;
    release_requested_ = true;
    cv_work_.notify_one();
    cv_done_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !release_requested_ || stop_ || release_generation_ != gen; });
}

void InferenceService::publish_sessions() {
    std::vector<SessionConfig> keys;
    for (const auto& s : sessions_) keys.push_back(s.first);
    std::lock_guard<std::mutex> lk(mu_);
    session_count_snapshot_ = keys.size();
    session_keys_snapshot_ = std::move(keys);
}

void InferenceService::evict_all_sessions() {
    sessions_.clear();
    ram_models_.clear();
    publish_sessions();
}

ModelSession& InferenceService::session_for(const InferenceJob& job, float& ms_create) {
    for (auto& [cfg, cs] : sessions_) {
        if (cfg == job.cfg) {
            cs.last_used = std::chrono::steady_clock::now();
            cs.offload_mode = job.offload_mode;
            return *cs.session;
        }
    }
    SessionConfig cfg = job.cfg;
    if (job.offload_mode == DMS_OFFLOAD_TO_RAM) {
        // Keep the model bytes in RAM so re-creating the GPU session after offload skips disk I/O.
        const std::string key = platform::path_to_utf8(cfg.model_path);
        auto it = ram_models_.find(key);
        if (it == ram_models_.end()) {
            std::ifstream f(cfg.model_path, std::ios::binary | std::ios::ate);
            if (f) {
                auto bytes = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(f.tellg()));
                f.seekg(0);
                f.read(reinterpret_cast<char*>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
                if (f) it = ram_models_.emplace(key, std::move(bytes)).first;
            }
        }
        if (it != ram_models_.end()) cfg.model_bytes = it->second;
    }
    const auto t0 = std::chrono::steady_clock::now();
    SessionFactory factory;
    {
        std::lock_guard<std::mutex> lk(mu_);
        factory = session_factory_;
    }
    std::unique_ptr<ModelSession> s;
    try {
        s = factory(*runtime_, cfg);
    } catch (const Error& e) {
        if (e.status() != DMS_ERR_OUT_OF_MEMORY || sessions_.empty()) throw;
        // Memory pressure: drop every other cached session and retry once.
        logf(DMS_LOG_WARN, "session creation OOM; evicting %zu cached sessions and retrying", sessions_.size());
        sessions_.clear();
        publish_sessions();
        s = factory(*runtime_, cfg);
    }
    ms_create = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CachedSession cs;
    cs.session = std::move(s);
    cs.last_used = std::chrono::steady_clock::now();
    cs.offload_mode = job.offload_mode;
    sessions_.emplace_back(job.cfg, std::move(cs));
    publish_sessions();
    return *sessions_.back().second.session;
}

void InferenceService::run_job(InferenceJob& job) {
    const auto t0 = std::chrono::steady_clock::now();
    try {
        ModelSession& s = session_for(job, job.ms_session);
        Ort::RunOptions ro;
        {
            std::lock_guard<std::mutex> lk(mu_);
            running_ro_ = &ro;
            if (job.cancelled) ro.SetTerminate();
        }
        struct ClearRo {
            InferenceService* self;
            ~ClearRo() {
                std::lock_guard<std::mutex> lk(self->mu_);
                self->running_ro_ = nullptr;
            }
        } clear_ro{this};
        const auto& out = s.run(job.input.data(), job.shape, job.out_shape, ro);
        size_t n = 1;
        for (int64_t d : job.out_shape) n *= static_cast<size_t>(d > 0 ? d : 1);
        if (n > out.size()) fail(DMS_ERR_INFERENCE_FAILED, "output shape larger than buffer");
        job.output.resize(n);  // job.output usually comes from the buffer pool with enough capacity
        std::copy(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n), job.output.begin());
        job.status = DMS_OK;
        last_backend_ = static_cast<int>(job.cfg.backend);
    } catch (const Error& e) {
        job.status = e.status();
        job.error = e.what();
    } catch (const Ort::Exception& e) {
        job.status = classify_ort_error(e.what());
        job.error = e.what();
    } catch (const std::bad_alloc&) {
        job.status = DMS_ERR_OUT_OF_MEMORY;
        job.error = "host out of memory";
    } catch (const std::exception& e) {
        job.status = DMS_ERR_INTERNAL;
        job.error = e.what();
    } catch (...) {
        job.status = DMS_ERR_INTERNAL;
        job.error = "unknown exception in inference worker";
    }
    job.ms_run = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count() - job.ms_session;
    if (job.status == DMS_ERR_OUT_OF_MEMORY) {
        // Free everything we hold so the next fallback step starts from a clean slate.
        logf(DMS_LOG_WARN, "OOM during inference (%s); releasing cached sessions", job.error.c_str());
        sessions_.clear();
        publish_sessions();
    }
}

void InferenceService::apply_offload_policy_locked_out() {
    const auto now = std::chrono::steady_clock::now();
    bool changed = false;
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        const auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.last_used).count();
        const bool gpu = it->first.backend != Backend::CPU;
        const int mode = it->second.offload_mode;
        if (idle >= static_cast<long long>(idle_ms_) && gpu && mode != DMS_OFFLOAD_KEEP_VRAM) {
            logf(DMS_LOG_INFO, "offloading idle %s session (%s)", backend_name(it->first.backend), mode == DMS_OFFLOAD_TO_RAM ? "to RAM" : "full unload");
            if (mode == DMS_OFFLOAD_FULL_UNLOAD) ram_models_.erase(platform::path_to_utf8(it->first.model_path));
            it = sessions_.erase(it);
            changed = true;
        } else {
            ++it;
        }
    }
    if (changed) publish_sessions();
}

void InferenceService::worker_loop() {
    for (;;) {
        std::shared_ptr<InferenceJob> job;
        bool do_release = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait_for(lk, std::chrono::milliseconds(idle_ms_ / 4 + 1), [&] { return stop_ || !queue_.empty() || release_requested_; });
            if (stop_) break;
            if (release_requested_) {
                do_release = true;
            } else if (!queue_.empty()) {
                job = queue_.front();
                queue_.pop_front();
                running_ = job.get();
                job->started = true;
            }
        }
        cv_done_.notify_all();  // queue space freed
        if (do_release) {
            evict_all_sessions();
            {
                std::lock_guard<std::mutex> lk(mu_);
                release_requested_ = false;
            }
            cv_done_.notify_all();
            continue;
        }
        if (!job) {
            apply_offload_policy_locked_out();
            continue;
        }
        if (job->cancelled) {
            job->status = DMS_ERR_ABORTED;
        } else {
            run_job(*job);
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            job->done = true;
            running_ = nullptr;
        }
        cv_done_.notify_all();
    }
    // Shutdown: sessions must die on this thread before the runtime is unloaded.
    sessions_.clear();
    ram_models_.clear();
}

}  // namespace dms
