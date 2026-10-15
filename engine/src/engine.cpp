// C ABI entry points. Every exported function body runs inside dms::guard().
#include "engine_impl.h"

#include "image_ops.h"
#include "platform.h"
#include "sha256.h"

#include <cstdlib>
#include <mutex>

#ifndef DMS_ENGINE_VERSION_STRING
#define DMS_ENGINE_VERSION_STRING "1.0.0"
#endif

namespace dms {

namespace {
// Process-wide singleton (rule: model/session/CUDA state is a refcounted singleton behind a mutex).
std::mutex g_engine_mu;
std::shared_ptr<Engine> g_engine;  // guarded by g_engine_mu
int g_engine_refs = 0;             // guarded by g_engine_mu
}  // namespace

EngineConfig resolve_config(const dms_engine_config* c) {
    EngineConfig out;
    const auto data = platform::default_data_dir();
    out.models_dir = data / "models";
    out.log_dir = data / "logs";
    out.deps_dir = platform::module_dir() / "DMS_deps";
    if (const char* env = std::getenv("DMS_MODELS_DIR"); env && *env) out.models_dir = platform::path_from_utf8(env);
    if (const char* env = std::getenv("DMS_LOG_DIR"); env && *env) out.log_dir = platform::path_from_utf8(env);
    if (const char* env = std::getenv("DMS_DEPS_DIR"); env && *env) out.deps_dir = platform::path_from_utf8(env);
    if (!c) return out;
    if (c->struct_size < sizeof(uint32_t)) fail(DMS_ERR_INVALID_ARG, "dms_engine_config.struct_size not set");
    if (c->models_dir && *c->models_dir) out.models_dir = platform::path_from_utf8(c->models_dir);
    if (c->log_dir && *c->log_dir) out.log_dir = platform::path_from_utf8(c->log_dir);
    if (c->deps_dir && *c->deps_dir) out.deps_dir = platform::path_from_utf8(c->deps_dir);
    out.log_level = clampi(c->log_level, DMS_LOG_ERROR, DMS_LOG_DEBUG);
    if (c->vram_reserve_bytes) out.vram_reserve_bytes = c->vram_reserve_bytes;
    if (c->cache_budget_bytes) out.cache_budget_bytes = c->cache_budget_bytes;
    if (c->inference_timeout_ms) out.inference_timeout_ms = c->inference_timeout_ms;
    if (c->max_neighbor_frames) out.max_neighbor_frames = c->max_neighbor_frames > 64 ? 64 : c->max_neighbor_frames;
    if (c->offload_idle_ms) out.offload_idle_ms = c->offload_idle_ms;
    // Cache budget never exceeds 1/4 of physical RAM.
    const auto mem = platform::system_memory();
    if (mem.total > 0 && out.cache_budget_bytes > mem.total / 4) out.cache_budget_bytes = mem.total / 4;
    return out;
}

Engine::Engine(EngineConfig cfg) : cfg_(std::move(cfg)) {
    logger_ = std::make_unique<Logger>(cfg_.log_dir, cfg_.log_level);
    set_global_logger(logger_.get());
    logf(DMS_LOG_INFO, "DMS engine %s starting; models=%s deps=%s", DMS_ENGINE_VERSION_STRING,
         platform::path_to_utf8(cfg_.models_dir).c_str(), platform::path_to_utf8(cfg_.deps_dir).c_str());
}

Engine::~Engine() {
    log(DMS_LOG_INFO, "DMS engine shutting down");
    set_global_logger(nullptr);
}

std::shared_ptr<Engine> lock_engine(dms_engine* h) {
    if (!h) fail(DMS_ERR_INVALID_ARG, "engine handle is NULL");
    std::lock_guard<std::mutex> lk(g_engine_mu);
    if (!g_engine || static_cast<dms_engine*>(g_engine.get()) != h) fail(DMS_ERR_NOT_INITIALIZED, "engine handle is not live");
    return g_engine;
}

}  // namespace dms

using namespace dms;

extern "C" {

dms_status DMS_API dms_engine_acquire(const dms_engine_config* config, dms_engine** out_engine) {
    return guard("dms_engine_acquire", [&]() -> dms_status {
        if (!out_engine) fail(DMS_ERR_INVALID_ARG, "out_engine is NULL");
        *out_engine = nullptr;
        std::lock_guard<std::mutex> lk(g_engine_mu);
        if (!g_engine) {
            g_engine = std::make_shared<Engine>(resolve_config(config));
            g_engine_refs = 0;
        }
        ++g_engine_refs;
        *out_engine = g_engine.get();
        logf(DMS_LOG_DEBUG, "engine acquired (refs=%d)", g_engine_refs);
        return DMS_OK;
    });
}

dms_status DMS_API dms_engine_release(dms_engine* engine) {
    return guard("dms_engine_release", [&]() -> dms_status {
        std::shared_ptr<Engine> doomed;
        {
            std::lock_guard<std::mutex> lk(g_engine_mu);
            if (!g_engine || static_cast<dms_engine*>(g_engine.get()) != engine) return DMS_ERR_NOT_INITIALIZED;
            if (--g_engine_refs > 0) return DMS_OK;
            doomed = std::move(g_engine);  // destroyed outside the lock once in-flight calls drop their refs
            g_engine.reset();
            g_engine_refs = 0;
        }
        doomed.reset();
        return DMS_OK;
    });
}

dms_status DMS_API dms_engine_set_log_level(dms_engine* engine, int32_t level) {
    return guard("dms_engine_set_log_level", [&]() -> dms_status {
        auto e = lock_engine(engine);
        e->logger().set_level(clampi(level, DMS_LOG_ERROR, DMS_LOG_DEBUG));
        return DMS_OK;
    });
}

const char* DMS_API dms_status_message(dms_status s) {
    switch (s) {
        case DMS_OK: return "OK";
        case DMS_ERR_INVALID_ARG: return "Invalid argument";
        case DMS_ERR_NOT_INITIALIZED: return "Engine not initialized";
        case DMS_ERR_MODEL_NOT_INSTALLED: return "Model not installed - open the effect Options to download it";
        case DMS_ERR_MODEL_LOAD_FAILED: return "Model failed to load";
        case DMS_ERR_BACKEND_UNAVAILABLE: return "No inference backend available";
        case DMS_ERR_OUT_OF_MEMORY: return "Out of memory";
        case DMS_ERR_INFERENCE_FAILED: return "Inference failed";
        case DMS_ERR_TIMEOUT: return "Timed out waiting for inference";
        case DMS_ERR_ABORTED: return "Aborted";
        case DMS_ERR_IO: return "File I/O error";
        case DMS_ERR_NETWORK: return "Network error";
        case DMS_ERR_CHECKSUM: return "Checksum mismatch";
        case DMS_ERR_INTERNAL: return "Internal error";
        case DMS_ERR_BUSY: return "Busy";
        case DMS_ERR_UNSUPPORTED: return "Unsupported";
        case DMS_ERR_NOT_FOUND: return "Not found";
        case DMS_ERR_DEGRADED: return "Result produced by a fallback";
    }
    return "Unknown status";
}

const char* DMS_API dms_last_error(void) { return dms::last_error(); }

const char* DMS_API dms_engine_version(void) { return DMS_ENGINE_VERSION_STRING; }

void DMS_API dms_log(dms_engine* engine, int32_t level, const char* message) {
    (void)guard("dms_log", [&]() -> dms_status {
        auto e = lock_engine(engine);
        e->logger().write(level, message ? message : "");
        return DMS_OK;
    });
}

dms_status DMS_API dms_depth_compute(dms_engine* engine, const dms_depth_request* request, dms_depth_result* result) {
    return guard("dms_depth_compute", [&]() -> dms_status {
        auto e = lock_engine(engine);
        check_struct(request, "dms_depth_request");
        check_struct(result, "dms_depth_result");
        return e->compute(*request, *result);
    });
}

dms_status DMS_API dms_resize_f32(const float* src, int32_t sw, int32_t sh, float* dst, int32_t dw, int32_t dh) {
    return guard("dms_resize_f32", [&]() -> dms_status {
        if (!src || !dst) fail(DMS_ERR_INVALID_ARG, "NULL buffer");
        check_dims(sw, sh, "src");
        check_dims(dw, dh, "dst");
        resize_bilinear(src, sw, sh, dst, dw, dh);
        return DMS_OK;
    });
}

dms_status DMS_API dms_sha256_hex(const void* data, size_t size, char out_hex[65]) {
    return guard("dms_sha256_hex", [&]() -> dms_status {
        if (!out_hex || (!data && size)) fail(DMS_ERR_INVALID_ARG, "NULL buffer");
        Sha256 h;
        if (size) h.update(data, size);
        copy_str(out_hex, 65, h.finish_hex());
        return DMS_OK;
    });
}

}  // extern "C"
