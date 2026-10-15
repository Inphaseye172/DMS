// C ABI entry points. Every exported function body runs inside dms::guard().
#include "engine_impl.h"

#include "image_ops.h"
#include "platform.h"
#include "sha256.h"

#include <cstdlib>
#include <cstring>
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

namespace {
std::unique_ptr<Logger> make_logger(const EngineConfig& c) {
    auto l = std::make_unique<Logger>(c.log_dir, c.log_level);
    set_global_logger(l.get());
    return l;
}
}  // namespace

Engine::Engine(EngineConfig cfg)
    : cfg_(std::move(cfg)), logger_(make_logger(cfg_)), manifest_(cfg_.models_dir), downloader_() {
    inference_ = std::make_unique<InferenceService>([this] { return make_runtime(); }, 8, cfg_.offload_idle_ms);
    logf(DMS_LOG_INFO, "DMS engine %s starting; models=%s deps=%s", DMS_ENGINE_VERSION_STRING,
         platform::path_to_utf8(cfg_.models_dir).c_str(), platform::path_to_utf8(cfg_.deps_dir).c_str());
}

Engine::~Engine() {
    log(DMS_LOG_INFO, "DMS engine shutting down");
    inference_.reset();  // stops the worker, destroys sessions, unloads ONNX Runtime
    set_global_logger(nullptr);
}

namespace {
// The CUDA build of ORT is only worth loading when every CUDA dependency is present next to it.
bool cuda_deps_present(const std::filesystem::path& deps, std::string& missing) {
#if defined(_WIN32)
    // NVIDIA DLLs live in the same folder as the CUDA build of ORT (see scripts/fetch-deps.ps1).
    const auto d = deps / "ort-cuda";
    const std::vector<std::filesystem::path> need = {d / "onnxruntime.dll", d / "onnxruntime_providers_cuda.dll", d / "cudart64_12.dll",
                                                     d / "cublas64_12.dll", d / "cublasLt64_12.dll", d / "cufft64_11.dll", d / "cudnn64_9.dll"};
#else
    const std::vector<std::filesystem::path> need = {deps / "ort-cuda" / "libonnxruntime_providers_cuda.so"};
#endif
    std::error_code ec;
    for (const auto& p : need) {
        if (!std::filesystem::is_regular_file(p, ec)) {
            missing = platform::path_to_utf8(p);
            return false;
        }
    }
    return true;
}
}  // namespace

std::unique_ptr<OrtRuntime> Engine::make_runtime() {
    std::string tried;
    const GpuSnapshot gs = gpu().snapshot();
    std::string missing;
    const bool nvidia_ok = gs.vendor == DMS_GPU_NVIDIA && gs.driver.at_least(kMinCudaDriverMajor, kMinCudaDriverMinor);
    const bool prefer_cuda = nvidia_ok && cuda_deps_present(cfg_.deps_dir, missing);
    if (gs.vendor == DMS_GPU_NVIDIA && !nvidia_ok)
        logf(DMS_LOG_WARN, "NVIDIA driver %s is older than %d.%02d required by CUDA 12 (cudaErrorInsufficientDriver); CUDA disabled",
             gs.driver.str().c_str(), kMinCudaDriverMajor, kMinCudaDriverMinor);
    else if (gs.vendor == DMS_GPU_NVIDIA && !prefer_cuda)
        logf(DMS_LOG_WARN, "CUDA dependencies incomplete (missing %s); using DirectML/CPU", missing.c_str());
    const bool prefer_dml = !prefer_cuda && gs.has_dxgi_adapter;
    for (const auto& p : OrtRuntime::candidate_libs(cfg_.deps_dir, prefer_cuda, prefer_dml)) {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(p, ec)) {
            tried += "\n  " + platform::path_to_utf8(p) + " (missing)";
            continue;
        }
        try {
            return std::make_unique<OrtRuntime>(p);
        } catch (const Error& e) {
            tried += "\n  " + platform::path_to_utf8(p) + " (" + e.what() + ")";
        }
    }
    fail(DMS_ERR_BACKEND_UNAVAILABLE, "ONNX Runtime could not be loaded; tried:" + tried);
}

void Engine::fill_model_info(const ModelSpec& m, dms_model_info& out) {
    const uint32_t size = out.struct_size;
    std::memset(&out, 0, sizeof(out) < size ? sizeof(out) : size);
    out.struct_size = size;
    copy_str(out.id, sizeof(out.id), m.id);
    copy_str(out.display_name, sizeof(out.display_name), m.display_name);
    out.kind = m.kind;
    out.window_size = m.window;
    out.patch_multiple = m.patch_multiple;
    out.expected_bytes = m.size;
    out.has_url = m.url.empty() ? 0 : 1;
    const auto p = manifest_.path_for(m);
    std::error_code ec;
    out.installed = std::filesystem::is_regular_file(p, ec) && (!m.size || std::filesystem::file_size(p, ec) == m.size) ? 1 : 0;
    {
        std::lock_guard<std::mutex> lk(verified_mu_);
        out.verified = verified_.count(m.id) ? 1 : 0;
    }
    copy_str(out.path, sizeof(out.path), platform::path_to_utf8(p));
}

void Engine::mark_verified(const std::string& id) {
    std::lock_guard<std::mutex> lk(verified_mu_);
    verified_.insert(id);
}

dms_status Engine::verify_model(const std::string& id) {
    const auto spec = manifest_.find(id);
    if (!spec) fail(DMS_ERR_NOT_FOUND, "unknown model " + id);
    const auto p = manifest_.path_for(*spec);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) fail(DMS_ERR_MODEL_NOT_INSTALLED, spec->display_name + " is not installed");
    if (spec->sha256.empty()) fail(DMS_ERR_UNSUPPORTED, "no checksum known for " + spec->display_name);
    const std::string got = sha256_file(p);
    if (got.empty()) fail(DMS_ERR_IO, "cannot read " + platform::path_to_utf8(p));
    if (got != spec->sha256) fail(DMS_ERR_CHECKSUM, spec->display_name + ": checksum mismatch (file is corrupt; delete it and download again)");
    mark_verified(id);
    return DMS_OK;
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

dms_status DMS_API dms_model_count(dms_engine* engine, int32_t* out_count) {
    return guard("dms_model_count", [&]() -> dms_status {
        auto e = lock_engine(engine);
        if (!out_count) fail(DMS_ERR_INVALID_ARG, "out_count is NULL");
        *out_count = static_cast<int32_t>(e->manifest().count());
        return DMS_OK;
    });
}

dms_status DMS_API dms_model_info_at(dms_engine* engine, int32_t index, dms_model_info* out_info) {
    return guard("dms_model_info_at", [&]() -> dms_status {
        auto e = lock_engine(engine);
        check_struct(out_info, "dms_model_info");
        if (index < 0) fail(DMS_ERR_INVALID_ARG, "negative index");
        const auto m = e->manifest().at(static_cast<size_t>(index));
        if (!m) fail(DMS_ERR_NOT_FOUND, "model index out of range");
        e->fill_model_info(*m, *out_info);
        return DMS_OK;
    });
}

dms_status DMS_API dms_model_info_by_id(dms_engine* engine, const char* model_id, dms_model_info* out_info) {
    return guard("dms_model_info_by_id", [&]() -> dms_status {
        auto e = lock_engine(engine);
        check_struct(out_info, "dms_model_info");
        if (!model_id) fail(DMS_ERR_INVALID_ARG, "model_id is NULL");
        const auto m = e->manifest().find(model_id);
        if (!m) fail(DMS_ERR_NOT_FOUND, std::string("unknown model ") + model_id);
        e->fill_model_info(*m, *out_info);
        return DMS_OK;
    });
}

dms_status DMS_API dms_model_verify(dms_engine* engine, const char* model_id) {
    return guard("dms_model_verify", [&]() -> dms_status {
        auto e = lock_engine(engine);
        if (!model_id) fail(DMS_ERR_INVALID_ARG, "model_id is NULL");
        return e->verify_model(model_id);
    });
}

dms_status DMS_API dms_model_reload_manifest(dms_engine* engine) {
    return guard("dms_model_reload_manifest", [&]() -> dms_status {
        auto e = lock_engine(engine);
        e->manifest().reload();
        return DMS_OK;
    });
}

dms_status DMS_API dms_download_start(dms_engine* engine, const char* model_id) {
    return guard("dms_download_start", [&]() -> dms_status {
        auto e = lock_engine(engine);
        if (!model_id) fail(DMS_ERR_INVALID_ARG, "model_id is NULL");
        const auto m = e->manifest().find(model_id);
        if (!m || m->custom_path) fail(DMS_ERR_NOT_FOUND, std::string("unknown model ") + model_id);
        logf(DMS_LOG_INFO, "download requested: %s from %s", m->id.c_str(), m->url.c_str());
        return e->downloader().start(*m, e->manifest().path_for(*m));
    });
}

dms_status DMS_API dms_download_query(dms_engine* engine, const char* model_id, dms_download_info* out_info) {
    return guard("dms_download_query", [&]() -> dms_status {
        auto e = lock_engine(engine);
        check_struct(out_info, "dms_download_info");
        if (!model_id) fail(DMS_ERR_INVALID_ARG, "model_id is NULL");
        const auto st = e->downloader().status(model_id);
        if (st.state == DMS_DL_DONE) e->mark_verified(model_id);  // SHA-256 checked by the downloader
        out_info->state = st.state;
        out_info->bytes_done = st.done;
        out_info->bytes_total = st.total;
        out_info->bytes_per_second = st.bytes_per_second;
        copy_str(out_info->message, sizeof(out_info->message), st.message);
        return DMS_OK;
    });
}

dms_status DMS_API dms_download_cancel(dms_engine* engine, const char* model_id) {
    return guard("dms_download_cancel", [&]() -> dms_status {
        auto e = lock_engine(engine);
        if (!model_id) fail(DMS_ERR_INVALID_ARG, "model_id is NULL");
        e->downloader().cancel(model_id);
        return DMS_OK;
    });
}

dms_status DMS_API dms_device_query(dms_engine* engine, dms_device_info* out_info) {
    return guard("dms_device_query", [&]() -> dms_status {
        auto e = lock_engine(engine);
        check_struct(out_info, "dms_device_info");
        const uint32_t size = out_info->struct_size;
        std::memset(out_info, 0, sizeof(*out_info));
        out_info->struct_size = size;
        const auto mem = platform::system_memory();
        out_info->ram_total_bytes = mem.total;
        out_info->ram_free_bytes = mem.available;
        const GpuSnapshot gs = e->gpu().snapshot();
        out_info->gpu_vendor = gs.vendor;
        copy_str(out_info->gpu_name, sizeof(out_info->gpu_name), gs.name);
        copy_str(out_info->driver_version, sizeof(out_info->driver_version), gs.driver.str());
        out_info->vram_total_bytes = gs.vram_total;
        out_info->vram_free_bytes = gs.vram_free;
        out_info->active_backend = e->inference().last_backend();
        std::string notes = gs.notes;
        // Loading the runtime here is cheap after the first time and never touches the network.
        try {
            e->inference().ensure_runtime();
        } catch (const Error& ex) {
            notes += std::string("ONNX Runtime: ") + ex.what() + " ";
        }
        copy_str(out_info->ort_version, sizeof(out_info->ort_version), e->inference().runtime_version());
        const bool driver_ok = gs.driver.at_least(kMinCudaDriverMajor, kMinCudaDriverMinor);
        out_info->cuda_available = gs.vendor == DMS_GPU_NVIDIA && driver_ok && e->inference().runtime_has_provider("CUDAExecutionProvider");
        out_info->directml_available = gs.has_dxgi_adapter && e->inference().runtime_has_provider("DmlExecutionProvider");
#if defined(DMS_ENABLE_TENSORRT)
        out_info->tensorrt_available = out_info->cuda_available && e->inference().runtime_has_provider("TensorrtExecutionProvider");
#endif
        if (gs.vendor == DMS_GPU_NVIDIA && !driver_ok) notes += "NVIDIA driver too old for CUDA 12. ";
        if (gs.vendor == DMS_GPU_NVIDIA && !gs.driver.at_least(kMinMemShareDriverMajor, kMinMemShareDriverMinor))
            notes += "CUDA Memory Sharing needs driver >= 560. ";
        copy_str(out_info->notes, sizeof(out_info->notes), notes);
        return DMS_OK;
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
