#include "ort_runtime.h"

#include "common.h"
#include "logging.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace dms {

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::CPU: return "CPU";
        case Backend::CUDA: return "CUDA";
        case Backend::DirectML: return "DirectML";
        case Backend::TensorRT: return "TensorRT";
    }
    return "?";
}

dms_status classify_ort_error(const std::string& what) noexcept {
    try {
        std::string s = what;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s.find("out of memory") != std::string::npos || s.find("failed to allocate") != std::string::npos ||
            s.find("bad_alloc") != std::string::npos || s.find("cudaerrormemoryallocation") != std::string::npos ||
            s.find("e_outofmemory") != std::string::npos || s.find("887a0005") != std::string::npos /* DXGI device removed */ ||
            s.find("available memory") != std::string::npos)
            return DMS_ERR_OUT_OF_MEMORY;
        if (s.find("exiting due to terminate flag") != std::string::npos || s.find("terminate") != std::string::npos) return DMS_ERR_ABORTED;
        if (s.find("insufficient driver") != std::string::npos || s.find("cudaerrorinsufficientdriver") != std::string::npos ||
            s.find("no cuda-capable device") != std::string::npos || s.find("loadlibrary failed") != std::string::npos ||
            s.find("error loading") != std::string::npos)
            return DMS_ERR_BACKEND_UNAVAILABLE;
    } catch (...) {  // NOLINT(bugprone-empty-catch): classification is best effort; fall through to the generic status
    }
    return DMS_ERR_INFERENCE_FAILED;
}

namespace {
void ORT_API_CALL ort_log(void*, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message) {
    const int level = severity >= ORT_LOGGING_LEVEL_ERROR ? DMS_LOG_ERROR : severity == ORT_LOGGING_LEVEL_WARNING ? DMS_LOG_WARN : DMS_LOG_DEBUG;
    logf(level, "[ORT %s %s] %s (%s)", category ? category : "", logid ? logid : "", message ? message : "", code_location ? code_location : "");
}

int default_cpu_threads() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read once at runtime creation; the engine never calls setenv
    if (const char* e = std::getenv("DMS_CPU_THREADS"); e && *e) return clampi(std::atoi(e), 1, 256);
    const unsigned hc = std::thread::hardware_concurrency();
    // Leave headroom for the host application's own render threads.
    return clampi(static_cast<int>(hc > 2 ? hc / 2 : 1), 1, 16);
}
}  // namespace

std::vector<std::filesystem::path> OrtRuntime::candidate_libs(const std::filesystem::path& deps, bool prefer_cuda, bool prefer_dml) {
#if defined(_WIN32)
    const char* name = "onnxruntime.dll";
#elif defined(__APPLE__)
    const char* name = "libonnxruntime.dylib";
#else
    const char* name = "libonnxruntime.so";
#endif
    std::vector<std::filesystem::path> out;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read once at runtime creation; the engine never calls setenv
    if (const char* e = std::getenv("DMS_ORT_LIB"); e && *e) out.push_back(platform::path_from_utf8(e));
    std::vector<const char*> subdirs;
    if (prefer_cuda) subdirs.push_back("ort-cuda");
    if (prefer_dml) subdirs.push_back("ort-dml");
    for (const char* s : {"ort-cuda", "ort-dml", "ort-cpu"})
        if (std::find(subdirs.begin(), subdirs.end(), std::string_view(s)) == subdirs.end()) subdirs.push_back(s);
    for (const char* s : subdirs) out.push_back(deps / s / name);
    out.push_back(deps / name);
    return out;
}

void OrtRuntime::preload_dependencies(const std::filesystem::path& dir) {
#if defined(_WIN32)
    static const char* kPrefixes[] = {"cudart", "cublaslt", "cublas", "cufft", "cudnn", "directml", "onnxruntime_providers_shared"};
    std::error_code ec;
    std::vector<std::filesystem::path> found;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        std::string name = platform::path_to_utf8(e.path().filename());
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (name.size() < 5 || name.substr(name.size() - 4) != ".dll" || name.find(".debug.") != std::string::npos) continue;
        for (const char* p : kPrefixes)
            if (name.rfind(p, 0) == 0) found.push_back(e.path());
    }
    // cudart / cuBLAS first, then cuDNN core before its sub-libraries.
    std::sort(found.begin(), found.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
        auto rank = [](const std::filesystem::path& p) {
            std::string n = platform::path_to_utf8(p.filename());
            std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (n.rfind("cudart", 0) == 0) return 0;
            if (n.rfind("cublaslt", 0) == 0) return 1;
            if (n.rfind("cublas", 0) == 0) return 2;
            if (n.rfind("cufft", 0) == 0) return 3;
            if (n.rfind("cudnn64", 0) == 0) return 4;
            if (n.rfind("cudnn", 0) == 0) return 5;
            return 6;
        };
        return rank(a) < rank(b);
    });
    for (const auto& p : found) {
        platform::DynLib l(p);
        if (l.ok()) {
            l.pin();
            logf(DMS_LOG_DEBUG, "preloaded %s", platform::path_to_utf8(p.filename()).c_str());
        } else {
            logf(DMS_LOG_WARN, "could not preload %s: %s", platform::path_to_utf8(p).c_str(), l.error().c_str());
        }
    }
#else
    (void)dir;
#endif
}

OrtRuntime::OrtRuntime(const std::filesystem::path& lib_path) : lib_path_(lib_path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(lib_path, ec)) fail(DMS_ERR_BACKEND_UNAVAILABLE, "ONNX Runtime not found at " + platform::path_to_utf8(lib_path));
#if defined(_WIN32)
    // A host may already have its own onnxruntime.dll loaded (e.g. for built-in AI features). We load
    // ours by full path, so it is a separate module, but log it: provider DLLs resolve by name.
    if (HMODULE existing = GetModuleHandleW(L"onnxruntime.dll")) {
        wchar_t buf[MAX_PATH * 2] = {0};
        GetModuleFileNameW(existing, buf, MAX_PATH * 2);
        logf(DMS_LOG_WARN, "another onnxruntime.dll is already loaded in this process: %s",
             platform::path_to_utf8(std::filesystem::path(buf)).c_str());
    }
#endif
    // Delay-loaded / runtime-loaded dependencies (DirectML.dll, cuDNN sub-libraries, CUDA runtime) are
    // resolved by module name with the process search order, which would pick up System32's older
    // DirectML.dll or miss cuDNN. Pre-load every dependency shipped next to onnxruntime by full path;
    // later by-name lookups then bind to these already-loaded modules.
    preload_dependencies(lib_path.parent_path());
    lib_ = platform::DynLib(lib_path);
    if (!lib_.ok()) fail(DMS_ERR_BACKEND_UNAVAILABLE, "cannot load ONNX Runtime: " + lib_.error());
    // Never unload ONNX Runtime: its static state, thread pools and provider libraries are not safe to
    // tear down mid-process (and each unload leaks its static allocations). Loaded once per process.
    lib_.pin();
    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
    auto get_base = lib_.fn<GetApiBaseFn>("OrtGetApiBase");
    if (!get_base) fail(DMS_ERR_BACKEND_UNAVAILABLE, "OrtGetApiBase not exported by " + platform::path_to_utf8(lib_path));
    const OrtApiBase* base = get_base();
    if (!base) fail(DMS_ERR_BACKEND_UNAVAILABLE, "OrtGetApiBase returned NULL");
    version_ = base->GetVersionString ? base->GetVersionString() : "?";
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) fail(DMS_ERR_BACKEND_UNAVAILABLE, "ONNX Runtime " + version_ + " does not support API version " + std::to_string(ORT_API_VERSION));
    Ort::InitApi(api);
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dms", ort_log, nullptr);
        providers_ = Ort::GetAvailableProviders();
    } catch (const Ort::Exception& e) {
        env_.reset();
        Ort::InitApi(nullptr);
        fail(DMS_ERR_BACKEND_UNAVAILABLE, std::string("ONNX Runtime init failed: ") + e.what());
    }
    cpu_threads_ = default_cpu_threads();
    std::string provs;
    for (const auto& p : providers_) provs += p + " ";
    logf(DMS_LOG_INFO, "ONNX Runtime %s loaded from %s; providers: %s", version_.c_str(), platform::path_to_utf8(lib_path).c_str(), provs.c_str());
}

OrtRuntime::~OrtRuntime() {
    env_.reset();
    Ort::InitApi(nullptr);
    // lib_ is pinned: the library stays loaded for the life of the host process.
}

bool OrtRuntime::has_provider(const std::string& name) const {
    return std::find(providers_.begin(), providers_.end(), name) != providers_.end();
}

namespace {

struct ProviderOptionList {
    std::vector<std::string> keys, values;
    void add(const char* k, const std::string& v) {
        keys.emplace_back(k);
        values.push_back(v);
    }
    std::vector<const char*> k() const {
        std::vector<const char*> o;
        o.reserve(keys.size());
        for (const auto& x : keys) o.push_back(x.c_str());
        return o;
    }
    std::vector<const char*> v() const {
        std::vector<const char*> o;
        o.reserve(values.size());
        for (const auto& x : values) o.push_back(x.c_str());
        return o;
    }
};

void append_cuda(OrtRuntime& rt, Ort::SessionOptions& so, const SessionConfig& cfg) {
    if (!rt.has_provider("CUDAExecutionProvider")) fail(DMS_ERR_BACKEND_UNAVAILABLE, "this ONNX Runtime build has no CUDA execution provider");
    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* raw = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
    std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)> opts(raw, api.ReleaseCUDAProviderOptions);
    ProviderOptionList o;
    o.add("device_id", std::to_string(cfg.device_id));
    o.add("arena_extend_strategy", "kSameAsRequested");  // grow only as needed; keeps headroom for the host
    o.add("cudnn_conv_algo_search", "HEURISTIC");         // no multi-second exhaustive search on first frame
    o.add("do_copy_in_default_stream", "1");
    if (cfg.gpu_mem_limit) o.add("gpu_mem_limit", std::to_string(cfg.gpu_mem_limit));
    // "Lower precision" without an FP16 model variant: allow TF32 tensor-core math (Ampere+).
    o.add("use_tf32", cfg.fp16 ? "1" : "0");
    const auto k = o.k();
    const auto v = o.v();
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(opts.get(), k.data(), v.data(), k.size()));
    so.AppendExecutionProvider_CUDA_V2(*opts);
    if (cfg.memshare && rt.cuda_shared_arena(cfg.device_id)) so.AddConfigEntry("session.use_env_allocators", "1");
}

void append_dml(OrtRuntime& rt, Ort::SessionOptions& so, const SessionConfig& cfg) {
    using AppendDml = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
    auto fn = rt.lib().fn<AppendDml>("OrtSessionOptionsAppendExecutionProvider_DML");
    if (!fn || !rt.has_provider("DmlExecutionProvider")) fail(DMS_ERR_BACKEND_UNAVAILABLE, "this ONNX Runtime build has no DirectML execution provider");
    // Required by the DirectML EP.
    so.DisableMemPattern();
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    Ort::ThrowOnError(fn(so, cfg.device_id));
}

void append_tensorrt(OrtRuntime& rt, Ort::SessionOptions& so, const SessionConfig& cfg) {
#if defined(DMS_ENABLE_TENSORRT)
    if (!rt.has_provider("TensorrtExecutionProvider")) fail(DMS_ERR_BACKEND_UNAVAILABLE, "this ONNX Runtime build has no TensorRT execution provider");
    const OrtApi& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* raw = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&raw));
    std::unique_ptr<OrtTensorRTProviderOptionsV2, void (*)(OrtTensorRTProviderOptionsV2*)> opts(raw, api.ReleaseTensorRTProviderOptions);
    ProviderOptionList o;
    o.add("device_id", std::to_string(cfg.device_id));
    o.add("trt_fp16_enable", cfg.fp16 ? "1" : "0");
    o.add("trt_engine_cache_enable", "1");
    o.add("trt_timing_cache_enable", "1");
    if (!cfg.trt_cache_dir.empty()) {
        o.add("trt_engine_cache_path", platform::path_to_utf8(cfg.trt_cache_dir));
        o.add("trt_timing_cache_path", platform::path_to_utf8(cfg.trt_cache_dir));
    }
    o.add("trt_max_workspace_size", std::to_string(1ull << 30));
    const auto k = o.k();
    const auto v = o.v();
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(opts.get(), k.data(), v.data(), k.size()));
    so.AppendExecutionProvider_TensorRT_V2(*opts);
    // Nodes TensorRT cannot take fall back to CUDA, then CPU.
    SessionConfig cuda_cfg = cfg;
    cuda_cfg.backend = Backend::CUDA;
    append_cuda(rt, so, cuda_cfg);
#else
    (void)rt;
    (void)so;
    (void)cfg;
    fail(DMS_ERR_UNSUPPORTED, "TensorRT support was not compiled in (configure with -DDMS_ENABLE_TENSORRT=ON)");
#endif
}

}  // namespace

void append_execution_provider(OrtRuntime& rt, Ort::SessionOptions& so, const SessionConfig& cfg) {
    switch (cfg.backend) {
        case Backend::CPU: return;
        case Backend::CUDA: append_cuda(rt, so, cfg); return;
        case Backend::DirectML: append_dml(rt, so, cfg); return;
        case Backend::TensorRT: append_tensorrt(rt, so, cfg); return;
    }
}

bool OrtRuntime::cuda_shared_arena(int device_id) {
    std::lock_guard<std::mutex> lk(arena_mu_);
    if (arena_registered_) return true;
    if (arena_failed_) return false;
    try {
        Ort::MemoryInfo mi("Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
        std::unordered_map<std::string, std::string> opts{{"arena_extend_strategy", "1"}};
        env_->CreateAndRegisterAllocatorV2("CUDAExecutionProvider", mi, opts, nullptr);
        arena_registered_ = true;
        log(DMS_LOG_INFO, "CUDA memory sharing: shared env arena registered");
    } catch (const Ort::Exception& e) {
        arena_failed_ = true;
        logf(DMS_LOG_WARN, "CUDA memory sharing unavailable (%s); continuing without it", e.what());
    }
    return arena_registered_;
}

ModelSession::ModelSession(OrtRuntime& rt, const SessionConfig& cfg)
    : rt_(rt), cfg_(cfg), cpu_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    try {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(rt.cpu_threads());
        so.SetInterOpNumThreads(1);
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
        append_execution_provider(rt, so, cfg);
        if (cfg.model_bytes && !cfg.model_bytes->empty())
            session_ = std::make_unique<Ort::Session>(rt.env(), cfg.model_bytes->data(), cfg.model_bytes->size(), so);
        else
            session_ = std::make_unique<Ort::Session>(rt.env(), cfg.model_path.c_str(), so);

        if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 1) fail(DMS_ERR_MODEL_LOAD_FAILED, "model has no inputs/outputs");
        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_->GetInputNameAllocated(0, alloc).get();
        out_name_ = session_->GetOutputNameAllocated(0, alloc).get();
        // Keep the TypeInfo owners alive: the TensorTypeAndShapeInfo views point into them.
        const Ort::TypeInfo in_type_info = session_->GetInputTypeInfo(0);
        const Ort::TypeInfo out_type_info = session_->GetOutputTypeInfo(0);
        if (in_type_info.GetONNXType() != ONNX_TYPE_TENSOR || out_type_info.GetONNXType() != ONNX_TYPE_TENSOR)
            fail(DMS_ERR_MODEL_LOAD_FAILED, "model input/output must be tensors");
        const auto in_info = in_type_info.GetTensorTypeAndShapeInfo();
        const auto out_info = out_type_info.GetTensorTypeAndShapeInfo();
        in_type_ = in_info.GetElementType();
        out_type_ = out_info.GetElementType();
        in_rank_ = in_info.GetDimensionsCount();
        out_rank_ = out_info.GetDimensionsCount();
        auto supported = [](ONNXTensorElementDataType t) {
            return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        };
        if (!supported(in_type_) || !supported(out_type_)) fail(DMS_ERR_MODEL_LOAD_FAILED, "model I/O must be float32 or float16");
        if (in_rank_ != 4 && in_rank_ != 5) fail(DMS_ERR_MODEL_LOAD_FAILED, "model input must be rank 4 (NCHW) or 5 (NTCHW)");
        if (out_rank_ < 3 || out_rank_ > 5) fail(DMS_ERR_MODEL_LOAD_FAILED, "unexpected model output rank " + std::to_string(out_rank_));
        logf(DMS_LOG_INFO, "session created: %s on %s (in %s rank %zu, out %s rank %zu)", platform::path_to_utf8(cfg.model_path.filename()).c_str(),
             backend_name(cfg.backend), in_name_.c_str(), in_rank_, out_name_.c_str(), out_rank_);
    } catch (const Ort::Exception& e) {
        const dms_status st = classify_ort_error(e.what());
        fail(st == DMS_ERR_INFERENCE_FAILED ? DMS_ERR_MODEL_LOAD_FAILED : st, std::string("session creation failed: ") + e.what());
    }
}

const std::vector<float>& ModelSession::run(const float* input, const std::vector<int64_t>& shape, std::vector<int64_t>& out_shape,
                                            Ort::RunOptions& ro) {
    if (shape.size() != in_rank_) fail(DMS_ERR_INVALID_ARG, "input rank mismatch");
    size_t count = 1;
    for (int64_t d : shape) {
        if (d <= 0) fail(DMS_ERR_INVALID_ARG, "invalid input shape");
        count *= static_cast<size_t>(d);
    }
    try {
        // Input: wrap caller memory (no copy) or stage into the reused FP16 buffer.
        Ort::Value in_val{nullptr};
        if (in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            if (in_half_.size() < count) in_half_.resize(count);
            for (size_t i = 0; i < count; ++i) in_half_[i] = Ort::Float16_t(input[i]);
            in_val = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_info_, in_half_.data(), count, shape.data(), shape.size());
        } else {
            in_val = Ort::Value::CreateTensor<float>(cpu_info_, const_cast<float*>(input), count, shape.data(), shape.size());
        }
        const char* in_names[] = {in_name_.c_str()};
        const char* out_names[] = {out_name_.c_str()};

        // Expected output: spatial dims equal to the input's (DA-style dense prediction).
        std::vector<int64_t> expect;
        const int64_t H = shape[shape.size() - 2], W = shape[shape.size() - 1];
        if (in_rank_ == 5) {  // [B,T,3,H,W] -> [B,T,H,W]
            expect = {shape[0], shape[1], H, W};
        } else {
            expect = {shape[0], H, W};
        }
        while (expect.size() < out_rank_) expect.insert(expect.begin() + 1, 1);
        size_t expect_count = 1;
        for (int64_t d : expect) expect_count *= static_cast<size_t>(d);

        if (!ort_allocates_output_ && out_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && expect.size() == out_rank_) {
            if (out_buf_.size() < expect_count) out_buf_.resize(expect_count);
            Ort::Value out_val = Ort::Value::CreateTensor<float>(cpu_info_, out_buf_.data(), expect_count, expect.data(), expect.size());
            try {
                session_->Run(ro, in_names, &in_val, 1, out_names, &out_val, 1);
                out_shape = expect;
                return out_buf_;
            } catch (const Ort::Exception& e) {
                const dms_status st = classify_ort_error(e.what());
                if (st == DMS_ERR_OUT_OF_MEMORY || st == DMS_ERR_ABORTED) throw;
                // Shape differs from our expectation: let ORT allocate from now on.
                ort_allocates_output_ = true;
                logf(DMS_LOG_DEBUG, "preallocated output rejected (%s); using ORT-allocated outputs", e.what());
            }
        }
        auto outs = session_->Run(ro, in_names, &in_val, 1, out_names, 1);
        if (outs.empty() || !outs[0].IsTensor()) fail(DMS_ERR_INFERENCE_FAILED, "model produced no tensor output");
        const auto info = outs[0].GetTensorTypeAndShapeInfo();
        out_shape = info.GetShape();
        const size_t n = info.GetElementCount();
        if (out_buf_.size() < n) out_buf_.resize(n);
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* h = outs[0].GetTensorData<Ort::Float16_t>();
            for (size_t i = 0; i < n; ++i) out_buf_[i] = h[i].ToFloat();
        } else {
            const float* f = outs[0].GetTensorData<float>();
            std::copy(f, f + n, out_buf_.begin());
        }
        return out_buf_;
    } catch (const Ort::Exception& e) {
        fail(classify_ort_error(e.what()), std::string("inference failed: ") + e.what());
    }
}

}  // namespace dms
