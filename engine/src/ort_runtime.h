// ONNX Runtime, loaded dynamically (LoadLibraryExW / dlopen) from an explicit directory, so a missing or
// mismatched onnxruntime library becomes an error code instead of a host crash at plugin load time.
// The engine never links against onnxruntime.
#pragma once

#include "dms/engine.h"
#include "platform.h"

// The engine never links onnxruntime: the C++ wrapper must not auto-initialize from OrtGetApiBase().
// Defined here (not only on the command line) so every includer agrees (ODR).
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dms {

enum class Backend : int { CPU = DMS_BACKEND_CPU, CUDA = DMS_BACKEND_CUDA, DirectML = DMS_BACKEND_DIRECTML, TensorRT = DMS_BACKEND_TENSORRT };
const char* backend_name(Backend b) noexcept;

struct SessionConfig {
    std::filesystem::path model_path;
    Backend backend = Backend::CPU;
    bool fp16 = false;            // using an FP16 model variant (or TensorRT FP16 mode)
    bool memshare = false;        // CUDA: shared arena + pinned host I/O
    int device_id = 0;
    uint64_t gpu_mem_limit = 0;   // CUDA arena cap (0 = unlimited)
    std::filesystem::path trt_cache_dir;
    std::shared_ptr<const std::vector<uint8_t>> model_bytes;  // optional in-RAM model (offload-to-RAM mode)

    bool operator==(const SessionConfig& o) const {
        return model_path == o.model_path && backend == o.backend && fp16 == o.fp16 && memshare == o.memshare && device_id == o.device_id;
    }
};

class OrtRuntime;

// One ORT session plus its reusable I/O buffers. Used only from the inference worker thread.
class ModelSession {
public:
    ModelSession(OrtRuntime& rt, const SessionConfig& cfg);
    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

    // Runs the model on an NCHW (or NTCHW) float tensor. Output buffer is owned by the session and reused.
    // Returns a view valid until the next run(). Throws dms::Error (OOM is reported as DMS_ERR_OUT_OF_MEMORY).
    const std::vector<float>& run(const float* input, const std::vector<int64_t>& shape, std::vector<int64_t>& out_shape, Ort::RunOptions& ro);

    const SessionConfig& config() const noexcept { return cfg_; }
    size_t input_rank() const noexcept { return in_rank_; }
    size_t output_rank() const noexcept { return out_rank_; }

private:
    OrtRuntime& rt_;
    SessionConfig cfg_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    ONNXTensorElementDataType in_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType out_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    size_t in_rank_ = 4, out_rank_ = 3;
    bool ort_allocates_output_ = false;  // set if binding a preallocated output failed once
    Ort::MemoryInfo cpu_info_;
    std::vector<float> out_buf_;               // reused across runs (grows only)
    std::vector<Ort::Float16_t> in_half_;      // reused FP16 staging
    std::vector<Ort::Float16_t> out_half_;
};

class OrtRuntime {
public:
    // Loads the onnxruntime shared library at `lib_path`. Throws dms::Error(DMS_ERR_BACKEND_UNAVAILABLE).
    explicit OrtRuntime(const std::filesystem::path& lib_path);
    ~OrtRuntime();
    OrtRuntime(const OrtRuntime&) = delete;
    OrtRuntime& operator=(const OrtRuntime&) = delete;

    Ort::Env& env() { return *env_; }
    const std::string& version() const noexcept { return version_; }
    const std::filesystem::path& lib_path() const noexcept { return lib_path_; }
    bool has_provider(const std::string& name) const;
    const std::vector<std::string>& providers() const noexcept { return providers_; }
    platform::DynLib& lib() noexcept { return lib_; }
    int cpu_threads() const noexcept { return cpu_threads_; }
    // Registers one CUDA arena in the env shared by all sessions ("CUDA Memory Sharing"). Thread-safe.
    bool cuda_shared_arena(int device_id);

    // Candidate library locations for a deps directory, in preference order.
    static std::vector<std::filesystem::path> candidate_libs(const std::filesystem::path& deps_dir, bool prefer_cuda, bool prefer_dml);

private:
    static void preload_dependencies(const std::filesystem::path& dir);
    platform::DynLib lib_;
    std::filesystem::path lib_path_;
    std::unique_ptr<Ort::Env> env_;
    std::string version_;
    std::vector<std::string> providers_;
    int cpu_threads_ = 2;
    std::mutex arena_mu_;
    bool arena_registered_ = false;
    bool arena_failed_ = false;
};

// Appends the execution provider for cfg.backend to the session options. Throws dms::Error.
void append_execution_provider(OrtRuntime& rt, Ort::SessionOptions& so, const SessionConfig& cfg);

// Classifies an ORT error message into a dms_status (OOM detection drives the fallback chain).
dms_status classify_ort_error(const std::string& what) noexcept;

}  // namespace dms
