// Backend selection, VRAM estimation and the fallback chain plan.
//   requested -> (FP16 failed) FP32 -> lower base resolution -> more tiles (smaller per-tile res) -> CPU
// The final rungs (last cached frame / flat gray) are applied by the caller (plugin render core),
// which owns the per-instance "last good" depth.
#pragma once

#include "dms/engine.h"
#include "gpu_probe.h"
#include "manifest.h"
#include "ort_runtime.h"

#include <string>
#include <vector>

namespace dms {

struct Attempt {
    Backend backend = Backend::CPU;
    bool fp16 = false;
    int res = 518;        // per-pass inference resolution (long side)
    int tiles = 1;        // 1, 4, 9, 16
    bool parallel = false;
    std::string label;    // human readable, for logs/messages
};

struct PlanInputs {
    bool hardware = true;
    bool fp16 = false;
    bool use_tensorrt = false;
    bool tensorrt_compiled = false;
    int base_res = 518;
    int tiles = 1;        // resolved (auto already applied)
    bool parallel = false;
    int patch = 14;
    GpuSnapshot gpu;
    bool runtime_has_cuda = false;
    bool runtime_has_dml = false;
    bool runtime_has_trt = false;
    bool cuda_sticky_unavailable = false;
    bool dml_sticky_unavailable = false;
};

std::vector<Attempt> build_plan(const PlanInputs& in);

// Resolves "Auto" (0) tiles for a frame; validates explicit values (1/4/9/16).
int resolve_tiles(int requested, int frame_w, int frame_h, int base_res) noexcept;

// Conservative VRAM estimate (bytes) for one inference run, heuristically sized for DINOv2-based models:
// weights (unless the session is already resident) + attention maps + activations.
uint64_t estimate_vram(const ModelSpec& spec, uint64_t model_file_bytes, int inf_w, int inf_h, int batch, bool fp16, bool session_resident) noexcept;

const char* fallback_reason(dms_status s) noexcept;

}  // namespace dms
