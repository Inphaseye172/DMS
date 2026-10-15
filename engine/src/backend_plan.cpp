#include "backend_plan.h"

#include "image_ops.h"

#include <algorithm>
#include <cmath>

namespace dms {

namespace {
constexpr int kResLadder[] = {1024, 768, 518, 384};

std::string label_for(Backend b, bool fp16, int res, int tiles, bool parallel) {
    std::string s = backend_name(b);
    s += fp16 ? " FP16" : " FP32";
    s += " @" + std::to_string(res);
    if (tiles > 1) s += " x" + std::to_string(tiles) + " tiles";
    if (parallel) s += " (parallel samples)";
    return s;
}
}  // namespace

int resolve_tiles(int requested, int w, int h, int base_res) noexcept {
    if (requested == 1 || requested == 4 || requested == 9 || requested == 16) return requested;
    // Auto: keep each tile close to the base resolution.
    const double ratio = static_cast<double>(std::max(w, h)) / static_cast<double>(base_res > 0 ? base_res : 518);
    if (ratio <= 2.5) return 1;
    if (ratio <= 4.0) return 4;
    if (ratio <= 6.0) return 9;
    return 16;
}

std::vector<Attempt> build_plan(const PlanInputs& in) {
    std::vector<Attempt> plan;
    auto add = [&](Backend b, bool fp16, int res, int tiles, bool parallel) {
        Attempt a;
        a.backend = b;
        a.fp16 = fp16;
        a.res = res;
        a.tiles = tiles;
        a.parallel = parallel;
        a.label = label_for(b, fp16, res, tiles, parallel);
        plan.push_back(a);
    };

    // Pick the GPU backend.
    bool have_gpu = false;
    Backend gpu = Backend::CPU;
    if (in.hardware) {
        const bool nvidia_ok = in.gpu.vendor == DMS_GPU_NVIDIA && in.gpu.driver.at_least(kMinCudaDriverMajor, kMinCudaDriverMinor);
        if (nvidia_ok && in.runtime_has_cuda && !in.cuda_sticky_unavailable) {
            gpu = (in.use_tensorrt && in.tensorrt_compiled && in.runtime_has_trt) ? Backend::TensorRT : Backend::CUDA;
            have_gpu = true;
        } else if (in.gpu.has_dxgi_adapter && in.runtime_has_dml && !in.dml_sticky_unavailable) {
            gpu = Backend::DirectML;
            have_gpu = true;
        }
    }

    if (have_gpu) {
        add(gpu, in.fp16, in.base_res, in.tiles, in.parallel);
        if (in.parallel) add(gpu, in.fp16, in.base_res, in.tiles, false);           // batch too big -> sequential samples
        if (in.fp16) add(gpu, false, in.base_res, in.tiles, false);                // FP16 failed -> FP32
        if (gpu == Backend::TensorRT) add(Backend::CUDA, false, in.base_res, in.tiles, false);
        const Backend g2 = gpu == Backend::TensorRT ? Backend::CUDA : gpu;
        int smallest = in.base_res;
        for (int r : kResLadder) {
            if (r < in.base_res) {
                add(g2, false, r, in.tiles, false);                                // lower resolution
                smallest = r;
            }
        }
        for (int t : {4, 9, 16}) {
            if (t <= in.tiles) continue;
            const int per_tile = round_to_multiple(static_cast<int>(std::lround(smallest / std::sqrt(static_cast<double>(t)))), in.patch);
            add(g2, false, std::max(per_tile, 2 * in.patch), t, false);           // more tiles, smaller peak memory
        }
    }
    add(Backend::CPU, false, in.base_res, in.tiles, false);                        // last inference rung
    return plan;
}

uint64_t estimate_vram(const ModelSpec& spec, uint64_t model_file_bytes, int iw, int ih, int batch, bool fp16, bool resident) noexcept {
    // ViT size class from parameter count (DINOv2 S/B/L/G).
    int heads = 6, dim = 384, layers = 12;
    if (spec.params_millions > 600) { heads = 24; dim = 1536; layers = 40; }
    else if (spec.params_millions > 200) { heads = 16; dim = 1024; layers = 24; }
    else if (spec.params_millions > 60) { heads = 12; dim = 768; layers = 12; }
    const int p = spec.patch_multiple > 0 ? spec.patch_multiple : 14;
    const double tokens = std::ceil(static_cast<double>(iw) / p) * std::ceil(static_cast<double>(ih) / p) + 1.0;
    const double bytes_per = fp16 ? 2.0 : 4.0;
    const double b = std::max(1, batch);
    const double attn = b * heads * tokens * tokens * bytes_per * 2.0;               // scores + softmax, one layer live
    const double acts = b * tokens * dim * bytes_per * 16.0;                          // MLP + residual buffers
    const double head = b * static_cast<double>(iw) * ih * 256.0 * bytes_per;          // DPT decoder feature maps
    const double weights = resident ? 0.0 : static_cast<double>(model_file_bytes) * (fp16 && spec.fp16_file.empty() ? 1.0 : 1.0) * 1.2;
    const double workspace = 256.0 * 1024 * 1024;                                      // cuDNN/cuBLAS workspace
    (void)layers;
    return static_cast<uint64_t>((weights + attn + acts + head + workspace) * 1.3);    // 30% safety margin
}

const char* fallback_reason(dms_status s) noexcept {
    switch (s) {
        case DMS_ERR_OUT_OF_MEMORY: return "out of memory";
        case DMS_ERR_BACKEND_UNAVAILABLE: return "backend unavailable";
        case DMS_ERR_MODEL_LOAD_FAILED: return "model failed to load on this backend";
        case DMS_ERR_INFERENCE_FAILED: return "inference failed";
        case DMS_ERR_UNSUPPORTED: return "unsupported";
        default: return dms_status_message(s);
    }
}

}  // namespace dms
