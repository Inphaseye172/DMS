// Depth computation pipeline. Milestone 1: deterministic gray ramp stub (proves the ABI + host plumbing).
#include "engine_impl.h"

#include <chrono>

namespace dms {

dms_status Engine::compute(const dms_depth_request& req, dms_depth_result& res) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!req.frames || req.frame_count <= 0 || req.center_index < 0 || req.center_index >= req.frame_count)
        fail(DMS_ERR_INVALID_ARG, "invalid frames / frame_count / center_index");
    const dms_frame& f = req.frames[req.center_index];
    check_dims(f.width, f.height, "frame");
    if (!res.depth || res.width != f.width || res.height != f.height) fail(DMS_ERR_INVALID_ARG, "result buffer does not match frame size");

    // Gray ramp: left = far (0), right = near (1).
    for (int32_t y = 0; y < f.height; ++y) {
        float* row = res.depth + static_cast<size_t>(y) * static_cast<size_t>(f.width);
        for (int32_t x = 0; x < f.width; ++x)
            row[x] = f.width > 1 ? static_cast<float>(x) / static_cast<float>(f.width - 1) : 0.5f;
    }
    res.from_cache = 0;
    res.backend_used = DMS_BACKEND_NONE;
    res.fallback_steps = 0;
    res.raw_min = 0.f;
    res.raw_max = 1.f;
    res.ms_total = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    copy_str(res.message, sizeof(res.message), "stub ramp (milestone 1)");
    return DMS_OK;
}

}  // namespace dms
