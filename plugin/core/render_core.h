// Host-independent SmartFX render pipeline. The AE glue layer checks out worlds and calls render();
// everything else (conversion, inference request, caching, fallbacks, mapping, output writing for the
// requested sub-rect) happens here, so it can be unit-tested without After Effects.
#pragma once

#include "params_model.h"
#include "world_view.h"

#include "dms/engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dms_plugin {

struct RenderJob {
    ParamSnapshot params;
    // Temporal window (ordered by time). For single-frame models this has exactly one entry.
    std::vector<WorldView> frames;
    int32_t center_index = 0;
    Rect input_rect;   // layer-space rect covered by frames[*] pixel (0,0)..(w,h)
    Rect output_rect;  // layer-space rect covered by the output world
    WorldView output;
    int64_t time_num = 0;   // host time of the center frame
    int64_t time_den = 1;
    int64_t time_step = 1;  // host time between frames (frames[k] is at time_num + (k - center_index) * time_step)
    uint8_t instance_guid[16] = {0};
    dms_abort_fn abort_fn = nullptr;
    dms_progress_fn progress_fn = nullptr;
    void* callback_user = nullptr;
    uint32_t timeout_ms = 0;
};

struct RenderOutcome {
    dms_status status = DMS_OK;  // DMS_OK, DMS_ERR_DEGRADED (output valid, via fallback), DMS_ERR_ABORTED
    bool output_written = false;
    bool report_to_user = false;  // first occurrence of this problem for the instance
    std::string message;
    float ms_total = 0.f;
    float ms_convert = 0.f;   // world -> RGB float (+ temporal downscale)
    float ms_engine = 0.f;    // dms_depth_compute (cache, inference, post-processing)
    float ms_copyout = 0.f;   // mapping + writing the output sub-rect
};

// Never throws. Always leaves the output world fully written unless status == DMS_ERR_ABORTED.
RenderOutcome render(dms_engine* engine, const RenderJob& job) noexcept;

// Writes the final output for `out` from a full-input-size depth map. Exposed for tests.
void write_output(const RenderJob& job, const std::vector<float>& mapped, const std::vector<float>* rgb_colormap,
                  const std::vector<float>* slice_mask, const std::vector<float>& input_alpha) noexcept;

}  // namespace dms_plugin
