// Depth computation pipeline (caller thread): validation, model resolution, pre-processing, inference
// via the serialized worker, post-processing back to frame resolution.
#include "engine_impl.h"

#include "image_ops.h"
#include "pipeline.h"
#include "platform.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace dms {

namespace {

using Clock = std::chrono::steady_clock;
float ms_since(Clock::time_point t) { return std::chrono::duration<float, std::milli>(Clock::now() - t).count(); }

FrameView frame_view(const dms_frame& f) {
    FrameView v;
    v.rgb = f.rgb;
    v.width = f.width;
    v.height = f.height;
    v.stride = f.row_stride_floats;
    return v;
}

void validate_frame(const dms_frame& f, int w, int h) {
    check_struct(&f, "dms_frame");
    if (!f.rgb) fail(DMS_ERR_INVALID_ARG, "frame rgb is NULL");
    check_dims(f.width, f.height, "frame");
    if (f.width != w || f.height != h) fail(DMS_ERR_INVALID_ARG, "all frames in a window must have the same size");
    if (f.row_stride_floats < f.width * 3) fail(DMS_ERR_INVALID_ARG, "row_stride_floats < width*3");
}

int sanitize_base_res(int r) {
    if (r <= 0) return 518;
    return clampi(r, 112, 2048);
}

// Extracts the [h,w] map for batch index b (and temporal index t) from a dense output tensor.
const float* output_plane(const std::vector<float>& out, const std::vector<int64_t>& shape, size_t b, size_t t, int& oh, int& ow) {
    if (shape.size() < 3) fail(DMS_ERR_INFERENCE_FAILED, "model output rank < 3");
    oh = static_cast<int>(shape[shape.size() - 2]);
    ow = static_cast<int>(shape[shape.size() - 1]);
    if (oh <= 0 || ow <= 0) fail(DMS_ERR_INFERENCE_FAILED, "model output has empty spatial dims");
    const size_t plane = static_cast<size_t>(oh) * static_cast<size_t>(ow);
    size_t planes_per_batch = 1;
    for (size_t i = 1; i + 2 < shape.size(); ++i) planes_per_batch *= static_cast<size_t>(shape[i]);
    const size_t index = b * planes_per_batch + (t < planes_per_batch ? t : planes_per_batch - 1);
    if ((index + 1) * plane > out.size()) fail(DMS_ERR_INFERENCE_FAILED, "model output smaller than its shape");
    return out.data() + index * plane;
}

}  // namespace

dms_status Engine::compute(const dms_depth_request& req, dms_depth_result& res) {
    const auto t_start = Clock::now();
    StageTimes times;

    // ---- validation (never trust the caller) ----
    if (!req.model_id || !*req.model_id) fail(DMS_ERR_INVALID_ARG, "model_id is empty");
    if (!req.frames || req.frame_count <= 0 || req.frame_count > 64 || req.center_index < 0 || req.center_index >= req.frame_count)
        fail(DMS_ERR_INVALID_ARG, "invalid frames / frame_count / center_index");
    const dms_frame& center = req.frames[req.center_index];
    check_dims(center.width, center.height, "frame");
    const int w = center.width, h = center.height;
    for (int32_t i = 0; i < req.frame_count; ++i) validate_frame(req.frames[i], w, h);
    if (!res.depth || res.width != w || res.height != h) fail(DMS_ERR_INVALID_ARG, "result buffer does not match frame size");
    const int base_res = sanitize_base_res(req.base_resolution);

    // ---- model resolution ----
    const auto spec = manifest_.find(req.model_id);
    if (!spec) fail(DMS_ERR_NOT_FOUND, std::string("unknown model '") + req.model_id + "'");
    const auto model_path = manifest_.path_for(*spec, false);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model_path, ec))
        fail(DMS_ERR_MODEL_NOT_INSTALLED, spec->display_name + " is not installed (expected at " + platform::path_to_utf8(model_path) + ")");
    if (spec->size) {
        const auto sz = std::filesystem::file_size(model_path, ec);
        if (ec || sz != spec->size) fail(DMS_ERR_MODEL_NOT_INSTALLED, spec->display_name + " file is incomplete or corrupt - re-download it from Options");
    }

    // ---- preprocess (caller thread) ----
    auto t = Clock::now();
    int iw = 0, ih = 0;
    inference_dims(w, h, base_res, spec->patch_multiple, iw, ih);
    auto job = std::make_shared<InferenceJob>();
    job->cfg.model_path = model_path;
    job->cfg.backend = Backend::CPU;
    job->offload_mode = clampi(req.backend.struct_size ? req.backend.offload_mode : 0, 0, 2);
    job->input.resize(static_cast<size_t>(3) * static_cast<size_t>(iw) * static_cast<size_t>(ih));
    preprocess_crop(frame_view(center), Crop{0, 0, w, h}, iw, ih, Augment{}, job->input.data());
    job->shape = {1, 3, ih, iw};
    times.preprocess = ms_since(t);

    // ---- inference (worker thread) ----
    t = Clock::now();
    const uint32_t timeout = req.timeout_ms ? req.timeout_ms : cfg_.inference_timeout_ms;
    const dms_status st = inference_->execute(job, timeout, req.abort_fn, req.user);
    if (st == DMS_ERR_ABORTED || st == DMS_ERR_TIMEOUT) {
        set_last_error(st == DMS_ERR_ABORTED ? "aborted by host" : "timed out waiting for inference");
        return st;
    }
    if (st != DMS_OK) fail(st, job->error.empty() ? std::string(dms_status_message(st)) : job->error);
    times.session = job->ms_session;
    times.inference = job->ms_run;

    // ---- postprocess (caller thread) ----
    t = Clock::now();
    int oh = 0, ow = 0;
    const float* plane = output_plane(job->output, job->out_shape, 0, 0, oh, ow);
    std::vector<float> small(plane, plane + static_cast<size_t>(oh) * static_cast<size_t>(ow));
    float mn = 0.f, mx = 0.f;
    if (!finite_min_max(small.data(), small.size(), mn, mx)) fail(DMS_ERR_INFERENCE_FAILED, "model output contains no finite values");
    sanitize(small.data(), small.size(), mn);
    resize_bilinear(small.data(), ow, oh, res.depth, w, h);
    res.raw_min = mn;
    res.raw_max = mx;
    times.postprocess = ms_since(t);

    res.from_cache = 0;
    res.backend_used = static_cast<int32_t>(job->cfg.backend);
    res.fallback_steps = 0;
    res.message[0] = '\0';
    times.total = ms_since(t_start);
    res.ms_total = times.total;
    logf(DMS_LOG_DEBUG, "depth %dx%d model=%s inf=%dx%d backend=%s | pre %.1f ms, session %.1f ms, inference %.1f ms, post %.1f ms, total %.1f ms",
         w, h, spec->id.c_str(), iw, ih, backend_name(job->cfg.backend), static_cast<double>(times.preprocess), static_cast<double>(times.session),
         static_cast<double>(times.inference), static_cast<double>(times.postprocess), static_cast<double>(times.total));
    return DMS_OK;
}

}  // namespace dms
