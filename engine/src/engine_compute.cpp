// Depth computation pipeline (caller thread): validation, model resolution, backend plan with the
// fallback chain, pre-processing, inference via the serialized worker, post-processing.
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

int sanitize_base_res(int r) { return r <= 0 ? 518 : clampi(r, 112, 2048); }

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

bool fallbackable(dms_status s) {
    return s == DMS_ERR_OUT_OF_MEMORY || s == DMS_ERR_BACKEND_UNAVAILABLE || s == DMS_ERR_MODEL_LOAD_FAILED || s == DMS_ERR_INFERENCE_FAILED ||
           s == DMS_ERR_UNSUPPORTED;
}

}  // namespace

struct Engine::ComputeCtx {
    const dms_depth_request* req = nullptr;
    ModelSpec spec;
    std::filesystem::path model_path;
    std::filesystem::path fp16_path;  // empty if no installed FP16 variant
    uint64_t model_bytes = 0;
    std::vector<FrameView> frames;
    int center = 0;
    int w = 0, h = 0;
    int samples = 1;
    bool memshare = false;
    uint32_t timeout_ms = 0;
    // progress accounting
    int units_total = 1;
    int units_done = 0;
    // results
    std::vector<float> depth;  // full-resolution output of the successful attempt
    StageTimes times;
    Backend used = Backend::CPU;
    bool full_quality = true;  // false if the successful attempt reduced resolution / changed tiling
};

GpuMonitor& Engine::gpu() {
    std::lock_guard<std::mutex> lk(gpu_mu_);
    if (!gpu_) gpu_ = make_system_gpu_monitor();
    return *gpu_;
}

void Engine::set_gpu_monitor(std::unique_ptr<GpuMonitor> m) {
    std::lock_guard<std::mutex> lk(gpu_mu_);
    gpu_ = std::move(m);
}

bool Engine::oom_recent(const std::string& key) {
    std::lock_guard<std::mutex> lk(oom_mu_);
    auto it = oom_memo_.find(key);
    return it != oom_memo_.end() && Clock::now() - it->second < std::chrono::seconds(30);
}

void Engine::note_oom(const std::string& key) {
    std::lock_guard<std::mutex> lk(oom_mu_);
    oom_memo_[key] = Clock::now();
}

dms_status Engine::run_pass(ComputeCtx& ctx, const Attempt& a, int x0, int y0, int x1, int y1, int iw, int ih, std::vector<float>& out_small,
                            int& ow, int& oh, std::string& err) {
    const dms_depth_request& req = *ctx.req;
    const FrameView& fv = ctx.frames[static_cast<size_t>(ctx.center)];
    const Crop crop{x0, y0, x1, y1};
    const int n = ctx.samples;
    const int batch = a.parallel ? n : 1;
    const size_t img = static_cast<size_t>(3) * static_cast<size_t>(iw) * static_cast<size_t>(ih);
    bool first = true;
    std::vector<float> accum;
    for (int s0 = 0; s0 < n; s0 += batch) {
        const int b = std::min(batch, n - s0);
        // Host abort + VRAM check before every batch.
        if (req.abort_fn && req.abort_fn(req.user)) return DMS_ERR_ABORTED;
        if (a.backend != Backend::CPU) {
            const uint64_t free = gpu().free_vram();
            SessionConfig probe_cfg;
            probe_cfg.model_path = a.fp16 && !ctx.fp16_path.empty() ? ctx.fp16_path : ctx.model_path;
            probe_cfg.backend = a.backend;
            probe_cfg.fp16 = a.fp16;
            probe_cfg.memshare = ctx.memshare && a.backend == Backend::CUDA;
            const bool resident = inference_->has_session(probe_cfg);
            const uint64_t need = estimate_vram(ctx.spec, ctx.model_bytes, iw, ih, b, a.fp16, resident);
            if (free > 0 && free < need + cfg_.vram_reserve_bytes) {
                err = "insufficient VRAM: free " + std::to_string(free >> 20) + " MB, need ~" + std::to_string(need >> 20) + " MB + " +
                      std::to_string(cfg_.vram_reserve_bytes >> 20) + " MB reserve";
                return DMS_ERR_OUT_OF_MEMORY;
            }
        }
        auto t = Clock::now();
        auto job = std::make_shared<InferenceJob>(pool_);
        job->cfg.model_path = a.fp16 && !ctx.fp16_path.empty() ? ctx.fp16_path : ctx.model_path;
        job->cfg.backend = a.backend;
        job->cfg.fp16 = a.fp16 && a.backend != Backend::CPU;
        job->cfg.memshare = ctx.memshare && a.backend == Backend::CUDA;
        job->cfg.trt_cache_dir = cfg_.models_dir / "trt_cache";
        job->offload_mode = clampi(req.backend.struct_size ? req.backend.offload_mode : 0, 0, 2);
        job->input = pool_->take(img * static_cast<size_t>(b));
        job->output = pool_->take(static_cast<size_t>(iw) * static_cast<size_t>(ih) * static_cast<size_t>(b));
        std::vector<Augment> augs;
        for (int k = 0; k < b; ++k) {
            augs.push_back(augment_for_sample(s0 + k, n));
            preprocess_crop(fv, crop, iw, ih, augs.back(), job->input.data() + img * static_cast<size_t>(k));
        }
        job->shape = {b, 3, ih, iw};
        ctx.times.preprocess += ms_since(t);

        const dms_status st = inference_->execute(job, ctx.timeout_ms, req.abort_fn, req.user);
        if (st != DMS_OK) {
            err = job->error;
            return st;
        }
        ctx.times.session += job->ms_session;
        ctx.times.inference += job->ms_run;

        t = Clock::now();
        for (int k = 0; k < b; ++k) {
            int poh = 0, pow_ = 0;
            const float* plane = output_plane(job->output, job->out_shape, static_cast<size_t>(k), 0, poh, pow_);
            const size_t pn = static_cast<size_t>(poh) * static_cast<size_t>(pow_);
            size_t bad = 0;
            for (size_t i = 0; i < pn; ++i)
                if (!std::isfinite(plane[i])) ++bad;
            if (bad * 100 > pn) {
                err = "model produced non-finite values (" + std::to_string(bad) + " of " + std::to_string(pn) + ")";
                return DMS_ERR_INFERENCE_FAILED;  // typical FP16 overflow -> fallback to FP32
            }
            std::vector<float> one(plane, plane + pn);
            float mn = 0.f, mx = 0.f;
            finite_min_max(one.data(), pn, mn, mx);
            sanitize(one.data(), pn, mn);
            deaugment(one.data(), pow_, poh, augs[static_cast<size_t>(k)]);
            if (first) {
                ow = pow_;
                oh = poh;
                accum.assign(pn, 0.f);
                first = false;
            } else if (pow_ != ow || poh != oh) {
                err = "inconsistent output sizes between samples";
                return DMS_ERR_INFERENCE_FAILED;
            }
            for (size_t i = 0; i < pn; ++i) accum[i] += one[i];
        }
        ctx.times.postprocess += ms_since(t);
        ++ctx.units_done;
        if (req.progress_fn &&
            req.progress_fn(req.user, static_cast<float>(ctx.units_done) / static_cast<float>(std::max(1, ctx.units_total))))
            return DMS_ERR_ABORTED;
    }
    const float inv = 1.f / static_cast<float>(n);
    for (auto& v : accum) v *= inv;
    out_small = std::move(accum);
    return DMS_OK;
}

dms_status Engine::run_attempt(ComputeCtx& ctx, const Attempt& a, std::string& err) {
    try {
        if (ctx.spec.kind == DMS_MODEL_KIND_TEMPORAL) return run_temporal(ctx, a, err);
        if (a.tiles > 1) return run_tiled(ctx, a, err);
        int iw = 0, ih = 0;
        inference_dims(ctx.w, ctx.h, a.res, ctx.spec.patch_multiple, iw, ih);
        ctx.units_total = (ctx.samples + (a.parallel ? ctx.samples : 1) - 1) / (a.parallel ? ctx.samples : 1);
        ctx.units_done = 0;
        std::vector<float> small;
        int ow = 0, oh = 0;
        const dms_status st = run_pass(ctx, a, 0, 0, ctx.w, ctx.h, iw, ih, small, ow, oh, err);
        if (st != DMS_OK) return st;
        const auto t = Clock::now();
        ctx.depth.resize(pixel_count(ctx.w, ctx.h));
        resize_bilinear(small.data(), ow, oh, ctx.depth.data(), ctx.w, ctx.h);
        ctx.times.postprocess += ms_since(t);
        return DMS_OK;
    } catch (const Error& e) {
        err = e.what();
        return e.status();
    } catch (const std::bad_alloc&) {
        err = "host out of memory";
        return DMS_ERR_OUT_OF_MEMORY;
    }
}

dms_status Engine::run_tiled(ComputeCtx&, const Attempt&, std::string& err) {
    err = "tiling not implemented yet";
    return DMS_ERR_UNSUPPORTED;
}

dms_status Engine::run_temporal(ComputeCtx&, const Attempt&, std::string& err) {
    err = "temporal models not implemented yet";
    return DMS_ERR_UNSUPPORTED;
}

dms_status Engine::compute(const dms_depth_request& req, dms_depth_result& res) {
    const auto t_start = Clock::now();

    // ---- validation (never trust the caller) ----
    if (!req.model_id || !*req.model_id) fail(DMS_ERR_INVALID_ARG, "model_id is empty");
    if (!req.frames || req.frame_count <= 0 || req.frame_count > 64 || req.center_index < 0 || req.center_index >= req.frame_count)
        fail(DMS_ERR_INVALID_ARG, "invalid frames / frame_count / center_index");
    const dms_frame& center = req.frames[req.center_index];
    check_dims(center.width, center.height, "frame");
    const int w = center.width, h = center.height;
    for (int32_t i = 0; i < req.frame_count; ++i) validate_frame(req.frames[i], w, h);
    if (!res.depth || res.width != w || res.height != h) fail(DMS_ERR_INVALID_ARG, "result buffer does not match frame size");
    if (!std::isfinite(req.edges_enhancement) || !std::isfinite(req.subpixel_smoothing)) fail(DMS_ERR_INVALID_ARG, "non-finite parameter");
    const int base_res = sanitize_base_res(req.base_resolution);
    const bool have_backend = req.backend.struct_size >= sizeof(uint32_t);

    // ---- model resolution ----
    ComputeCtx ctx;
    ctx.req = &req;
    const auto spec = manifest_.find(req.model_id);
    if (!spec) fail(DMS_ERR_NOT_FOUND, std::string("unknown model '") + req.model_id + "'");
    ctx.spec = *spec;
    ctx.model_path = manifest_.path_for(*spec, false);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(ctx.model_path, ec))
        fail(DMS_ERR_MODEL_NOT_INSTALLED, spec->display_name + " is not installed (expected at " + platform::path_to_utf8(ctx.model_path) + ")");
    ctx.model_bytes = std::filesystem::file_size(ctx.model_path, ec);
    if (spec->size && (ec || ctx.model_bytes != spec->size))
        fail(DMS_ERR_MODEL_NOT_INSTALLED, spec->display_name + " file is incomplete or corrupt - re-download it from Options");
    if (!spec->fp16_file.empty()) {
        const auto p16 = manifest_.path_for(*spec, true);
        if (std::filesystem::is_regular_file(p16, ec) && (!spec->fp16_size || std::filesystem::file_size(p16, ec) == spec->fp16_size)) ctx.fp16_path = p16;
    }
    for (int32_t i = 0; i < req.frame_count; ++i) ctx.frames.push_back(frame_view(req.frames[i]));
    ctx.center = req.center_index;
    ctx.w = w;
    ctx.h = h;
    ctx.samples = have_backend ? clampi(req.backend.samples, 1, 16) : 1;
    ctx.timeout_ms = req.timeout_ms ? req.timeout_ms : cfg_.inference_timeout_ms;

    // ---- frame cache ----
    const bool cacheable = req.cache_key.struct_size >= sizeof(uint32_t) && req.cache_key.layer_hash != 0;
    CacheKey key;
    if (cacheable) {
        key = cache_key(req.cache_key, spec->id, base_res, have_backend ? &req.backend : nullptr, req.edges_enhancement, req.subpixel_smoothing, w, h);
        CacheEntry ce;
        if (cache_.get(key, ce) && ce.depth && ce.depth->size() == pixel_count(w, h)) {
            std::copy(ce.depth->begin(), ce.depth->end(), res.depth);
            res.from_cache = 1;
            res.backend_used = DMS_BACKEND_NONE;
            res.fallback_steps = 0;
            res.raw_min = ce.raw_min;
            res.raw_max = ce.raw_max;
            res.message[0] = '\0';
            res.ms_total = ms_since(t_start);
            return DMS_OK;
        }
    }

    // ---- backend plan ----
    inference_->ensure_runtime();  // throws BACKEND_UNAVAILABLE if no ONNX Runtime at all
    const GpuSnapshot gs = gpu().snapshot();
    PlanInputs pin;
    pin.hardware = have_backend ? req.backend.hardware_acceleration != 0 : true;
    pin.fp16 = have_backend && req.backend.fp16 != 0;
    pin.use_tensorrt = have_backend && req.backend.use_tensorrt != 0;
#if defined(DMS_ENABLE_TENSORRT)
    pin.tensorrt_compiled = true;
#endif
    pin.base_res = base_res;
    pin.tiles = resolve_tiles(have_backend ? req.backend.tiles : 0, w, h, base_res);
    pin.parallel = have_backend && req.backend.parallel_samples != 0 && ctx.samples > 2;
    pin.patch = spec->patch_multiple;
    pin.gpu = gs;
    pin.runtime_has_cuda = inference_->runtime_has_provider("CUDAExecutionProvider");
    pin.runtime_has_dml = inference_->runtime_has_provider("DmlExecutionProvider");
    pin.runtime_has_trt = inference_->runtime_has_provider("TensorrtExecutionProvider");
    pin.cuda_sticky_unavailable = cuda_sticky_unavailable_;
    pin.dml_sticky_unavailable = dml_sticky_unavailable_;
    // CUDA Memory Sharing: NVIDIA only, and only on known-good drivers (silently disabled otherwise).
    ctx.memshare = have_backend && req.backend.cuda_memory_sharing != 0 && gs.vendor == DMS_GPU_NVIDIA;
    if (ctx.memshare && !gs.driver.at_least(kMinMemShareDriverMajor, kMinMemShareDriverMinor)) {
        ctx.memshare = false;
        if (!memshare_warned_.exchange(true))
            logf(DMS_LOG_WARN, "CUDA Memory Sharing disabled: driver %s is older than the known-good %d.%02d", gs.driver.str().c_str(),
                 kMinMemShareDriverMajor, kMinMemShareDriverMinor);
    }
    const std::vector<Attempt> plan = build_plan(pin);

    // ---- execute the plan (fallback chain) ----
    std::string trail;
    int steps = 0;
    dms_status last = DMS_ERR_BACKEND_UNAVAILABLE;
    std::string last_err;
    bool ok = false;
    for (size_t i = 0; i < plan.size(); ++i) {
        const Attempt& a = plan[i];
        const std::string memo_key = ctx.spec.id + "|" + a.label;
        if ((a.backend == Backend::CUDA && cuda_sticky_unavailable_) || (a.backend == Backend::DirectML && dml_sticky_unavailable_)) continue;
        if (i + 1 < plan.size() && a.backend != Backend::CPU && oom_recent(memo_key)) {
            trail += a.label + ": skipped (out of memory recently); ";
            ++steps;
            continue;
        }
        std::string err;
        const dms_status st = run_attempt(ctx, a, err);
        if (st == DMS_OK) {
            ctx.used = a.backend;
            ctx.full_quality = a.res == base_res && a.tiles == pin.tiles;
            ok = true;
            break;
        }
        if (st == DMS_ERR_ABORTED || st == DMS_ERR_TIMEOUT) {
            set_last_error(st == DMS_ERR_ABORTED ? "aborted by host" : "timed out waiting for inference");
            return st;
        }
        last = st;
        last_err = err;
        logf(DMS_LOG_WARN, "attempt %s failed: %s (%s)", a.label.c_str(), fallback_reason(st), err.c_str());
        trail += a.label + ": " + fallback_reason(st) + "; ";
        ++steps;
        if (st == DMS_ERR_OUT_OF_MEMORY) note_oom(memo_key);
        if (st == DMS_ERR_BACKEND_UNAVAILABLE && a.backend == Backend::CUDA && !cuda_sticky_unavailable_.exchange(true))
            logf(DMS_LOG_WARN, "CUDA disabled for this session: %s", err.c_str());
        if (st == DMS_ERR_BACKEND_UNAVAILABLE && a.backend == Backend::DirectML) dml_sticky_unavailable_ = true;
        if (!fallbackable(st)) break;
        if (a.backend == Backend::CPU && st == DMS_ERR_MODEL_LOAD_FAILED) break;  // the file itself is bad
    }
    if (!ok) fail(last, last_err.empty() ? std::string(dms_status_message(last)) : last_err + (trail.empty() ? "" : " [" + trail + "]"));

    // ---- copy out ----
    auto t = Clock::now();
    float mn = 0.f, mx = 0.f;
    if (!finite_min_max(ctx.depth.data(), ctx.depth.size(), mn, mx)) fail(DMS_ERR_INFERENCE_FAILED, "no finite depth values");
    std::copy(ctx.depth.begin(), ctx.depth.end(), res.depth);
    if (cacheable && ctx.full_quality) {  // reduced-quality fallbacks are not cached (GPU may recover)
        CacheEntry ce;
        ce.depth = std::make_shared<const std::vector<float>>(std::move(ctx.depth));
        ce.raw_min = mn;
        ce.raw_max = mx;
        cache_.put(key, std::move(ce));
    }
    ctx.times.postprocess += ms_since(t);

    res.from_cache = 0;
    res.backend_used = static_cast<int32_t>(ctx.used);
    res.fallback_steps = steps;
    res.raw_min = mn;
    res.raw_max = mx;
    res.message[0] = '\0';
    if (steps > 0) copy_str(res.message, sizeof(res.message), "fallback: " + trail);
    ctx.times.total = ms_since(t_start);
    res.ms_total = ctx.times.total;
    logf(DMS_LOG_DEBUG, "depth %dx%d model=%s backend=%s steps=%d | pre %.1f ms, session %.1f ms, inference %.1f ms, post %.1f ms, total %.1f ms", w,
         h, spec->id.c_str(), backend_name(ctx.used), steps, static_cast<double>(ctx.times.preprocess), static_cast<double>(ctx.times.session),
         static_cast<double>(ctx.times.inference), static_cast<double>(ctx.times.postprocess), static_cast<double>(ctx.times.total));
    return steps > 0 ? DMS_ERR_DEGRADED : DMS_OK;
}

}  // namespace dms
