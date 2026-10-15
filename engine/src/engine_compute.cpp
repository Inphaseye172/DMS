// Depth computation pipeline (caller thread): validation, model resolution, backend plan with the
// fallback chain, pre-processing, inference via the serialized worker, post-processing.
#include "engine_impl.h"

#include "image_ops.h"
#include "pipeline.h"
#include "platform.h"
#include "postprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>

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
    std::vector<FrameView> frames;  // model input frames (may be downscaled when a guide is given)
    FrameView guide;                // full-resolution center frame (output size + edge guide)
    int center = 0;
    int w = 0, h = 0;               // output size (= guide size)
    bool memopt = false;
    bool from_temporal_store = false;
    std::string temporal_key_prefix;  // model|layer_hash|res|fp16|samples|
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

dms_status Engine::run_pass(ComputeCtx& ctx, const Attempt& a, const FrameView& fv, int x0, int y0, int x1, int y1, int iw, int ih,
                            std::vector<float>& out_small, int& ow, int& oh, std::string& err) {
    const dms_depth_request& req = *ctx.req;
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
        const FrameView& src = ctx.frames[static_cast<size_t>(ctx.center)];
        inference_dims(src.width, src.height, a.res, ctx.spec.patch_multiple, iw, ih);
        const dms_status st = run_pass(ctx, a, src, 0, 0, src.width, src.height, iw, ih, small, ow, oh, err);
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

// Tiled inference (detail merge): a global low-resolution pass provides the low spatial frequencies
// (overall depth layout, which a tile cannot know from its crop alone); every tile is inferred at the
// attempt resolution, affinely aligned to the global pass over its area (relative depth models have a
// per-run scale/shift), high-pass filtered, and only that detail is blended back with feathered weights
// over the overlap. Low frequencies never come from tiles, so tile-to-tile layout differences cannot
// produce seams.
dms_status Engine::run_tiled(ComputeCtx& ctx, const Attempt& a, std::string& err) {
    const FrameView& src = ctx.guide;  // tiles are cut from the full-resolution frame
    const int W = src.width, H = src.height;
    int g = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.tiles))));
    const int patch = ctx.spec.patch_multiple > 0 ? ctx.spec.patch_multiple : 14;
    // Tiles must stay meaningfully large; shrink the grid for small frames.
    while (g > 1 && (W / g < 4 * patch || H / g < 4 * patch)) --g;
    if (g <= 1) {
        Attempt single = a;
        single.tiles = 1;
        return run_attempt(ctx, single, err);
    }
    const int batches_per_pass = a.parallel ? 1 : ctx.samples;
    ctx.units_total = (g * g + 1) * batches_per_pass;
    ctx.units_done = 0;
    const size_t npx = pixel_count(ctx.w, ctx.h);

    // 1) Global pass (low resolution reference).
    std::vector<float> global(npx);
    {
        const FrameView& gsrc = ctx.frames[static_cast<size_t>(ctx.center)];
        int iw = 0, ih = 0, ow = 0, oh = 0;
        inference_dims(gsrc.width, gsrc.height, a.res, patch, iw, ih);
        std::vector<float> small;
        const dms_status st = run_pass(ctx, a, gsrc, 0, 0, gsrc.width, gsrc.height, iw, ih, small, ow, oh, err);
        if (st != DMS_OK) return st;
        resize_bilinear(small.data(), ow, oh, global.data(), ctx.w, ctx.h);
    }

    // 2) Tile layout with overlap = 32 px at tile inference resolution (converted to source pixels).
    const int approx_tile = std::max((W + g - 1) / g, (H + g - 1) / g);
    const int overlap = std::max(8, static_cast<int>(std::lround(32.0 * approx_tile / std::max(1, a.res))));
    const int tw = std::min(W, (W + (g - 1) * overlap + g - 1) / g);
    const int th = std::min(H, (H + (g - 1) * overlap + g - 1) / g);
    auto origin = [g](int i, int total, int tile) { return g > 1 ? static_cast<int>(std::lround(static_cast<double>(i) * (total - tile) / (g - 1))) : 0; };
    const float sx = static_cast<float>(ctx.w) / static_cast<float>(W);  // source -> output scale (normally 1)
    const float sy = static_cast<float>(ctx.h) / static_cast<float>(H);
    // Band split: structures larger than ~1/8 of a tile come from the global pass.
    const int lp_radius = std::max(2, static_cast<int>(std::lround(std::min(static_cast<float>(tw) * sx, static_cast<float>(th) * sy) / 16.f)));
    auto lowpass = [lp_radius](const float* in, float* out, int w, int h) {
        std::vector<float> tmp(static_cast<size_t>(w) * static_cast<size_t>(h)), t2(tmp.size());
        box_blur(in, out, tmp.data(), w, h, lp_radius);  // 3 box passes ~ Gaussian
        box_blur(out, t2.data(), tmp.data(), w, h, lp_radius);
        box_blur(t2.data(), out, tmp.data(), w, h, lp_radius);
    };

    std::vector<float> detail(npx, 0.f), wsum(npx, 0.f);
    for (int ty = 0; ty < g; ++ty) {
        for (int tx = 0; tx < g; ++tx) {
            const int x0 = origin(tx, W, tw), y0 = origin(ty, H, th);
            const int x1 = std::min(W, x0 + tw), y1 = std::min(H, y0 + th);
            int iw = 0, ih = 0, ow = 0, oh = 0;
            inference_dims(x1 - x0, y1 - y0, a.res, patch, iw, ih);
            std::vector<float> small;
            const dms_status st = run_pass(ctx, a, src, x0, y0, x1, y1, iw, ih, small, ow, oh, err);
            if (st != DMS_OK) return st;
            // Tile region in output coordinates.
            const int ox0 = static_cast<int>(std::lround(static_cast<float>(x0) * sx));
            const int oy0 = static_cast<int>(std::lround(static_cast<float>(y0) * sy));
            const int ox1 = std::max(ox0 + 1, static_cast<int>(std::lround(static_cast<float>(x1) * sx)));
            const int oy1 = std::max(oy0 + 1, static_cast<int>(std::lround(static_cast<float>(y1) * sy)));
            const int rw = std::min(ox1, ctx.w) - ox0, rh = std::min(oy1, ctx.h) - oy0;
            if (rw <= 0 || rh <= 0) continue;
            std::vector<float> tile(static_cast<size_t>(rw) * static_cast<size_t>(rh));
            resize_bilinear(small.data(), ow, oh, tile.data(), rw, rh);
            // Align the tile to the global reference over the tile area (scales the detail correctly).
            AffineAccumulator fit;
            std::vector<float> gref(tile.size());
            for (int y = 0; y < rh; ++y)
                for (int x = 0; x < rw; ++x)
                    gref[static_cast<size_t>(y) * static_cast<size_t>(rw) + static_cast<size_t>(x)] =
                        global[static_cast<size_t>(oy0 + y) * static_cast<size_t>(ctx.w) + static_cast<size_t>(ox0 + x)];
            fit.add(tile.data(), gref.data(), tile.size());
            Affine af;
            if (!fit.solve(af)) continue;  // featureless tile: contributes no detail
            for (auto& v : tile) v = static_cast<float>(af.s * static_cast<double>(v) + af.b);
            std::vector<float> tile_lp(tile.size());
            lowpass(tile.data(), tile_lp.data(), rw, rh);
            // Feathered weights: ramp over the overlap on interior edges only.
            const float ovx = std::max(1.f, static_cast<float>(overlap) * sx), ovy = std::max(1.f, static_cast<float>(overlap) * sy);
            for (int y = 0; y < rh; ++y) {
                const float dy0 = ty == 0 ? 1e9f : static_cast<float>(y) + 0.5f;
                const float dy1 = ty == g - 1 ? 1e9f : static_cast<float>(rh - y) - 0.5f;
                const float wy = std::min(1.f, std::min(dy0, dy1) / ovy);
                for (int x = 0; x < rw; ++x) {
                    const float dx0 = tx == 0 ? 1e9f : static_cast<float>(x) + 0.5f;
                    const float dx1 = tx == g - 1 ? 1e9f : static_cast<float>(rw - x) - 0.5f;
                    const float wx = std::min(1.f, std::min(dx0, dx1) / ovx);
                    const float wgt = std::max(1e-4f, wx * wy);
                    const size_t ti = static_cast<size_t>(y) * static_cast<size_t>(rw) + static_cast<size_t>(x);
                    const size_t o = static_cast<size_t>(oy0 + y) * static_cast<size_t>(ctx.w) + static_cast<size_t>(ox0 + x);
                    detail[o] += wgt * (tile[ti] - tile_lp[ti]);
                    wsum[o] += wgt;
                }
            }
        }
    }
    // 3) Merge: low band from the global pass + blended high band from the tiles.
    std::vector<float> global_lp(npx);
    lowpass(global.data(), global_lp.data(), ctx.w, ctx.h);
    ctx.depth.resize(npx);
    for (size_t i = 0; i < npx; ++i) ctx.depth[i] = wsum[i] > 0.f ? global_lp[i] + detail[i] / wsum[i] : global[i];
    return DMS_OK;
}

dms_status Engine::run_temporal(ComputeCtx& ctx, const Attempt& a, std::string& err) {
    const dms_depth_request& req = *ctx.req;
    // Restrict to the effective window around the center (callers normally pass exactly the window).
    const int wmax = effective_window(ctx.spec.window, ctx.memopt);
    int first = 0, count = static_cast<int>(ctx.frames.size());
    if (count > wmax) {
        first = clampi(ctx.center - (wmax - 1) / 2, 0, count - wmax);
        count = wmax;
    }
    const int center = ctx.center - first;
    const FrameView& f0 = ctx.frames[static_cast<size_t>(first)];
    int iw = 0, ih = 0;
    inference_dims(f0.width, f0.height, a.res, ctx.spec.patch_multiple, iw, ih);
    const size_t img = static_cast<size_t>(3) * static_cast<size_t>(iw) * static_cast<size_t>(ih);
    const size_t plane = static_cast<size_t>(iw) * static_cast<size_t>(ih);
    ctx.units_total = ctx.samples;
    ctx.units_done = 0;

    std::vector<std::vector<float>> acc(static_cast<size_t>(count));
    int ow = 0, oh = 0;
    for (int s = 0; s < ctx.samples; ++s) {
        if (req.abort_fn && req.abort_fn(req.user)) return DMS_ERR_ABORTED;
        Augment aug = augment_for_sample(s, ctx.samples);
        if (a.backend != Backend::CPU) {
            const uint64_t free = gpu().free_vram();
            const uint64_t need = estimate_vram(ctx.spec, ctx.model_bytes, iw, ih, count, a.fp16, false);
            if (free > 0 && free < need + cfg_.vram_reserve_bytes) {
                err = "insufficient VRAM for a " + std::to_string(count) + "-frame window (free " + std::to_string(free >> 20) + " MB, need ~" +
                      std::to_string(need >> 20) + " MB + reserve); enable Memory Optimization";
                return DMS_ERR_OUT_OF_MEMORY;
            }
        }
        auto t = Clock::now();
        auto job = std::make_shared<InferenceJob>(pool_);
        job->cfg.model_path = a.fp16 && !ctx.fp16_path.empty() ? ctx.fp16_path : ctx.model_path;
        job->cfg.backend = a.backend;
        job->cfg.fp16 = a.fp16 && a.backend != Backend::CPU;
        job->cfg.memshare = ctx.memshare && a.backend == Backend::CUDA;
        job->offload_mode = clampi(req.backend.struct_size ? req.backend.offload_mode : 0, 0, 2);
        job->input = pool_->take(img * static_cast<size_t>(count));
        job->output = pool_->take(plane * static_cast<size_t>(count));
        for (int k = 0; k < count; ++k) {
            const FrameView& fv = ctx.frames[static_cast<size_t>(first) + static_cast<size_t>(k)];
            preprocess_crop(fv, Crop{0, 0, fv.width, fv.height}, iw, ih, aug, job->input.data() + img * static_cast<size_t>(k));
        }
        job->shape = {1, count, 3, ih, iw};
        ctx.times.preprocess += ms_since(t);
        const dms_status st = inference_->execute(job, ctx.timeout_ms, req.abort_fn, req.user);
        if (st != DMS_OK) {
            err = job->error;
            return st;
        }
        ctx.times.session += job->ms_session;
        ctx.times.inference += job->ms_run;
        t = Clock::now();
        for (int k = 0; k < count; ++k) {
            int poh = 0, pow_ = 0;
            const float* p = output_plane(job->output, job->out_shape, 0, static_cast<size_t>(k), poh, pow_);
            const size_t pn = static_cast<size_t>(poh) * static_cast<size_t>(pow_);
            std::vector<float> one(p, p + pn);
            float mn = 0.f, mx = 0.f;
            if (!finite_min_max(one.data(), pn, mn, mx)) {
                err = "temporal model produced no finite values";
                return DMS_ERR_INFERENCE_FAILED;
            }
            sanitize(one.data(), pn, mn);
            deaugment(one.data(), pow_, poh, aug);
            auto& dst = acc[static_cast<size_t>(k)];
            if (dst.empty()) dst.assign(pn, 0.f);
            if (dst.size() != pn) {
                err = "inconsistent temporal output sizes";
                return DMS_ERR_INFERENCE_FAILED;
            }
            for (size_t i = 0; i < pn; ++i) dst[i] += one[i];
            ow = pow_;
            oh = poh;
        }
        ctx.times.postprocess += ms_since(t);
        ++ctx.units_done;
        if (req.progress_fn && req.progress_fn(req.user, static_cast<float>(ctx.units_done) / static_cast<float>(ctx.samples))) return DMS_ERR_ABORTED;
    }
    const auto t = Clock::now();
    const float inv = 1.f / static_cast<float>(ctx.samples);
    for (auto& v : acc)
        for (auto& x : v) x *= inv;

    // Cross-window alignment: frames this window shares with the previous one have anchors (aligned,
    // small planes keyed by frame content). Fit one affine (scale/shift) for the whole window.
    constexpr int kAnchorSide = 128;
    const int aw = std::max(1, ow * kAnchorSide / std::max(ow, oh));
    const int ah = std::max(1, oh * kAnchorSide / std::max(ow, oh));
    std::vector<uint64_t> fps(static_cast<size_t>(count));
    AffineAccumulator fit;
    std::vector<float> small(static_cast<size_t>(aw) * static_cast<size_t>(ah));
    for (int k = 0; k < count; ++k) {
        const FrameView& fv = ctx.frames[static_cast<size_t>(first) + static_cast<size_t>(k)];
        fps[static_cast<size_t>(k)] = frame_fingerprint(fv.rgb, fv.width, fv.height, fv.stride);
        PlaneStore::Plane anchor;
        if (anchors_.get(ctx.spec.id + "|" + std::to_string(fps[static_cast<size_t>(k)]), anchor) && anchor.w == aw && anchor.h == ah) {
            resize_bilinear(acc[static_cast<size_t>(k)].data(), ow, oh, small.data(), aw, ah);
            fit.add(small.data(), anchor.data->data(), small.size());
        }
    }
    Affine af;
    if (fit.count() > 0 && fit.solve(af)) {
        for (auto& v : acc)
            for (auto& x : v) x = static_cast<float>(af.s * static_cast<double>(x) + af.b);
        logf(DMS_LOG_DEBUG, "temporal window aligned to previous window: scale %.4f shift %.4f (%zu samples)", af.s, af.b, fit.count());
    }
    for (int k = 0; k < count; ++k) {
        auto pl = std::make_shared<std::vector<float>>(static_cast<size_t>(aw) * static_cast<size_t>(ah));
        resize_bilinear(acc[static_cast<size_t>(k)].data(), ow, oh, pl->data(), aw, ah);
        anchors_.put(ctx.spec.id + "|" + std::to_string(fps[static_cast<size_t>(k)]), PlaneStore::Plane{pl, aw, ah});
        // Every frame of the window is kept so its own render does not re-run the model.
        if (!ctx.temporal_key_prefix.empty()) {
            const dms_frame& fr = req.frames[first + k];
            temporal_store_.put(ctx.temporal_key_prefix + std::to_string(ctx.frames[static_cast<size_t>(first) + static_cast<size_t>(k)].stride) + "|" +
                                    std::to_string(fr.time_num) + "/" + std::to_string(fr.time_den),
                                PlaneStore::Plane{std::make_shared<const std::vector<float>>(acc[static_cast<size_t>(k)]), ow, oh});
        }
    }
    ctx.depth.resize(pixel_count(ctx.w, ctx.h));
    resize_bilinear(acc[static_cast<size_t>(center)].data(), ow, oh, ctx.depth.data(), ctx.w, ctx.h);
    ctx.times.postprocess += ms_since(t);
    return DMS_OK;
}

dms_status Engine::compute(const dms_depth_request& req, dms_depth_result& res) {
    const auto t_start = Clock::now();

    // ---- validation (never trust the caller) ----
    if (!req.model_id || !*req.model_id) fail(DMS_ERR_INVALID_ARG, "model_id is empty");
    if (!req.frames || req.frame_count <= 0 || req.frame_count > 64 || req.center_index < 0 || req.center_index >= req.frame_count)
        fail(DMS_ERR_INVALID_ARG, "invalid frames / frame_count / center_index");
    const dms_frame& center = req.frames[req.center_index];
    check_dims(center.width, center.height, "frame");
    for (int32_t i = 0; i < req.frame_count; ++i) validate_frame(req.frames[i], center.width, center.height);
    // The guide field is the last member: only read it if the caller's struct is new enough to contain it.
    constexpr size_t kGuideEnd = offsetof(dms_depth_request, guide) + sizeof(void*);
    const bool has_guide = req.struct_size >= kGuideEnd && req.guide != nullptr;
    if (has_guide) validate_frame(*req.guide, req.guide->width, req.guide->height);
    const int w = has_guide ? req.guide->width : center.width;
    const int h = has_guide ? req.guide->height : center.height;
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
    ctx.guide = has_guide ? frame_view(*req.guide) : frame_view(center);
    ctx.center = req.center_index;
    ctx.memopt = have_backend && req.backend.memory_optimization != 0 && spec->kind == DMS_MODEL_KIND_TEMPORAL;
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
    // Temporal models always run untiled (the window already bounds memory; tiles would break coherence).
    pin.tiles = spec->kind == DMS_MODEL_KIND_TEMPORAL ? 1 : resolve_tiles(have_backend ? req.backend.tiles : 0, w, h, base_res);
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

    // Temporal models: a previous window may already contain this frame (pre post-processing).
    if (spec->kind == DMS_MODEL_KIND_TEMPORAL && cacheable) {
        ctx.temporal_key_prefix = spec->id + "|" + std::to_string(req.cache_key.layer_hash) + "|" + std::to_string(base_res) + "|" +
                                  std::to_string(pin.fp16 ? 1 : 0) + "|" + std::to_string(ctx.samples) + "|" + std::to_string(ctx.memopt ? 1 : 0) + "|";
        PlaneStore::Plane pl;
        const std::string k = ctx.temporal_key_prefix + std::to_string(ctx.frames[static_cast<size_t>(ctx.center)].stride) + "|" +
                              std::to_string(req.frames[req.center_index].time_num) + "/" + std::to_string(req.frames[req.center_index].time_den);
        if (temporal_store_.get(k, pl)) {
            ctx.depth.resize(pixel_count(w, h));
            resize_bilinear(pl.data->data(), pl.w, pl.h, ctx.depth.data(), w, h);
            ctx.from_temporal_store = true;
        }
    }

    // ---- execute the plan (fallback chain) ----
    std::string trail;
    int steps = 0;
    dms_status last = DMS_ERR_BACKEND_UNAVAILABLE;
    std::string last_err;
    bool ok = ctx.from_temporal_store;
    for (size_t i = 0; i < plan.size() && !ok; ++i) {
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

    // ---- edge enhancement / subpixel smoothing (full resolution, part of the cached result) ----
    auto t = Clock::now();
    const float edges = clampf(req.edges_enhancement, 0.f, 100.f);
    const float smoothing = clampf(req.subpixel_smoothing, 0.f, 100.f);
    if (edges > 0.f) refine_edges(ctx.depth.data(), ctx.guide.rgb, ctx.guide.stride, w, h, edges, ctx.depth.data());
    if (smoothing > 0.f) smooth_depth(ctx.depth.data(), w, h, smoothing, ctx.depth.data());
    ctx.times.postprocess += ms_since(t);

    // ---- copy out ----
    t = Clock::now();
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

    res.from_cache = ctx.from_temporal_store ? 1 : 0;
    res.backend_used = ctx.from_temporal_store ? DMS_BACKEND_NONE : static_cast<int32_t>(ctx.used);
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
