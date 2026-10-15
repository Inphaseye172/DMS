// SmartFX PRE_RENDER / SMART_RENDER and the legacy RENDER path (Premiere Pro / non-smart hosts).
// Rules enforced here:
//   - every checked-out layer is checked back in on every path (SmartLayerCheckout RAII),
//   - at most kMaxWindow neighbor frames are requested from the host,
//   - only the requested sub-rect is written,
//   - PF_ABORT / PF_PROGRESS are honoured (engine polls them between tiles and samples),
//   - aborted renders return PF_Interrupt_CANCEL; failures still produce a valid output buffer.
#include "DMS_Depth.h"

#include "ae_raii.h"
#include "pixels.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace dms_ae {

using namespace dms_plugin;

namespace {

constexpr uint32_t kPreRenderMagic = 0x52504D44u;  // 'DMPR'
constexpr int32_t kMaxWindow = 32;                  // hard cap on temporal neighbor checkouts
constexpr A_long kCheckoutBase = 100;               // checkout ids kCheckoutBase + k

struct PreRenderData {
    uint32_t magic = kPreRenderMagic;
    ParamSnapshot params;
    Rect input_rect;
    Rect output_rect;
    int32_t frame_count = 0;   // number of checkouts issued (ids kCheckoutBase .. +frame_count-1)
    int32_t center_index = 0;
    A_long time = 0;
    A_long time_step = 0;
    A_u_long time_scale = 1;
};

void DeletePreRenderData(void* p) {
    // Ownership was released to the host in pre-render; reclaim it into a unique_ptr to destroy.
    std::unique_ptr<PreRenderData> owned(static_cast<PreRenderData*>(p));
}

Rect to_rect(const PF_LRect& r) { return Rect{r.left, r.top, r.right, r.bottom}; }
PF_LRect to_lrect(const Rect& r) {
    PF_LRect o;
    o.left = r.left;
    o.top = r.top;
    o.right = r.right;
    o.bottom = r.bottom;
    return o;
}

int32_t scaled(A_long v, const PF_RationalScale& s) {
    if (s.den == 0) return static_cast<int32_t>(v);
    const double r = static_cast<double>(v) * static_cast<double>(s.num) / static_cast<double>(s.den);
    return static_cast<int32_t>(std::ceil(r));
}

bool pixel_depth_of(PF_InData* in_data, PF_EffectWorld* w, PixelDepth& out) {
    if (!w) return false;
    AEFX_SuiteScoper<DmsWorldSuite> ws(in_data, DMS_WORLD_SUITE);
    PF_PixelFormat fmt = PF_PixelFormat_INVALID;
    if (ws->PF_GetPixelFormat(w, &fmt) != PF_Err_NONE) return false;
    switch (fmt) {
        case PF_PixelFormat_ARGB32: out = PixelDepth::U8; return true;
        case PF_PixelFormat_ARGB64: out = PixelDepth::U16; return true;
        case PF_PixelFormat_ARGB128: out = PixelDepth::F32; return true;
        default: return false;  // unsupported format: caller writes a safe output
    }
}

WorldView view_of(PF_EffectWorld* w, PixelDepth d) {
    WorldView v;
    if (!w) return v;
    v.data = w->data;
    v.width = w->width;
    v.height = w->height;
    v.rowbytes = w->rowbytes;
    v.depth = d;
    return v;
}

int DMS_API abort_cb(void* user) {
    auto* in = static_cast<PF_InData*>(user);
    return (in && PF_ABORT(in) != PF_Err_NONE) ? 1 : 0;
}

int DMS_API progress_cb(void* user, float fraction) {
    auto* in = static_cast<PF_InData*>(user);
    if (!in) return 0;
    const float f = std::isfinite(fraction) ? std::clamp(fraction, 0.f, 1.f) : 0.f;
    return PF_PROGRESS(in, static_cast<A_long>(f * 1000.f), 1000) != PF_Err_NONE ? 1 : 0;
}

PF_Err finish(PF_InData* in_data, PF_OutData* out_data, const RenderOutcome& o) {
    if (o.status == DMS_ERR_ABORTED) return PF_Interrupt_CANCEL;
    if (!o.message.empty()) log_line(in_data, o.status == DMS_OK ? DMS_LOG_DEBUG : DMS_LOG_WARN, o.message);
    if (o.report_to_user && !o.message.empty()) report_message(in_data, out_data, o.message);
    return PF_Err_NONE;
}

}  // namespace

PF_Err smart_pre_render(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra) {
    (void)out_data;
    if (!extra || !extra->input || !extra->output || !extra->cb) return PF_Err_BAD_CALLBACK_PARAM;
    const GlobalData g = read_global(in_data);

    auto pre = std::make_unique<PreRenderData>();
    pre->params = sanitize(read_params_checkout(in_data), dms_colormap_count(), gpu_is_nvidia(g.engine));
    pre->time = in_data->current_time;
    pre->time_step = in_data->time_step < 0 ? -in_data->time_step : in_data->time_step;
    pre->time_scale = in_data->time_scale;

    // Temporal window (bounded, deterministic so neighbouring frames share one inference).
    // Single-frame models use exactly one checkout.
    int64_t frame_index = 0, win_start = 0;
    int32_t window = 1;
    const A_long step = in_data->time_step < 0 ? -in_data->time_step : in_data->time_step;
    if (model_is_temporal(pre->params.model_index) && g.engine && step != 0) {
        frame_index = static_cast<int64_t>(in_data->current_time / step);
        if (dms_temporal_window(g.engine, model_id(pre->params.model_index), frame_index, pre->params.memopt, &win_start, &window) != DMS_OK) {
            win_start = frame_index;
            window = 1;
        }
        window = std::clamp(window, 1, kMaxWindow);
    } else {
        win_start = frame_index;
    }
    const int32_t half = static_cast<int32_t>(std::clamp<int64_t>(frame_index - win_start, 0, window - 1));

    // Depth needs the whole frame: request the full layer extent, not just the output tile.
    PF_RenderRequest req = extra->input->output_request;
    req.rect.left = 0;
    req.rect.top = 0;
    req.rect.right = scaled(in_data->width, in_data->downsample_x);
    req.rect.bottom = scaled(in_data->height, in_data->downsample_y);
    req.preserve_rgb_of_zero_alpha = FALSE;

    PF_CheckoutResult center_result;
    AEFX_CLR_STRUCT(center_result);
    for (int32_t k = 0; k < window; ++k) {
        const A_long t = window == 1 ? in_data->current_time : static_cast<A_long>((win_start + k) * step);
        PF_CheckoutResult res;
        AEFX_CLR_STRUCT(res);
        PF_Err err = extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, kCheckoutBase + k, &req, t, in_data->time_step,
                                               in_data->time_scale, &res);
        if (err) return err;
        pre->frame_count = k + 1;
        if (k == half) center_result = res;
    }
    pre->center_index = half;
    pre->input_rect = to_rect(center_result.result_rect);

    const Rect requested = to_rect(extra->input->output_request.rect);
    const Rect out_rect = intersect(requested, pre->input_rect);
    pre->output_rect = out_rect;
    extra->output->result_rect = to_lrect(out_rect);
    extra->output->max_result_rect = center_result.max_result_rect;
    extra->output->solid = FALSE;
    // GPU rendering is possible only for frameworks we accepted in GPU_DEVICE_SETUP (CUDA).
    if (extra->input->what_gpu != PF_GPU_Framework_NONE && gpu_framework_supported(extra->input->what_gpu))
        extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    extra->output->pre_render_data = pre.release();  // host owns it now
    extra->output->delete_pre_render_data_func = DeletePreRenderData;
    return PF_Err_NONE;
}

PF_Err smart_render_common(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool gpu, const WorldAcquire& acquire,
                           const OutputCommit& commit) {
    if (!extra || !extra->input || !extra->cb) return PF_Err_BAD_CALLBACK_PARAM;
    const auto* pre = static_cast<const PreRenderData*>(extra->input->pre_render_data);
    if (!pre || pre->magic != kPreRenderMagic) return PF_Err_BAD_CALLBACK_PARAM;
    const GlobalData g = read_global(in_data);

    const auto t_checkout = std::chrono::steady_clock::now();
    // Check out every frame requested in pre-render; RAII checks each one back in on every path.
    std::vector<std::unique_ptr<SmartLayerCheckout>> checkouts;
    checkouts.reserve(static_cast<size_t>(pre->frame_count));
    for (int32_t k = 0; k < pre->frame_count; ++k)
        checkouts.push_back(std::make_unique<SmartLayerCheckout>(in_data, extra, kCheckoutBase + k));

    PF_EffectWorld* output = nullptr;
    PF_Err err = extra->cb->checkout_output(in_data->effect_ref, &output);
    if (err) return err;
    if (!output) return PF_Err_NONE;  // nothing to render (empty result rect)

    RenderJob job;
    std::vector<uint8_t> host_out;  // GPU path: host staging for the output world
    if (gpu) {
        if (output->width <= 0 || output->height <= 0 || output->rowbytes < output->width * static_cast<A_long>(sizeof(PixelBGRA)))
            return PF_Err_NONE;
        host_out.assign(static_cast<size_t>(output->rowbytes) * static_cast<size_t>(output->height), 0);
        job.output.data = host_out.data();
        job.output.width = output->width;
        job.output.height = output->height;
        job.output.rowbytes = output->rowbytes;
        job.output.depth = PixelDepth::F32_BGRA;
    } else {
        PixelDepth out_depth = PixelDepth::U8;
        if (!pixel_depth_of(in_data, output, out_depth)) {
            report_message(in_data, out_data, "DMS: unsupported output pixel format");
            return PF_Err_NONE;
        }
        job.output = view_of(output, out_depth);
    }
    job.params = pre->params;
    job.input_rect = pre->input_rect;
    job.output_rect = pre->output_rect;
    job.time_num = pre->time;
    job.time_den = static_cast<int64_t>(pre->time_scale ? pre->time_scale : 1);
    job.time_step = pre->time_step ? pre->time_step : 1;
    for (size_t k = 0; k < checkouts.size(); ++k) {
        WorldView v;
        if (!acquire(checkouts[k]->world(), v)) v = WorldView{};  // missing neighbour: render core drops it
        job.frames.push_back(v);
    }
    job.center_index = std::clamp(pre->center_index, 0, static_cast<int32_t>(job.frames.size()) - 1);
    read_sequence_guid(in_data, job.instance_guid);
    job.abort_fn = abort_cb;
    job.progress_fn = progress_cb;
    job.callback_user = in_data;
    const float ms_checkout = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_checkout).count();

    RenderOutcome outcome;
    if (!g.engine) {
        fill_world(job.output, 0.5f, 1.f);
        outcome.status = DMS_ERR_DEGRADED;
        outcome.output_written = true;
        outcome.report_to_user = true;
        outcome.message = "DMS: engine failed to initialize - see the log in %LOCALAPPDATA%\\DMS\\logs";
    } else {
        outcome = render(g.engine, job);
    }
    const auto t_commit = std::chrono::steady_clock::now();
    if (outcome.status != DMS_ERR_ABORTED && !commit(output, job.output)) {
        log_line(in_data, DMS_LOG_ERROR, "failed to publish the output world (GPU upload)");
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    const float ms_commit = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_commit).count();
    if (g.engine) {
        char line[256];
        std::snprintf(line, sizeof(line),
                      "render %dx%d frames=%zu%s | checkout %.1f ms, convert %.1f ms, engine %.1f ms, copy-out %.1f ms, upload %.1f ms",
                      job.output.width, job.output.height, job.frames.size(), gpu ? " (GPU)" : "", static_cast<double>(ms_checkout),
                      static_cast<double>(outcome.ms_convert), static_cast<double>(outcome.ms_engine), static_cast<double>(outcome.ms_copyout),
                      static_cast<double>(ms_commit));
        log_line(in_data, DMS_LOG_DEBUG, line);
    }
    return finish(in_data, out_data, outcome);
}

PF_Err smart_render(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    return smart_render_common(
        in_data, out_data, extra, /*gpu=*/false,
        [in_data](PF_EffectWorld* w, WorldView& v) {
            PixelDepth d = PixelDepth::U8;
            if (!w || !pixel_depth_of(in_data, w, d)) return false;
            v = view_of(w, d);
            return true;
        },
        [](PF_EffectWorld*, const WorldView&) { return true; });
}

PF_Err legacy_render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output) {
    if (!params || !params[P_INPUT] || !output) return PF_Err_BAD_CALLBACK_PARAM;
    const GlobalData g = read_global(in_data);
    PF_EffectWorld* input = &params[P_INPUT]->u.ld;
    PixelDepth in_depth = PixelDepth::U8, out_depth = PixelDepth::U8;
    if (!pixel_depth_of(in_data, output, out_depth)) return PF_Err_NONE;
    RenderJob job;
    job.params = sanitize(read_params_array(in_data, params), dms_colormap_count(), gpu_is_nvidia(g.engine));
    job.output = view_of(output, out_depth);
    if (pixel_depth_of(in_data, input, in_depth)) job.frames.push_back(view_of(input, in_depth));
    else job.frames.push_back(WorldView{});
    job.center_index = 0;
    job.input_rect = Rect{0, 0, input->width, input->height};
    job.output_rect = Rect{0, 0, output->width, output->height};
    job.time_num = in_data->current_time;
    job.time_den = in_data->time_scale ? in_data->time_scale : 1;
    read_sequence_guid(in_data, job.instance_guid);
    job.abort_fn = abort_cb;
    job.progress_fn = progress_cb;
    job.callback_user = in_data;
    if (!g.engine) {
        fill_world(job.output, 0.5f, 1.f);
        return PF_Err_NONE;
    }
    return finish(in_data, out_data, render(g.engine, job));
}

}  // namespace dms_ae
