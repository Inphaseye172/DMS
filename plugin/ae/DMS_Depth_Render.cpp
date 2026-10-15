// SmartFX PRE_RENDER / SMART_RENDER and the legacy RENDER path (Premiere Pro / non-smart hosts).
// Rules enforced here:
//   - every checked-out layer is checked back in on every path (SmartLayerCheckout RAII),
//   - at most kMaxWindow neighbor frames are requested from the host,
//   - only the requested sub-rect is written,
//   - PF_ABORT / PF_PROGRESS are honoured (engine polls them between tiles and samples),
//   - aborted renders return PF_Interrupt_CANCEL; failures still produce a valid output buffer.
#include "DMS_Depth.h"

#include "ae_raii.h"

#include <algorithm>
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
    pre->time_step = in_data->time_step;
    pre->time_scale = in_data->time_scale;

    // Temporal window size (bounded). Single-frame models use exactly one checkout.
    int32_t window = 1;
    if (model_is_temporal(pre->params.model_index) && g.engine && in_data->time_step != 0) {
        dms_model_info mi{};
        mi.struct_size = sizeof(mi);
        if (dms_model_info_by_id(g.engine, model_id(pre->params.model_index), &mi) == DMS_OK) window = mi.window_size;
        window = std::clamp(window, 1, kMaxWindow);
    }
    const int32_t half = (window - 1) / 2;

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
        const A_long t = in_data->current_time + static_cast<A_long>(k - half) * in_data->time_step;
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
    extra->output->pre_render_data = pre.release();  // host owns it now
    extra->output->delete_pre_render_data_func = DeletePreRenderData;
    return PF_Err_NONE;
}

PF_Err smart_render(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    if (!extra || !extra->input || !extra->cb) return PF_Err_BAD_CALLBACK_PARAM;
    const auto* pre = static_cast<const PreRenderData*>(extra->input->pre_render_data);
    if (!pre || pre->magic != kPreRenderMagic) return PF_Err_BAD_CALLBACK_PARAM;
    const GlobalData g = read_global(in_data);

    // Check out every frame requested in pre-render; RAII checks each one back in.
    std::vector<std::unique_ptr<SmartLayerCheckout>> checkouts;
    checkouts.reserve(static_cast<size_t>(pre->frame_count));
    for (int32_t k = 0; k < pre->frame_count; ++k)
        checkouts.push_back(std::make_unique<SmartLayerCheckout>(in_data, extra, kCheckoutBase + k));

    PF_EffectWorld* output = nullptr;
    PF_Err err = extra->cb->checkout_output(in_data->effect_ref, &output);
    if (err) return err;
    if (!output) return PF_Err_NONE;  // nothing to render (empty result rect)

    PixelDepth out_depth;
    if (!pixel_depth_of(in_data, output, out_depth)) {
        report_message(in_data, out_data, "DMS: unsupported output pixel format");
        return PF_Err_NONE;
    }
    RenderJob job;
    job.params = pre->params;
    job.output = view_of(output, out_depth);
    job.input_rect = pre->input_rect;
    job.output_rect = pre->output_rect;
    job.time_num = pre->time;
    job.time_den = static_cast<int64_t>(pre->time_scale ? pre->time_scale : 1);
    for (size_t k = 0; k < checkouts.size(); ++k) {
        PixelDepth d;
        PF_EffectWorld* w = checkouts[k]->world();
        if (w && pixel_depth_of(in_data, w, d)) job.frames.push_back(view_of(w, d));
        else job.frames.push_back(WorldView{});  // missing neighbor (outside layer time): render core drops it
    }
    job.center_index = std::clamp(pre->center_index, 0, static_cast<int32_t>(job.frames.size()) - 1);
    read_sequence_guid(in_data, job.instance_guid);
    job.abort_fn = abort_cb;
    job.progress_fn = progress_cb;
    job.callback_user = in_data;

    if (!g.engine) {
        fill_world(job.output, 0.5f, 1.f);
        report_message(in_data, out_data, "DMS: engine failed to initialize - see log in %LOCALAPPDATA%\\DMS\\logs");
        return PF_Err_NONE;
    }
    return finish(in_data, out_data, render(g.engine, job));
}

PF_Err legacy_render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output) {
    if (!params || !params[P_INPUT] || !output) return PF_Err_BAD_CALLBACK_PARAM;
    const GlobalData g = read_global(in_data);
    PF_EffectWorld* input = &params[P_INPUT]->u.ld;
    PixelDepth in_depth, out_depth;
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
