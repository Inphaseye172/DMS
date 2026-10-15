// Parameter registration, reading, supervision (show/hide), and dynamic flags.
//
// NEVER change parameter count or order after release: append only, bump DMS_PARAMS_VERSION and PF_VERSION.
#include "DMS_Depth.h"

#include "ae_raii.h"

#include <memory>
#include <string>

namespace dms_ae {

using namespace dms_plugin;

namespace {

constexpr PF_ParamFlags kNoKeyframes = PF_ParamFlag_CANNOT_TIME_VARY;

std::string colormap_popup_string() {
    std::string s;
    const int n = dms_colormap_count();
    for (int i = 0; i < n; ++i) {
        if (i) s += '|';
        const char* name = dms_colormap_name(i);
        s += (name && *name) ? name : "?";
    }
    return s.empty() ? std::string("None") : s;
}

double fixed_to_double(PF_Fixed f) { return static_cast<double>(f) / 65536.0; }

}  // namespace

PF_Err params_setup(PF_InData* in_data, PF_OutData* out_data) {
    PF_ParamDef def;
    // Strings must outlive PF_ADD_PARAM; AE copies popup strings during the call.
    const std::string colormaps = colormap_popup_string();
    const int colormap_count = dms_colormap_count() > 0 ? dms_colormap_count() : 1;

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Model", static_cast<A_short>(kModelIds.size()), 1, kModelPopup, PF_ParamFlag_SUPERVISE | kNoKeyframes, ID_MODEL);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Base Resolution", static_cast<A_short>(kBaseRes.size()), 2, kBaseResPopup, kNoKeyframes, ID_BASE_RES);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Edges Enhancement", 0, 100, 0, 100, 0, PF_Precision_TENTHS, PF_ValueDisplayFlag_NONE, 0, ID_EDGES);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Subpixel Smoothing", 0, 100, 0, 100, 0, PF_Precision_TENTHS, PF_ValueDisplayFlag_NONE, 0, ID_SMOOTHING);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Mapping Mode", 3, 2, kMappingPopup, PF_ParamFlag_SUPERVISE, ID_MAPPING);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Near Guide Point", 50, 90, false, ID_NEAR_POINT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT("Far Guide Point", 50, 10, false, ID_FAR_POINT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Near Value", 0, 1, 0, 1, 1, PF_Precision_THOUSANDTHS, PF_ValueDisplayFlag_NONE, 0, ID_NEAR_VALUE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Far Value", 0, 1, 0, 1, 0, PF_Precision_THOUSANDTHS, PF_ValueDisplayFlag_NONE, 0, ID_FAR_VALUE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Mode", 2, 1, kModePopup, PF_ParamFlag_SUPERVISE, ID_MODE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Colormap", static_cast<A_short>(colormap_count), 1, colormaps.c_str(), 0, ID_COLORMAP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Minimum", 0, 1, 0, 1, 0, PF_Precision_THOUSANDTHS, PF_ValueDisplayFlag_NONE, 0, ID_SLICE_MIN);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Maximum", 0, 1, 0, 1, 1, PF_Precision_THOUSANDTHS, PF_ValueDisplayFlag_NONE, 0, ID_SLICE_MAX);

    AEFX_CLR_STRUCT(def);
    def.flags = PF_ParamFlag_START_COLLAPSED;
    PF_ADD_TOPIC("Backend & Performance", ID_BACKEND_TOPIC);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Hardware Acceleration", TRUE, PF_ParamFlag_SUPERVISE | kNoKeyframes, ID_HW_ACCEL);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Lower Precision (FP16)", FALSE, kNoKeyframes, ID_FP16);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("CUDA Memory Sharing", FALSE, kNoKeyframes, ID_CUDA_MEMSHARE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Model Offloading", 3, 1, kOffloadPopup, kNoKeyframes, ID_OFFLOAD);
    AEFX_CLR_STRUCT(def);
    def.flags = PF_ParamFlag_SUPERVISE | kNoKeyframes;
    PF_ADD_SLIDER("Samples", 1, 16, 1, 16, 1, ID_SAMPLES);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Parallel Samples", FALSE, kNoKeyframes, ID_PARALLEL);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Computation Tiles", static_cast<A_short>(kTiles.size()), 1, kTilesPopup, kNoKeyframes, ID_TILES);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Memory Optimization", FALSE, kNoKeyframes, ID_MEMOPT);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_BACKEND_TOPIC_END);

    out_data->num_params = P_NUM_PARAMS;
    return PF_Err_NONE;
}

namespace {
// Reads one param definition into the raw struct (shared by params[] and checkout paths).
void read_one(RawParamValues& r, int index, const PF_ParamDef& d) {
    switch (index) {
        case P_MODEL: r.model_popup = d.u.pd.value; break;
        case P_BASE_RES: r.base_res_popup = d.u.pd.value; break;
        case P_EDGES: r.edges = d.u.fs_d.value; break;
        case P_SMOOTHING: r.smoothing = d.u.fs_d.value; break;
        case P_MAPPING: r.mapping_popup = d.u.pd.value; break;
        case P_NEAR_POINT: r.near_x = fixed_to_double(d.u.td.x_value); r.near_y = fixed_to_double(d.u.td.y_value); break;
        case P_FAR_POINT: r.far_x = fixed_to_double(d.u.td.x_value); r.far_y = fixed_to_double(d.u.td.y_value); break;
        case P_NEAR_VALUE: r.near_value = d.u.fs_d.value; break;
        case P_FAR_VALUE: r.far_value = d.u.fs_d.value; break;
        case P_MODE: r.mode_popup = d.u.pd.value; break;
        case P_COLORMAP: r.colormap_popup = d.u.pd.value; break;
        case P_SLICE_MIN: r.slice_min = d.u.fs_d.value; break;
        case P_SLICE_MAX: r.slice_max = d.u.fs_d.value; break;
        case P_HW_ACCEL: r.hw_accel = d.u.bd.value; break;
        case P_FP16: r.fp16 = d.u.bd.value; break;
        case P_CUDA_MEMSHARE: r.cuda_memshare = d.u.bd.value; break;
        case P_OFFLOAD: r.offload_popup = d.u.pd.value; break;
        case P_SAMPLES: r.samples = d.u.sd.value; break;
        case P_PARALLEL: r.parallel = d.u.bd.value; break;
        case P_TILES: r.tiles_popup = d.u.pd.value; break;
        case P_MEMOPT: r.memopt = d.u.bd.value; break;
        default: break;
    }
}
}  // namespace

RawParamValues read_params_array(PF_InData* in_data, PF_ParamDef* params[]) {
    (void)in_data;
    RawParamValues r;
    if (!params) return r;
    for (int i = P_MODEL; i < P_NUM_PARAMS; ++i)
        if (params[i]) read_one(r, i, *params[i]);
    return r;
}

RawParamValues read_params_checkout(PF_InData* in_data) {
    RawParamValues r;
    for (int i = P_MODEL; i < P_NUM_PARAMS; ++i) {
        if (i == P_BACKEND_TOPIC || i == P_BACKEND_TOPIC_END) continue;
        ParamCheckout pc(in_data, i);  // checked back in at end of scope, on every path
        if (pc.ok()) read_one(r, i, pc.def());
    }
    return r;
}

bool gpu_is_nvidia(dms_engine* engine) noexcept {
    if (!engine) return false;
    int32_t vendor = DMS_GPU_UNKNOWN;  // GPU probe only: never loads ONNX Runtime on the UI thread
    return dms_gpu_vendor_query(engine, &vendor) == DMS_OK && vendor == DMS_GPU_NVIDIA;
}

PF_Err update_params_ui(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]) {
    if (!params) return PF_Err_NONE;
    const GlobalData g = read_global(in_data);
    const RawParamValues raw = read_params_array(in_data, params);
    const ParamUiState ui = compute_ui_state(raw, gpu_is_nvidia(g.engine));

    AEFX_SuiteScoper<DmsParamUtilsSuite> pu(in_data, DMS_PARAMUTILS_SUITE, out_data);

    // Enabled/disabled via PF_PUI_DISABLED (works in AE and Premiere).
    // Visibility: AE uses AEGP dynamic stream flags; Premiere honours PF_PUI_INVISIBLE.
    std::unique_ptr<AegpEffectRef> effect;
    if (!g.is_premiere && g.plugin_id) effect = std::make_unique<AegpEffectRef>(in_data, g.plugin_id);

    for (int i = P_MODEL; i < P_NUM_PARAMS; ++i) {
        if (i == P_BACKEND_TOPIC || i == P_BACKEND_TOPIC_END || !params[i]) continue;
        PF_ParamDef copy = *params[i];
        const PF_ParamUIFlags before = copy.ui_flags;
        if (ui.enabled[static_cast<size_t>(i)]) copy.ui_flags &= ~PF_PUI_DISABLED;
        else copy.ui_flags |= PF_PUI_DISABLED;
        if (g.is_premiere) {
            if (ui.visible[static_cast<size_t>(i)]) copy.ui_flags &= ~PF_PUI_INVISIBLE;
            else copy.ui_flags |= PF_PUI_INVISIBLE;
        }
        if (copy.ui_flags != before) {
            PF_Err err = pu->PF_UpdateParamUI(in_data->effect_ref, i, &copy);
            if (err) return err;
        }
        if (effect && effect->get()) {
            AegpStreamRef stream(in_data, g.plugin_id, effect->get(), i);
            if (stream.get()) {
                AEFX_SuiteScoper<DmsDynStreamSuite> dyn(in_data, DMS_DYNSTREAM_SUITE, out_data);
                dyn->AEGP_SetDynamicStreamFlag(stream.get(), AEGP_DynStreamFlag_HIDDEN, FALSE, ui.visible[static_cast<size_t>(i)] ? FALSE : TRUE);
            }
        }
    }
    return PF_Err_NONE;
}

PF_Err query_dynamic_flags(PF_InData* in_data, PF_OutData* out_data) {
    // NON_PARAM_VARY only for the temporal model (its output depends on neighbor frames / window state);
    // single-frame models keep it cleared so AE's frame cache stays effective.
    ParamCheckout model(in_data, P_MODEL);
    const int32_t popup = model.ok() ? model.def().u.pd.value : 1;
    const int32_t index = (popup >= 1 && popup <= static_cast<int32_t>(kModelIds.size())) ? popup - 1 : 0;
    if (model_is_temporal(index)) out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
    else out_data->out_flags &= ~PF_OutFlag_NON_PARAM_VARY;
    return PF_Err_NONE;
}

}  // namespace dms_ae
