#include "params_model.h"

#include <cmath>

namespace dms_plugin {
namespace {
double fin(double v, double dflt) { return std::isfinite(v) ? v : dflt; }
float clampf(double v, double lo, double hi, double dflt) {
    v = fin(v, dflt);
    return static_cast<float>(v < lo ? lo : (v > hi ? hi : v));
}
int32_t popup(int32_t v, int32_t count, int32_t dflt) { return (v >= 1 && v <= count) ? v : dflt; }
}  // namespace

bool model_is_temporal(int32_t model_index) {
    return model_index >= 0 && model_index < static_cast<int32_t>(kModelIds.size()) &&
           kModelIds[static_cast<size_t>(model_index)][0] == 'v';
}

const char* model_id(int32_t model_index) {
    if (model_index < 0 || model_index >= static_cast<int32_t>(kModelIds.size())) return kModelIds[0];
    return kModelIds[static_cast<size_t>(model_index)];
}

ParamSnapshot sanitize(const RawParamValues& r, int colormap_count, bool is_nvidia) {
    ParamSnapshot s;
    s.model_index = popup(r.model_popup, static_cast<int32_t>(kModelIds.size()), 1) - 1;
    s.base_res = kBaseRes[static_cast<size_t>(popup(r.base_res_popup, static_cast<int32_t>(kBaseRes.size()), 2) - 1)];
    s.edges = clampf(r.edges, 0, 100, 0);
    s.smoothing = clampf(r.smoothing, 0, 100, 0);
    s.mapping = static_cast<Mapping>(popup(r.mapping_popup, 3, 2) - 1);
    // Guide points: reject NaN/Inf; extreme values are clamped to a sane coordinate range.
    s.near_x = clampf(r.near_x, -1e6, 1e6, 0);
    s.near_y = clampf(r.near_y, -1e6, 1e6, 0);
    s.far_x = clampf(r.far_x, -1e6, 1e6, 0);
    s.far_y = clampf(r.far_y, -1e6, 1e6, 0);
    s.near_value = clampf(r.near_value, 0, 1, 1);
    s.far_value = clampf(r.far_value, 0, 1, 0);
    s.mode = static_cast<Mode>(popup(r.mode_popup, 2, 1) - 1);
    s.colormap = popup(r.colormap_popup, colormap_count > 0 ? colormap_count : 1, 1) - 1;
    s.slice_min = clampf(r.slice_min, 0, 1, 0);
    s.slice_max = clampf(r.slice_max, 0, 1, 1);
    if (s.slice_max < s.slice_min) {
        const float t = s.slice_min;
        s.slice_min = s.slice_max;
        s.slice_max = t;
    }
    s.hw_accel = r.hw_accel ? 1 : 0;
    s.fp16 = r.fp16 ? 1 : 0;
    s.cuda_memshare = (r.cuda_memshare && is_nvidia && s.hw_accel) ? 1 : 0;
    s.offload = popup(r.offload_popup, 3, 1) - 1;
    s.samples = r.samples < 1 ? 1 : (r.samples > 16 ? 16 : r.samples);
    s.parallel = (r.parallel && s.samples > 2) ? 1 : 0;
    s.tiles = kTiles[static_cast<size_t>(popup(r.tiles_popup, static_cast<int32_t>(kTiles.size()), 1) - 1)];
    s.memopt = (r.memopt && model_is_temporal(s.model_index)) ? 1 : 0;
    return s;
}

ParamUiState compute_ui_state(const RawParamValues& r, bool is_nvidia) {
    ParamUiState u;
    u.visible.fill(true);
    u.enabled.fill(true);
    const bool guided = r.mapping_popup == 3;
    u.visible[P_NEAR_POINT] = u.visible[P_FAR_POINT] = u.visible[P_NEAR_VALUE] = u.visible[P_FAR_VALUE] = guided;
    const bool slicing = r.mode_popup == 2;
    u.visible[P_COLORMAP] = !slicing;
    u.visible[P_SLICE_MIN] = u.visible[P_SLICE_MAX] = slicing;
    u.visible[P_CUDA_MEMSHARE] = is_nvidia;
    u.enabled[P_CUDA_MEMSHARE] = is_nvidia && r.hw_accel != 0;
    u.enabled[P_FP16] = r.hw_accel != 0;
    u.enabled[P_PARALLEL] = r.samples > 2;
    const int32_t model_index = (r.model_popup >= 1 && r.model_popup <= static_cast<int32_t>(kModelIds.size())) ? r.model_popup - 1 : 0;
    u.enabled[P_MEMOPT] = model_is_temporal(model_index);
    return u;
}

}  // namespace dms_plugin
