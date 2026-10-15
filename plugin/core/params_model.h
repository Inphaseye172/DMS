// AE-independent parameter model. The order of ParamIndex values IS the on-disk parameter order:
// NEVER reorder or remove entries after the first release. Append only, and bump DMS_PARAMS_VERSION
// (and the effect PF_VERSION) when adding.
#pragma once

#include <array>
#include <cstdint>

namespace dms_plugin {

constexpr uint32_t DMS_PARAMS_VERSION = 1;

enum ParamIndex : int {
    P_INPUT = 0,
    P_MODEL,             // popup
    P_BASE_RES,          // popup 384/518/768/1024
    P_EDGES,             // float 0..100
    P_SMOOTHING,         // float 0..100
    P_MAPPING,           // popup Raw/Normalize/Guided
    P_NEAR_POINT,        // point (guided)
    P_FAR_POINT,         // point (guided)
    P_NEAR_VALUE,        // float 0..1 (guided)
    P_FAR_VALUE,         // float 0..1 (guided)
    P_MODE,              // popup Depth Map/Slicing
    P_COLORMAP,          // popup (depth map)
    P_SLICE_MIN,         // float 0..1 (slicing)
    P_SLICE_MAX,         // float 0..1 (slicing)
    P_BACKEND_TOPIC,     // topic start "Backend & Performance"
    P_HW_ACCEL,          // checkbox
    P_FP16,              // checkbox
    P_CUDA_MEMSHARE,     // checkbox (NVIDIA only)
    P_OFFLOAD,           // popup Keep in VRAM/Offload to RAM/Full Unload
    P_SAMPLES,           // int slider 1..16
    P_PARALLEL,          // checkbox (Samples > 2)
    P_TILES,             // popup Auto/1/4/9/16
    P_MEMOPT,            // checkbox (video model only)
    P_BACKEND_TOPIC_END, // topic end
    P_NUM_PARAMS
};

// Stable disk IDs (PF_ParamDef::uu.id). Must never change for a given parameter.
enum ParamDiskId : int {
    ID_MODEL = 1, ID_BASE_RES, ID_EDGES, ID_SMOOTHING, ID_MAPPING, ID_NEAR_POINT, ID_FAR_POINT, ID_NEAR_VALUE, ID_FAR_VALUE,
    ID_MODE, ID_COLORMAP, ID_SLICE_MIN, ID_SLICE_MAX, ID_BACKEND_TOPIC, ID_HW_ACCEL, ID_FP16, ID_CUDA_MEMSHARE, ID_OFFLOAD,
    ID_SAMPLES, ID_PARALLEL, ID_TILES, ID_MEMOPT, ID_BACKEND_TOPIC_END
};

// Popup choices. Indices are 1-based in AE; these arrays are 0-based.
// Model popup entries map to engine model ids. Append only.
inline constexpr std::array<const char*, 6> kModelIds = {"da2-small", "da2-base", "da2-large", "da3-small", "da3-base", "video-da-small"};
inline constexpr const char* kModelPopup =
    "Depth Anything v2 Small|Depth Anything v2 Base|Depth Anything v2 Large|Depth Anything v3 Small|Depth Anything v3 Base|Video Depth (temporal)";
inline constexpr std::array<int, 4> kBaseRes = {384, 518, 768, 1024};
inline constexpr const char* kBaseResPopup = "384|518|768|1024";
inline constexpr const char* kMappingPopup = "Raw|Normalize|Guided";
inline constexpr const char* kModePopup = "Depth Map|Slicing";
inline constexpr const char* kOffloadPopup = "Keep in VRAM|Offload to RAM|Full Unload";
inline constexpr std::array<int, 5> kTiles = {0, 1, 4, 9, 16};
inline constexpr const char* kTilesPopup = "Auto|1|4|9|16";

enum class Mapping : int32_t { Raw = 0, Normalize = 1, Guided = 2 };
enum class Mode : int32_t { DepthMap = 0, Slicing = 1 };

// Flat, pointer-free snapshot of all render-relevant parameters (also used as pre-render data).
struct ParamSnapshot {
    uint32_t version = DMS_PARAMS_VERSION;
    int32_t model_index = 0;  // 0-based into kModelIds
    int32_t base_res = 518;
    float edges = 0.f;
    float smoothing = 0.f;
    Mapping mapping = Mapping::Normalize;
    float near_x = 0.f, near_y = 0.f;  // input-layer pixel coordinates at the current downsample
    float far_x = 0.f, far_y = 0.f;
    float near_value = 1.f, far_value = 0.f;
    Mode mode = Mode::DepthMap;
    int32_t colormap = 0;  // 0 = None (grayscale)
    float slice_min = 0.f, slice_max = 1.f;
    int32_t hw_accel = 1;
    int32_t fp16 = 0;
    int32_t cuda_memshare = 0;
    int32_t offload = 0;
    int32_t samples = 1;
    int32_t parallel = 0;
    int32_t tiles = 0;  // 0 = auto
    int32_t memopt = 0;
};

// Converts raw UI values (as read from the host) into a sanitized snapshot. Every slider is clamped
// in code regardless of UI limits, NaN/Inf are replaced by defaults, popups are range-checked.
struct RawParamValues {
    int32_t model_popup = 1;  // 1-based
    int32_t base_res_popup = 2;
    double edges = 0, smoothing = 0;
    int32_t mapping_popup = 2;
    double near_x = 0, near_y = 0, far_x = 0, far_y = 0;
    double near_value = 1, far_value = 0;
    int32_t mode_popup = 1;
    int32_t colormap_popup = 1;
    double slice_min = 0, slice_max = 1;
    int32_t hw_accel = 1, fp16 = 0, cuda_memshare = 0;
    int32_t offload_popup = 1;
    int32_t samples = 1;
    int32_t parallel = 0;
    int32_t tiles_popup = 1;
    int32_t memopt = 0;
};

ParamSnapshot sanitize(const RawParamValues& raw, int colormap_count, bool is_nvidia);

bool model_is_temporal(int32_t model_index);
const char* model_id(int32_t model_index);

// Visibility/enabled state for supervised params (PF_Cmd_UPDATE_PARAMS_UI).
struct ParamUiState {
    std::array<bool, P_NUM_PARAMS> visible{};
    std::array<bool, P_NUM_PARAMS> enabled{};
};
ParamUiState compute_ui_state(const RawParamValues& raw, bool is_nvidia);

}  // namespace dms_plugin
