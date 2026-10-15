// Internal declarations shared by the AE glue translation units.
#pragma once

#include "ae_includes.h"
#include "dms_version.h"
#include "params_model.h"
#include "render_core.h"
#include "sequence_data.h"

#include "dms/engine.h"

#include <functional>
#include <string>

namespace dms_ae {

// in_data->appl_id of Premiere Pro ('PrMr'), spelled out to avoid multi-character literals.
constexpr A_long kPremiereApplId = (static_cast<A_long>('P') << 24) | (static_cast<A_long>('r') << 16) | (static_cast<A_long>('M') << 8) | 'r';

// Global data (PF_OutData::global_data). Written once in GLOBAL_SETUP, read-only afterwards (MFR safe).
struct GlobalData {
    uint32_t magic;            // 'DMSG'
    AEGP_PluginID plugin_id;   // AE only
    int32_t is_premiere;
    dms_engine* engine;        // process-wide engine (refcounted inside the engine)
};
constexpr uint32_t kGlobalMagic = 0x47534D44u;

// Returns a copy of the global data (or zeros if unavailable). Never throws.
GlobalData read_global(PF_InData* in_data) noexcept;

// Writes `msg` into out_data->return_msg via the host's sprintf and asks AE to display it.
void report_message(PF_InData* in_data, PF_OutData* out_data, const std::string& msg) noexcept;

// Logs through the engine log (if available).
void log_line(PF_InData* in_data, int level, const std::string& msg) noexcept;

// Params (DMS_Depth_Params.cpp)
PF_Err params_setup(PF_InData* in_data, PF_OutData* out_data);
dms_plugin::RawParamValues read_params_array(PF_InData* in_data, PF_ParamDef* params[]);
dms_plugin::RawParamValues read_params_checkout(PF_InData* in_data);
PF_Err update_params_ui(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]);
PF_Err query_dynamic_flags(PF_InData* in_data, PF_OutData* out_data);
bool gpu_is_nvidia(dms_engine* engine) noexcept;

// Render (DMS_Depth_Render.cpp)
PF_Err smart_pre_render(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra);
PF_Err smart_render(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra);
PF_Err legacy_render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output);
// Shared SmartFX render: `acquire` turns a checked-out world into a host-readable view (CPU: direct;
// GPU: download), `commit` publishes the host output (CPU: no-op; GPU: upload).
using WorldAcquire = std::function<bool(PF_EffectWorld*, dms_plugin::WorldView&)>;
using OutputCommit = std::function<bool(PF_EffectWorld*, const dms_plugin::WorldView&)>;
PF_Err smart_render_common(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool gpu, const WorldAcquire& acquire,
                           const OutputCommit& commit);

// GPU (DMS_Depth_GPU.cpp)
bool gpu_framework_supported(PF_GPU_Framework fw) noexcept;
PF_Err gpu_device_setup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra);
PF_Err gpu_device_setdown(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetdownExtra* extra);
PF_Err smart_render_gpu(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra);

// Dialogs (DMS_Depth_Dialog.cpp) — UI thread only, never called from render.
PF_Err do_options_dialog(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]);
// Shows a modal download dialog with progress + cancel for `model_id`. Returns true when installed.
bool run_download_dialog(dms_engine* engine, const char* model_id, const char* display_name);

// Sequence data access for render threads (PF_EffectSequenceDataSuite1, read-only).
bool read_sequence_guid(PF_InData* in_data, uint8_t guid[16]) noexcept;

}  // namespace dms_ae
