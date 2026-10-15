// Internal declarations shared by the AE glue translation units.
#pragma once

#include "ae_includes.h"
#include "dms_version.h"
#include "params_model.h"
#include "render_core.h"
#include "sequence_data.h"

#include "dms/engine.h"

#include <string>

namespace dms_ae {

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

// Dialogs (DMS_Depth_Dialog.cpp) — UI thread only, never called from render.
PF_Err do_options_dialog(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]);
// Shows a modal download dialog with progress + cancel for `model_id`. Returns true when installed.
bool run_download_dialog(dms_engine* engine, const char* model_id, const char* display_name);

// Sequence data access for render threads (PF_EffectSequenceDataSuite1, read-only).
bool read_sequence_guid(PF_InData* in_data, uint8_t guid[16]) noexcept;

}  // namespace dms_ae
