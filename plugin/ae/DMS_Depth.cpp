// After Effects entry point and command dispatch.
//
// Stability contract: every PF_Cmd handler runs inside dispatch()'s try/catch. No exception escapes
// to the host; failures become a PF_Err plus a message in out_data->return_msg, and are logged.
#include "DMS_Depth.h"

#include "ae_raii.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <new>

namespace dms_ae {

static_assert(DMS_PIPL_OUTFLAGS ==
                  (PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_I_DO_DIALOG | PF_OutFlag_DEEP_COLOR_AWARE |
                   PF_OutFlag_SEND_UPDATE_PARAMS_UI),
              "PiPL outflags out of sync with GLOBAL_SETUP");
static_assert(DMS_PIPL_OUTFLAGS2 ==
                  (PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
                   PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                   PF_OutFlag2_SUPPORTS_GET_FLATTENED_SEQUENCE_DATA | PF_OutFlag2_SUPPORTS_THREADED_RENDERING),
              "PiPL outflags2 out of sync with GLOBAL_SETUP");
static_assert(DMS_PIPL_EFFECT_VERSION ==
                  PF_VERSION(DMS_MAJOR_VERSION, DMS_MINOR_VERSION, DMS_BUG_VERSION, DMS_STAGE_VERSION, DMS_BUILD_VERSION),
              "PiPL effect version out of sync");

GlobalData read_global(PF_InData* in_data) noexcept {
    GlobalData g{};
    try {
        if (!in_data || !in_data->global_data) return g;
        HandleLock lk(in_data, in_data->global_data);
        if (lk.data() && lk.size() >= sizeof(GlobalData)) {
            std::memcpy(&g, lk.data(), sizeof(GlobalData));
            if (g.magic != kGlobalMagic) g = GlobalData{};
        }
    } catch (...) {
        g = GlobalData{};
    }
    return g;
}

void log_line(PF_InData* in_data, int level, const std::string& msg) noexcept {
    try {
        const GlobalData g = read_global(in_data);
        if (g.engine) dms_log(g.engine, level, msg.c_str());
    } catch (...) {
    }
}

void report_message(PF_InData* in_data, PF_OutData* out_data, const std::string& msg) noexcept {
    if (!in_data || !out_data) return;
    // return_msg holds PF_MAX_EFFECT_MSG_LEN+1 chars and the host sprintf is unbounded: truncate first.
    char buf[PF_MAX_EFFECT_MSG_LEN - 8];
    std::snprintf(buf, sizeof(buf), "%s", msg.c_str());
    if (in_data->utils && in_data->utils->ansi.sprintf) {
        in_data->utils->ansi.sprintf(out_data->return_msg, "%s", buf);
    } else {
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s", buf);
    }
    out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
    log_line(in_data, DMS_LOG_WARN, msg);
}

bool read_sequence_guid(PF_InData* in_data, uint8_t guid[16]) noexcept {
    std::memset(guid, 0, 16);
    try {
        AEFX_SuiteScoper<DmsSeqDataSuite> seq(in_data, DMS_SEQDATA_SUITE);
        PF_ConstHandle h = nullptr;
        if (seq->PF_GetConstSequenceData(in_data->effect_ref, &h) != PF_Err_NONE || !h || !*h) return false;
        const auto* sd = reinterpret_cast<const dms_plugin::SequenceData*>(*h);
        if (sd->magic != dms_plugin::kSeqMagic) return false;
        std::memcpy(guid, sd->instance_guid, 16);
        return true;
    } catch (...) {
        return false;
    }
}

namespace {

PF_Err about(PF_InData* in_data, PF_OutData* out_data) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "%s v%d.%d.%d\rAI depth maps (Depth Anything) - engine %s", DMS_EFFECT_NAME, DMS_MAJOR_VERSION,
                  DMS_MINOR_VERSION, DMS_BUG_VERSION, dms_engine_version());
    in_data->utils->ansi.sprintf(out_data->return_msg, "%s", msg);
    return PF_Err_NONE;
}

PF_Err global_setup(PF_InData* in_data, PF_OutData* out_data) {
    out_data->my_version = PF_VERSION(DMS_MAJOR_VERSION, DMS_MINOR_VERSION, DMS_BUG_VERSION, DMS_STAGE_VERSION, DMS_BUILD_VERSION);
    out_data->out_flags = DMS_PIPL_OUTFLAGS;
    out_data->out_flags2 = DMS_PIPL_OUTFLAGS2;

    HostHandle h(in_data, out_data);
    if (!h.allocate(sizeof(GlobalData))) return PF_Err_OUT_OF_MEMORY;
    GlobalData g{};
    g.magic = kGlobalMagic;
    g.is_premiere = in_data->appl_id == 'PrMr' ? 1 : 0;
    if (!g.is_premiere) {
        AEFX_SuiteScoper<DmsUtilitySuite> util(in_data, DMS_UTILITY_SUITE, out_data);
        if (util->AEGP_RegisterWithAEGP(nullptr, DMS_EFFECT_NAME, &g.plugin_id) != A_Err_NONE) g.plugin_id = 0;
    }
    dms_engine_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.log_level = DMS_LOG_INFO;
    if (dms_engine_acquire(&cfg, &g.engine) != DMS_OK) {
        g.engine = nullptr;  // effect still loads; renders will output gray + message
    }
    {
        HandleLock lk(in_data, h.get());
        if (!lk.data()) return PF_Err_OUT_OF_MEMORY;
        std::memcpy(lk.data(), &g, sizeof(g));
    }
    out_data->global_data = h.release();
    if (g.engine) dms_log(g.engine, DMS_LOG_INFO, g.is_premiere ? "GLOBAL_SETUP (Premiere Pro)" : "GLOBAL_SETUP (After Effects)");
    return PF_Err_NONE;
}

PF_Err global_setdown(PF_InData* in_data, PF_OutData* out_data) {
    const GlobalData g = read_global(in_data);
    if (g.engine) {
        dms_log(g.engine, DMS_LOG_INFO, "GLOBAL_SETDOWN");
        dms_engine_release(g.engine);  // last ref releases sessions, CUDA context, model weights
    }
    if (in_data->global_data) {
        AEFX_SuiteScoper<DmsHandleSuite> hs(in_data, DMS_HANDLE_SUITE, out_data);
        hs->host_dispose_handle(in_data->global_data);
        out_data->global_data = nullptr;
    }
    return PF_Err_NONE;
}

// Allocates a fresh sequence-data handle containing `sd`.
PF_Err make_sequence_handle(PF_InData* in_data, PF_OutData* out_data, const dms_plugin::SequenceData& sd, PF_Handle* out) {
    HostHandle h(in_data, out_data);
    if (!h.allocate(sizeof(dms_plugin::SequenceData))) return PF_Err_OUT_OF_MEMORY;
    {
        HandleLock lk(in_data, h.get());
        if (!lk.data()) return PF_Err_OUT_OF_MEMORY;
        std::memcpy(lk.data(), &sd, sizeof(sd));
    }
    *out = h.release();
    return PF_Err_NONE;
}

void open_instance(PF_InData* in_data, const dms_plugin::SequenceData& sd) {
    const GlobalData g = read_global(in_data);
    if (g.engine) dms_instance_open(g.engine, sd.instance_guid);
}

PF_Err sequence_setup(PF_InData* in_data, PF_OutData* out_data) {
    dms_plugin::SequenceData sd;
    dms_plugin::init_sequence_data(sd);
    PF_Handle h = nullptr;
    PF_Err err = make_sequence_handle(in_data, out_data, sd, &h);
    if (err) return err;
    out_data->sequence_data = h;
    open_instance(in_data, sd);
    return PF_Err_NONE;
}

PF_Err sequence_resetup(PF_InData* in_data, PF_OutData* out_data) {
    // Data may come from ANY plugin version (or be missing). Upgrade into a fresh handle.
    dms_plugin::SequenceData sd;
    bool recognized = false;
    if (in_data->sequence_data) {
        HandleLock lk(in_data, in_data->sequence_data);
        recognized = dms_plugin::upgrade_sequence_data(lk.data(), lk.size(), sd);
    } else {
        dms_plugin::init_sequence_data(sd);
    }
    if (!recognized && in_data->sequence_data) log_line(in_data, DMS_LOG_WARN, "SEQUENCE_RESETUP: unrecognized sequence data, using defaults");
    PF_Handle h = nullptr;
    PF_Err err = make_sequence_handle(in_data, out_data, sd, &h);
    if (err) return err;
    if (in_data->sequence_data) {
        AEFX_SuiteScoper<DmsHandleSuite> hs(in_data, DMS_HANDLE_SUITE, out_data);
        hs->host_dispose_handle(in_data->sequence_data);
    }
    out_data->sequence_data = h;
    open_instance(in_data, sd);
    return PF_Err_NONE;
}

PF_Err sequence_flatten(PF_InData* in_data, PF_OutData* out_data) {
    // In-memory form is already flat (no pointers): hand the same handle back.
    out_data->sequence_data = in_data->sequence_data;
    return PF_Err_NONE;
}

PF_Err get_flattened_sequence_data(PF_InData* in_data, PF_OutData* out_data) {
    // Must return a NEW handle and leave the original untouched (MFR requirement).
    dms_plugin::SequenceData sd;
    dms_plugin::init_sequence_data(sd);
    if (in_data->sequence_data) {
        HandleLock lk(in_data, in_data->sequence_data);
        if (lk.data() && lk.size() >= sizeof(sd)) std::memcpy(&sd, lk.data(), sizeof(sd));
    }
    PF_Handle h = nullptr;
    PF_Err err = make_sequence_handle(in_data, out_data, sd, &h);
    if (err) return err;
    out_data->sequence_data = h;
    return PF_Err_NONE;
}

PF_Err sequence_setdown(PF_InData* in_data, PF_OutData* out_data) {
    if (in_data->sequence_data) {
        {
            HandleLock lk(in_data, in_data->sequence_data);
            if (lk.data() && lk.size() >= sizeof(dms_plugin::SequenceData)) {
                const auto* sd = static_cast<const dms_plugin::SequenceData*>(lk.data());
                const GlobalData g = read_global(in_data);
                if (g.engine && sd->magic == dms_plugin::kSeqMagic) dms_instance_close(g.engine, sd->instance_guid);
            }
        }
        AEFX_SuiteScoper<DmsHandleSuite> hs(in_data, DMS_HANDLE_SUITE, out_data);
        hs->host_dispose_handle(in_data->sequence_data);
    }
    out_data->sequence_data = nullptr;
    return PF_Err_NONE;
}

PF_Err user_changed_param(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], const PF_UserChangedParamExtra* extra) {
    if (!extra || extra->param_index != dms_plugin::P_MODEL) return PF_Err_NONE;
    const GlobalData g = read_global(in_data);
    if (!g.engine) return PF_Err_NONE;
    const auto raw = read_params_array(in_data, params);
    const auto snap = dms_plugin::sanitize(raw, dms_colormap_count(), false);
    dms_model_info mi{};
    mi.struct_size = sizeof(mi);
    if (dms_model_info_by_id(g.engine, dms_plugin::model_id(snap.model_index), &mi) != DMS_OK || mi.installed) return PF_Err_NONE;
    if (!mi.has_url) {
        report_message(in_data, out_data,
                       std::string("DMS: '") + mi.display_name + "' has no download source configured. Copy the model file to: " + mi.path);
        return PF_Err_NONE;
    }
    // UI thread: offer the download with progress (never from render).
    if (run_download_dialog(g.engine, mi.id, mi.display_name)) out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
    return PF_Err_NONE;
}

PF_Err dispatch(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra) {
    switch (cmd) {
        case PF_Cmd_ABOUT: return about(in_data, out_data);
        case PF_Cmd_GLOBAL_SETUP: return global_setup(in_data, out_data);
        case PF_Cmd_GLOBAL_SETDOWN: return global_setdown(in_data, out_data);
        case PF_Cmd_PARAMS_SETUP: return params_setup(in_data, out_data);
        case PF_Cmd_SEQUENCE_SETUP: return sequence_setup(in_data, out_data);
        case PF_Cmd_SEQUENCE_RESETUP: return sequence_resetup(in_data, out_data);
        case PF_Cmd_SEQUENCE_FLATTEN: return sequence_flatten(in_data, out_data);
        case PF_Cmd_GET_FLATTENED_SEQUENCE_DATA: return get_flattened_sequence_data(in_data, out_data);
        case PF_Cmd_SEQUENCE_SETDOWN: return sequence_setdown(in_data, out_data);
        case PF_Cmd_DO_DIALOG: return do_options_dialog(in_data, out_data, params);
        case PF_Cmd_USER_CHANGED_PARAM:
            return user_changed_param(in_data, out_data, params, static_cast<const PF_UserChangedParamExtra*>(extra));
        case PF_Cmd_UPDATE_PARAMS_UI: return update_params_ui(in_data, out_data, params);
        case PF_Cmd_QUERY_DYNAMIC_FLAGS: return query_dynamic_flags(in_data, out_data);
        case PF_Cmd_SMART_PRE_RENDER: return smart_pre_render(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
        case PF_Cmd_SMART_RENDER: return smart_render(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
        case PF_Cmd_RENDER: return legacy_render(in_data, out_data, params, output);
        default: return PF_Err_NONE;
    }
}

}  // namespace
}  // namespace dms_ae

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* inSPBasicSuitePtr, const char* inHostName, const char* inHostVersion) {
    (void)inSPBasicSuitePtr;
    (void)inHostName;
    (void)inHostVersion;
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, DMS_EFFECT_NAME, DMS_EFFECT_MATCH_NAME, DMS_EFFECT_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", DMS_SUPPORT_URL);
    return result;
}

extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output,
                                       void* extra) {
    PF_Err err = PF_Err_NONE;
    if (!in_data || !out_data) return PF_Err_BAD_CALLBACK_PARAM;
    try {
        err = dms_ae::dispatch(cmd, in_data, out_data, params, output, extra);
    } catch (PF_Err thrown) {  // AEFX_SuiteScoper throws A_Err/PF_Err when a suite is missing
        err = thrown;
        dms_ae::log_line(in_data, DMS_LOG_ERROR, "host suite unavailable (PF_Err " + std::to_string(thrown) + ") in cmd " + std::to_string(cmd));
    } catch (const std::bad_alloc&) {
        dms_ae::report_message(in_data, out_data, "DMS: out of memory");
        err = PF_Err_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        dms_ae::report_message(in_data, out_data, std::string("DMS: internal error: ") + e.what());
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    } catch (...) {
        dms_ae::report_message(in_data, out_data, "DMS: unknown internal error");
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
