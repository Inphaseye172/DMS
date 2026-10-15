// Options + model download dialogs. UI thread only (PF_Cmd_DO_DIALOG / PF_Cmd_USER_CHANGED_PARAM).
// The download itself runs on the engine's worker thread; this code only polls its progress on a
// TaskDialog timer, so the host UI never blocks on network I/O and the user can always cancel.
#include "DMS_Depth.h"

#include "params_model.h"

#include <cstdio>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#endif

namespace dms_ae {

#if defined(_WIN32)
namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

HMODULE this_module() {
    HMODULE hm = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&this_module), &hm);
    return hm;
}

// Activates the comctl32 v6 manifest embedded in the .aex (resource id 2) for the scope, so
// TaskDialogIndirect is available even if the host process was not manifested for v6.
class ComCtlActivation {
public:
    ComCtlActivation() {
        ACTCTXW ctx{};
        ctx.cbSize = sizeof(ctx);
        ctx.dwFlags = ACTCTX_FLAG_HMODULE_VALID | ACTCTX_FLAG_RESOURCE_NAME_VALID;
        ctx.hModule = this_module();
        ctx.lpResourceName = MAKEINTRESOURCEW(2);
        handle_ = CreateActCtxW(&ctx);
        if (handle_ != INVALID_HANDLE_VALUE && !ActivateActCtx(handle_, &cookie_)) cookie_ = 0;
    }
    ~ComCtlActivation() {
        if (cookie_) DeactivateActCtx(0, cookie_);
        if (handle_ != INVALID_HANDLE_VALUE) ReleaseActCtx(handle_);
    }
    ComCtlActivation(const ComCtlActivation&) = delete;
    ComCtlActivation& operator=(const ComCtlActivation&) = delete;

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    ULONG_PTR cookie_ = 0;
};

using TaskDialogIndirectFn = HRESULT(WINAPI*)(const TASKDIALOGCONFIG*, int*, int*, BOOL*);

TaskDialogIndirectFn task_dialog_fn() {
    HMODULE cc = GetModuleHandleW(L"comctl32.dll");
    if (!cc) cc = LoadLibraryExW(L"comctl32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return cc ? reinterpret_cast<TaskDialogIndirectFn>(GetProcAddress(cc, "TaskDialogIndirect")) : nullptr;
}

HWND owner_window() {
    HWND h = GetActiveWindow();
    return h ? h : GetForegroundWindow();
}

struct DownloadCtx {
    dms_engine* engine;
    std::string model_id;
    bool done = false;
    bool failed = false;
    std::wstring error;
};

HRESULT CALLBACK download_cb(HWND hwnd, UINT msg, WPARAM wparam, LPARAM, LONG_PTR ref) {
    auto* ctx = reinterpret_cast<DownloadCtx*>(ref);
    switch (msg) {
        case TDN_CREATED:
            SendMessageW(hwnd, TDM_SET_PROGRESS_BAR_RANGE, 0, MAKELPARAM(0, 1000));
            break;
        case TDN_TIMER: {
            dms_download_info di{};
            di.struct_size = sizeof(di);
            if (dms_download_query(ctx->engine, ctx->model_id.c_str(), &di) != DMS_OK) break;
            const double frac = di.bytes_total ? static_cast<double>(di.bytes_done) / static_cast<double>(di.bytes_total) : 0.0;
            SendMessageW(hwnd, TDM_SET_PROGRESS_BAR_POS, static_cast<WPARAM>(frac * 1000.0), 0);
            wchar_t line[256];
            swprintf(line, 256, L"%s  %.1f / %.1f MB  (%.1f MB/s)", di.state == DMS_DL_VERIFYING ? L"Verifying checksum..." : L"Downloading...",
                     static_cast<double>(di.bytes_done) / 1e6, static_cast<double>(di.bytes_total) / 1e6,
                     static_cast<double>(di.bytes_per_second) / 1e6);
            SendMessageW(hwnd, TDM_SET_ELEMENT_TEXT, TDE_CONTENT, reinterpret_cast<LPARAM>(line));
            if (di.state == DMS_DL_DONE) {
                ctx->done = true;
                SendMessageW(hwnd, TDM_CLICK_BUTTON, IDOK, 0);
            } else if (di.state == DMS_DL_FAILED || di.state == DMS_DL_CANCELLED) {
                ctx->failed = true;
                ctx->error = widen(di.message);
                SendMessageW(hwnd, TDM_CLICK_BUTTON, IDOK, 0);
            }
            break;
        }
        case TDN_BUTTON_CLICKED:
            if (static_cast<int>(wparam) == IDCANCEL) dms_download_cancel(ctx->engine, ctx->model_id.c_str());
            break;
        default: break;
    }
    return S_OK;
}

void open_folder(const std::string& path) {
    const std::wstring w = widen(path);
    ShellExecuteW(nullptr, L"open", w.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

}  // namespace

bool run_download_dialog(dms_engine* engine, const char* model_id, const char* display_name) {
    if (!engine || !model_id) return false;
    const dms_status st = dms_download_start(engine, model_id);
    if (st != DMS_OK && st != DMS_ERR_BUSY) {
        const std::wstring msg = widen(std::string("Could not start the download: ") + dms_last_error());
        MessageBoxW(owner_window(), msg.c_str(), L"DMS Depth Map", MB_OK | MB_ICONWARNING);
        return false;
    }
    ComCtlActivation act;
    TaskDialogIndirectFn td = task_dialog_fn();
    if (!td) {
        MessageBoxW(owner_window(), L"The model is downloading in the background. Re-open Options to check progress.", L"DMS Depth Map",
                    MB_OK | MB_ICONINFORMATION);
        return false;
    }
    DownloadCtx ctx{engine, model_id};
    const std::wstring title = widen(std::string("Downloading ") + (display_name ? display_name : model_id));
    TASKDIALOGCONFIG cfg{};
    cfg.cbSize = sizeof(cfg);
    cfg.hwndParent = owner_window();
    cfg.dwFlags = TDF_SHOW_PROGRESS_BAR | TDF_CALLBACK_TIMER | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
    cfg.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    cfg.pszWindowTitle = L"DMS Depth Map";
    cfg.pszMainInstruction = title.c_str();
    cfg.pszContent = L"Starting...";
    cfg.pfCallback = download_cb;
    cfg.lpCallbackData = reinterpret_cast<LONG_PTR>(&ctx);
    int button = 0;
    if (FAILED(td(&cfg, &button, nullptr, nullptr))) return false;
    if (!ctx.done && !ctx.failed) dms_download_cancel(engine, model_id);  // closed via Cancel/Esc
    if (ctx.failed && !ctx.error.empty()) MessageBoxW(owner_window(), ctx.error.c_str(), L"DMS Depth Map - download failed", MB_OK | MB_ICONWARNING);
    return ctx.done;
}

PF_Err do_options_dialog(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]) {
    const GlobalData g = read_global(in_data);
    if (!g.engine) {
        report_message(in_data, out_data, "DMS: engine failed to initialize - see %LOCALAPPDATA%\\DMS\\logs\\dms.log");
        return PF_Err_NONE;
    }
    const auto snap = dms_plugin::sanitize(read_params_array(in_data, params), dms_colormap_count(), false);
    ComCtlActivation act;
    TaskDialogIndirectFn td = task_dialog_fn();
    enum { kDownload = 1001, kVerify, kOpenModels, kOpenLogs };
    for (int guard = 0; guard < 32; ++guard) {
        dms_model_info mi{};
        mi.struct_size = sizeof(mi);
        const bool have_model = dms_model_info_by_id(g.engine, dms_plugin::model_id(snap.model_index), &mi) == DMS_OK;
        dms_device_info di{};
        di.struct_size = sizeof(di);
        dms_device_query(g.engine, &di);
        static const char* kBackends[] = {"none", "CPU", "CUDA", "DirectML", "TensorRT"};
        const int ab = (di.active_backend >= 0 && di.active_backend <= 4) ? di.active_backend : 0;
        char body[2048];
        std::snprintf(body, sizeof(body),
                      "Engine %s, ONNX Runtime %s\nGPU: %s (driver %s), VRAM free %llu / %llu MB\n"
                      "Backends: CUDA %s, DirectML %s, TensorRT %s. Last used: %s\n%s%s\n\n"
                      "Model: %s\nStatus: %s\nFile: %s",
                      dms_engine_version(), di.ort_version[0] ? di.ort_version : "(not loaded)", di.gpu_name[0] ? di.gpu_name : "unknown",
                      di.driver_version[0] ? di.driver_version : "-", static_cast<unsigned long long>(di.vram_free_bytes >> 20),
                      static_cast<unsigned long long>(di.vram_total_bytes >> 20), di.cuda_available ? "yes" : "no",
                      di.directml_available ? "yes" : "no", di.tensorrt_available ? "yes" : "no", kBackends[ab],
                      di.notes[0] ? "Notes: " : "", di.notes, have_model ? mi.display_name : "?",
                      !have_model ? "unknown model" : mi.installed ? "installed" : (mi.has_url ? "not installed (download available)" : "not installed (no download source configured)"),
                      have_model ? mi.path : "");
        const std::wstring wbody = widen(body);
        if (!td) {
            MessageBoxW(owner_window(), wbody.c_str(), L"DMS Depth Map - Options", MB_OK);
            return PF_Err_NONE;
        }
        TASKDIALOG_BUTTON buttons[4];
        UINT nb = 0;
        if (have_model && !mi.installed && mi.has_url) buttons[nb++] = {kDownload, L"Download model"};
        if (have_model && mi.installed) buttons[nb++] = {kVerify, L"Verify model checksum"};
        buttons[nb++] = {kOpenModels, L"Open models folder"};
        buttons[nb++] = {kOpenLogs, L"Open log folder"};
        TASKDIALOGCONFIG cfg{};
        cfg.cbSize = sizeof(cfg);
        cfg.hwndParent = owner_window();
        cfg.dwFlags = TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
        cfg.dwCommonButtons = TDCBF_CLOSE_BUTTON;
        cfg.pszWindowTitle = L"DMS Depth Map - Options";
        cfg.pszMainInstruction = L"DMS Depth Map";
        cfg.pszContent = wbody.c_str();
        cfg.pButtons = buttons;
        cfg.cButtons = nb;
        int button = 0;
        if (FAILED(td(&cfg, &button, nullptr, nullptr))) return PF_Err_NONE;
        switch (button) {
            case kDownload:
                if (run_download_dialog(g.engine, mi.id, mi.display_name)) out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
                break;
            case kVerify: {
                const dms_status vs = dms_model_verify(g.engine, mi.id);  // explicit user action; may take seconds
                const std::wstring m = widen(vs == DMS_OK ? std::string("Checksum OK.") : std::string("Verification failed: ") + dms_last_error());
                MessageBoxW(owner_window(), m.c_str(), L"DMS Depth Map", MB_OK | (vs == DMS_OK ? MB_ICONINFORMATION : MB_ICONWARNING));
                break;
            }
            case kOpenModels: {
                std::string p = mi.path;
                const size_t slash = p.find_last_of("\\/");
                open_folder(slash == std::string::npos ? p : p.substr(0, slash));
                break;
            }
            case kOpenLogs: {
                wchar_t local[MAX_PATH] = {0};
                const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
                if (n > 0 && n < MAX_PATH) {
                    const std::wstring dir = std::wstring(local) + L"\\DMS\\logs";
                    ShellExecuteW(nullptr, L"open", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                }
                break;
            }
            default: return PF_Err_NONE;  // Close
        }
    }
    return PF_Err_NONE;
}

#else  // non-Windows hosts: no native dialogs (macOS is out of scope for v1)

bool run_download_dialog(dms_engine*, const char*, const char*) { return false; }
PF_Err do_options_dialog(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef*[]) {
    report_message(in_data, out_data, "DMS: Options dialog is only implemented on Windows in v1");
    return PF_Err_NONE;
}

#endif

}  // namespace dms_ae
