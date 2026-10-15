#include "gpu_probe.h"

#include "logging.h"
#include "platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#include <dxgi1_4.h>
#endif

namespace dms {

std::string DriverVersion::str() const {
    if (!valid()) return {};
    char b[32];
    std::snprintf(b, sizeof(b), "%d.%02d", major, minor);
    return b;
}

DriverVersion DriverVersion::parse(const std::string& s) noexcept {
    DriverVersion v;
    int a = 0, b = 0;
    if (std::sscanf(s.c_str(), "%d.%d", &a, &b) >= 1 && a > 0 && a < 100000 && b >= 0 && b < 100000) {
        v.major = a;
        v.minor = b;
    }
    return v;
}

namespace {

// Minimal NVML ABI (stable since NVML 4; we only use these entry points).
using nvmlReturn_t = int;
using nvmlDevice_t = void*;
struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
constexpr nvmlReturn_t NVML_SUCCESS = 0;

class SystemGpuMonitor : public GpuMonitor {
public:
    SystemGpuMonitor() { probe(); }
    ~SystemGpuMonitor() override {
        if (nvml_shutdown_ && nvml_ok_) nvml_shutdown_();
    }
    SystemGpuMonitor(const SystemGpuMonitor&) = delete;
    SystemGpuMonitor& operator=(const SystemGpuMonitor&) = delete;

    GpuSnapshot snapshot() override {
        std::lock_guard<std::mutex> lk(mu_);
        GpuSnapshot s = snap_;
        s.vram_free = free_vram_locked();
        return s;
    }
    uint64_t free_vram() override {
        std::lock_guard<std::mutex> lk(mu_);
        return free_vram_locked();
    }

private:
    uint64_t free_vram_locked() {
        if (nvml_ok_ && device_ && get_mem_) {
            nvmlMemory_t m{};
            if (get_mem_(device_, &m) == NVML_SUCCESS) return m.free;
        }
#if defined(_WIN32)
        if (adapter3_) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
                return info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage : 0;
        }
#endif
        return 0;
    }

    void probe() {
        std::lock_guard<std::mutex> lk(mu_);
        snap_.vendor = DMS_GPU_NONE;
        probe_nvml();
#if defined(_WIN32)
        probe_dxgi();
#endif
        logf(DMS_LOG_INFO, "GPU probe: vendor=%d name='%s' driver=%s vram=%llu MB dxgi=%d %s", snap_.vendor, snap_.name.c_str(),
             snap_.driver.str().c_str(), static_cast<unsigned long long>(snap_.vram_total >> 20), snap_.has_dxgi_adapter ? 1 : 0,
             snap_.notes.c_str());
    }

    void probe_nvml() {
#if defined(_WIN32)
        // nvml.dll ships with the driver in System32 (R418+); older drivers used the NVSMI folder.
        lib_ = platform::DynLib(L"nvml.dll");
        if (!lib_.ok()) {
            wchar_t pf[MAX_PATH] = {0};
            if (GetEnvironmentVariableW(L"ProgramW6432", pf, MAX_PATH) > 0)
                lib_ = platform::DynLib(std::filesystem::path(pf) / L"NVIDIA Corporation" / L"NVSMI" / L"nvml.dll");
        }
#else
        lib_ = platform::DynLib("libnvidia-ml.so.1");
#endif
        if (!lib_.ok()) {
            snap_.notes += "NVML not found (no NVIDIA driver). ";
            return;
        }
        lib_.pin();  // never unload a driver library
        auto init = lib_.fn<nvmlReturn_t (*)()>("nvmlInit_v2");
        nvml_shutdown_ = lib_.fn<nvmlReturn_t (*)()>("nvmlShutdown");
        auto driver = lib_.fn<nvmlReturn_t (*)(char*, unsigned)>("nvmlSystemGetDriverVersion");
        auto count = lib_.fn<nvmlReturn_t (*)(unsigned*)>("nvmlDeviceGetCount_v2");
        auto handle = lib_.fn<nvmlReturn_t (*)(unsigned, nvmlDevice_t*)>("nvmlDeviceGetHandleByIndex_v2");
        auto name = lib_.fn<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned)>("nvmlDeviceGetName");
        get_mem_ = lib_.fn<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*)>("nvmlDeviceGetMemoryInfo");
        if (!init || !driver || !count || !handle || !get_mem_) {
            snap_.notes += "NVML is missing required entry points. ";
            return;
        }
        if (init() != NVML_SUCCESS) {
            snap_.notes += "nvmlInit failed. ";
            return;
        }
        nvml_ok_ = true;
        char buf[96] = {0};
        if (driver(buf, sizeof(buf)) == NVML_SUCCESS) snap_.driver = DriverVersion::parse(buf);
        unsigned n = 0;
        if (count(&n) != NVML_SUCCESS || n == 0) {
            snap_.notes += "no NVIDIA devices. ";
            return;
        }
        if (handle(0, &device_) != NVML_SUCCESS) return;
        snap_.vendor = DMS_GPU_NVIDIA;
        char nm[128] = {0};
        if (name && name(device_, nm, sizeof(nm)) == NVML_SUCCESS) snap_.name = nm;
        nvmlMemory_t m{};
        if (get_mem_(device_, &m) == NVML_SUCCESS) snap_.vram_total = m.total;
    }

#if defined(_WIN32)
    void probe_dxgi() {
        dxgi_ = platform::DynLib(L"dxgi.dll");
        if (!dxgi_.ok()) return;
        dxgi_.pin();
        using CreateFactory1 = HRESULT(WINAPI*)(REFIID, void**);
        auto create = dxgi_.fn<CreateFactory1>("CreateDXGIFactory1");
        if (!create) return;
        IDXGIFactory1* factory = nullptr;
        if (FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory) return;
        IDXGIAdapter1* best = nullptr;
        SIZE_T best_mem = 0;
        for (UINT i = 0;; ++i) {
            IDXGIAdapter1* a = nullptr;
            if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 d{};
            if (SUCCEEDED(a->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.DedicatedVideoMemory >= best_mem) {
                if (best) best->Release();
                best = a;
                best_mem = d.DedicatedVideoMemory;
            } else {
                a->Release();
            }
        }
        factory->Release();
        if (!best) return;
        DXGI_ADAPTER_DESC1 d{};
        best->GetDesc1(&d);
        snap_.has_dxgi_adapter = true;
        if (snap_.vendor != DMS_GPU_NVIDIA) {
            snap_.vendor = d.VendorId == 0x10DE ? DMS_GPU_NVIDIA : d.VendorId == 0x1002 ? DMS_GPU_AMD : d.VendorId == 0x8086 ? DMS_GPU_INTEL : DMS_GPU_UNKNOWN;
            char nm[128] = {0};
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nm, sizeof(nm) - 1, nullptr, nullptr);
            snap_.name = nm;
            snap_.vram_total = d.DedicatedVideoMemory;
            best->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3_));
        }
        best->Release();
    }
    IDXGIAdapter3* adapter3_ = nullptr;  // intentionally kept for the process lifetime (budget queries)
    platform::DynLib dxgi_;
#endif

    std::mutex mu_;
    GpuSnapshot snap_;
    platform::DynLib lib_;
    bool nvml_ok_ = false;
    nvmlDevice_t device_ = nullptr;
    nvmlReturn_t (*nvml_shutdown_)() = nullptr;
    nvmlReturn_t (*get_mem_)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
};

}  // namespace

std::unique_ptr<GpuMonitor> make_system_gpu_monitor() { return std::make_unique<SystemGpuMonitor>(); }

}  // namespace dms
