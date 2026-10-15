// GPU discovery without linking any GPU SDK: NVML (nvml.dll / libnvidia-ml.so.1, shipped with the
// NVIDIA driver) for NVIDIA driver version + VRAM, DXGI (Windows) for AMD/Intel adapters + VRAM budget.
#pragma once

#include "dms/engine.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace dms {

struct DriverVersion {
    int major = 0;
    int minor = 0;
    bool valid() const noexcept { return major > 0; }
    bool at_least(int maj, int min) const noexcept { return major > maj || (major == maj && minor >= min); }
    std::string str() const;
    static DriverVersion parse(const std::string& s) noexcept;  // "576.02" / "550.54.15"
};

struct GpuSnapshot {
    int vendor = DMS_GPU_UNKNOWN;
    std::string name;
    DriverVersion driver;
    uint64_t vram_total = 0;
    uint64_t vram_free = 0;
    bool has_dxgi_adapter = false;  // a D3D12-capable hardware adapter exists (DirectML candidate)
    std::string notes;
};

class GpuMonitor {
public:
    virtual ~GpuMonitor() = default;
    // Static info (vendor, name, driver); cached by implementations.
    virtual GpuSnapshot snapshot() = 0;
    // Current free VRAM in bytes (0 if unknown). Must be cheap: called before every session/batch.
    virtual uint64_t free_vram() = 0;
};

std::unique_ptr<GpuMonitor> make_system_gpu_monitor();

// Test double.
class FakeGpuMonitor : public GpuMonitor {
public:
    explicit FakeGpuMonitor(GpuSnapshot s) : snap_(std::move(s)) {}
    GpuSnapshot snapshot() override {
        std::lock_guard<std::mutex> lk(mu_);
        return snap_;
    }
    uint64_t free_vram() override {
        std::lock_guard<std::mutex> lk(mu_);
        return snap_.vram_free;
    }
    void set_free(uint64_t b) {
        std::lock_guard<std::mutex> lk(mu_);
        snap_.vram_free = b;
    }

private:
    std::mutex mu_;
    GpuSnapshot snap_;
};

// Driver policy (documented in docs/STABILITY.md).
// CUDA 12 runtime (ORT 1.22 CUDA build) needs >= 527.41 (Windows) / 525.60.13 (Linux).
constexpr int kMinCudaDriverMajor = 527;
constexpr int kMinCudaDriverMinor = 41;
// CUDA Memory Sharing (shared env arena + pinned I/O) is only enabled on drivers known to be stable with it.
constexpr int kMinMemShareDriverMajor = 560;
constexpr int kMinMemShareDriverMinor = 0;

}  // namespace dms
