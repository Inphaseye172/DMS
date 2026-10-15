// GPU render path (PF_OutFlag2_SUPPORTS_GPU_RENDER_F32), phase 2.
//
// Scope: CUDA GPU worlds (PF_PixelFormat_GPU_BGRA128). The input world is copied to host memory
// through the CUDA Driver API on After Effects' own stream/context, rendered by the same render core
// as the CPU path, and the result copied back into the GPU output world. This keeps the host's GPU
// pipeline from falling back to CPU for the layer chain. Feeding ONNX Runtime directly from the GPU
// world (zero copy) is future work. Any other framework (DirectX, OpenCL, Metal) is declined in
// GPU_DEVICE_SETUP, so the host uses the CPU path for this effect.
#include "DMS_Depth.h"

#include "ae_raii.h"

#include "AE_EffectGPUSuites.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace dms_ae {

namespace {

// Minimal CUDA Driver API surface (stable ABI since CUDA 3.2; nvcuda.dll ships with the NVIDIA driver).
using CUresult = int;
using CUstream = void*;
using CUdeviceptr = unsigned long long;
using CUcontext = void*;
enum CUmemorytype { CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2 };
struct CUDA_MEMCPY2D {
    size_t srcXInBytes, srcY;
    CUmemorytype srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    void* srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    CUmemorytype dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    void* dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
};

struct CudaDriver {
    bool ok = false;
    CUresult (*memcpy2d_async)(const CUDA_MEMCPY2D*, CUstream) = nullptr;
    CUresult (*stream_sync)(CUstream) = nullptr;
    CUresult (*ctx_get_current)(CUcontext*) = nullptr;
};

// Process-wide, initialized once, immutable afterwards (MFR safe).
const CudaDriver& cuda_driver() {
    static CudaDriver drv;
    static std::once_flag once;
    std::call_once(once, [] {
#if defined(_WIN32)
        HMODULE m = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!m) return;  // never freed: driver library stays loaded for the process
        drv.memcpy2d_async = reinterpret_cast<CUresult (*)(const CUDA_MEMCPY2D*, CUstream)>(GetProcAddress(m, "cuMemcpy2DAsync_v2"));
        drv.stream_sync = reinterpret_cast<CUresult (*)(CUstream)>(GetProcAddress(m, "cuStreamSynchronize"));
        drv.ctx_get_current = reinterpret_cast<CUresult (*)(CUcontext*)>(GetProcAddress(m, "cuCtxGetCurrent"));
        drv.ok = drv.memcpy2d_async && drv.stream_sync && drv.ctx_get_current;
#endif
    });
    return drv;
}

bool copy_2d(const CudaDriver& d, CUstream stream, bool to_host, void* host, const void* device, size_t pitch, size_t width_bytes, size_t rows) {
    CUDA_MEMCPY2D c{};
    c.WidthInBytes = width_bytes;
    c.Height = rows;
    if (to_host) {
        c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcDevice = reinterpret_cast<CUdeviceptr>(device);
        c.srcPitch = pitch;
        c.dstMemoryType = CU_MEMORYTYPE_HOST;
        c.dstHost = host;
        c.dstPitch = pitch;
    } else {
        c.srcMemoryType = CU_MEMORYTYPE_HOST;
        c.srcHost = host;
        c.srcPitch = pitch;
        c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.dstDevice = reinterpret_cast<CUdeviceptr>(const_cast<void*>(device));
        c.dstPitch = pitch;
    }
    if (d.memcpy2d_async(&c, stream) != 0) return false;
    return d.stream_sync(stream) == 0;
}

}  // namespace

bool gpu_framework_supported(PF_GPU_Framework fw) noexcept {
    if (fw != PF_GPU_Framework_CUDA) return false;
    try {
        return cuda_driver().ok;
    } catch (...) {
        return false;
    }
}

PF_Err gpu_device_setup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    if (!extra || !extra->input || !extra->output) return PF_Err_BAD_CALLBACK_PARAM;
    if (gpu_framework_supported(extra->input->what_gpu)) {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        log_line(in_data, DMS_LOG_INFO, "GPU_DEVICE_SETUP: CUDA device " + std::to_string(extra->input->device_index) + " accepted");
    } else {
        out_data->out_flags2 &= ~PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        log_line(in_data, DMS_LOG_INFO, "GPU_DEVICE_SETUP: framework " + std::to_string(static_cast<int>(extra->input->what_gpu)) +
                                            " not supported; CPU path will be used");
    }
    extra->output->gpu_data = nullptr;  // no per-device state
    return PF_Err_NONE;
}

PF_Err gpu_device_setdown(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetdownExtra* extra) {
    (void)in_data;
    (void)out_data;
    (void)extra;
    return PF_Err_NONE;
}

PF_Err smart_render_gpu(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    if (!extra || !extra->input || !extra->cb) return PF_Err_BAD_CALLBACK_PARAM;
    if (extra->input->what_gpu != PF_GPU_Framework_CUDA) return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    const CudaDriver& drv = cuda_driver();
    if (!drv.ok) return PF_Err_UNRECOGNIZED_PARAM_TYPE;

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo info{};
    PF_Err err = gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info);
    if (err) return err;
    CUcontext ctx = nullptr;
    if (drv.ctx_get_current(&ctx) != 0 || !ctx) {
        log_line(in_data, DMS_LOG_ERROR, "SMART_RENDER_GPU: no current CUDA context");
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    auto stream = static_cast<CUstream>(info.command_queuePV);

    // Host staging buffers (one per checked-out world), filled from the GPU worlds.
    std::vector<std::vector<uint8_t>> staging;
    auto download = [&](PF_EffectWorld* w, dms_plugin::WorldView& view) -> bool {
        if (!w) return false;
        void* dev = nullptr;
        if (gpu->GetGPUWorldData(in_data->effect_ref, w, &dev) != PF_Err_NONE || !dev) return false;
        const size_t pitch = static_cast<size_t>(w->rowbytes);
        const size_t width_bytes = static_cast<size_t>(w->width) * sizeof(dms_plugin::PixelBGRA);
        if (pitch < width_bytes || w->height <= 0) return false;
        staging.emplace_back(pitch * static_cast<size_t>(w->height));
        if (!copy_2d(drv, stream, true, staging.back().data(), dev, pitch, width_bytes, static_cast<size_t>(w->height))) return false;
        view.data = staging.back().data();
        view.width = w->width;
        view.height = w->height;
        view.rowbytes = w->rowbytes;
        view.depth = dms_plugin::PixelDepth::F32_BGRA;
        return true;
    };
    staging.reserve(40);
    return smart_render_common(in_data, out_data, extra, /*gpu=*/true, download, [&](PF_EffectWorld* out_world, const dms_plugin::WorldView& host_out) -> bool {
        void* dev = nullptr;
        if (gpu->GetGPUWorldData(in_data->effect_ref, out_world, &dev) != PF_Err_NONE || !dev) return false;
        const size_t width_bytes = static_cast<size_t>(host_out.width) * sizeof(dms_plugin::PixelBGRA);
        return copy_2d(drv, stream, false, host_out.data, dev, static_cast<size_t>(host_out.rowbytes), width_bytes, static_cast<size_t>(host_out.height));
    });
}

}  // namespace dms_ae
