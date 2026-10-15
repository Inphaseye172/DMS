#!/usr/bin/env python3
"""Renders docs/STABILITY.md from docs/STABILITY.md.in, replacing {{ref:KEY}} with path:line of the code
that enforces each rule (located by regex below, so references never go stale).
  python3 -I tools/gen_stability_doc.py          # write docs/STABILITY.md
  python3 -I tools/gen_stability_doc.py --check  # exit 1 if docs/STABILITY.md is out of date
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REFS = {
    "effectmain_try": ("plugin/ae/DMS_Depth.cpp", r"err = dms_ae::dispatch\("),
    "effectmain_catch": ("plugin/ae/DMS_Depth.cpp", r"catch \(PF_Err thrown\)"),
    "report_message": ("plugin/ae/DMS_Depth.cpp", r"^void report_message"),
    "sprintf": ("plugin/ae/DMS_Depth.cpp", r"in_data->utils->ansi.sprintf\(out_data->return_msg"),
    "guard": ("engine/src/common.h", r"^dms_status guard"),
    "logger_rotate": ("engine/src/logging.cpp", r"^void Logger::rotate_locked"),
    "fallback_plan": ("engine/src/backend_plan.cpp", r"^std::vector<Attempt> build_plan"),
    "fallback_loop": ("engine/src/engine_compute.cpp", r"// ---- execute the plan \(fallback chain\) ----"),
    "render_fallback": ("plugin/core/render_core.cpp", r"// Fallback: last good depth of this instance, else flat mid-gray."),
    "render_noexcept": ("plugin/core/render_core.cpp", r"^RenderOutcome render\("),
    "vram_check": ("engine/src/engine_compute.cpp", r"if \(free > 0 && free < need \+ cfg_.vram_reserve_bytes\)"),
    "vram_reserve_default": ("engine/src/engine_impl.h", r"uint64_t vram_reserve_bytes = 1ull << 30;"),
    "vram_estimate": ("engine/src/backend_plan.cpp", r"^uint64_t estimate_vram"),
    "abort_poll": ("engine/src/inference_service.cpp", r"const dms_status a = check_abort\(\);  // host callback"),
    "abort_batch": ("engine/src/engine_compute.cpp", r"// Host abort \+ VRAM check before every batch."),
    "progress": ("engine/src/engine_compute.cpp", r"req.progress_fn\(req.user, static_cast<float>\(ctx.units_done\)"),
    "abort_ae": ("plugin/ae/DMS_Depth_Render.cpp", r"^int DMS_API abort_cb"),
    "interrupt": ("plugin/ae/DMS_Depth_Render.cpp", r"return PF_Interrupt_CANCEL;"),
    "terminate": ("engine/src/inference_service.cpp", r"if \(running_ == job.get\(\) && running_ro_\) running_ro_->SetTerminate\(\);"),
    "raii_handle": ("plugin/ae/ae_raii.h", r"^class HostHandle"),
    "raii_checkout": ("plugin/ae/ae_raii.h", r"^class SmartLayerCheckout"),
    "raii_param": ("plugin/ae/ae_raii.h", r"^class ParamCheckout"),
    "prerender_owner": ("plugin/ae/DMS_Depth_Render.cpp", r"^void DeletePreRenderData"),
    "seq_struct": ("plugin/core/sequence_data.h", r"^struct SequenceData"),
    "seq_upgrade": ("plugin/core/sequence_data.cpp", r"^bool upgrade_sequence_data"),
    "seq_resetup": ("plugin/ae/DMS_Depth.cpp", r"^PF_Err sequence_resetup"),
    "seq_flatten": ("plugin/ae/DMS_Depth.cpp", r"^PF_Err get_flattened_sequence_data"),
    "seq_const": ("plugin/ae/DMS_Depth.cpp", r"seq->PF_GetConstSequenceData"),
    "singleton": ("engine/src/engine.cpp", r"^std::shared_ptr<Engine> g_engine;"),
    "acquire": ("engine/src/engine.cpp", r"^dms_status DMS_API dms_engine_acquire"),
    "lock_engine": ("engine/src/engine.cpp", r"^std::shared_ptr<Engine> lock_engine"),
    "global_setdown": ("plugin/ae/DMS_Depth.cpp", r"dms_engine_release\(g.engine\);"),
    "driver_policy": ("engine/src/gpu_probe.h", r"constexpr int kMinCudaDriverMajor"),
    "insufficient_driver": ("engine/src/engine.cpp", r"cudaErrorInsufficientDriver"),
    "memshare_disable": ("engine/src/engine_compute.cpp", r"// CUDA Memory Sharing: NVIDIA only, and only on known-good drivers"),
    "sticky": ("engine/src/engine_compute.cpp", r"CUDA disabled for this session"),
    "queue": ("engine/src/inference_service.cpp", r"// Bounded queue: wait for space"),
    "worker": ("engine/src/inference_service.cpp", r"^void InferenceService::worker_loop"),
    "job_owns": ("engine/src/inference_service.h", r"^struct InferenceJob"),
    "download_ui": ("plugin/ae/DMS_Depth_Dialog.cpp", r"^bool run_download_dialog"),
    "download_worker": ("engine/src/downloader.cpp", r"^dms_status Downloader::start"),
    "download_verify": ("engine/src/downloader.cpp", r"if \(got != spec.sha256\)"),
    "sanitize_params": ("plugin/core/params_model.cpp", r"^ParamSnapshot sanitize"),
    "nan_engine": ("engine/src/engine_compute.cpp", r"non-finite parameter"),
    "dims": ("engine/src/common.h", r"inline void check_dims"),
    "pixfmt": ("plugin/ae/DMS_Depth_Render.cpp", r"^bool pixel_depth_of"),
    "zero_size": ("plugin/core/render_core.cpp", r"// Zero-size / missing input: write transparent output"),
    "alpha": ("plugin/core/pixels.cpp", r"const float k = 0.5f \* \(1.f - a\);"),
    "subrect": ("plugin/core/render_core.cpp", r"^void write_output"),
    "prerender_rect": ("plugin/ae/DMS_Depth_Render.cpp", r"const Rect out_rect = intersect\(requested, pre->input_rect\);"),
    "maxwindow": ("plugin/ae/DMS_Depth_Render.cpp", r"constexpr int32_t kMaxWindow = 32;"),
    "dll_preload": ("engine/src/ort_runtime.cpp", r"preload_dependencies\(lib_path.parent_path\(\)\);"),
    "dll_pin": ("engine/src/ort_runtime.cpp", r"lib_.pin\(\);"),
    "dynamic_flags": ("plugin/ae/DMS_Depth_Params.cpp", r"^PF_Err query_dynamic_flags"),
    "pipl_assert": ("plugin/ae/DMS_Depth.cpp", r"^static_assert\(DMS_PIPL_OUTFLAGS =="),
    "param_order": ("plugin/core/params_model.h", r"^enum ParamIndex"),
    "param_frozen_test": ("tests/unit/test_plugin_core.cpp", r"TEST\(ParamsModel, ParamOrderIsFrozen\)"),
    "model_size_check": ("engine/src/engine_compute.cpp", r"file is incomplete or corrupt"),
    "oom_memo": ("engine/src/engine_compute.cpp", r"skipped \(out of memory recently\)"),
    "nonfinite_out": ("engine/src/engine_compute.cpp", r"model produced non-finite values"),
    "cache_pressure": ("engine/src/frame_cache.cpp", r"// Memory pressure: when the machine is low on RAM"),
    "pool": ("engine/src/inference_service.h", r"^class BufferPool"),
    "timer": ("engine/src/engine_compute.cpp", r"\| pre %.1f ms, session %.1f ms"),
    "gpu_once": ("plugin/ae/DMS_Depth_GPU.cpp", r"std::call_once\(once"),
    "message_dedup": ("engine/src/instance_registry.cpp", r"^bool InstanceRegistry::should_report"),
}


def locate(key):
    path, pat = REFS[key]
    with open(os.path.join(ROOT, path)) as f:
        s = f.read()
    m = re.search(pat, s, re.M)
    if not m:
        raise SystemExit(f"reference '{key}' not found: /{pat}/ in {path}")
    return f"`{path}:{s[:m.start()].count(chr(10)) + 1}`"


def render():
    with open(os.path.join(ROOT, "docs", "STABILITY.md.in")) as f:
        tmpl = f.read()
    return "<!-- Generated from STABILITY.md.in by tools/gen_stability_doc.py. Edit the template. -->\n" + re.sub(
        r"\{\{ref:([a-z0-9_]+)\}\}", lambda m: locate(m.group(1)), tmpl)


if __name__ == "__main__":
    out = render()
    dst = os.path.join(ROOT, "docs", "STABILITY.md")
    if "--check" in sys.argv:
        cur = open(dst).read() if os.path.exists(dst) else ""
        if cur != out:
            print("docs/STABILITY.md is out of date: run python3 -I tools/gen_stability_doc.py")
            sys.exit(1)
        print("docs/STABILITY.md is up to date")
    else:
        with open(dst, "w") as f:
            f.write(out)
        print("wrote", dst)
