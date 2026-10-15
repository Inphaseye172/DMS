# Architecture

DMS is an After Effects SmartFX effect that turns footage into per-pixel depth maps with local ONNX
inference (Depth Anything v2/v3 and a temporal video model). Stability inside the host comes first.
The design pushes almost all logic into code that can be built and tested without After Effects.

```
┌──────────────────────────────── After Effects / aerender ────────────────────────────────┐
│ plugin/ae  (Windows, AE SDK)          thin glue: PF_Cmd dispatch, params, sequence data,  │
│   DMS_Depth.cpp  _Params.cpp           SmartFX checkouts, dialogs, GPU worlds             │
│   _Render.cpp  _Dialog.cpp  _GPU.cpp   every handler inside try/catch -> PF_Err            │
└───────────────┬──────────────────────────────────────────────────────────────────────────┘
                │ WorldView (pointer + rowbytes + pixel layout), ParamSnapshot (POD)
┌───────────────▼──────────────────────────────────────────────────────────────────────────┐
│ plugin/core  (any OS, unit-tested)    pixel conversion 8/16/32 bpc + GPU BGRA128, alpha,   │
│   render_core: render(job) noexcept    content hashing, temporal window downscale, mapping/ │
│   params_model, sequence_data          slicing/colormap output for the requested sub-rect,  │
│                                        fallback to last-good / mid-gray                     │
└───────────────┬──────────────────────────────────────────────────────────────────────────┘
                │ C ABI: engine/include/dms/engine.h  (extern "C", status codes, no exceptions)
┌───────────────▼──────────────────────────────────────────────────────────────────────────┐
│ engine  (static lib, any OS)                                                               │
│   Engine singleton (refcounted) ── Manifest (models.json + override) ── Downloader (worker) │
│        │                                                                                    │
│        ├─ compute(): validate → frame cache → temporal store → backend plan → attempts      │
│        │     pre-process (caller thread) ──► InferenceService (1 worker thread, bounded    │
│        │     post-process / tiles / samples ◄── queue, sessions, ORT env, offload policy)  │
│        ├─ FrameCache (LRU, bytes)  InstanceRegistry (last good, message de-dup)            │
│        ├─ GpuMonitor (NVML / DXGI, dynamic)   OrtRuntime (onnxruntime loaded by path)      │
│        └─ postprocess: mapping, slicing, guided-filter edges, smoothing, colormaps          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
tools/depthcli: the same engine from the command line (tests, benchmarks, stress)
```

## Frame lifecycle (SmartFX)

1. **`PF_Cmd_SMART_PRE_RENDER`** reads every parameter once through RAII `PF_CHECKOUT_PARAM`. It sanitizes them into a POD `ParamSnapshot`, which becomes the pre-render data.
   - It checks out the full layer, because depth needs the whole frame. For the temporal model it also checks out the frames of the window chosen by `dms_temporal_window`, at most 32.
   - It returns `result_rect` = the requested rect ∩ the layer, and sets `GPU_RENDER_POSSIBLE` for CUDA.
2. **`PF_Cmd_SMART_RENDER` / `_GPU`** checks out the pixels (each checkout has an RAII check-in) and wraps them as `WorldView`s. GPU worlds are first copied to host memory through the CUDA Driver API. Then it calls `dms_plugin::render()`.
3. **`render()`** in plugin/core does the following:
   - Converts worlds to RGB float, compositing straight alpha over 50% gray.
   - Downscales temporal neighbours, keeping a full-resolution guide frame.
   - Hashes the content: this hash is the cache's "layer checksum".
   - Calls `dms_depth_compute`, maps the depth (Raw / Normalize / Guided), and writes only the output sub-rect, as grayscale, colormap, or slice (source RGB with alpha = mask).
   - On failure it uses the instance's last good depth, else flat mid-gray, and reports one message.
4. **`Engine::compute()`** in the engine does the following:
   - Validates everything.
   - Checks the frame cache, then the temporal store.
   - Builds the backend plan and walks it.
   - Applies edge enhancement and subpixel smoothing at full resolution, then caches the result.

Per-stage timings (pre-process, session, inference, post-process, total) are logged at debug level. The plugin adds checkout and copy-out time.

## Threading and Multi-Frame Rendering

- **No mutable globals in the plugin.** Global data is written once in `GLOBAL_SETUP`. Sequence data is read in render via `PF_GetConstSequenceData`. Per-instance runtime state lives in the engine, keyed by a GUID stored in the 96-byte POD sequence data.
- **Engine shared state** (cache, registry, plane stores, buffer pool, manifest, downloader) is guarded by mutexes. The engine singleton is a mutex-guarded `shared_ptr`, so an in-flight call keeps it alive.
- **All model execution happens on one worker thread** (`InferenceService`). That thread owns the ONNX Runtime env, the sessions and therefore the CUDA context. Render threads:
  - do their own pre-processing and post-processing in parallel;
  - submit jobs to a bounded queue;
  - wait on a condition variable in 50 ms slices, polling `PF_ABORT` on their own thread;
  - give up after a timeout (default 120 s).
- **Abandoned jobs are cancelled.** A job that is already running is terminated through `RunOptions::SetTerminate`. Jobs own their buffers through `shared_ptr`.

## Backends and the fallback chain

- **Library selection.** One ONNX Runtime library is loaded per process, by full path from `DMS_deps`, and never unloaded.
  - `ort-cuda` is used on NVIDIA with driver ≥ 527.41 when all CUDA DLLs are present.
  - Otherwise `ort-dml` (DirectML) is used when a D3D12 adapter exists.
  - The CPU execution provider is always available inside either build.
  - TensorRT is compiled in only with `DMS_ENABLE_TENSORRT`; plain CUDA is the default, as agreed.
- **Plan.** The requested configuration comes first. Then, if FP16 failed, FP32. Then lower base resolutions (1024 → 768 → 518 → 384). Then more tiles at a smaller per-tile resolution, which lowers peak memory. Then the CPU.
  - Before every session creation and every batch, free VRAM (NVML, or the DXGI budget) must exceed the estimate plus a 1 GiB reserve.
  - Out-of-memory rungs are memoized for 30 s. A CUDA initialization failure (for example an insufficient driver) disables CUDA for the session.
  - Results from reduced-quality rungs are not cached, so quality returns when the GPU recovers.
- **Lower precision.** This uses an FP16 model variant when the manifest provides one, TF32 math on CUDA otherwise, or TensorRT FP16. Non-finite output counts as a failure, which leads to the FP32 rung.
- **CUDA Memory Sharing** registers one shared CUDA arena in the ORT env, used by all sessions. It is enabled only on NVIDIA drivers ≥ 560.
- **Model Offloading.** "Keep in VRAM" keeps sessions resident. "Offload to RAM" drops idle GPU sessions but keeps the model bytes in RAM for fast re-creation. "Full Unload" drops both. Idle time defaults to 10 s.

## Quality features

- **Samples.** N runs with horizontal-flip and sub-pixel-jitter test-time augmentation, undone and averaged. When Parallel is on and Samples > 2, they are batched into one run; on out-of-memory this falls back to sequential runs.
- **Tiles (Auto / 4 / 9 / 16)** use a detail merge:
  - A global low-resolution pass supplies the low spatial frequencies.
  - Each tile is inferred at the base resolution with a 32 px overlap at tile inference resolution.
  - Each tile is affinely aligned to the global pass, because relative depth has a per-run scale and shift.
  - Each tile is high-pass filtered, and only that detail is blended in, with feathered weights.
  - Tiles therefore add detail but cannot introduce layout seams. On very textureless content at 16 tiles a faint residual step can remain at tile boundaries.
- **Temporal model.** Windows of W frames (W/2 with Memory Optimization) are scheduled with stride W − W/4. Each window runs once, and every frame's result is kept in a bounded temporal store.
  - Consecutive windows are aligned with a least-squares scale/shift fit on their shared frames, keyed by frame content, which removes flicker at boundaries.
  - Neighbours are downscaled in the plugin to bound host memory; a full-resolution guide frame drives the output size and edge refinement.
- **Edges Enhancement** is a guided filter of the depth, with image luminance as the guide. **Subpixel Smoothing** is an edge-preserving self-guided filter. Both radii scale with the up-sampling factor.
- **Mapping and colormaps.** Normalize maps to [0, 1] with white = near. Guided maps linearly through two keyframeable points, with robust 5×5 sampling, and handles degenerate cases. There are 20 append-only matplotlib LUTs.

## Models

- **Not bundled.** Model files are downloaded on first use to `%LOCALAPPDATA%\DMS\models` (overridable with `DMS_MODELS_DIR`).
- **Manifest.** `engine/src/models_manifest.json` is compiled in. A user `models.json` in the models folder overrides entries by id, adds models, and supplies the Depth Anything v3 and video URLs that this environment could not reach.
- **Downloader.** It runs on its own worker thread and resumes through an HTTP Range request plus a sidecar file that stores the SHA-256 mid-state. It checks disk space first and detects oversized responses and ignored Range requests. The file is moved into place only after SHA-256 verification. Only `https://` and `file://` sources are accepted.

## Parameters

Parameters are defined in `plugin/core/params_model.h`. The order is frozen, and a unit test fails if it changes. Disk IDs are stable. New parameters are appended, and `DMS_PARAMS_VERSION` and the effect version are bumped. Backend parameters and Model / Base Resolution are not keyframeable. The guided mapping parameters are keyframeable. Supervised parameters drive show/hide through AEGP dynamic stream flags in After Effects, `PF_PUI_INVISIBLE` in Premiere, and `PF_PUI_DISABLED` for enable state.

## Repository map

| Path | Contents |
|---|---|
| `engine/include/dms/engine.h` | The C ABI. It is the only header the plugin and the CLI include. |
| `engine/src/` | Engine implementation, one module per concern |
| `plugin/core/` | Host-independent render pipeline, params, sequence data |
| `plugin/ae/` | After Effects glue (Windows) |
| `tools/depthcli/` | CLI |
| `tools/ae_sdk_mock/` | Compile-only AE SDK subset for Linux CI |
| `tools/*.py` | Fixtures, colormaps, synthetic models, DLL closure check, stability doc generator |
| `tests/unit/` | GoogleTest suite (98 tests) |
| `tests/stress/` | `cli_leak_run.py` (any OS), `aerender_stress.ps1` + `make_stress_project.jsx` (Windows + AE) |
| `scripts/` | `fetch-deps.ps1`, `fetch-deps.sh`, `install.ps1` |
