# Verification record

This file records what was actually run, as opposed to what was only written. Everything below ran on 2026-10-15 in a Linux x86-64 container: 4 vCPU, 16 GB RAM, **no GPU**, GCC 13.3, Clang 18, CMake 3.28, and ONNX Runtime 1.22.0 (CPU build).

## Ran and passed

| Check | Result |
|---|---|
| Unit tests, `linux-debug` (GCC, `-Werror`, ASan + UBSan + LSan) | 99/99 pass |
| Unit tests, `linux-release` and a Clang 18 build | 99/99 pass |
| clang-tidy (`linux-tidy`, warnings as errors) on engine, plugin core, AE glue (against `tools/ae_sdk_mock`) and CLI | clean |
| AE glue compile check against the SDK mock | compiles; found and fixed a missing include and a non-portable `'PrMr'` literal |
| Depth Anything v2 Small downloaded by the engine's own downloader (99 MB, SHA-256 verified) | OK |
| DA2-small engine output vs an independent Python ONNX Runtime reference | Pearson correlation 1.00000 on two fixtures |
| DA2-small depth ordering on the floor scene (near floor > horizon) | correct (opt-in test `RealModel.*`) |
| Fallback chain with injected CUDA OOM, driver failure and FP16 NaN, and a below-reserve VRAM check (fake NVIDIA GPU) | all rungs behave as specified |
| Tiling seam test (per-input-normalizing model, 4/9/16 tiles) | matches the single pass within 5% of range; fails by 30–50% if alignment is disabled |
| Temporal flicker test (window 8, 24 frames) | 4 inferences instead of 24; boundary jump under 2× the in-window change (7× without alignment) |
| `scripts/fetch-deps.ps1` under PowerShell 7.4 (download cache pre-seeded from the same pinned files) | every SHA-256 verified, layout created |
| `tools/check_deps.py` on that layout (17 CUDA-folder DLLs, 3 DirectML-folder DLLs) | every import resolves |
| PowerShell parse check of `fetch-deps.ps1`, `install.ps1`, `aerender_stress.ps1` | parse OK |

## CLI stress / leak matrix (`tests/stress/cli_leak_run.py`)

Each configuration ran in one depthcli process with the cache disabled, so every frame ran inference. Pass criteria: exit code 0, under 600 s, and RSS growth and trend under max(48 MB, 8%) after warm-up.

| Model | Mappings | Tiles | Frames per run | Result | RSS growth after warm-up |
|---|---|---|---|---|---|
| Synthetic single-frame (per-input-normalizing) | raw, normalize, guided | 1, 4, 16 | 500–504 | 9/9 PASS | −1.8 to +4.8 MB, peak 43 MB |
| Synthetic temporal (window 8) | raw, normalize, guided | 1, 4, 16 | 504 | 9/9 PASS | −3.4 to +1.3 MB, peak 85 MB |
| Depth Anything v2 Small (real model, CPU) | raw, normalize, guided | 1, 4, 16 | 60 | 9/9 PASS | 0 to +4.0 MB, peak 326 MB |

27 of 27 runs passed. The DA2-small 16-tile runs took 583–600 s on this 4-vCPU CPU-only machine, right at the 600 s limit. That is CPU throughput (17 inferences per frame), not a hang. On a GPU these runs are expected to be far faster, but that was not measured.

## Windows run on the developer's machine (2026-10-15, `scripts/setup-windows.ps1`)

Machine: Windows 10 Pro 19045, Visual Studio Community 2026 18.11 (CMake 4.3.1, generator "Visual Studio 18 2026"),
Windows SDK 10.0.26100, NVIDIA GeForce RTX 3080 10 GB with driver 591.44, After Effects 2026, After Effects SDK 26.5.

| Check | Result |
|---|---|
| Runtime dependencies fetched; every DLL import resolves | OK |
| `DMS_Depth.aex` built against the real AE SDK 26.5 (`/W4 /WX`, static CRT) | OK, 0.8 MB, no errors and no warnings |
| Unit tests (MSVC build) | 99/99 pass |
| Engine device report | CUDA available, ONNX Runtime 1.22.0 (CUDA build loaded; DirectML is the fallback build for non-NVIDIA GPUs) |
| DA2-small model download through WinHTTP (SHA-256 verified) | OK |
| Inference with hardware acceleration, 640×360, 6 frames, cache off | **CUDA** backend, average 171.7 ms (includes the first frame's session creation), no fallbacks |
| Same on the CPU | average 486.9 ms |
| 4K (3840×2160), 4 tiles, edges + smoothing, Turbo colormap | CUDA, 1582 ms, no fallbacks |
| Install into After Effects 2026 (elevated) | OK |

Known issue from this run: on a re-run, `fetch-deps.ps1` re-hashed and re-extracted every archive silently, which looked
like a hang. This is fixed: re-runs now finish in about a second and show progress.

## Not run yet

- Loading the plug-in in After Effects 2026 and the UI checklist in `docs/BUILD_WINDOWS.md` (in progress).
- `tests/stress/aerender_stress.ps1`.
- DirectML and TensorRT backends, CUDA Memory Sharing, the GPU render path (`SMART_RENDER_GPU`).
- Depth Anything v3 and the video model (no download source configured yet).

## Originally not run (needs Windows / an NVIDIA GPU / After Effects)

- Building `DMS_Depth.aex` with MSVC against the real AE SDK, including PiPL compilation.
- Loading the plugin in After Effects 2026, the Effect Controls UI (show/hide, Options dialog, download dialog), and MFR rendering.
- `tests/stress/aerender_stress.ps1` (4K × 500 frames per model × mapping × tiles).
- CUDA, DirectML and TensorRT execution providers; real NVML and DXGI readings; CUDA Memory Sharing; the GPU render path (`SMART_RENDER_GPU`).
- Depth Anything v3 and the video model: no download source was reachable from this environment, so the manifest entries need URLs and checksums. The temporal path ran only with a synthetic model of the same I/O signature.

The items to confirm first on Windows are listed under "Known risks" in `docs/STABILITY.md`.
