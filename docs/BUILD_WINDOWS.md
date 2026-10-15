# Building, installing and verifying on Windows

## Quick start (recommended)

```powershell
git clone -b claude/ae-depth-map-plugin-5iubui https://github.com/Inphaseye172/DMS.git C:\src\dms
cd C:\src\dms
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 -AeSdk "C:\path\to\unzipped\AfterEffectsSDK" -Install
```

`scripts\setup-windows.ps1` runs these steps:

1. Checks the prerequisites and prints the exact `winget` command for anything missing.
2. Fetches the pinned DLLs and checks that every DLL import resolves.
3. Builds the plug-in and the CLI, then runs the unit tests.
4. Downloads Depth Anything v2 Small and times it on your GPU and on the CPU.
5. With `-Install`, installs the plug-in into After Effects 2026 (one UAC prompt; After Effects must be closed).

Everything goes to `build\setup-log.txt`. It is safe to re-run.

- **No AE SDK yet:** leave out `-AeSdk` and the script still builds and tests the engine, the CLI and the real GPU path.
- **Rough timing:** expect about 2.3 GB of downloads and about 10 minutes the first time.

Then run the **After Effects checklist** at the end of this page.

The manual steps below are what the script automates.

The engine, the CLI and the tests build and run on Linux too (see the README). This page covers the
`.aex` for After Effects 2026, which needs Windows, MSVC and the After Effects SDK.

## Prerequisites

- Windows 10/11 x64, Visual Studio 2022 or 2026 (Desktop C++ workload), CMake ≥ 3.25 that knows your Visual Studio version (the one bundled with VS is fine), Git. Presets use the newest installed Visual Studio.
- After Effects SDK 2025 or 2026 from <https://developer.adobe.com/after-effects/>. Unzip it, then set
  `AE_SDK_ROOT` to the folder that contains `Examples` (or to `Examples` itself).
- For the NVIDIA path, an NVIDIA driver ≥ 527.41 (≥ 560 for CUDA Memory Sharing). No CUDA toolkit is needed.
- Optional: Python 3 for `tools/check_deps.py`.

## 1. Fetch runtime dependencies (pinned, SHA-256 verified)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch-deps.ps1          # ~2.3 GB download, cached in build\_downloads
python tools\check_deps.py build\DMS_deps                                 # every DLL import must resolve: expect "OK"
```

## 2. Build

```powershell
$env:AE_SDK_ROOT = "C:\SDKs\AfterEffectsSDK"
cmake --preset windows-release
cmake --build --preset windows-release          # -> build\DMS_Depth.aex, build\windows-release\bin\depthcli.exe
ctest --preset windows-release                  # unit tests (engine, plugin core) on Windows
```

Other presets:

| Preset | Purpose |
|---|---|
| `windows-engine-only` | Engine, CLI and tests without the AE SDK |
| `windows-debug` | AddressSanitizer on engine/CLI/tests. The plugin is not built, because ASan must never be loaded into AE |
| `windows-plugin-debug` | Debug `.aex` for attaching Visual Studio to `AfterFX.exe` |
| `windows-analyze` | MSVC `/analyze` |

All targets link the MSVC runtime statically (`/MT`), so the `.aex` never binds to the older
`msvcp140.dll` that After Effects ships. That makes building with Visual Studio 2026 safe for the plug-in.

If the SDK headers differ from what the glue expects, every SDK-specific name is centralized in `plugin/ae/ae_includes.h` (suite versions) and the four `plugin/ae/DMS_Depth*.cpp` files. The items to check first are listed under "Known risks" in `docs/STABILITY.md`.

## 3. Install (requires elevation)

```powershell
# From an elevated PowerShell ("Run as administrator"); After Effects must be closed.
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#   -> C:\Program Files\Adobe\Adobe After Effects 2026\Support Files\Plug-ins\DMS\DMS_Depth.aex (+ DMS_deps\)
# Use -MediaCore to install into the shared Adobe MediaCore folder (also seen by Premiere Pro).
# Use -Uninstall to remove it.
```

## 4. First run

1. In After Effects 2026, apply **Effect > DMS > DMS Depth Map** to a footage layer.
2. With no model installed, the layer renders flat gray and shows one message. Click **Options** in Effect Controls and choose **Download model**. The dialog shows progress and has Cancel; the download resumes if interrupted.
3. The log is `%LOCALAPPDATA%\DMS\logs\dms.log`. It records the backend, the fallbacks and per-stage timings; set the log level to debug for timings.

Depth Anything v3 and the video model need download sources. Put them in `%LOCALAPPDATA%\DMS\models\models.json`:

```json
{"models": [
  {"id": "video-da-small", "kind": "temporal", "window": 32, "input_layout": "BTCHW",
   "file": "video_depth_anything_vits.onnx", "url": "https://.../video_depth_anything_vits.onnx",
   "sha256": "<64 hex>", "size": 123456789}
]}
```

## 5. CLI on Windows

```powershell
build\windows-release\bin\depthcli.exe --device
build\windows-release\bin\depthcli.exe --download da2-small
build\windows-release\bin\depthcli.exe -i tests\fixtures\scene_4k.png -o out --colormap Turbo --tiles 4 --log-level 3
```

`depthcli` looks for `DMS_deps` next to the executable. Copy `build\DMS_deps` there, or set `DMS_DEPS_DIR`.

## 6. Headless stress test (aerender)

```powershell
python tools\make_fixtures.py --4k     # only if tests\fixtures\scene_4k.png is missing
powershell -ExecutionPolicy Bypass -File tests\stress\aerender_stress.ps1 -Models 1,2,3
```

The script builds a 4K × 500-frame comp for every model × mapping × tiles (1/4/16) combination with `make_stress_project.jsx`. It renders each comp with `aerender`. A comp fails on a non-zero exit, a timeout over 10 minutes, missing frames, or a working-set/VRAM upward trend after warm-up.

The report is written to `tests\stress\build\aerender_stress_report.json`. Pass `-OMTemplate` if your After Effects has no "PNG Sequence" output-module template.

## What to send back if something fails

- **Build:** the first 30 compiler errors.
- **Load or render problems:** `%LOCALAPPDATA%\DMS\logs\dms.log`, the AE version, the GPU model and driver version, and the output of `depthcli --device`.
- **Stress failures:** `aerender_stress_report.json` and the matching `<comp>.log`.

## After Effects checklist (first session)

Open the log in a text editor while you test: `%LOCALAPPDATA%\DMS\logs\dms.log`.

| # | Do | Expect |
|---|---|---|
| 1 | Start AE 2026, open the **Effects & Presets** panel | **DMS > DMS Depth Map** is listed. If it is missing, check that `Plug-ins\DMS\DMS_Depth.aex` exists, then look at the log |
| 2 | Apply the effect to a 1080p footage layer | A grayscale depth map appears, white = near. The first frame is slower because the session is created |
| 3 | Scrub a few frames, then go back | Revisited frames are instant (frame cache) |
| 4 | **Mapping Mode = Guided** | Near/Far Guide Points and Near/Far Values appear. Moving a point changes the mapping |
| 5 | **Mode = Slicing** | Minimum/Maximum appear, Colormap hides; only the selected depth band stays visible |
| 6 | **Mode = Depth Map, Colormap = Turbo** | A colored depth map |
| 7 | **Samples = 3** | **Parallel samples** becomes enabled |
| 8 | Open **Backend & Performance**, turn **Hardware Acceleration** off and on | Renders on CPU (slower) and back on the GPU. The log line `depth ... backend=` shows CUDA or CPU |
| 9 | **Model = Depth Anything v2 Base** (not downloaded) | The download dialog opens with progress and Cancel. Cancel, then reopen: it resumes |
| 10 | **Model = Video Depth** | One message: no download source configured. **Memory Optimization** becomes enabled |
| 11 | Click **Options** in Effect Controls | The dialog shows the GPU, driver, VRAM, backends and model status |
| 12 | Set the project to 32 bpc and render 100 frames with Multi-Frame Rendering on | No errors; memory stays flat in Task Manager |
| 13 | Save, close and reopen the project | Parameters are preserved and the effect renders |
| 14 | Turn **Computation Tiles = 16** on a 4K layer | Slower but works, with no visible tile seams |

Send back anything that differs, along with `dms.log`.
