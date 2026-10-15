# Building, installing and verifying on Windows

The engine, the CLI and the tests build and run on Linux too (see the README). This page covers the
`.aex` for After Effects 2026, which needs Windows, MSVC and the After Effects SDK.

## Prerequisites

- Windows 11 x64, Visual Studio 2022 (Desktop C++ workload), CMake ≥ 3.25 (the VS one is fine), Git.
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
cmake --build --preset windows-release          # -> build\DMS_Depth.aex, build\windows-release\bin\Release\depthcli.exe
ctest --preset windows-release                  # unit tests (engine, plugin core) on Windows
```

Other presets:

| Preset | Purpose |
|---|---|
| `windows-engine-only` | Engine, CLI and tests without the AE SDK |
| `windows-debug` | AddressSanitizer on engine/CLI/tests. The plugin is not built, because ASan must never be loaded into AE |
| `windows-plugin-debug` | Debug `.aex` for attaching Visual Studio to `AfterFX.exe` |
| `windows-analyze` | MSVC `/analyze` |

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
build\windows-release\bin\Release\depthcli.exe --device
build\windows-release\bin\Release\depthcli.exe --download da2-small
build\windows-release\bin\Release\depthcli.exe -i tests\fixtures\scene_4k.png -o out --colormap Turbo --tiles 4 --log-level 3
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
