# Third-party components

All versions are pinned. Every downloaded artifact is verified with SHA-256 before use
(`cmake/FetchORT.cmake`, `scripts/fetch-deps.ps1`, `scripts/fetch-deps.sh`). Hashes below were
computed from the actual downloads on 2026-10-15.

## Runtime (shipped next to the plugin in `DMS_deps/`, loaded dynamically)

| Component | Version | Artifact | SHA-256 | License |
|---|---|---|---|---|
| ONNX Runtime (CUDA build, Windows) | 1.22.0 | `onnxruntime-win-x64-gpu-1.22.0.zip` (GitHub release) | `5b5241716b2628c1ab5e79ee620be767531021149ee68f30fc46c16263fb94dd` | MIT |
| ONNX Runtime (DirectML build, Windows) | 1.22.0 | `Microsoft.ML.OnnxRuntime.DirectML.1.22.0.nupkg` (GitHub release) | `b38d92c31d202d1dc772af9416fe30b6abae8527651e742c380c2d131e5f1817` | MIT |
| DirectML redistributable (required by the ORT DirectML build, which delay-loads `DirectML.dll` >= 1.15) | 1.15.4 | `microsoft.ai.directml.1.15.4.nupkg` (api.nuget.org) | `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9` | Microsoft DirectML license (redistributable) |
| ONNX Runtime (CPU, Linux; dev/CI only) | 1.22.0 | `onnxruntime-linux-x64-1.22.0.tgz` (GitHub release) | `8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3` | MIT |
| NVIDIA CUDA runtime | 12.9.79 | `nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl` (PyPI) | `8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891` | NVIDIA EULA (redistributable) |
| NVIDIA cuBLAS / cuBLASLt | 12.9.2.10 | `nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl` (PyPI) | `623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661` | NVIDIA EULA (redistributable) |
| NVIDIA cuFFT | 11.4.1.4 | `nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl` (PyPI) | `8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80` | NVIDIA EULA (redistributable) |
| NVIDIA cuDNN | 9.27.0.42 | `nvidia_cudnn_cu12-9.27.0.42-py3-none-win_amd64.whl` (PyPI) | `06e9b0026f3bad97d2b58666330fabec04fe1672f776661ecb0ce0029c27f142` | NVIDIA cuDNN SLA (redistributable) |

The NVIDIA DLL set was derived from the PE import table of `onnxruntime_providers_cuda.dll` 1.22.0:
`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `cufft64_11.dll`, `cudnn64_9.dll`
(plus the `cudnn_*64_9.dll` sub-libraries that cuDNN loads on demand). cuRAND is not imported and is
not shipped. `nvml.dll` comes with the NVIDIA driver and is never redistributed.

All NVIDIA DLLs are placed in the same folder as the CUDA build of ONNX Runtime (`DMS_deps/ort-cuda/`) and
`DirectML.dll` next to the DirectML build (`DMS_deps/ort-dml/`). The engine pre-loads them by full path
before ONNX Runtime needs them, so by-name resolution (delay loads, cuDNN sub-libraries) never picks up an
older `DirectML.dll` from System32 or a different CUDA from `PATH`.

## Models (downloaded on first use, never bundled; checksums in `engine/src/models_manifest.json`)

| Model | Source | Size | SHA-256 | License |
|---|---|---|---|---|
| Depth Anything v2 Small (dynamic shapes) | github.com/fabio-sim/Depth-Anything-ONNX release v2.0.0 `depth_anything_v2_vits_dynamic.onnx` | 99,092,268 B | `46c4e8eeda3a27f34701831b6a2ec7753d7b38779b215acb5633424703deed8f` | Apache-2.0 |
| Depth Anything v2 Base (dynamic shapes) | same release, `depth_anything_v2_vitb_dynamic.onnx` | 388,949,900 B | `d823dddd4f363cbd0ef73cb9ade3a6ab56b60c7a425ebbbab3fa8c0cd936e33e` | CC-BY-NC-4.0 (check before commercial use) |
| Depth Anything v2 Large (dynamic shapes) | same release, `depth_anything_v2_vitl_dynamic.onnx` | 1,336,975,076 B | `65432320391cf5e18ceb36d1c6eeb1a59281b5ed9ae4600d26bf25519043c8b0` | CC-BY-NC-4.0 (check before commercial use) |
| Depth Anything v3, Video Depth | **URL + checksum to be supplied** (HuggingFace was unreachable from the build environment) | - | - | - |

## Build / test only

| Component | Version | How it is obtained | Hash | License |
|---|---|---|---|---|
| GoogleTest | 1.15.2 | CMake FetchContent, git tag pinned to commit `b514bdc898e2951020cbdca1304b75f5950d1f59` | git commit | BSD-3-Clause |
| stb_image | 2.30 | vendored `third_party/stb/stb_image.h` | `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3` | Public domain / MIT |
| stb_image_write | 1.16 | vendored `third_party/stb/stb_image_write.h` | `cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05` | Public domain / MIT |
| nlohmann/json | 3.11.3 | vendored `third_party/nlohmann/json.hpp` | `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6` | MIT |
| Adobe After Effects SDK | 2025/2026 | user-supplied via `AE_SDK_ROOT` (not redistributable) | - | Adobe SDK license |

Python tooling (`tools/*.py`, fixtures and colormap generation only, never shipped): numpy, Pillow,
onnx, matplotlib.
