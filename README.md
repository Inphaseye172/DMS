# DMS Depth: native AI depth maps for After Effects

DMS Depth is a native C++ SmartFX effect for After Effects 2026 (Windows, NVIDIA first). It generates per-pixel depth maps from footage and stills with local ONNX inference: Depth Anything v2 and v3, plus a temporal video model for flicker-free output. The effect is MFR-safe. It falls back gracefully on every GPU or model failure and never downloads or blocks inside a render.

| Component | Path | Builds / runs on |
|---|---|---|
| Inference engine (static lib, C ABI) | `engine/` (`engine/include/dms/engine.h`) | Linux, Windows |
| Host-independent render pipeline | `plugin/core/` | Linux, Windows |
| After Effects glue → `DMS_Depth.aex` | `plugin/ae/` | Windows + AE SDK |
| Command-line runner | `tools/depthcli/` | Linux, Windows |
| Unit tests (GoogleTest) | `tests/unit/` | Linux, Windows |
| Stress tests | `tests/stress/` | CLI: any OS; aerender: Windows + AE |

Documentation:

- `docs/ARCHITECTURE.md`: design and data flow.
- `docs/STABILITY.md`: every crash-prevention rule, with code locations and verification status.
- `docs/BUILD_WINDOWS.md`: build, install, first run and stress testing.
- `THIRD_PARTY.md`: pinned versions and SHA-256 hashes.

## Quick start on Linux (engine + CLI, CPU)

```bash
cmake --preset linux-release && cmake --build --preset linux-release && ctest --preset linux-release
./build/linux-release/bin/depthcli --download da2-small                    # 99 MB, SHA-256 verified, resumable
./build/linux-release/bin/depthcli -i tests/fixtures/scene_floor.png -o out --colormap Turbo
./build/linux-release/bin/depthcli -i tests/fixtures/scene_4k.png -o out --tiles 4 --edges 50 --smoothing 20
python3 -I tests/stress/cli_leak_run.py --cli build/linux-release/bin/depthcli --models-dir ~/.local/state/dms/models
```

Presets: `linux-debug` (ASan/UBSan), `linux-release`, `linux-tidy` (Clang + clang-tidy). The configure step downloads the pinned ONNX Runtime 1.22.0 and verifies its hash.

## Windows / After Effects

See `docs/BUILD_WINDOWS.md`. In short:

1. Run `scripts\fetch-deps.ps1`.
2. Set `AE_SDK_ROOT`.
3. Run `cmake --preset windows-release`, then build.
4. Run `scripts\install.ps1` from an elevated shell.
