# DMS Depth — native AI depth maps for After Effects

Native C++ After Effects (2026, Windows, NVIDIA-first) SmartFX effect that generates per-pixel depth
maps from footage with local ONNX inference (Depth Anything v2/v3 and a temporal video model).

* `engine/` — inference engine static library behind a C ABI (`engine/include/dms/engine.h`).
* `plugin/core/` — host-independent render pipeline (unit-tested on any OS).
* `plugin/ae/` — After Effects glue (Windows + AE SDK).
* `tools/depthcli/` — command-line runner for the engine.
* `docs/ARCHITECTURE.md`, `docs/STABILITY.md`, `docs/BUILD_WINDOWS.md`, `THIRD_PARTY.md`.

Quick start (Linux dev box, CPU):

```bash
cmake --preset linux-release && cmake --build --preset linux-release && ctest --preset linux-release
./build/linux-release/bin/depthcli --download da2-small
./build/linux-release/bin/depthcli -i tests/fixtures/scene_floor.png -o out --colormap Turbo
```

Windows / After Effects: see `docs/BUILD_WINDOWS.md`.
