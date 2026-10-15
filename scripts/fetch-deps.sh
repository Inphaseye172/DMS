#!/usr/bin/env bash
# Linux developer setup: fetches the pinned CPU build of ONNX Runtime used by the engine tests and the
# CLI (CMake also does this automatically at configure time via cmake/FetchORT.cmake).
set -euo pipefail
VER=1.22.0
SHA=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3
DEST="${1:-$(dirname "$0")/../build/DMS_deps}"
CACHE="$(dirname "$0")/../build/_downloads"
mkdir -p "$CACHE" "$DEST/ort-cpu"
TGZ="$CACHE/onnxruntime-linux-x64-$VER.tgz"
if [ ! -f "$TGZ" ] || ! echo "$SHA  $TGZ" | sha256sum -c --status; then
  curl -fL --retry 4 -o "$TGZ.part" "https://github.com/microsoft/onnxruntime/releases/download/v$VER/onnxruntime-linux-x64-$VER.tgz"
  echo "$SHA  $TGZ.part" | sha256sum -c -
  mv "$TGZ.part" "$TGZ"
fi
tar -xzf "$TGZ" -C "$CACHE"
cp -L "$CACHE/onnxruntime-linux-x64-$VER/lib/libonnxruntime.so" "$DEST/ort-cpu/"
echo "ONNX Runtime $VER (CPU) in $DEST/ort-cpu"
