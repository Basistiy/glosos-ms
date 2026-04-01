#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ONNX_RUNTIME_LIB_PATH:-}" ]; then
  ONNX_RUNTIME_LIB_PATH="$(python3 - <<'PY'
import pathlib
import onnxruntime
print(pathlib.Path(onnxruntime.__file__).resolve().parent / "capi" / "libonnxruntime.so")
PY
)"
  export ONNX_RUNTIME_LIB_PATH
fi

exec npm --prefix /app/bridge start
