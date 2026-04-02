#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ONNX_RUNTIME_LIB_PATH:-}" ]; then
  ONNX_RUNTIME_LIB_PATH="$(/opt/venv/bin/python3 - <<'PY'
import pathlib
import onnxruntime
base = pathlib.Path(onnxruntime.__file__).resolve().parent / "capi"
candidates = sorted(base.glob("libonnxruntime.so*"))
if not candidates:
    candidates = sorted(base.glob("libonnxruntime*.so*"))
if not candidates:
    raise SystemExit(f"No ONNX Runtime shared library found under {base}")
print(candidates[0])
PY
)"
  export ONNX_RUNTIME_LIB_PATH
fi

exec npm --prefix /app/bridge start
