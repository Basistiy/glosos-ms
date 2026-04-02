#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ONNX_RUNTIME_LIB_PATH:-}" ]; then
  ONNX_RUNTIME_LIB_PATH="$(python3 - <<'PY'
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

RUNTIME_PID=""

if [ "${AUTO_START_RUNTIME_SERVERS:-1}" != "0" ]; then
  npm --prefix /app/bridge run start:runtime &
  RUNTIME_PID="$!"
fi

cleanup() {
  if [ -n "${RUNTIME_PID}" ]; then
    kill "${RUNTIME_PID}" 2>/dev/null || true
    wait "${RUNTIME_PID}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

npm --prefix /app/bridge start
