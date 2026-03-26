#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$ROOT_DIR/models"
MODEL_PATH="${SILERO_VAD_MODEL_PATH:-$MODEL_DIR/silero_vad.onnx}"
ONNX_RUNTIME_LIB_PATH_DEFAULT="${ONNX_RUNTIME_LIB_PATH:-/opt/homebrew/lib/libonnxruntime.dylib}"
SILERO_VAD_MODEL_URL="${SILERO_VAD_MODEL_URL:-https://github.com/snakers4/silero-vad/raw/refs/tags/v5.0/files/silero_vad.onnx}"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    exit 1
  fi
}

run_step() {
  local title="$1"
  shift
  echo ""
  echo "==> $title"
  "$@"
}

model_file_looks_valid() {
  local path="$1"
  [ -f "$path" ] || return 1
  local size
  size="$(wc -c <"$path")"
  [ "$size" -ge 1000000 ] || return 1
}

install_with_brew_if_missing() {
  local binary="$1"
  local formula="$2"
  if command -v "$binary" >/dev/null 2>&1; then
    return
  fi
  if ! command -v brew >/dev/null 2>&1; then
    echo "Missing required command: $binary" >&2
    echo "Install Homebrew or provide $binary manually." >&2
    exit 1
  fi
  run_step "Installing $formula with Homebrew" brew install "$formula"
}

ensure_brew_formula() {
  local formula="$1"
  if brew list "$formula" >/dev/null 2>&1; then
    return
  fi
  run_step "Installing $formula with Homebrew" brew install "$formula"
}

require_command uv
require_command npm
install_with_brew_if_missing pkg-config pkg-config
if ! command -v brew >/dev/null 2>&1; then
  echo "Missing required command: brew" >&2
  echo "Install Homebrew or provide pkg-config, opus, and onnxruntime manually." >&2
  exit 1
fi
ensure_brew_formula opus
ensure_brew_formula onnxruntime

if [ ! -f "$ONNX_RUNTIME_LIB_PATH_DEFAULT" ]; then
  echo "Expected ONNX Runtime shared library at $ONNX_RUNTIME_LIB_PATH_DEFAULT" >&2
  echo "Set ONNX_RUNTIME_LIB_PATH to your local libonnxruntime.dylib path before running the peer." >&2
fi

run_step "Syncing Python environment with uv" \
  uv sync --project "$ROOT_DIR"

run_step "Preparing Go dependencies" \
  env GOCACHE=/tmp/go-build go mod download

run_step "Installing bridge dependencies" \
  npm install --prefix "$ROOT_DIR/bridge"

if [ -f "$ROOT_DIR/functions/package.json" ]; then
  run_step "Installing Firebase functions dependencies" \
    npm install --prefix "$ROOT_DIR/functions"
fi

if ! model_file_looks_valid "$MODEL_PATH"; then
  require_command curl
  mkdir -p "$MODEL_DIR"
  rm -f "$MODEL_PATH"
  run_step "Downloading Silero VAD model" \
    curl -fL "$SILERO_VAD_MODEL_URL" -o "$MODEL_PATH"
fi

echo ""
echo "Install complete."
echo "Next step: cd \"$ROOT_DIR/bridge\" && npm start"
