#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

require_command uv
require_command npm

run_step "Syncing Python environment with uv" \
  uv sync --project "$ROOT_DIR"

run_step "Installing bridge dependencies" \
  npm install --prefix "$ROOT_DIR/bridge"

if [ -f "$ROOT_DIR/functions/package.json" ]; then
  run_step "Installing Firebase functions dependencies" \
    npm install --prefix "$ROOT_DIR/functions"
fi

echo ""
echo "Install complete."
echo "Next step: cd \"$ROOT_DIR/bridge\" && npm start"
