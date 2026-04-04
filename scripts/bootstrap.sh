#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/basistiy/glosos-ms:latest}"
RUNTIME_DIR="${RUNTIME_DIR:-$PWD/glosos-runtime}"
CONTAINER_NAME="${CONTAINER_NAME:-glosos-ms}"
BRIDGE_PORT="${BRIDGE_PORT:-8080}"
MLX_AUDIO_HOST="${MLX_AUDIO_HOST:-127.0.0.1}"
MLX_AUDIO_PORT="${MLX_AUDIO_PORT:-8001}"
STT_MODEL="${AGENT_STT_MODEL:-mlx-community/Qwen3-ASR-0.6B-4bit}"
TTS_MODEL="${AGENT_TTS_MODEL:-mlx-community/kitten-tts-mini-0.8-8bit}"
STT_LANGUAGE="${AGENT_STT_LANGUAGE:-en}"
TTS_VOICE="${PION_TTS_VOICE:-expr-voice-5-m}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

wait_http() {
  local url="$1"
  local retries="${2:-60}"
  local delay="${3:-1}"
  local i
  for ((i=1; i<=retries; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

need_cmd docker
need_cmd curl
need_cmd python3

mkdir -p "$RUNTIME_DIR/logs"
ENV_FILE="$RUNTIME_DIR/bridge.env"
MLX_AUDIO_LOG="$RUNTIME_DIR/logs/mlx-audio.log"

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing required command: uv" >&2
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

ESPEAK_BIN=""
ESPEAK_LIB=""
if command -v brew >/dev/null 2>&1 && brew --prefix espeak-ng >/dev/null 2>&1; then
  ESPEAK_PREFIX="$(brew --prefix espeak-ng)"
  if [ -x "$ESPEAK_PREFIX/bin/espeak-ng" ]; then
    ESPEAK_BIN="$ESPEAK_PREFIX/bin/espeak-ng"
  fi
  if [ -f "$ESPEAK_PREFIX/lib/libespeak-ng.dylib" ]; then
    ESPEAK_LIB="$ESPEAK_PREFIX/lib/libespeak-ng.dylib"
  fi
fi
if [ -z "$ESPEAK_BIN" ] || [ -z "$ESPEAK_LIB" ]; then
  echo "Warning: espeak-ng not auto-detected. Kitten TTS may fail. Install with: brew install espeak-ng" >&2
fi

BRIDGE_EMAIL_VAL="${BRIDGE_EMAIL:-}"
BRIDGE_PASSWORD_VAL="${BRIDGE_PASSWORD:-}"
AGENT_API_KEY_VAL="${AGENT_API_KEY:-}"

if [ -z "$BRIDGE_EMAIL_VAL" ]; then
  read -r -p "BRIDGE_EMAIL: " BRIDGE_EMAIL_VAL
fi
if [ -z "$BRIDGE_PASSWORD_VAL" ]; then
  read -r -s -p "BRIDGE_PASSWORD: " BRIDGE_PASSWORD_VAL
  echo ""
fi
if [ -z "$AGENT_API_KEY_VAL" ]; then
  read -r -s -p "AGENT_API_KEY (Gemini): " AGENT_API_KEY_VAL
  echo ""
fi

if [ -z "$BRIDGE_EMAIL_VAL" ] || [ -z "$BRIDGE_PASSWORD_VAL" ] || [ -z "$AGENT_API_KEY_VAL" ]; then
  echo "BRIDGE_EMAIL, BRIDGE_PASSWORD and AGENT_API_KEY are required." >&2
  exit 1
fi

cat > "$ENV_FILE" <<EOT
BRIDGE_EMAIL=$BRIDGE_EMAIL_VAL
BRIDGE_PASSWORD=$BRIDGE_PASSWORD_VAL

BRIDGE_PORT=8080
WEBRTC_CALLS_COLLECTION=webrtc_calls

AUTO_START_MLX_SERVER=0
AUTO_START_MLX_AUDIO_SERVER=0
AUTO_START_SAY_TTS_SERVER=0
AUTO_CONFIGURE_PION_TTS_BASE_URL=0

AGENT_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
AGENT_API_KEY=$AGENT_API_KEY_VAL
AGENT_MODEL=${AGENT_MODEL:-gemini-2.5-flash}

AGENT_STT_PROVIDER=external
AGENT_STT_BASE_URL=http://host.docker.internal:${MLX_AUDIO_PORT}/v1
AGENT_STT_API_KEY=mlx-audio
AGENT_STT_MODEL=$STT_MODEL
AGENT_STT_LANGUAGE=$STT_LANGUAGE

AGENT_TTS_BASE_URL=http://host.docker.internal:${MLX_AUDIO_PORT}/v1
AGENT_TTS_API_KEY=mlx-audio
AGENT_TTS_MODEL=$TTS_MODEL
PION_TTS_VOICE=$TTS_VOICE
PION_TTS_LANGUAGE=en
EOT

VENV_DIR="$RUNTIME_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  uv venv --python "${UV_PYTHON:-3.11}" "$VENV_DIR"
fi

# Install via uv against the venv interpreter (works even when pip module is absent).
uv pip install --python "$VENV_DIR/bin/python" --quiet --upgrade pip
uv pip install --python "$VENV_DIR/bin/python" --quiet \
  "setuptools<81" \
  mlx-audio \
  soundfile \
  uvicorn \
  fastapi \
  python-multipart \
  webrtcvad \
  phonemizer

if wait_http "http://${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}/openapi.json" 2 1; then
  echo "mlx-audio already running at ${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}"
else
  echo "Starting mlx-audio server on ${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}..."
  PHONEMIZER_ESPEAK_PATH="$ESPEAK_BIN" \
  PHONEMIZER_ESPEAK_LIBRARY="$ESPEAK_LIB" \
  nohup "$VENV_DIR/bin/python" -m mlx_audio.server \
    --host "$MLX_AUDIO_HOST" \
    --port "$MLX_AUDIO_PORT" \
    >"$MLX_AUDIO_LOG" 2>&1 &
fi

if ! wait_http "http://${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}/openapi.json" 90 1; then
  echo "mlx-audio did not become ready. Check: $MLX_AUDIO_LOG" >&2
  exit 1
fi

echo "Warming up STT/TTS models..."
SILENCE_WAV="$RUNTIME_DIR/silence.wav"
"$VENV_DIR/bin/python" - <<PY
import wave
import struct
from pathlib import Path
p = Path("$SILENCE_WAV")
with wave.open(str(p), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    frames = b"".join(struct.pack("<h", 0) for _ in range(16000))
    wf.writeframes(frames)
PY

curl -fsS --max-time 120 \
  -X POST "http://${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}/v1/audio/transcriptions" \
  -H "Authorization: Bearer mlx-audio" \
  -F "model=$STT_MODEL" \
  -F "language=$STT_LANGUAGE" \
  -F "file=@${SILENCE_WAV};type=audio/wav" \
  >/dev/null || true

curl -fsS --max-time 120 \
  -X POST "http://${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mlx-audio" \
  --data "{\"model\":\"$TTS_MODEL\",\"input\":\"warmup\",\"response_format\":\"wav\"}" \
  >/dev/null || true

echo "Pulling container image: $IMAGE"
docker pull "$IMAGE"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting container: $CONTAINER_NAME"
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${BRIDGE_PORT}:8080" \
  --env-file "$ENV_FILE" \
  "$IMAGE" >/dev/null

echo ""
echo "Done."
echo "Bridge health:  http://127.0.0.1:${BRIDGE_PORT}/health"
echo "mlx-audio:      http://${MLX_AUDIO_HOST}:${MLX_AUDIO_PORT}/openapi.json"
echo "Container logs: docker logs -f ${CONTAINER_NAME}"
echo "Stop container: docker rm -f ${CONTAINER_NAME}"
echo "Runtime dir:    ${RUNTIME_DIR}"
