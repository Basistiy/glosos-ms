# syntax=docker/dockerfile:1.7

FROM golang:1.25-bookworm AS go-builder
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libopus-dev \
  && rm -rf /var/lib/apt/lists/*

COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY cmd ./cmd
COPY internal ./internal

RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=1 go build -tags nolibopusfile -o /out/pion_bridge_peer ./cmd/pion_bridge_peer

FROM node:20-bookworm-slim AS py-builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      uv \
      openai \
      onnxruntime \
      fastapi \
      uvicorn \
      python-multipart \
      webrtcvad \
      phonemizer \
      mlx-audio

FROM node:20-bookworm-slim AS runtime
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    libopus0 \
    espeak-ng \
  && rm -rf /var/lib/apt/lists/*

COPY --from=py-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY bridge/package*.json ./bridge/
RUN npm install --prefix ./bridge --omit=dev

COPY bridge/index.js ./bridge/index.js
COPY bridge/runtime_servers.js ./bridge/runtime_servers.js
COPY bridge/runtime_servers_runner.js ./bridge/runtime_servers_runner.js
COPY agent.py ./agent.py
COPY models ./models
COPY services/say_tts_server.py ./services/say_tts_server.py
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
COPY docker/start.sh ./docker/start.sh
COPY --from=go-builder /out/pion_bridge_peer /app/bin/pion_bridge_peer
RUN chmod +x /app/bin/pion_bridge_peer /app/docker/start.sh

ENV AUTO_START_MLX_SERVER=0 \
    AUTO_START_MLX_AUDIO_SERVER=0 \
    AUTO_START_SAY_TTS_SERVER=0 \
    MLX_RUNNER_COMMAND=python3 \
    MLX_AUDIO_RUNNER_COMMAND=python3 \
    SAY_TTS_RUNNER_COMMAND=python3 \
    PION_PEER_BINARY_PATH=/app/bin/pion_bridge_peer \
    SILERO_VAD_MODEL_PATH=/app/models/silero_vad.onnx

EXPOSE 8080

CMD ["/app/docker/start.sh"]
