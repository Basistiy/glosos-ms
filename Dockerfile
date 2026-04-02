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

FROM node:20-bookworm AS runtime
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ca-certificates \
    curl \
    libopus0 \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir uv openai onnxruntime

COPY bridge/package*.json ./bridge/
RUN npm install --prefix ./bridge --omit=dev

COPY . .
COPY --from=go-builder /out/pion_bridge_peer /app/bin/pion_bridge_peer
RUN chmod +x /app/bin/pion_bridge_peer /app/docker/start.sh

ENV AUTO_START_MLX_SERVER=0 \
    AUTO_START_MLX_AUDIO_SERVER=0 \
    AUTO_START_SAY_TTS_SERVER=0 \
    PION_PEER_BINARY_PATH=/app/bin/pion_bridge_peer \
    SILERO_VAD_MODEL_PATH=/app/models/silero_vad.onnx

EXPOSE 8080

CMD ["/app/docker/start.sh"]
