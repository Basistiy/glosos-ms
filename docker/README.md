# Docker quick start

## Build

```bash
docker build -t glosos-ms:local .
```

## Run

```bash
docker run --rm -it \
  -p 8080:8080 \
  -e BRIDGE_EMAIL='your-firebase-email' \
  -e BRIDGE_PASSWORD='your-firebase-password' \
  -e AGENT_API_BASE='https://your-llm-endpoint.example/v1' \
  -e AGENT_API_KEY='your-llm-api-key' \
  -e AGENT_MODEL='your-llm-model' \
  -e AGENT_STT_BASE_URL='https://your-stt-endpoint.example/v1' \
  -e AGENT_STT_API_KEY='your-stt-api-key' \
  -e AGENT_STT_MODEL='your-stt-model' \
  -e AGENT_TTS_BASE_URL='https://your-tts-endpoint.example/v1' \
  -e AGENT_TTS_API_KEY='your-tts-api-key' \
  -e AGENT_TTS_MODEL='your-tts-model' \
  glosos-ms:local
```

Notes:
- The container auto-detects `ONNX_RUNTIME_LIB_PATH` from the installed Python `onnxruntime` package.
- Silero VAD model path defaults to `/app/models/silero_vad.onnx`.
- MLX and macOS `say` autostart are disabled in-container.
