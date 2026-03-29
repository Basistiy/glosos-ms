# say TTS Server (Python)

Minimal OpenAI-compatible TTS endpoint for macOS.

## Run

```bash
cd /Users/evgeniibasistyi/Documents/GitHub/glosos-ms
uv run python services/say_tts_server.py --host 127.0.0.1 --port 8112
```

## Test

```bash
curl -i http://127.0.0.1:8112/health

curl -sS --max-time 40 \
  -X POST http://127.0.0.1:8112/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data '{"model":"any","input":"Hello from simple python server","voice":"Samantha","response_format":"wav"}' \
  --output /tmp/tts-say.wav

file /tmp/tts-say.wav
```

## Pion config

```bash
export PION_TTS_BASE_URL=http://127.0.0.1:8112
```

`pion_bridge_peer` will call:

- `POST $PION_TTS_BASE_URL/v1/audio/speech`
