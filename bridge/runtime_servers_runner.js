import "dotenv/config";
import path from "path";
import { fileURLToPath } from "url";
import { createRuntimeServers } from "./runtime_servers.js";

function intEnv(name, fallback) {
  const raw = (process.env[name] || "").trim();
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return parsed;
}

function prefixStream(stream, prefix) {
  let buffer = "";
  stream.on("data", (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.length > 0) {
        console.log(`${prefix} ${line}`);
      }
    }
  });
  stream.on("end", () => {
    if (buffer.length > 0) {
      console.log(`${prefix} ${buffer}`);
      buffer = "";
    }
  });
}

const bridgeDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(bridgeDir, "..");

const config = {
  autoStartMlxServer: (process.env.AUTO_START_MLX_SERVER || "1").trim() !== "0",
  mlxRunnerCommand: (process.env.MLX_RUNNER_COMMAND || "uv").trim(),
  mlxModel: (process.env.MLX_MODEL || "mlx-community/Qwen3-4B-Instruct-2507-4bit").trim(),
  mlxHost: (process.env.MLX_HOST || "127.0.0.1").trim(),
  mlxPort: intEnv("MLX_PORT", 8000),
  mlxRestartDelayMs: intEnv("MLX_RESTART_DELAY_MS", 3000),
  autoStartMlxAudioServer: (process.env.AUTO_START_MLX_AUDIO_SERVER || "1").trim() !== "0",
  mlxAudioRunnerCommand: (process.env.MLX_AUDIO_RUNNER_COMMAND || "uv").trim(),
  mlxAudioHost: (process.env.MLX_AUDIO_HOST || "127.0.0.1").trim(),
  mlxAudioPort: intEnv("MLX_AUDIO_PORT", 8001),
  mlxAudioRestartDelayMs: intEnv("MLX_AUDIO_RESTART_DELAY_MS", 3000),
  autoStartSayTTSServer: (process.env.AUTO_START_SAY_TTS_SERVER || "0").trim() !== "0",
  sayTtsRunnerCommand: (process.env.SAY_TTS_RUNNER_COMMAND || "uv").trim(),
  sayTtsHost: (process.env.SAY_TTS_HOST || "127.0.0.1").trim(),
  sayTtsPort: intEnv("SAY_TTS_PORT", 8112),
  sayTtsRestartDelayMs: intEnv("SAY_TTS_RESTART_DELAY_MS", 3000),
  sayTtsScriptPath: (process.env.SAY_TTS_SCRIPT_PATH || path.join(repoRoot, "services", "say_tts_server.py")).trim(),
  agentSttProvider: (process.env.AGENT_STT_PROVIDER || "mlx_audio").trim()
};

const runtimeServers = createRuntimeServers({ config, repoRoot, prefixStream });

runtimeServers.startMlxServer();
runtimeServers.startMlxAudioServer();
runtimeServers.startSayTtsServer();

console.log("[runtime] runtime servers started");

function shutdown() {
  runtimeServers.stopMlxServer();
  runtimeServers.stopMlxAudioServer();
  runtimeServers.stopSayTtsServer();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
