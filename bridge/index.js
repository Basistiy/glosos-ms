import "dotenv/config";
import express from "express";
import crypto from "crypto";
import { spawn, spawnSync } from "child_process";
import path from "path";
import { createInterface } from "readline/promises";
import { fileURLToPath } from "url";
import { deleteApp, initializeApp } from "firebase/app";
import {
  getAuth,
  signInWithEmailAndPassword
} from "firebase/auth";
import {
  getFunctions,
  httpsCallable
} from "firebase/functions";
import {
  getFirestore,
  addDoc,
  collection,
  doc,
  getDoc,
  onSnapshot,
  serverTimestamp,
  setDoc,
  updateDoc,
  deleteField
} from "firebase/firestore";

function mustEnv(name) {
  const value = (process.env[name] || "").trim();
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

function intEnv(name, fallback) {
  const raw = (process.env[name] || "").trim();
  if (!raw) return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return parsed;
}

const firebaseConfig = {
  apiKey: "AIzaSyA1iO6LzNaq9dwPb71m014p29_lUHwnkbs",
  authDomain: "glosos-103f7.firebaseapp.com",
  projectId: "glosos-103f7",
  storageBucket: "glosos-103f7.firebasestorage.app",
  messagingSenderId: "314422729512",
  appId: "1:314422729512:web:4fb8cb0278e64a5c374e1d",
};

const bridgeDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(bridgeDir, "..");

const config = {
  firebase: firebaseConfig,
  bridgeEmail: (process.env.BRIDGE_EMAIL || "").trim(),
  bridgePassword: (process.env.BRIDGE_PASSWORD || "").trim(),
  callsCollection: (process.env.WEBRTC_CALLS_COLLECTION || "webrtc_calls").trim(),
  autoRestartFirebase: (process.env.AUTO_RESTART_FIREBASE || "1").trim() !== "0",
  firebaseRestartIntervalMs: intEnv("FIREBASE_RESTART_INTERVAL_MS", 15 * 60 * 1000),
  port: intEnv("BRIDGE_PORT", 8080),
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
  autoConfigurePionTtsBaseUrl: (process.env.AUTO_CONFIGURE_PION_TTS_BASE_URL || "1").trim() !== "0",
  turnFunctionRegion: (process.env.TURN_FUNCTION_REGION || "europe-west1").trim(),
  turnFunctionName: (process.env.TURN_FUNCTION_NAME || "getTurnCredentials").trim(),
  turnServer: (process.env.TURN_SERVER || "54.37.235.123:3478").trim(),
  turnAuthSecret: (process.env.TURN_AUTH_SECRET || "").trim(),
  turnTTLSeconds: intEnv("TURN_TTL_SECONDS", 86400),
  turnRefreshIntervalSeconds: intEnv("TURN_REFRESH_INTERVAL_SECONDS", 82800),
  autoStartPionPeer: (process.env.AUTO_START_PION_PEER || "1").trim() !== "0",
  pionPeerRestartDelayMs: intEnv("PION_PEER_RESTART_DELAY_MS", 3000),
  autoStartAgent: (process.env.AUTO_START_AGENT || "1").trim() !== "0",
  agentRunnerCommand: (process.env.AGENT_RUNNER_COMMAND || "uv").trim(),
  agentApiBase: (process.env.AGENT_API_BASE || "").trim(),
  agentApiKey: (process.env.AGENT_API_KEY || "mlx-local").trim(),
  agentModel: (process.env.AGENT_MODEL || "").trim(),
  agentSttProvider: (process.env.AGENT_STT_PROVIDER || "mlx_audio").trim(),
  agentSttBaseUrl: (process.env.AGENT_STT_BASE_URL || "").trim(),
  agentSttApiKey: (process.env.AGENT_STT_API_KEY || "mlx-audio").trim(),
  agentSttModel: (process.env.AGENT_STT_MODEL || "mlx-community/Qwen3-ASR-0.6B-4bit").trim(),
  agentSttLanguage: (process.env.AGENT_STT_LANGUAGE || "en").trim(),
  agentTtsModel: (process.env.AGENT_TTS_MODEL || "mlx-community/kitten-tts-mini-0.8-8bit").trim(),
  googleSttProjectId: (process.env.GOOGLE_STT_PROJECT_ID || "").trim(),
  googleSttLocation: (process.env.GOOGLE_STT_LOCATION || "global").trim(),
  googleSttRecognizer: (process.env.GOOGLE_STT_RECOGNIZER || "_").trim(),
  googleSttModel: (process.env.GOOGLE_STT_MODEL || "").trim(),
  agentDisableAudioTranscription: (process.env.AGENT_DISABLE_AUDIO_TRANSCRIPTION || "0").trim() === "1",
  agentRestartDelayMs: intEnv("AGENT_RESTART_DELAY_MS", 3000),
  agentRequiredImports: (process.env.AGENT_REQUIRED_IMPORTS || "openai").split(",")
    .map((item) => item.trim())
    .filter(Boolean)
};

const turnState = {
  cachedCredentials: null,
  refreshPromise: null,
  refreshTimer: null
};

const peerState = {
  process: null,
  restartTimer: null,
  shuttingDown: false
};

const mlxState = {
  process: null,
  restartTimer: null,
  shuttingDown: false
};

const mlxAudioState = {
  process: null,
  restartTimer: null,
  shuttingDown: false
};

const sayTtsState = {
  process: null,
  restartTimer: null,
  shuttingDown: false
};

const agentState = {
  process: null,
  restartTimer: null,
  shuttingDown: false,
  lineBuffer: "",
  nextRequestID: 1,
  pending: new Map()
};

const firebaseState = {
  app: null,
  auth: null,
  functions: null,
  db: null,
  ownerUID: null,
  credentials: null,
  restartTimer: null,
  reinitializing: false
};

async function getBridgeCredentials() {
  let email = config.bridgeEmail;
  let password = config.bridgePassword;

  if (email && password) {
    return { email, password };
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout
  });

  try {
    if (!email) {
      email = (await rl.question("Firebase user email: ")).trim();
    }
    if (!password) {
      password = (await rl.question("Firebase password: ")).trim();
    }
  } finally {
    rl.close();
  }

  if (!email) {
    throw new Error("Firebase user email is required");
  }
  if (!password) {
    throw new Error("Firebase password is required");
  }

  return { email, password };
}

const app = express();
app.use(express.json({ limit: "1mb" }));

const sessions = new Map();

function normalizeSDP(sdp) {
  if (typeof sdp !== "string") return "";
  // Normalize any line ending style to CRLF for strict SDP parsers.
  return sdp.replace(/\r\n|\r|\n/g, "\r\n");
}

function timestampMillis(value) {
  if (!value) return 0;
  if (typeof value.toMillis === "function") return value.toMillis();
  if (value instanceof Date) return value.getTime();
  if (typeof value.seconds === "number") return (value.seconds * 1000);
  return 0;
}

function makeTurnCredentials({ peerId, ttlSeconds }) {
  if (!config.turnAuthSecret) {
    throw new Error("TURN_AUTH_SECRET is not configured");
  }
  const ttl = Math.max(60, Math.min(86400, Number.isFinite(ttlSeconds) ? ttlSeconds : config.turnTTLSeconds));
  const expiry = Math.floor(Date.now() / 1000) + ttl;
  const username = `${expiry}:${peerId}`;
  const credential = crypto
    .createHmac("sha1", config.turnAuthSecret)
    .update(username)
    .digest("base64");

  return {
    turnServer: config.turnServer,
    username,
    credential,
    ttlSeconds: ttl,
    expiresAtUnix: expiry
  };
}

async function fetchTurnCredentialsFromFirebase(functions, { ttlSeconds }) {
  const callable = httpsCallable(functions, config.turnFunctionName);
  const result = await callable({
    ttlSeconds,
    turnServer: config.turnServer
  });
  return result.data;
}

function turnCredentialsAreFresh(credentials, minRemainingSeconds = 120) {
  if (!credentials?.expiresAtUnix) {
    return false;
  }
  const nowUnix = Math.floor(Date.now() / 1000);
  return credentials.expiresAtUnix - nowUnix > minRemainingSeconds;
}

async function refreshTurnCredentials(functions) {
  if (turnState.refreshPromise) {
    return turnState.refreshPromise;
  }

  turnState.refreshPromise = (async () => {
    const credentials = await fetchTurnCredentialsFromFirebase(functions, {
      ttlSeconds: config.turnTTLSeconds
    });
    turnState.cachedCredentials = credentials;
    const expiresIn = credentials?.expiresAtUnix
      ? credentials.expiresAtUnix - Math.floor(Date.now() / 1000)
      : null;
    console.log(
      `[bridge] TURN credentials refreshed${Number.isFinite(expiresIn) ? ` expires_in=${expiresIn}s` : ""}`
    );
    return credentials;
  })();

  try {
    return await turnState.refreshPromise;
  } finally {
    turnState.refreshPromise = null;
  }
}

async function getTurnCredentials(functions) {
  if (turnCredentialsAreFresh(turnState.cachedCredentials)) {
    return turnState.cachedCredentials;
  }
  return refreshTurnCredentials(functions);
}

function startTurnRefreshLoop(functions) {
  const intervalMs = Math.max(60, config.turnRefreshIntervalSeconds) * 1000;
  if (turnState.refreshTimer) {
    clearInterval(turnState.refreshTimer);
  }
  turnState.refreshTimer = setInterval(() => {
    refreshTurnCredentials(functions).catch((err) => {
      console.error("[bridge] TURN refresh failed", err);
    });
  }, intervalMs);
}

function stopTurnRefreshLoop() {
  if (turnState.refreshTimer) {
    clearInterval(turnState.refreshTimer);
    turnState.refreshTimer = null;
  }
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

function clearPeerRestartTimer() {
  if (peerState.restartTimer) {
    clearTimeout(peerState.restartTimer);
    peerState.restartTimer = null;
  }
}

function clearFirebaseRestartTimer() {
  if (firebaseState.restartTimer) {
    clearTimeout(firebaseState.restartTimer);
    firebaseState.restartTimer = null;
  }
}

function mlxApiBase() {
  return `http://${config.mlxHost}:${config.mlxPort}/v1`;
}

function mlxAudioApiBase() {
  return `http://${config.mlxAudioHost}:${config.mlxAudioPort}/v1`;
}

function sayTtsApiBase() {
  return `http://${config.sayTtsHost}:${config.sayTtsPort}/v1`;
}

function clearMlxRestartTimer() {
  if (mlxState.restartTimer) {
    clearTimeout(mlxState.restartTimer);
    mlxState.restartTimer = null;
  }
}

function clearMlxAudioRestartTimer() {
  if (mlxAudioState.restartTimer) {
    clearTimeout(mlxAudioState.restartTimer);
    mlxAudioState.restartTimer = null;
  }
}

function clearSayTtsRestartTimer() {
  if (sayTtsState.restartTimer) {
    clearTimeout(sayTtsState.restartTimer);
    sayTtsState.restartTimer = null;
  }
}

function ensureMlxRuntimeReady() {
  const runnerCheck = spawnSync(config.mlxRunnerCommand, ["--version"], {
    cwd: repoRoot,
    env: process.env,
    encoding: "utf8"
  });
  if (runnerCheck.error) {
    throw new Error(
      `MLX runner '${config.mlxRunnerCommand}' is not available: ${runnerCheck.error.message}`
    );
  }
  if (runnerCheck.status !== 0) {
    throw new Error(
      `MLX runner '${config.mlxRunnerCommand}' failed its version check: ${runnerCheck.stderr || runnerCheck.stdout}`
    );
  }

  const importCheck = spawnSync(config.mlxRunnerCommand, ["run", "python", "-c", "import mlx_lm"], {
    cwd: repoRoot,
    env: process.env,
    encoding: "utf8"
  });
  if (importCheck.error) {
    throw new Error(`MLX dependency check failed: ${importCheck.error.message}`);
  }
  if (importCheck.status !== 0) {
    throw new Error(
      `Missing Python dependencies for MLX server. Run 'uv sync' in ${repoRoot}. ` +
      `Details: ${importCheck.stderr || importCheck.stdout}`
    );
  }
}

function ensureMlxAudioRuntimeReady() {
  const runnerCheck = spawnSync(config.mlxAudioRunnerCommand, ["--version"], {
    cwd: repoRoot,
    env: process.env,
    encoding: "utf8"
  });
  if (runnerCheck.error) {
    throw new Error(
      `MLX audio runner '${config.mlxAudioRunnerCommand}' is not available: ${runnerCheck.error.message}`
    );
  }
  if (runnerCheck.status !== 0) {
    throw new Error(
      `MLX audio runner '${config.mlxAudioRunnerCommand}' failed its version check: ${runnerCheck.stderr || runnerCheck.stdout}`
    );
  }

  const importCheck = spawnSync(
    config.mlxAudioRunnerCommand,
    ["run", "python", "-c", "import mlx_audio, uvicorn, fastapi, webrtcvad, multipart, phonemizer"],
    {
      cwd: repoRoot,
      env: process.env,
      encoding: "utf8"
    }
  );
  if (importCheck.error) {
    throw new Error(`MLX audio dependency check failed: ${importCheck.error.message}`);
  }
  if (importCheck.status !== 0) {
    throw new Error(
      `Missing Python dependencies for MLX audio server. Run 'uv sync' in ${repoRoot}. ` +
      `Details: ${importCheck.stderr || importCheck.stdout}`
    );
  }

}

function ensureSayTtsRuntimeReady() {
  const runnerCheck = spawnSync(config.sayTtsRunnerCommand, ["--version"], {
    cwd: repoRoot,
    env: process.env,
    encoding: "utf8"
  });
  if (runnerCheck.error) {
    throw new Error(
      `say-tts runner '${config.sayTtsRunnerCommand}' is not available: ${runnerCheck.error.message}`
    );
  }
  if (runnerCheck.status !== 0) {
    throw new Error(
      `say-tts runner '${config.sayTtsRunnerCommand}' failed its version check: ${runnerCheck.stderr || runnerCheck.stdout}`
    );
  }

  const importCheck = spawnSync(
    config.sayTtsRunnerCommand,
    ["run", "python", "-c", "import fastapi, uvicorn"],
    {
      cwd: repoRoot,
      env: process.env,
      encoding: "utf8"
    }
  );
  if (importCheck.error) {
    throw new Error(`say-tts dependency check failed: ${importCheck.error.message}`);
  }
  if (importCheck.status !== 0) {
    throw new Error(
      `Missing Python dependencies for say-tts server. Run 'uv sync' in ${repoRoot}. ` +
      `Details: ${importCheck.stderr || importCheck.stdout}`
    );
  }
}

function scheduleMlxRestart() {
  if (mlxState.shuttingDown || !config.autoStartMlxServer) {
    return;
  }
  clearMlxRestartTimer();
  mlxState.restartTimer = setTimeout(() => {
    startMlxServer();
  }, Math.max(1000, config.mlxRestartDelayMs));
}

function scheduleMlxAudioRestart() {
  if (mlxAudioState.shuttingDown || !config.autoStartMlxAudioServer || config.agentSttProvider !== "mlx_audio") {
    return;
  }
  clearMlxAudioRestartTimer();
  mlxAudioState.restartTimer = setTimeout(() => {
    startMlxAudioServer();
  }, Math.max(1000, config.mlxAudioRestartDelayMs));
}

function scheduleSayTtsRestart() {
  if (sayTtsState.shuttingDown || !config.autoStartSayTTSServer) {
    return;
  }
  clearSayTtsRestartTimer();
  sayTtsState.restartTimer = setTimeout(() => {
    startSayTtsServer();
  }, Math.max(1000, config.sayTtsRestartDelayMs));
}

function startMlxServer() {
  if (!config.autoStartMlxServer || mlxState.process) {
    return;
  }

  ensureMlxRuntimeReady();

  const args = [
    "run",
    "python",
    "-m",
    "mlx_lm.server",
    "--model",
    config.mlxModel,
    "--host",
    config.mlxHost,
    "--port",
    String(config.mlxPort)
  ];
  const child = spawn(config.mlxRunnerCommand, args, {
    cwd: repoRoot,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"]
  });

  mlxState.process = child;
  console.log(`[bridge] started MLX server model=${config.mlxModel} base=${mlxApiBase()}`);
  prefixStream(child.stdout, "[mlx]");
  prefixStream(child.stderr, "[mlx]");

  child.on("error", (err) => {
    console.error("[bridge] failed to start MLX server", err);
  });

  child.on("exit", (code, signal) => {
    mlxState.process = null;
    console.log(`[bridge] MLX server exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    scheduleMlxRestart();
  });
}

function startMlxAudioServer() {
  if (!config.autoStartMlxAudioServer || config.agentSttProvider !== "mlx_audio" || mlxAudioState.process) {
    return;
  }

  ensureMlxAudioRuntimeReady();

  const args = [
    "run",
    "python",
    "-m",
    "mlx_audio.server",
    "--host",
    config.mlxAudioHost,
    "--port",
    String(config.mlxAudioPort)
  ];
  const child = spawn(config.mlxAudioRunnerCommand, args, {
    cwd: repoRoot,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"]
  });

  mlxAudioState.process = child;
  console.log(`[bridge] started MLX audio server base=${mlxAudioApiBase()}`);
  prefixStream(child.stdout, "[mlx-audio]");
  prefixStream(child.stderr, "[mlx-audio]");

  child.on("error", (err) => {
    console.error("[bridge] failed to start MLX audio server", err);
  });

  child.on("exit", (code, signal) => {
    mlxAudioState.process = null;
    console.log(`[bridge] MLX audio server exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    scheduleMlxAudioRestart();
  });
}

function startSayTtsServer() {
  if (!config.autoStartSayTTSServer || sayTtsState.process) {
    return;
  }

  ensureSayTtsRuntimeReady();

  const args = [
    "run",
    "python",
    config.sayTtsScriptPath,
    "--host",
    config.sayTtsHost,
    "--port",
    String(config.sayTtsPort)
  ];
  const child = spawn(config.sayTtsRunnerCommand, args, {
    cwd: repoRoot,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"]
  });

  sayTtsState.process = child;
  console.log(`[bridge] started say-tts server base=${sayTtsApiBase()}`);
  prefixStream(child.stdout, "[say-tts]");
  prefixStream(child.stderr, "[say-tts]");

  child.on("error", (err) => {
    console.error("[bridge] failed to start say-tts server", err);
  });

  child.on("exit", (code, signal) => {
    sayTtsState.process = null;
    console.log(`[bridge] say-tts server exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    scheduleSayTtsRestart();
  });
}

function stopMlxServer() {
  mlxState.shuttingDown = true;
  clearMlxRestartTimer();
  if (mlxState.process) {
    mlxState.process.kill("SIGTERM");
  }
}

function stopMlxAudioServer() {
  mlxAudioState.shuttingDown = true;
  clearMlxAudioRestartTimer();
  if (mlxAudioState.process) {
    mlxAudioState.process.kill("SIGTERM");
  }
}

function stopSayTtsServer() {
  sayTtsState.shuttingDown = true;
  clearSayTtsRestartTimer();
  if (sayTtsState.process) {
    sayTtsState.process.kill("SIGTERM");
  }
}

function schedulePeerRestart(bridgeURL) {
  if (peerState.shuttingDown || !config.autoStartPionPeer) {
    return;
  }
  clearPeerRestartTimer();
  peerState.restartTimer = setTimeout(() => {
    startPionPeer(bridgeURL);
  }, Math.max(1000, config.pionPeerRestartDelayMs));
}

function startPionPeer(bridgeURL) {
  if (!config.autoStartPionPeer) {
    return;
  }
  if (peerState.process) {
    return;
  }

  const args = [
    "run",
    "-tags",
    "nolibopusfile",
    "./cmd/pion_bridge_peer",
    "-bridge-url",
    bridgeURL
  ];
  const childEnv = { ...process.env };
  const selectedPionTtsModel = (childEnv.PION_TTS_MODEL || childEnv.AGENT_TTS_MODEL || config.agentTtsModel || "").trim().toLowerCase();
  const useSayTts = selectedPionTtsModel === "say-tts";
  if (config.autoStartSayTTSServer && config.autoConfigurePionTtsBaseUrl && useSayTts) {
    const ttsBaseURL = `http://${config.sayTtsHost}:${config.sayTtsPort}`;
    if (!childEnv.PION_TTS_BASE_URL || !childEnv.PION_TTS_BASE_URL.trim()) {
      childEnv.PION_TTS_BASE_URL = ttsBaseURL;
    }
    if (!childEnv.AGENT_TTS_BASE_URL || !childEnv.AGENT_TTS_BASE_URL.trim()) {
      childEnv.AGENT_TTS_BASE_URL = ttsBaseURL;
    }
  }
  const child = spawn("go", args, {
    cwd: repoRoot,
    env: childEnv,
    stdio: ["ignore", "pipe", "pipe"]
  });

  peerState.process = child;
  console.log("[bridge] started pion peer");
  prefixStream(child.stdout, "[pion]");
  prefixStream(child.stderr, "[pion]");

  child.on("error", (err) => {
    console.error("[bridge] failed to start pion peer", err);
  });

  child.on("exit", (code, signal) => {
    peerState.process = null;
    console.log(`[bridge] pion peer exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    schedulePeerRestart(bridgeURL);
  });
}

function shutdownPeer() {
  peerState.shuttingDown = true;
  clearPeerRestartTimer();
  if (peerState.process) {
    peerState.process.kill("SIGTERM");
  }
}

function clearAgentRestartTimer() {
  if (agentState.restartTimer) {
    clearTimeout(agentState.restartTimer);
    agentState.restartTimer = null;
  }
}

function ensureAgentRuntimeReady() {
  const runnerCheck = spawnSync(config.agentRunnerCommand, ["--version"], {
    cwd: repoRoot,
    env: process.env,
    encoding: "utf8"
  });
  if (runnerCheck.error) {
    throw new Error(
      `Agent runner '${config.agentRunnerCommand}' is not available: ${runnerCheck.error.message}`
    );
  }
  if (runnerCheck.status !== 0) {
    throw new Error(
      `Agent runner '${config.agentRunnerCommand}' failed its version check: ${runnerCheck.stderr || runnerCheck.stdout}`
    );
  }

  const requiredImports = [...config.agentRequiredImports];
  if (requiredImports.length === 0) {
    return;
  }

  const importScript = requiredImports.map((name) => `import ${name}`).join("; ");
  const importCheck = spawnSync(config.agentRunnerCommand, ["run", "python", "-c", importScript], {
    cwd: repoRoot,
    env: process.env,
    encoding: "utf8"
  });
  if (importCheck.error) {
    throw new Error(`Agent dependency check failed: ${importCheck.error.message}`);
  }
  if (importCheck.status !== 0) {
    throw new Error(
      `Missing Python dependencies for agent: ${requiredImports.join(", ")}. ` +
      `Run 'uv sync' in ${repoRoot}. Details: ${importCheck.stderr || importCheck.stdout}`
    );
  }
}

function scheduleAgentRestart() {
  if (agentState.shuttingDown || !config.autoStartAgent) {
    return;
  }
  clearAgentRestartTimer();
  agentState.restartTimer = setTimeout(() => {
    startAgentProcess();
  }, Math.max(1000, config.agentRestartDelayMs));
}

function rejectPendingAgentRequests(message) {
  for (const { reject } of agentState.pending.values()) {
    reject(new Error(message));
  }
  agentState.pending.clear();
}

function detachSessionListeners(session) {
  for (const timer of Object.values(session.retryTimers)) {
    if (timer) {
      clearTimeout(timer);
    }
  }
  session.retryTimers = {};
  for (const unsubscribe of session.unsubscribers) {
    try {
      unsubscribe();
    } catch (_) {
      // no-op
    }
  }
  session.unsubscribers = [];
}

function attachSessionListeners(session, db) {
  session.callRef = doc(db, config.callsCollection, session.callId);
  session.callerCandidatesRef = collection(session.callRef, "callerCandidates");
  session.calleeCandidatesRef = collection(session.callRef, "calleeCandidates");

  const remoteDescriptionField = session.role === "caller" ? "answer" : "offer";
  const remoteCandidatesRef = session.role === "caller" ? session.calleeCandidatesRef : session.callerCandidatesRef;

  const startDescriptionWatch = () => {
    const unsubscribe = onSnapshot(session.callRef, (snap) => {
      if (!snap.exists()) return;
      const data = snap.data() || {};
      const desc = data[remoteDescriptionField];
      if (!desc || typeof desc.sdp !== "string" || typeof desc.type !== "string") return;
      if (session.remoteDescription && session.remoteDescription.sdp === desc.sdp) return;
      session.remoteDescription = { type: desc.type, sdp: normalizeSDP(desc.sdp) };
      session.remoteDescriptionVersion += 1;
      console.log(
        `[bridge] call=${session.callId} role=${session.role} remote description updated (${remoteDescriptionField})`
      );
    }, (err) => {
      console.error(`[bridge] call=${session.callId} doc watch error`, err);
      scheduleWatcherRetry(session, "description", startDescriptionWatch);
    });
    session.unsubscribers.push(unsubscribe);
  };

  const startCandidatesWatch = () => {
    const unsubscribe = onSnapshot(remoteCandidatesRef, (snap) => {
      for (const change of snap.docChanges()) {
        if (change.type !== "added") continue;
        const data = change.doc.data();
        if (!data?.candidate) continue;
        const createdAtMillis = timestampMillis(data.createdAt);
        if (createdAtMillis > 0 && createdAtMillis < session.startedAt) {
          continue;
        }
        const key = `${change.doc.id}:${data.candidate}`;
        if (session.seenRemoteCandidateKeys.has(key)) {
          continue;
        }
        session.seenRemoteCandidateKeys.add(key);
        const item = {
          id: session.nextCandidateID++,
          candidate: data.candidate,
          sdpMid: data.sdpMid ?? null,
          sdpMLineIndex: data.sdpMLineIndex ?? null,
          usernameFragment: data.usernameFragment ?? null
        };
        session.remoteCandidates.push(item);
      }
    }, (err) => {
      console.error(`[bridge] call=${session.callId} candidates watch error`, err);
      scheduleWatcherRetry(session, "candidates", startCandidatesWatch);
    });
    session.unsubscribers.push(unsubscribe);
  };

  startDescriptionWatch();
  startCandidatesWatch();
}

async function initializeFirebaseRuntime(credentials) {
  const firebaseApp = initializeApp(config.firebase, `bridge-${Date.now()}`);
  const auth = getAuth(firebaseApp);
  const functions = getFunctions(firebaseApp, config.turnFunctionRegion);
  const db = getFirestore(firebaseApp);

  await signInWithEmailAndPassword(auth, credentials.email, credentials.password);
  const user = auth.currentUser;
  if (!user) {
    throw new Error("firebase sign-in succeeded but currentUser is missing");
  }

  const selfCheckDoc = doc(db, config.callsCollection, "_bridge_healthcheck");
  try {
    const snap = await getDoc(selfCheckDoc);
    console.log(`[bridge] firebase auth ok uid=${user.uid} healthcheck_doc_exists=${snap.exists()}`);
  } catch (err) {
    console.warn(`[bridge] firebase auth ok uid=${user.uid}; healthcheck read skipped: ${err.message || String(err)}`);
  }

  const previousApp = firebaseState.app;
  stopTurnRefreshLoop();

  firebaseState.app = firebaseApp;
  firebaseState.auth = auth;
  firebaseState.functions = functions;
  firebaseState.db = db;
  firebaseState.ownerUID = user.uid;
  firebaseState.credentials = credentials;

  app.locals.db = db;
  app.locals.functions = functions;
  app.locals.ownerUID = user.uid;

  for (const session of sessions.values()) {
    detachSessionListeners(session);
    attachSessionListeners(session, db);
  }

  await refreshTurnCredentials(functions);
  startTurnRefreshLoop(functions);

  if (previousApp) {
    await deleteApp(previousApp).catch(() => {});
  }
}

function scheduleFirebaseRestart() {
  if (!config.autoRestartFirebase) {
    return;
  }
  clearFirebaseRestartTimer();
  firebaseState.restartTimer = setTimeout(async () => {
    if (firebaseState.reinitializing || !firebaseState.credentials) {
      scheduleFirebaseRestart();
      return;
    }
    firebaseState.reinitializing = true;
    console.log(
      `[bridge] refreshing firebase runtime interval_ms=${Math.max(60_000, config.firebaseRestartIntervalMs)}`
    );
    try {
      await initializeFirebaseRuntime(firebaseState.credentials);
    } catch (err) {
      console.error("[bridge] firebase runtime refresh failed", err);
    } finally {
      firebaseState.reinitializing = false;
      scheduleFirebaseRestart();
    }
  }, Math.max(60_000, config.firebaseRestartIntervalMs));
}

function handleAgentStdout(chunk) {
  agentState.lineBuffer += chunk.toString();
  const lines = agentState.lineBuffer.split(/\r?\n/);
  agentState.lineBuffer = lines.pop() || "";

  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }
    let payload;
    try {
      payload = JSON.parse(line);
    } catch (err) {
      console.log(`[agent] ${line}`);
      continue;
    }

    const requestID = payload?.id;
    if (!requestID || !agentState.pending.has(requestID)) {
      console.log(`[agent] unexpected response ${line}`);
      continue;
    }

    const pending = agentState.pending.get(requestID);
    if (payload.error) {
      agentState.pending.delete(requestID);
      pending.reject(new Error(payload.error));
      continue;
    }

    if (pending.stream) {
      if (typeof pending.onEvent === "function") {
        try {
          pending.onEvent(payload);
        } catch (err) {
          console.error("[bridge] agent stream event handler error", err);
        }
      }
      if (payload.done) {
        agentState.pending.delete(requestID);
        pending.resolve(payload);
      }
      continue;
    }

    agentState.pending.delete(requestID);
    pending.resolve(payload);
  }
}

function startAgentProcess() {
  if (!config.autoStartAgent || agentState.process) {
    return;
  }

  ensureAgentRuntimeReady();

  const agentApiBase = config.agentApiBase || mlxApiBase();
  const agentModel = config.agentModel || config.mlxModel;
  const args = [
    "run",
    "agent.py",
    "--stdio",
    "--api-base",
    agentApiBase,
    "--api-key",
    config.agentApiKey,
    "--model",
    agentModel
  ];
  const child = spawn(config.agentRunnerCommand, args, {
    cwd: repoRoot,
    env: process.env,
    stdio: ["pipe", "pipe", "pipe"]
  });

  agentState.process = child;
  agentState.lineBuffer = "";
  console.log("[bridge] started agent process");

  child.stdout.on("data", handleAgentStdout);
  prefixStream(child.stderr, "[agent]");

  child.on("error", (err) => {
    console.error("[bridge] failed to start agent", err);
  });

  child.on("exit", (code, signal) => {
    agentState.process = null;
    rejectPendingAgentRequests("agent process exited");
    console.log(`[bridge] agent exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    scheduleAgentRestart();
  });
}

function stopAgentProcess() {
  agentState.shuttingDown = true;
  clearAgentRestartTimer();
  if (agentState.process) {
    agentState.process.kill("SIGTERM");
  }
}

async function chatWithAgent(message, { reset = false } = {}) {
  if (!message.trim()) {
    throw new Error("message is required");
  }
  const response = await sendAgentRequest({ type: "chat", message, reset });
  return String(response?.response ?? "");
}

async function streamChatWithAgent(message, { reset = false, onDelta } = {}) {
  if (!message.trim()) {
    throw new Error("message is required");
  }
  const response = await sendAgentRequest(
    { type: "chat", message, reset, stream: true },
    {
      onEvent: (event) => {
        const delta = typeof event?.delta === "string" ? event.delta : "";
        if (!delta || typeof onDelta !== "function") {
          return;
        }
        onDelta(delta);
      }
    }
  );
  return String(response?.response ?? "");
}

async function sendAgentRequest(request, { onEvent } = {}) {
  if (!agentState.process || !agentState.process.stdin || agentState.process.killed) {
    throw new Error("agent process is not running");
  }

  const id = `req-${agentState.nextRequestID++}`;
  const payload = JSON.stringify({ id, ...request });
  const stream = Boolean(request?.stream);

  return new Promise((resolve, reject) => {
    agentState.pending.set(id, { resolve, reject, stream, onEvent });
    agentState.process.stdin.write(`${payload}\n`, (err) => {
      if (!err) {
        return;
      }
      agentState.pending.delete(id);
      reject(err);
    });
  });
}

function scheduleWatcherRetry(session, watchName, startWatch) {
  if (session.closed) {
    return;
  }
  if (session.retryTimers[watchName]) {
    return;
  }
  session.retryTimers[watchName] = setTimeout(() => {
    session.retryTimers[watchName] = null;
    if (!session.closed) {
      startWatch();
    }
  }, 2000);
}

function makeSession(callId, role, db) {
  const session = {
    callId,
    role,
    callRef: null,
    callerCandidatesRef: null,
    calleeCandidatesRef: null,
    remoteDescription: null,
    remoteDescriptionVersion: 0,
    remoteCandidates: [],
    nextCandidateID: 1,
    unsubscribers: [],
    retryTimers: {},
    seenRemoteCandidateKeys: new Set(),
    startedAt: Date.now(),
    closed: false
  };
  attachSessionListeners(session, db);
  return session;
}

function closeSession(session) {
  session.closed = true;
  detachSessionListeners(session);
}

function userSettingsRef(db, ownerUID) {
  return doc(db, "user_settings", ownerUID);
}

async function publishActiveCall(db, ownerUID, callId) {
  if (!ownerUID) {
    return;
  }
  await setDoc(userSettingsRef(db, ownerUID), {
    activeCallId: callId,
    activeCallUpdatedAt: serverTimestamp()
  }, { merge: true });
}

async function clearActiveCall(db, ownerUID) {
  if (!ownerUID) {
    return;
  }
  await setDoc(userSettingsRef(db, ownerUID), {
    activeCallId: deleteField(),
    activeCallUpdatedAt: serverTimestamp()
  }, { merge: true });
}

function asyncHandler(fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.post("/agent/chat", asyncHandler(async (req, res) => {
  const message = String(req.body?.message || "").trim();
  const reset = Boolean(req.body?.reset);
  const response = await chatWithAgent(message, { reset });
  return res.json({ response });
}));

app.post("/agent/chat-stream", async (req, res, next) => {
  const message = String(req.body?.message || "").trim();
  const reset = Boolean(req.body?.reset);
  if (!message) {
    return res.status(400).json({ error: "message is required" });
  }

  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    const response = await streamChatWithAgent(message, {
      reset,
      onDelta: (delta) => {
        res.write(`${JSON.stringify({ delta })}\n`);
      }
    });
    res.write(`${JSON.stringify({ done: true, response })}\n`);
    return res.end();
  } catch (err) {
    if (!res.headersSent) {
      return next(err);
    }
    res.write(`${JSON.stringify({ done: true, error: err?.message || String(err) })}\n`);
    return res.end();
  }
});

app.post("/session/start", asyncHandler(async (req, res) => {
  const callId = String(req.body?.callId || "").trim();
  const role = String(req.body?.role || "").trim();
  if (!callId) {
    return res.status(400).json({ error: "callId is required" });
  }
  if (role !== "caller" && role !== "callee") {
    return res.status(400).json({ error: "role must be caller or callee" });
  }

  if (sessions.has(callId)) {
    closeSession(sessions.get(callId));
    sessions.delete(callId);
  }

  const callRef = doc(req.app.locals.db, config.callsCollection, callId);
  const ownerUID = req.app.locals.ownerUID || null;

  const baseDoc = {
    ownerUid: ownerUID,
    updatedAt: serverTimestamp(),
    source: "bridge"
  };

  if (role === "caller") {
    await setDoc(callRef, {
      ...baseDoc,
      ownerUid: ownerUID,
      status: "offer_pending",
      offer: deleteField(),
      answer: deleteField(),
    }, { merge: true });
  } else {
    await setDoc(callRef, {
      ...baseDoc,
      status: "callee_waiting"
    }, { merge: true });
  }

  await publishActiveCall(req.app.locals.db, ownerUID, callId);

  const session = makeSession(callId, role, req.app.locals.db);
  sessions.set(callId, session);

  return res.json({
    ok: true,
    callId,
    role
  });
}));

app.post("/session/:callId/stop", asyncHandler(async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.json({ ok: true, alreadyStopped: true });
  }
  await clearActiveCall(req.app.locals.db, req.app.locals.ownerUID || null);
  closeSession(session);
  sessions.delete(callId);
  return res.json({ ok: true });
}));

app.post("/session/:callId/local-description", asyncHandler(async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.status(404).json({ error: "session not started" });
  }
  const type = String(req.body?.type || "").trim();
  const sdp = normalizeSDP(String(req.body?.sdp || ""));
  if (!type || !sdp) {
    return res.status(400).json({ error: "type and sdp are required" });
  }

  const ownerUID = req.app.locals.ownerUID || null;
  if (session.role === "caller") {
    await setDoc(session.callRef, {
      ownerUid: ownerUID,
      offer: { type, sdp },
      status: "offer_ready",
      updatedAt: serverTimestamp()
    }, { merge: true });
  } else {
    await updateDoc(session.callRef, {
      answer: { type, sdp },
      status: "answered",
      updatedAt: serverTimestamp()
    });
  }

  return res.json({ ok: true });
}));

app.get("/session/:callId/remote-description", asyncHandler(async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.status(404).json({ error: "session not started" });
  }
  const version = Number.parseInt(String(req.query.version || "0"), 10) || 0;
  if (!session.remoteDescription || session.remoteDescriptionVersion <= version) {
    return res.json({ hasUpdate: false, version: session.remoteDescriptionVersion });
  }
  const normalized = {
    type: session.remoteDescription.type,
    sdp: normalizeSDP(session.remoteDescription.sdp)
  };
  return res.json({
    hasUpdate: true,
    version: session.remoteDescriptionVersion,
    description: normalized
  });
}));

app.post("/session/:callId/local-candidate", asyncHandler(async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.status(404).json({ error: "session not started" });
  }
  const candidate = String(req.body?.candidate || "").trim();
  if (!candidate) {
    return res.status(400).json({ error: "candidate is required" });
  }

  const payload = {
    candidate,
    sdpMid: req.body?.sdpMid ?? null,
    sdpMLineIndex: req.body?.sdpMLineIndex ?? null,
    usernameFragment: req.body?.usernameFragment ?? null,
    createdAt: serverTimestamp()
  };

  const targetRef = session.role === "caller" ? session.callerCandidatesRef : session.calleeCandidatesRef;
  await addDoc(targetRef, payload);
  return res.json({ ok: true });
}));

app.get("/session/:callId/remote-candidates", asyncHandler(async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.status(404).json({ error: "session not started" });
  }
  const since = Number.parseInt(String(req.query.since || "0"), 10) || 0;
  const items = session.remoteCandidates.filter((item) => item.id > since);
  return res.json({ items });
}));

async function boot() {
  const credentials = await getBridgeCredentials();
  await initializeFirebaseRuntime(credentials);
  scheduleFirebaseRestart();
  startMlxServer();
  startMlxAudioServer();
  startSayTtsServer();
  startAgentProcess();
  app.listen(config.port, () => {
    const bridgeURL = `http://127.0.0.1:${config.port}`;
    console.log(`[bridge] listening on ${bridgeURL}`);
    startPionPeer(bridgeURL);
  });
}

app.post("/turn-credentials", asyncHandler(async (req, res) => {
  try {
    const creds = await getTurnCredentials(req.app.locals.functions);
    return res.json(creds);
  } catch (err) {
    const peerId = String(req.body?.peerId || "").trim() || "peer";
    const ttlSeconds = Number.parseInt(String(req.body?.ttlSeconds || ""), 10);
    if (config.turnAuthSecret) {
      try {
        const fallbackCreds = makeTurnCredentials({ peerId, ttlSeconds });
        return res.json(fallbackCreds);
      } catch (_) {
        // Ignore and return the original Firebase error below.
      }
    }
    return res.status(500).json({ error: err.message || String(err) });
  }
}));

app.use((err, _req, res, _next) => {
  const message = err?.message || String(err);
  console.error("[bridge] request error", err);
  if (res.headersSent) {
    return;
  }
  res.status(500).json({ error: message });
});

boot().catch((err) => {
  console.error("[bridge] fatal", err);
  process.exit(1);
});

process.on("SIGINT", () => {
  clearFirebaseRestartTimer();
  stopTurnRefreshLoop();
  stopAgentProcess();
  stopMlxServer();
  stopMlxAudioServer();
  stopSayTtsServer();
  shutdownPeer();
  process.exit(0);
});

process.on("SIGTERM", () => {
  clearFirebaseRestartTimer();
  stopTurnRefreshLoop();
  stopAgentProcess();
  stopMlxServer();
  stopMlxAudioServer();
  stopSayTtsServer();
  shutdownPeer();
  process.exit(0);
});
