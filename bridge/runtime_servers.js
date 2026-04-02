import { spawn, spawnSync } from "child_process";

export function createRuntimeServers({ config, repoRoot, prefixStream }) {
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

  function mlxApiBase() {
    return `http://${config.mlxHost}:${config.mlxPort}/v1`;
  }

  function mlxAudioApiBase() {
    return `http://${config.mlxAudioHost}:${config.mlxAudioPort}/v1`;
  }

  function sayTtsApiBase() {
    return `http://${config.sayTtsHost}:${config.sayTtsPort}/v1`;
  }

  function clearRestartTimer(state) {
    if (state.restartTimer) {
      clearTimeout(state.restartTimer);
      state.restartTimer = null;
    }
  }

  function pythonInvocation(runnerCommand, pythonArgs) {
    const runner = (runnerCommand || "").trim();
    if (runner === "uv") {
      return { command: "uv", args: ["run", "python", ...pythonArgs] };
    }
    return { command: runner || "python3", args: pythonArgs };
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

    const mlxImport = pythonInvocation(config.mlxRunnerCommand, ["-c", "import mlx_lm"]);
    const importCheck = spawnSync(mlxImport.command, mlxImport.args, {
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

    const mlxAudioImport = pythonInvocation(
      config.mlxAudioRunnerCommand,
      ["-c", "import mlx_audio, uvicorn, fastapi, webrtcvad, multipart, phonemizer"],
    );
    const importCheck = spawnSync(mlxAudioImport.command, mlxAudioImport.args, {
      cwd: repoRoot,
      env: process.env,
      encoding: "utf8"
    });
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

    const sayImport = pythonInvocation(config.sayTtsRunnerCommand, ["-c", "import fastapi, uvicorn"]);
    const importCheck = spawnSync(sayImport.command, sayImport.args, {
      cwd: repoRoot,
      env: process.env,
      encoding: "utf8"
    });
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
    clearRestartTimer(mlxState);
    mlxState.restartTimer = setTimeout(() => {
      startMlxServer();
    }, Math.max(1000, config.mlxRestartDelayMs));
  }

  function scheduleMlxAudioRestart() {
    if (mlxAudioState.shuttingDown || !config.autoStartMlxAudioServer || config.agentSttProvider !== "mlx_audio") {
      return;
    }
    clearRestartTimer(mlxAudioState);
    mlxAudioState.restartTimer = setTimeout(() => {
      startMlxAudioServer();
    }, Math.max(1000, config.mlxAudioRestartDelayMs));
  }

  function scheduleSayTtsRestart() {
    if (sayTtsState.shuttingDown || !config.autoStartSayTTSServer) {
      return;
    }
    clearRestartTimer(sayTtsState);
    sayTtsState.restartTimer = setTimeout(() => {
      startSayTtsServer();
    }, Math.max(1000, config.sayTtsRestartDelayMs));
  }

  function startMlxServer() {
    if (!config.autoStartMlxServer || mlxState.process) {
      return;
    }

    ensureMlxRuntimeReady();

    const mlxServer = pythonInvocation(config.mlxRunnerCommand, [
      "-m",
      "mlx_lm.server",
      "--model",
      config.mlxModel,
      "--host",
      config.mlxHost,
      "--port",
      String(config.mlxPort)
    ]);
    const child = spawn(mlxServer.command, mlxServer.args, {
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

    const mlxAudioServer = pythonInvocation(config.mlxAudioRunnerCommand, [
      "-m",
      "mlx_audio.server",
      "--host",
      config.mlxAudioHost,
      "--port",
      String(config.mlxAudioPort)
    ]);
    const child = spawn(mlxAudioServer.command, mlxAudioServer.args, {
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

    const sayServer = pythonInvocation(config.sayTtsRunnerCommand, [
      config.sayTtsScriptPath,
      "--host",
      config.sayTtsHost,
      "--port",
      String(config.sayTtsPort)
    ]);
    const child = spawn(sayServer.command, sayServer.args, {
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
    clearRestartTimer(mlxState);
    if (mlxState.process) {
      mlxState.process.kill("SIGTERM");
    }
  }

  function stopMlxAudioServer() {
    mlxAudioState.shuttingDown = true;
    clearRestartTimer(mlxAudioState);
    if (mlxAudioState.process) {
      mlxAudioState.process.kill("SIGTERM");
    }
  }

  function stopSayTtsServer() {
    sayTtsState.shuttingDown = true;
    clearRestartTimer(sayTtsState);
    if (sayTtsState.process) {
      sayTtsState.process.kill("SIGTERM");
    }
  }

  return {
    startMlxServer,
    startMlxAudioServer,
    startSayTtsServer,
    stopMlxServer,
    stopMlxAudioServer,
    stopSayTtsServer,
    mlxApiBase,
    sayTtsApiBase
  };
}
