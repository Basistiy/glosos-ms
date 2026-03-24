import express from "express";
import crypto from "crypto";
import { spawn } from "child_process";
import path from "path";
import { createInterface } from "readline/promises";
import { fileURLToPath } from "url";
import { initializeApp } from "firebase/app";
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
  port: intEnv("BRIDGE_PORT", 8080),
  turnFunctionRegion: (process.env.TURN_FUNCTION_REGION || "europe-west1").trim(),
  turnFunctionName: (process.env.TURN_FUNCTION_NAME || "getTurnCredentials").trim(),
  turnServer: (process.env.TURN_SERVER || "54.37.235.123:3478").trim(),
  turnAuthSecret: (process.env.TURN_AUTH_SECRET || "").trim(),
  turnTTLSeconds: intEnv("TURN_TTL_SECONDS", 86400),
  turnRefreshIntervalSeconds: intEnv("TURN_REFRESH_INTERVAL_SECONDS", 82800),
  autoStartPionPeer: (process.env.AUTO_START_PION_PEER || "1").trim() !== "0",
  pionPeerRole: (process.env.PION_PEER_ROLE || "callee").trim(),
  pionPeerRestartDelayMs: intEnv("PION_PEER_RESTART_DELAY_MS", 3000)
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
  if (config.pionPeerRole !== "caller" && config.pionPeerRole !== "callee") {
    throw new Error("PION_PEER_ROLE must be caller or callee");
  }
  if (peerState.process) {
    return;
  }

  const args = [
    "run",
    "./cmd/pion_bridge_peer",
    "-role",
    config.pionPeerRole,
    "-bridge-url",
    bridgeURL
  ];
  const child = spawn("go", args, {
    cwd: repoRoot,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"]
  });

  peerState.process = child;
  console.log(`[bridge] started pion peer role=${config.pionPeerRole}`);
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

function makeSession(callId, role, db) {
  const callRef = doc(db, config.callsCollection, callId);
  const callerCandidatesRef = collection(callRef, "callerCandidates");
  const calleeCandidatesRef = collection(callRef, "calleeCandidates");

  const session = {
    callId,
    role,
    callRef,
    callerCandidatesRef,
    calleeCandidatesRef,
    remoteDescription: null,
    remoteDescriptionVersion: 0,
    remoteCandidates: [],
    nextCandidateID: 1,
    unsubscribers: [],
    startedAt: Date.now()
  };

  const remoteDescriptionField = role === "caller" ? "answer" : "offer";
  const remoteCandidatesRef = role === "caller" ? calleeCandidatesRef : callerCandidatesRef;

  session.unsubscribers.push(
    onSnapshot(callRef, (snap) => {
      if (!snap.exists()) return;
      const data = snap.data() || {};
      const desc = data[remoteDescriptionField];
      if (!desc || typeof desc.sdp !== "string" || typeof desc.type !== "string") return;
      if (session.remoteDescription && session.remoteDescription.sdp === desc.sdp) return;
      session.remoteDescription = { type: desc.type, sdp: normalizeSDP(desc.sdp) };
      session.remoteDescriptionVersion += 1;
      console.log(
        `[bridge] call=${callId} role=${role} remote description updated (${remoteDescriptionField})`
      );
    }, (err) => {
      console.error(`[bridge] call=${callId} doc watch error`, err);
    })
  );

  session.unsubscribers.push(
    onSnapshot(remoteCandidatesRef, (snap) => {
      for (const change of snap.docChanges()) {
        if (change.type !== "added") continue;
        const data = change.doc.data();
        if (!data?.candidate) continue;
        const createdAtMillis = timestampMillis(data.createdAt);
        if (createdAtMillis > 0 && createdAtMillis < session.startedAt) {
          continue;
        }
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
      console.error(`[bridge] call=${callId} candidates watch error`, err);
    })
  );

  return session;
}

function closeSession(session) {
  for (const unsubscribe of session.unsubscribers) {
    try {
      unsubscribe();
    } catch (_) {
      // no-op
    }
  }
  session.unsubscribers = [];
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
  const firebaseApp = initializeApp(config.firebase);
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

  app.locals.db = db;
  app.locals.functions = functions;
  app.locals.ownerUID = user.uid;
  await refreshTurnCredentials(functions);
  startTurnRefreshLoop(functions);
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
  shutdownPeer();
  process.exit(0);
});

process.on("SIGTERM", () => {
  shutdownPeer();
  process.exit(0);
});
