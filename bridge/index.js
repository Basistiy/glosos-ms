import express from "express";
import crypto from "crypto";
import { initializeApp } from "firebase/app";
import {
  getAuth,
  signInWithEmailAndPassword
} from "firebase/auth";
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

const config = {
  firebase: {
    apiKey: mustEnv("FIREBASE_API_KEY"),
    authDomain: mustEnv("FIREBASE_AUTH_DOMAIN"),
    projectId: mustEnv("FIREBASE_PROJECT_ID"),
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET || undefined,
    messagingSenderId: process.env.FIREBASE_MESSAGING_SENDER_ID || undefined,
    appId: process.env.FIREBASE_APP_ID || undefined
  },
  bridgeEmail: mustEnv("BRIDGE_EMAIL"),
  bridgePassword: mustEnv("BRIDGE_PASSWORD"),
  callsCollection: (process.env.WEBRTC_CALLS_COLLECTION || "webrtc_calls").trim(),
  port: intEnv("BRIDGE_PORT", 8080),
  turnServer: (process.env.TURN_SERVER || "54.37.235.123:3478").trim(),
  turnAuthSecret: (process.env.TURN_AUTH_SECRET || "").trim(),
  turnTTLSeconds: intEnv("TURN_TTL_SECONDS", 600)
};

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
  const ttl = Math.max(60, Math.min(3600, Number.isFinite(ttlSeconds) ? ttlSeconds : config.turnTTLSeconds));
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

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.post("/session/start", async (req, res) => {
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

  const session = makeSession(callId, role, req.app.locals.db);
  sessions.set(callId, session);

  return res.json({
    ok: true,
    callId,
    role
  });
});

app.post("/session/:callId/stop", async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.json({ ok: true, alreadyStopped: true });
  }
  closeSession(session);
  sessions.delete(callId);
  return res.json({ ok: true });
});

app.post("/session/:callId/local-description", async (req, res) => {
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
});

app.get("/session/:callId/remote-description", async (req, res) => {
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
});

app.post("/session/:callId/local-candidate", async (req, res) => {
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
});

app.get("/session/:callId/remote-candidates", async (req, res) => {
  const callId = String(req.params.callId || "").trim();
  const session = sessions.get(callId);
  if (!session) {
    return res.status(404).json({ error: "session not started" });
  }
  const since = Number.parseInt(String(req.query.since || "0"), 10) || 0;
  const items = session.remoteCandidates.filter((item) => item.id > since);
  return res.json({ items });
});

async function boot() {
  const firebaseApp = initializeApp(config.firebase);
  const auth = getAuth(firebaseApp);
  const db = getFirestore(firebaseApp);

  await signInWithEmailAndPassword(auth, config.bridgeEmail, config.bridgePassword);
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
  app.locals.ownerUID = user.uid;
  app.listen(config.port, () => {
    console.log(`[bridge] listening on http://127.0.0.1:${config.port}`);
  });
}

app.post("/turn-credentials", async (req, res) => {
  const peerId = String(req.body?.peerId || "").trim() || "peer";
  const ttlSeconds = Number.parseInt(String(req.body?.ttlSeconds || ""), 10);
  try {
    const creds = makeTurnCredentials({ peerId, ttlSeconds });
    return res.json(creds);
  } catch (err) {
    return res.status(500).json({ error: err.message || String(err) });
  }
});

boot().catch((err) => {
  console.error("[bridge] fatal", err);
  process.exit(1);
});
