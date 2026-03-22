"use strict";

const crypto = require("crypto");
const { onCall, HttpsError } = require("firebase-functions/v2/https");
const { defineSecret, defineString } = require("firebase-functions/params");

const TURN_AUTH_SECRET = defineSecret("TURN_AUTH_SECRET");
const TURN_SERVER = defineString("TURN_SERVER", { default: "54.37.235.123:3478" });
const TURN_TTL_SECONDS = defineString("TURN_TTL_SECONDS", { default: "600" });

exports.getTurnCredentials = onCall(
  { region: "europe-west1", secrets: [TURN_AUTH_SECRET] },
  async (request) => {
    if (!request.auth) {
      throw new HttpsError("unauthenticated", "Authentication required");
    }

    const uid = request.auth.uid;
    const requestedServer = typeof request.data?.turnServer === "string" ? request.data.turnServer.trim() : "";
    const turnServer = requestedServer || TURN_SERVER.value();

    if (!turnServer) {
      throw new HttpsError("failed-precondition", "TURN server is not configured");
    }

    const defaultTTL = Number.parseInt(TURN_TTL_SECONDS.value(), 10);
    const requestedTTL = Number.parseInt(String(request.data?.ttlSeconds ?? ""), 10);
    let ttlSeconds = Number.isFinite(requestedTTL) ? requestedTTL : defaultTTL;
    if (!Number.isFinite(ttlSeconds)) ttlSeconds = 600;
    ttlSeconds = Math.min(Math.max(ttlSeconds, 60), 3600);

    const unixExpiry = Math.floor(Date.now() / 1000) + ttlSeconds;
    const username = `${unixExpiry}:${uid}`;

    const secret = TURN_AUTH_SECRET.value();
    const credential = crypto
      .createHmac("sha1", secret)
      .update(username)
      .digest("base64");

    return {
      turnServer,
      username,
      credential,
      ttlSeconds,
      expiresAtUnix: unixExpiry
    };
  }
);
