# Firebase signaling for client-to-client WebRTC

This flow is pure client-to-client. There is no Go signaling server and no service-account key in the client.

## 1) What you need

- Firebase project with Firestore enabled.
- Firebase Authentication enabled with `Email/Password` sign-in provider.
- Firebase Web app config (apiKey, authDomain, projectId, appId, etc).
- Firestore security rules that allow your test users to read/write one call document and its candidate subcollections.

You do **not** need `GOOGLE_APPLICATION_CREDENTIALS` for this browser-only flow.

## 2) Firestore structure

Collection default: `webrtc_calls`

Document: `webrtc_calls/{callId}`

Fields:

- `offer` object (`type`, `sdp`) written by caller
- `answer` object (`type`, `sdp`) written by callee
- `status` string

Subcollections:

- `webrtc_calls/{callId}/callerCandidates`
- `webrtc_calls/{callId}/calleeCandidates`

Each subcollection stores ICE candidate JSON from the corresponding peer.

## 3) Run test

Open this file on both peers:

- `/Users/evgeni/Documents/GitHub/glosos-ms/firebase_datachannel_test.html`

Steps:

1. Paste the same Firebase config JSON on both peers.
2. Enter email/password on both peers (first login can auto-create the account if user does not exist).
3. Use the same collection and call ID.
4. Peer A selects `Caller` and clicks `Start`.
5. Peer B selects `Callee` and clicks `Start`.
6. Wait for `Connected`, then send messages both directions.

## 4) Recommended next hardening

- Scope access by authenticated user/session in Firestore rules.
- Add cleanup for old call docs and candidate docs.
- Add reconnect/ICE restart flow for mobile network changes.

## 5) Short-lived TURN tokens (recommended)

Use Firebase Functions to mint TURN credentials per session. Do not ship long-lived TURN password in the client.

### 5.1 Coturn config for REST auth

In `/etc/turnserver.conf` use:

```conf
lt-cred-mech
use-auth-secret
static-auth-secret=REPLACE_WITH_LONG_RANDOM_SECRET
realm=turn.glosos.local
```

Remove static user/password line:

```conf
# user=turnuser:turnpassword
```

Restart:

```bash
sudo systemctl restart coturn
```

### 5.2 Firebase Function

Files added in this repo:

- `/Users/evgeni/Documents/GitHub/glosos-ms/functions/index.js`
- `/Users/evgeni/Documents/GitHub/glosos-ms/functions/package.json`

Deploy:

```bash
cd /Users/evgeni/Documents/GitHub/glosos-ms/functions
npm install
firebase functions:secrets:set TURN_AUTH_SECRET
firebase deploy --only functions:getTurnCredentials
```

Notes:

- Function defaults are already in code (`TURN_SERVER=54.37.235.123:3478`, `TURN_TTL_SECONDS=600`).
- You can override requested TURN server from the client page if needed.

### 5.3 Client page settings

In `/Users/evgeni/Documents/GitHub/glosos-ms/firebase_datachannel_test.html`:

- `Use TURN` = `Yes`
- `TURN Token Function Region` = your deployed region (default in this repo is `europe-west1`)
- `TURN Token Function Name` = `getTurnCredentials`
- TURN username/password are no longer present in the page UI; credentials are fetched from the function per session.

## 6) Starter Firestore rules (auth required)

```js
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /webrtc_calls/{callId} {
      allow read, write: if request.auth != null;
      match /{sub=**} {
        allow read, write: if request.auth != null;
      }
    }
  }
}
```
