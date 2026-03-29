#!/usr/bin/env python3
"""Minimal OpenAI-compatible TTS endpoint for macOS using `say`.

POST /v1/audio/speech
Request JSON:
  {"model":"...","input":"hello","voice":"Samantha","response_format":"wav"}
Response:
  audio/wav bytes
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str = Field(min_length=1)
    voice: Optional[str] = None
    response_format: str = "wav"


app = FastAPI(title="say-tts-server")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


def _run(cmd: list[str], timeout: int) -> None:
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"command timed out: {' '.join(cmd)}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or f"command failed: {' '.join(cmd)}"
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/v1/audio/speech")
def audio_speech(payload: SpeechRequest) -> Response:
    text = payload.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` is required")

    response_format = (payload.response_format or "wav").strip().lower()
    if response_format != "wav":
        raise HTTPException(status_code=400, detail="Only response_format=wav is supported")

    with tempfile.TemporaryDirectory(prefix="say-tts-") as tmp_dir:
        tmp = Path(tmp_dir)
        aiff_path = tmp / "speech.aiff"
        wav_path = tmp / "speech.wav"

        say_cmd = ["say", "-o", str(aiff_path)]
        if payload.voice and payload.voice.strip():
            say_cmd.extend(["-v", payload.voice.strip()])
        say_cmd.append(text)

        try:
            _run(say_cmd, timeout=30)
        except HTTPException as exc:
            # If requested voice is invalid, retry with system default voice once.
            if payload.voice and "Voice" in str(exc.detail):
                _run(["say", "-o", str(aiff_path), text], timeout=30)
            else:
                raise

        _run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16",
                str(aiff_path),
                str(wav_path),
            ],
            timeout=30,
        )

        wav_bytes = wav_path.read_bytes()

    headers = {"Content-Disposition": 'inline; filename="speech.wav"'}
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.exception_handler(HTTPException)
def http_error(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": True, "reason": exc.detail})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run say-based TTS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
