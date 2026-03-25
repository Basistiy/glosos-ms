#!/usr/bin/env python3
"""Interactive local chat backed by an OpenAI-compatible MLX server."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "mlx-community/Orchestrator-8B-6bit"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "mlx-local"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local coding agent. Be concise, practical, and respond "
    "like a normal assistant. Do not output hidden reasoning or XML-style tags."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive local Python chat")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id exposed by the local server")
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible base URL, for example http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Dummy key for local servers that require the field to be present",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for the conversation",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run as a JSONL stdin/stdout worker process for the bridge",
    )
    parser.add_argument(
        "--audio-output-dir",
        default="agent_recordings",
        help="Directory where speech-only audio chunks are written",
    )
    parser.add_argument(
        "--vad-target-rate",
        type=int,
        default=16000,
        help="Sample rate for VAD processing and saved speech chunks",
    )
    parser.add_argument(
        "--vad-min-silence-chunks",
        type=int,
        default=2,
        help="How many consecutive silent chunks end the active speech segment",
    )
    return parser


def build_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )


def new_messages(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def complete_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def run_turn(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    user_text: str,
) -> str:
    messages.append({"role": "user", "content": user_text})
    reply = complete_chat(client, model, messages)
    messages.append({"role": "assistant", "content": reply})
    return reply


class SpeechChunkRecorder:
    def __init__(
        self,
        output_dir: str,
        target_rate: int = 16000,
        min_silence_chunks: int = 2,
        pre_speech_chunks: int = 2,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_rate = target_rate
        self.min_silence_chunks = max(1, min_silence_chunks)
        self.pre_speech_chunks = max(0, pre_speech_chunks)
        self.sequence = 0
        self.pending_speech_parts: list[Any] = []
        self.trailing_silence_chunks = 0
        self.pre_speech_buffer: deque[Any] = deque(maxlen=self.pre_speech_chunks or None)
        self._runtime: dict[str, Any] | None = None

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        import av  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from silero_vad import (  # pylint: disable=import-outside-toplevel
            get_speech_timestamps,
            load_silero_vad,
        )

        self._runtime = {
            "av": av,
            "np": np,
            "torch": torch,
            "model": load_silero_vad(),
            "get_speech_timestamps": get_speech_timestamps,
        }
        return self._runtime

    def process_chunk(
        self,
        *,
        call_id: str,
        stream_id: str,
        track_id: str,
        seq: int,
        final: bool,
        audio_bytes: bytes,
    ) -> dict[str, Any]:
        runtime = self._ensure_runtime()
        np = runtime["np"]
        torch = runtime["torch"]
        get_speech_timestamps = runtime["get_speech_timestamps"]
        model = runtime["model"]

        samples = self._decode_chunk(audio_bytes, runtime)
        speech_timestamps: list[dict[str, int]] = []

        if samples.size > 0:
            audio_tensor = torch.from_numpy(samples.copy())
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                model,
                sampling_rate=self.target_rate,
            )
        saved_paths: list[str] = []
        if speech_timestamps:
            if self.pre_speech_buffer:
                self.pending_speech_parts.extend(self.pre_speech_buffer)
                self.pre_speech_buffer.clear()
            self.pending_speech_parts.append(samples)
            self.trailing_silence_chunks = 0
        elif self.pending_speech_parts:
            if samples.size > 0:
                self.pending_speech_parts.append(samples)
            self.trailing_silence_chunks += 1
        elif samples.size > 0 and self.pre_speech_chunks > 0:
            self.pre_speech_buffer.append(samples)

        if self.pending_speech_parts and (final or self.trailing_silence_chunks >= self.min_silence_chunks):
            saved_paths.append(self._flush_pending(call_id, stream_id, track_id, runtime))

        if final:
            self.pre_speech_buffer.clear()

        return {
            "ok": True,
            "response": "audio chunk processed",
            "speechDetected": bool(speech_timestamps),
            "speechFrames": int(sum(item["end"] - item["start"] for item in speech_timestamps)),
            "savedPaths": saved_paths,
            "seq": seq,
        }

    def _decode_chunk(self, audio_bytes: bytes, runtime: dict[str, Any]):
        av = runtime["av"]
        np = runtime["np"]

        container = av.open(io.BytesIO(audio_bytes), mode="r", format="ogg")
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=self.target_rate,
        )

        chunks: list[Any] = []
        try:
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    data = resampled.to_ndarray()
                    if data.ndim == 2:
                        chunks.append(data[0].astype(np.float32, copy=False))
                    else:
                        chunks.append(data.astype(np.float32, copy=False))

            tail = resampler.resample(None)
            for resampled in tail or []:
                data = resampled.to_ndarray()
                if data.ndim == 2:
                    chunks.append(data[0].astype(np.float32, copy=False))
                else:
                    chunks.append(data.astype(np.float32, copy=False))
        finally:
            container.close()

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _flush_pending(self, call_id: str, stream_id: str, track_id: str, runtime: dict[str, Any]) -> str:
        np = runtime["np"]

        samples = np.concatenate(self.pending_speech_parts).astype(np.float32, copy=False)
        self.pending_speech_parts = []
        self.trailing_silence_chunks = 0

        safe_call_id = sanitize_path_part(call_id or "call")
        safe_stream_id = sanitize_path_part(stream_id or "stream")
        safe_track_id = sanitize_path_part(track_id or "track")
        file_name = (
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_"
            f"{safe_call_id}_{safe_stream_id}_{safe_track_id}_{self.sequence:04d}.wav"
        )
        self.sequence += 1
        file_path = self.output_dir / file_name
        write_pcm16_wave(file_path, samples, self.target_rate, np)
        print(f"[audio] saved speech chunk {file_path}", file=sys.stderr)
        return str(file_path)


def sanitize_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "unknown"


def write_pcm16_wave(path: Path, samples: Any, sample_rate: int, np: Any) -> None:
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def run_stdio(args: argparse.Namespace) -> None:
    client = build_client(args)
    messages = new_messages(args.system_prompt)
    audio_recorder = SpeechChunkRecorder(
        args.audio_output_dir,
        target_rate=args.vad_target_rate,
        min_silence_chunks=args.vad_min_silence_chunks,
    )

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        request_id: str | None = None
        try:
            payload: dict[str, Any] = json.loads(raw_line)
            request_id = payload.get("id")
            request_type = str(payload.get("type", "chat")).strip() or "chat"

            if request_type == "chat":
                user_text = str(payload.get("message", "")).strip()
                if bool(payload.get("reset", False)):
                    messages = new_messages(args.system_prompt)
                if not user_text:
                    raise ValueError("message is required")

                response = {
                    "id": request_id,
                    "response": run_turn(client, args.model, messages, user_text),
                }
            elif request_type == "audio_chunk":
                audio_base64 = str(payload.get("audioBase64", "")).strip()
                if not audio_base64:
                    raise ValueError("audioBase64 is required")

                response = {
                    "id": request_id,
                    **audio_recorder.process_chunk(
                        call_id=str(payload.get("callId", "")).strip(),
                        stream_id=str(payload.get("streamId", "")).strip(),
                        track_id=str(payload.get("trackId", "")).strip(),
                        seq=int(payload.get("seq", 0)),
                        final=bool(payload.get("final", False)),
                        audio_bytes=base64.b64decode(audio_base64),
                    ),
                }
            else:
                raise ValueError(f"unsupported request type: {request_type}")
        except Exception as exc:  # pylint: disable=broad-except
            response = {"id": request_id, "error": f"{type(exc).__name__}: {exc}"}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def main() -> None:
    args = build_parser().parse_args()
    if args.stdio:
        run_stdio(args)
        return

    client = build_client(args)
    messages = new_messages(args.system_prompt)

    print(f"[info] Using model: {args.model}")
    print(f"[info] Server: {args.api_base}")
    print("[info] Commands: /exit, /quit, /reset")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] Bye.")
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("[info] Bye.")
            break
        if user_text == "/reset":
            messages = new_messages(args.system_prompt)
            print("[info] Conversation reset.")
            continue

        try:
            result = run_turn(client, args.model, messages, user_text)
            print(f"\nassistant> {result}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n[error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
