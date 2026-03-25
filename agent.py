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
from stt_provider import (
    DEFAULT_STT_API_KEY,
    DEFAULT_STT_BASE_URL,
    DEFAULT_STT_MODEL,
    DEFAULT_STT_PROVIDER,
    create_transcription_provider,
)

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
    parser.add_argument(
        "--stt-provider",
        default=DEFAULT_STT_PROVIDER,
        help="Speech-to-text provider to use: mlx_audio or google",
    )
    parser.add_argument(
        "--stt-base-url",
        default=DEFAULT_STT_BASE_URL,
        help="Base URL for an OpenAI-compatible STT server such as mlx-audio",
    )
    parser.add_argument(
        "--stt-api-key",
        default=DEFAULT_STT_API_KEY,
        help="API key for an OpenAI-compatible STT server",
    )
    parser.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help="Speech-to-text model identifier",
    )
    parser.add_argument(
        "--stt-language",
        default="",
        help="Optional language hint passed to the STT provider",
    )
    parser.add_argument(
        "--google-stt-project-id",
        default="",
        help="Google Cloud project id used when stt_provider=google",
    )
    parser.add_argument(
        "--google-stt-location",
        default="global",
        help="Google Cloud Speech location used when stt_provider=google",
    )
    parser.add_argument(
        "--google-stt-recognizer",
        default="_",
        help="Google Cloud Speech recognizer id used when stt_provider=google",
    )
    parser.add_argument(
        "--google-stt-model",
        default="",
        help="Optional Google Cloud Speech model name",
    )
    parser.add_argument(
        "--disable-audio-transcription",
        action="store_true",
        help="Disable speech-to-text for saved speech chunks",
    )
    parser.add_argument(
        "--transcription-model",
        dest="stt_model",
        default=argparse.SUPPRESS,
        help="Deprecated alias for --stt-model",
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


def build_audio_user_text(transcript: str) -> str:
    return f"Transcribed user speech:\n{transcript.strip()}"


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
        self.streams: dict[str, StreamState] = {}
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
        ssrc: int,
        seq: int,
        final: bool,
        audio_bytes: bytes,
    ) -> dict[str, Any]:
        runtime = self._ensure_runtime()
        np = runtime["np"]
        torch = runtime["torch"]
        get_speech_timestamps = runtime["get_speech_timestamps"]
        model = runtime["model"]

        stream_key = build_stream_key(call_id, stream_id, track_id, ssrc)
        state = self.streams.setdefault(stream_key, StreamState(pre_speech_chunks=self.pre_speech_chunks))
        state.ogg_bytes.extend(audio_bytes)

        decoded = self._decode_chunk(bytes(state.ogg_bytes), runtime)
        if decoded.size < state.decoded_sample_count:
            state.decoded_sample_count = 0

        samples = decoded[state.decoded_sample_count:].astype(np.float32, copy=False)
        state.decoded_sample_count = int(decoded.size)
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
            if state.pre_speech_buffer:
                state.pending_speech_parts.extend(state.pre_speech_buffer)
                state.pre_speech_buffer.clear()
            state.pending_speech_parts.append(samples)
            state.trailing_silence_chunks = 0
        elif state.pending_speech_parts:
            if samples.size > 0:
                state.pending_speech_parts.append(samples)
            state.trailing_silence_chunks += 1
        elif samples.size > 0 and self.pre_speech_chunks > 0:
            state.pre_speech_buffer.append(samples)

        if state.pending_speech_parts and (final or state.trailing_silence_chunks >= self.min_silence_chunks):
            saved_paths.append(self._flush_pending(call_id, stream_id, track_id, state, runtime))

        if final:
            state.pre_speech_buffer.clear()
            self.streams.pop(stream_key, None)

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

    def _flush_pending(
        self,
        call_id: str,
        stream_id: str,
        track_id: str,
        state: "StreamState",
        runtime: dict[str, Any],
    ) -> str:
        np = runtime["np"]

        samples = np.concatenate(state.pending_speech_parts).astype(np.float32, copy=False)
        state.pending_speech_parts = []
        state.trailing_silence_chunks = 0

        safe_call_id = sanitize_path_part(call_id or "call")
        safe_stream_id = sanitize_path_part(stream_id or "stream")
        safe_track_id = sanitize_path_part(track_id or "track")
        file_name = (
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_"
            f"{safe_call_id}_{safe_stream_id}_{safe_track_id}_{state.sequence:04d}.wav"
        )
        state.sequence += 1
        file_path = self.output_dir / file_name
        write_pcm16_wave(file_path, samples, self.target_rate, np)
        print(f"[audio] saved speech chunk {file_path}", file=sys.stderr)
        return str(file_path)


class StreamState:
    def __init__(self, pre_speech_chunks: int) -> None:
        self.ogg_bytes = bytearray()
        self.decoded_sample_count = 0
        self.sequence = 0
        self.pending_speech_parts: list[Any] = []
        self.trailing_silence_chunks = 0
        self.pre_speech_buffer: deque[Any] = deque(maxlen=pre_speech_chunks or None)


def build_stream_key(call_id: str, stream_id: str, track_id: str, ssrc: int) -> str:
    return "::".join([call_id.strip(), stream_id.strip(), track_id.strip(), str(ssrc)])


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
    transcription_provider = create_transcription_provider(args)
    if transcription_provider:
        print("[audio] warming transcription provider", file=sys.stderr)
        transcription_provider.warmup()
        print("[audio] transcription provider ready", file=sys.stderr)

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
                        ssrc=int(payload.get("ssrc", 0)),
                        seq=int(payload.get("seq", 0)),
                        final=bool(payload.get("final", False)),
                        audio_bytes=base64.b64decode(audio_base64),
                    ),
                }
                response["audioResponse"] = response.get("response", "")
                if transcription_provider and response.get("savedPaths"):
                    transcripts: list[dict[str, Any]] = []
                    transcript_texts: list[str] = []
                    for saved_path in response["savedPaths"]:
                        result = transcription_provider.transcribe_file(str(saved_path))
                        transcripts.append(result.to_dict())
                        if result.text:
                            transcript_texts.append(result.text)
                    response["transcriptions"] = transcripts
                    if transcript_texts:
                        response["transcriptText"] = "\n\n".join(transcript_texts)
                        response["assistantResponse"] = run_turn(
                            client,
                            args.model,
                            messages,
                            build_audio_user_text(response["transcriptText"]),
                        )
                        # Mirror the text-chat contract so callers can handle
                        # typed and transcribed input through the same field.
                        response["response"] = response["assistantResponse"]
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
