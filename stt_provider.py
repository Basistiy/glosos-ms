from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
import wave

from openai import NOT_GIVEN, OpenAI


DEFAULT_STT_PROVIDER = "mlx_audio"
DEFAULT_STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"
DEFAULT_STT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_STT_API_KEY = "mlx-audio"


@dataclass
class TranscriptionResult:
    path: str
    text: str
    provider: str
    language: str | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "text": self.text,
            "provider": self.provider,
        }
        if self.language:
            payload["language"] = self.language
        if self.raw is not None:
            payload["raw"] = self.raw
        return payload


class BaseTranscriptionProvider:
    provider_name = "base"

    def warmup(self) -> None:
        return None

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        raise NotImplementedError


class MlxAudioTranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "mlx_audio"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        language: str | None = None,
    ) -> None:
        self.client = OpenAI(base_url=normalize_openai_base_url(base_url), api_key=api_key)
        self.model = model
        self.language = language

    def warmup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            write_silence_wav(Path(temp_file.name))
            self.transcribe_file(temp_file.name)

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        with Path(audio_path).open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=self.language or NOT_GIVEN,
            )

        text = str(getattr(response, "text", "") or "").strip()
        language = getattr(response, "language", None)
        raw = model_dump(response)
        return TranscriptionResult(
            path=audio_path,
            text=text,
            provider=self.provider_name,
            language=language,
            raw=raw,
        )


class GoogleSpeechTranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "google"

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        recognizer: str,
        language: str | None = None,
        model: str | None = None,
    ) -> None:
        from google.cloud import speech_v2  # pylint: disable=import-outside-toplevel
        from google.cloud.speech_v2.types import cloud_speech  # pylint: disable=import-outside-toplevel

        self.speech_v2 = speech_v2
        self.cloud_speech = cloud_speech
        self.client = speech_v2.SpeechClient()
        self.project_id = project_id
        self.location = location
        self.recognizer = recognizer
        self.language = language
        self.model = model

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        with Path(audio_path).open("rb") as audio_file:
            content = audio_file.read()

        config = self.cloud_speech.RecognitionConfig(
            auto_decoding_config=self.cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language] if self.language else [],
            model=self.model or "",
        )
        request = self.cloud_speech.RecognizeRequest(
            recognizer=self.client.recognizer_path(
                self.project_id,
                self.location,
                self.recognizer,
            ),
            config=config,
            content=content,
        )
        response = self.client.recognize(request=request)

        texts: list[str] = []
        language = None
        for result in response.results:
            if getattr(result, "language_code", None) and language is None:
                language = result.language_code
            for alternative in result.alternatives:
                transcript = str(getattr(alternative, "transcript", "") or "").strip()
                if transcript:
                    texts.append(transcript)

        return TranscriptionResult(
            path=audio_path,
            text=" ".join(texts).strip(),
            provider=self.provider_name,
            language=language or self.language,
            raw=model_dump(response),
        )


def create_transcription_provider(args: Any) -> BaseTranscriptionProvider | None:
    if getattr(args, "disable_audio_transcription", False):
        return None

    provider_name = str(getattr(args, "stt_provider", DEFAULT_STT_PROVIDER) or DEFAULT_STT_PROVIDER).strip().lower()
    if provider_name == "mlx_audio":
        return MlxAudioTranscriptionProvider(
            base_url=getattr(args, "stt_base_url", DEFAULT_STT_BASE_URL),
            api_key=getattr(args, "stt_api_key", DEFAULT_STT_API_KEY),
            model=getattr(args, "stt_model", DEFAULT_STT_MODEL),
            language=getattr(args, "stt_language", None),
        )
    if provider_name == "google":
        project_id = str(getattr(args, "google_stt_project_id", "") or "").strip()
        if not project_id:
            raise ValueError("google_stt_project_id is required when stt_provider=google")
        return GoogleSpeechTranscriptionProvider(
            project_id=project_id,
            location=str(getattr(args, "google_stt_location", "global") or "global").strip(),
            recognizer=str(getattr(args, "google_stt_recognizer", "_") or "_").strip(),
            language=str(getattr(args, "stt_language", "") or "").strip() or None,
            model=str(getattr(args, "google_stt_model", "") or "").strip() or None,
        )
    raise ValueError(f"unsupported stt_provider: {provider_name}")


def normalize_openai_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def model_dump(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return value
    return None


def write_silence_wav(path: Path, sample_rate: int = 16000, duration_ms: int = 250) -> None:
    frame_count = max(1, sample_rate * duration_ms // 1000)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
