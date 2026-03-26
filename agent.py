#!/usr/bin/env python3

import argparse
import json
import sys
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "mlx-local"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful local coding assistant. "
    "Be concise, accurate, and practical."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcript-only local agent worker")
    parser.add_argument("--stdio", action="store_true", help="Run JSONL request/response loop on stdio")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser


def new_messages(system_prompt: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": system_prompt}]


def extract_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()


def complete_chat(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    choice = response.choices[0]
    return extract_text(choice.message)


def run_turn(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    user_text: str,
) -> str:
    messages.append({"role": "user", "content": user_text})
    assistant_text = complete_chat(client, model, messages)
    messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text


def run_stdio(args: argparse.Namespace) -> int:
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    messages = new_messages(args.system_prompt)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id: str | None = None
        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            if payload.get("type") != "chat":
                raise ValueError(f"unsupported request type: {payload.get('type')!r}")

            if payload.get("reset"):
                messages = new_messages(args.system_prompt)

            message = str(payload.get("message", "")).strip()
            if not message:
                raise ValueError("message is required")

            response_text = run_turn(client, args.model, messages, message)
            response = {"id": request_id, "response": response_text}
        except Exception as exc:  # noqa: BLE001
            response = {"id": request_id, "error": str(exc)}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.stdio:
        return run_stdio(args)
    raise SystemExit("--stdio is required")


if __name__ == "__main__":
    raise SystemExit(main())
