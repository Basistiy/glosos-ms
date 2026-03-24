#!/usr/bin/env python3
"""Interactive local chat backed by an OpenAI-compatible MLX server."""

from __future__ import annotations

import argparse
import json
import sys
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


def run_stdio(args: argparse.Namespace) -> None:
    client = build_client(args)
    messages = new_messages(args.system_prompt)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        request_id: str | None = None
        try:
            payload: dict[str, Any] = json.loads(raw_line)
            request_id = payload.get("id")
            user_text = str(payload.get("message", "")).strip()
            if bool(payload.get("reset", False)):
                messages = new_messages(args.system_prompt)
            if not user_text:
                raise ValueError("message is required")

            response = {
                "id": request_id,
                "response": run_turn(client, args.model, messages, user_text),
            }
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
