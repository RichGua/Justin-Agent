from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .config import (
    AgentConfig,
    PROVIDER_LOCAL,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)
from .runtime import JustinRuntime, build_runtime_bundle
from .server import serve
from .types import to_plain_dict

SETUP_ENV_KEYS = [
    "JUSTIN_MODEL_PROVIDER",
    "JUSTIN_MODEL_NAME",
    "JUSTIN_API_BASE",
    "JUSTIN_API_KEY",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Justin: a local-first agent.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("serve", help="Run the HTTP server and Web UI.")
    subparsers.add_parser("setup", help="Run first-time provider/API setup wizard.")

    chat_parser = subparsers.add_parser("chat", help="Chat in one-off or interactive mode.")
    chat_parser.add_argument("--session", help="Reuse an existing session id.")
    chat_parser.add_argument("--message", help="Send one message and print the result.")

    candidate_parser = subparsers.add_parser("candidate", help="Inspect and review memory candidates.")
    candidate_subparsers = candidate_parser.add_subparsers(dest="candidate_command", required=True)
    candidate_subparsers.add_parser("list", help="List candidates.")
    confirm_parser = candidate_subparsers.add_parser("confirm", help="Approve a candidate.")
    confirm_parser.add_argument("candidate_id")
    reject_parser = candidate_subparsers.add_parser("reject", help="Reject a candidate.")
    reject_parser.add_argument("candidate_id")
    reject_parser.add_argument("--note")

    memory_parser = subparsers.add_parser("memory", help="Inspect approved memories.")
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_subparsers.add_parser("list", help="List memories.")
    search_parser = memory_subparsers.add_parser("search", help="Search memories.")
    search_parser.add_argument("query")

    session_parser = subparsers.add_parser("session", help="Inspect sessions.")
    session_subparsers = session_parser.add_subparsers(dest="session_command", required=True)
    session_subparsers.add_parser("list", help="List sessions.")

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv:
        argv = ["chat"]

    args = build_parser().parse_args(argv)
    config = AgentConfig.from_env()

    if args.command == "setup":
        updated = run_setup_wizard(config)
        _print_provider_summary(updated)
        return

    if args.command == "chat" and args.message is None:
        config = _maybe_prompt_first_run_setup(config)

    if args.command == "serve":
        serve(config)
        return

    bundle = build_runtime_bundle(config)
    runtime = JustinRuntime(bundle)
    try:
        match args.command:
            case "chat":
                _run_chat(runtime, args.session, args.message)
            case "candidate":
                _run_candidate_commands(runtime, args)
            case "memory":
                _run_memory_commands(runtime, args)
            case "session":
                _print_json([to_plain_dict(item) for item in runtime.list_sessions()])
    finally:
        runtime.close()


def _run_chat(runtime: JustinRuntime, session_id: str | None, message: str | None) -> None:
    if message:
        result = _send_with_feedback(runtime, message, session_id)
        if result is None:
            return
        _print_json(to_plain_dict(result))
        return

    _print_cli_banner(runtime.config)
    _print_cli_help()
    active_session_id = session_id
    while True:
        prompt = input("you> ").strip()
        if not prompt:
            continue
        if prompt.startswith("/"):
            active_session_id, should_exit = _handle_slash_command(prompt, runtime, active_session_id)
            if should_exit:
                return
            continue
        if prompt in {"/exit", "/quit"}:
            return
        result = _send_with_feedback(runtime, prompt, active_session_id)
        if result is None:
            continue
        active_session_id = result.session.id
        print(f"\nJustin> {result.assistant_message.content}\n")
        if result.candidates:
            print("candidate memories:")
            for candidate in result.candidates:
                print(f"  - {candidate.id} [{candidate.kind}] {candidate.content}")


def _send_with_feedback(runtime: JustinRuntime, content: str, session_id: str | None):
    started_at = time.perf_counter()
    print("Justin is thinking...", file=sys.stderr, flush=True)
    try:
        result = runtime.send_message(content=content, session_id=session_id)
    except Exception as exc:  # keep CLI stable and show user-facing errors.
        elapsed = time.perf_counter() - started_at
        _print_runtime_error(exc, elapsed)
        return None

    elapsed = time.perf_counter() - started_at
    print(f"Justin responded in {elapsed:.1f}s.", file=sys.stderr, flush=True)
    return result


def _print_runtime_error(exc: Exception, elapsed_seconds: float) -> None:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if isinstance(exc, TimeoutError) or "timeout" in lowered or "timed out" in lowered:
        print(
            f"Justin timed out after {elapsed_seconds:.1f}s waiting for the model. "
            "Check provider/network health, then retry or run `Justin setup` to switch providers.",
            file=sys.stderr,
            flush=True,
        )
        return

    print(
        f"Justin request failed after {elapsed_seconds:.1f}s: {detail}",
        file=sys.stderr,
        flush=True,
    )


def _run_candidate_commands(runtime: JustinRuntime, args: argparse.Namespace) -> None:
    match args.candidate_command:
        case "list":
            _print_json([to_plain_dict(item) for item in runtime.list_candidates()])
        case "confirm":
            _print_json(to_plain_dict(runtime.confirm_candidate(args.candidate_id)))
        case "reject":
            _print_json(to_plain_dict(runtime.reject_candidate(args.candidate_id, args.note)))


def _run_memory_commands(runtime: JustinRuntime, args: argparse.Namespace) -> None:
    match args.memory_command:
        case "list":
            _print_json([to_plain_dict(item) for item in runtime.list_memories()])
        case "search":
            _print_json([to_plain_dict(item) for item in runtime.search_memories(args.query)])


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _maybe_prompt_first_run_setup(config: AgentConfig) -> AgentConfig:
    if not sys.stdin.isatty():
        return config
    if config.has_user_settings():
        return config
    if any(os.getenv(name) for name in SETUP_ENV_KEYS):
        return config

    print("First run detected. Let's configure your model provider for Justin.")
    answer = input("Run setup wizard now? [Y/n]: ").strip().lower()
    if answer in {"", "y", "yes"}:
        return run_setup_wizard(config)

    print("Skipping setup. Justin will run with local fallback model.")
    return config


def run_setup_wizard(config: AgentConfig | None = None) -> AgentConfig:
    config = config or AgentConfig.from_env()
    config.ensure_directories()

    print("\n=== Justin Setup Wizard ===")
    print("Choose your provider:")
    print("  1) OPENAI")
    print("  2) Ollama")
    print("  3) Nvidia NIM")
    print("  4) Local fallback (no API)")
    choice = _ask_choice({"1", "2", "3", "4"}, default="1")

    if choice == "1":
        config.model_provider = PROVIDER_OPENAI
        config.api_base = _ask_text("OpenAI API base", default="https://api.openai.com/v1")
        config.model_name = _ask_text("OpenAI model", default="gpt-4.1-mini")
        config.api_key = _ask_text("OpenAI API key", secret=True)
    elif choice == "2":
        config.model_provider = PROVIDER_OLLAMA
        config.api_base = _ask_text("Ollama API base", default="http://localhost:11434/v1")
        config.model_name = _ask_text("Ollama model", default="llama3.1")
        key = _ask_text("Ollama API key (optional, Enter to skip)", required=False, secret=True)
        config.api_key = key or None
    elif choice == "3":
        config.model_provider = PROVIDER_NVIDIA_NIM
        config.api_base = _ask_text("NVIDIA NIM API base", default="https://integrate.api.nvidia.com/v1")
        config.model_name = _ask_text("NVIDIA model", default="meta/llama-3.1-70b-instruct")
        config.api_key = _ask_text("NVIDIA NIM API key", secret=True)
    else:
        config.model_provider = PROVIDER_LOCAL
        config.model_name = "local-fallback"
        config.api_base = None
        config.api_key = None

    config.save_settings()
    _apply_env(config)

    print(f"Saved setup to {config.settings_path}")
    print("You can re-run this anytime with: Justin setup\n")
    return config


def _apply_env(config: AgentConfig) -> None:
    os.environ["JUSTIN_MODEL_PROVIDER"] = config.model_provider
    os.environ["JUSTIN_MODEL_NAME"] = config.model_name
    os.environ["JUSTIN_HOST"] = str(config.host)
    os.environ["JUSTIN_PORT"] = str(config.port)
    if config.api_base:
        os.environ["JUSTIN_API_BASE"] = config.api_base
    else:
        os.environ.pop("JUSTIN_API_BASE", None)
    if config.api_key:
        os.environ["JUSTIN_API_KEY"] = config.api_key
    else:
        os.environ.pop("JUSTIN_API_KEY", None)


def _ask_choice(allowed: set[str], default: str) -> str:
    while True:
        raw = input(f"Select provider [{default}]: ").strip()
        picked = raw or default
        if picked in allowed:
            return picked
        print(f"Invalid choice: {picked}. Expected one of {sorted(allowed)}.")


def _ask_text(label: str, default: str | None = None, required: bool = True, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            value = default
        if value or not required:
            if secret and value:
                print(f"{label} set.")
            return value
        print(f"{label} is required.")


def _provider_title(config: AgentConfig) -> str:
    mapping = {
        PROVIDER_OPENAI: "OPENAI",
        PROVIDER_OLLAMA: "OLLAMA",
        PROVIDER_NVIDIA_NIM: "NVIDIA NIM",
        PROVIDER_LOCAL: "LOCAL",
    }
    return mapping.get(config.model_provider, config.model_provider.upper())


def _print_provider_summary(config: AgentConfig) -> None:
    print(f"Provider: {_provider_title(config)}")
    print(f"Model: {config.model_name}")
    if config.api_base:
        print(f"API base: {config.api_base}")


def _print_cli_banner(config: AgentConfig) -> None:
    title = "Justin CLI"
    line = "=" * 64
    print(line)
    print(f"{title:^64}")
    print(line)
    print(f"provider: {_provider_title(config)}")
    print(f"model:    {config.model_name}")
    if config.api_base:
        print(f"api base: {config.api_base}")
    print(line)


def _print_cli_help() -> None:
    print("Commands:")
    print("  /help                  Show command list")
    print("  /session               Show active session id")
    print("  /provider              Show current provider config")
    print("  /setup                 Re-run setup wizard")
    print("  /candidates            List pending memory candidates")
    print("  /approve <id>          Approve a candidate")
    print("  /reject <id> [note]    Reject a candidate")
    print("  /memories [query]      List/search approved memories")
    print("  /clear                 Clear terminal")
    print("  /exit                  Quit\n")


def _handle_slash_command(
    raw_command: str,
    runtime: JustinRuntime,
    active_session_id: str | None,
) -> tuple[str | None, bool]:
    tokens = raw_command.split()
    command = tokens[0].lower()

    if command in {"/exit", "/quit"}:
        return active_session_id, True
    if command == "/help":
        _print_cli_help()
        return active_session_id, False
    if command == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        _print_cli_banner(runtime.config)
        return active_session_id, False
    if command == "/session":
        print(f"active session: {active_session_id or '(new session on next message)'}")
        return active_session_id, False
    if command == "/provider":
        _print_provider_summary(runtime.config)
        return active_session_id, False
    if command == "/setup":
        updated = run_setup_wizard(runtime.config)
        runtime.apply_config(updated)
        print("Runtime provider config reloaded.")
        return active_session_id, False
    if command == "/candidates":
        items = runtime.list_candidates(status="pending")
        if not items:
            print("No pending candidates.")
            return active_session_id, False
        for item in items[:20]:
            print(f"- {item.id} [{item.kind}] {item.content}")
        return active_session_id, False
    if command == "/approve":
        if len(tokens) < 2:
            print("usage: /approve <candidate-id>")
            return active_session_id, False
        try:
            memory = runtime.confirm_candidate(tokens[1])
            print(f"approved -> memory {memory.id}")
        except KeyError as exc:
            print(str(exc))
        return active_session_id, False
    if command == "/reject":
        if len(tokens) < 2:
            print("usage: /reject <candidate-id> [note]")
            return active_session_id, False
        note = " ".join(tokens[2:]) if len(tokens) > 2 else None
        try:
            runtime.reject_candidate(tokens[1], note)
            print("candidate rejected.")
        except KeyError as exc:
            print(str(exc))
        return active_session_id, False
    if command == "/memories":
        query = " ".join(tokens[1:]).strip()
        items = runtime.search_memories(query) if query else runtime.list_memories()
        if not items:
            print("No memories found.")
            return active_session_id, False
        for item in items[:20]:
            score = f" score={item.score:.2f}" if hasattr(item, "score") else ""
            print(f"- {item.id} [{item.kind}]{score} {item.content}")
        return active_session_id, False

    print(f"Unknown command: {raw_command}. Type /help.")
    return active_session_id, False
