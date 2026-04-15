from __future__ import annotations

import argparse
import json

from .config import AgentConfig
from .runtime import PersonalAgentRuntime, build_runtime_bundle
from .server import serve
from .types import to_plain_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first personal agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Run the HTTP server and Web UI.")

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


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "serve":
        serve()
        return

    bundle = build_runtime_bundle(AgentConfig.from_env())
    runtime = PersonalAgentRuntime(bundle)
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


def _run_chat(runtime: PersonalAgentRuntime, session_id: str | None, message: str | None) -> None:
    if message:
        result = runtime.send_message(content=message, session_id=session_id)
        _print_json(to_plain_dict(result))
        return

    print("Interactive mode. Type /exit to quit.")
    active_session_id = session_id
    while True:
        prompt = input("you> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return
        result = runtime.send_message(content=prompt, session_id=active_session_id)
        active_session_id = result.session.id
        print(f"agent> {result.assistant_message.content}")
        if result.candidates:
            print("candidate memories:")
            for candidate in result.candidates:
                print(f"  - {candidate.id} [{candidate.kind}] {candidate.content}")


def _run_candidate_commands(runtime: PersonalAgentRuntime, args: argparse.Namespace) -> None:
    match args.candidate_command:
        case "list":
            _print_json([to_plain_dict(item) for item in runtime.list_candidates()])
        case "confirm":
            _print_json(to_plain_dict(runtime.confirm_candidate(args.candidate_id)))
        case "reject":
            _print_json(to_plain_dict(runtime.reject_candidate(args.candidate_id, args.note)))


def _run_memory_commands(runtime: PersonalAgentRuntime, args: argparse.Namespace) -> None:
    match args.memory_command:
        case "list":
            _print_json([to_plain_dict(item) for item in runtime.list_memories()])
        case "search":
            _print_json([to_plain_dict(item) for item in runtime.search_memories(args.query)])


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
