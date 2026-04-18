from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager

from .config import (
    AgentConfig,
    PROVIDER_LOCAL,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)
from .runtime import JustinRuntime, build_runtime_bundle
from .types import to_plain_dict

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.formatted_text import HTML
    PROMPT_TOOLKIT_AVAILABLE = True
except ModuleNotFoundError:
    PROMPT_TOOLKIT_AVAILABLE = False

try:  # Optional UX dependency; CLI still works without Rich installed.
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    Console = None
    Group = None
    Panel = None
    Table = None
    Text = None
    RICH_AVAILABLE = False

SETUP_ENV_KEYS = [
    "JUSTIN_MODEL_PROVIDER",
    "JUSTIN_MODEL_NAME",
    "JUSTIN_API_BASE",
    "JUSTIN_API_KEY",
]

JUSTIN_ASCII_LOGO = r"""
      _ _   _  ____ _____ ___ _   _
     | | | | |/ ___|_   _|_ _| \ | |
  _  | | | | |\___ \ | |  | ||  \| |
 | |_| | |_| | ___) || |  | || |\  |
  \___/ \___/ |____/ |_| |___|_| \_|
"""

JUSTIN_SOLID_LOGO = "\n".join(
    [
        "     ██╗██╗   ██╗███████╗████████╗██╗███╗   ██╗",
        "     ██║██║   ██║██╔════╝╚══██╔══╝██║████╗  ██║",
        "     ██║██║   ██║███████╗   ██║   ██║██╔██╗ ██║",
        "██   ██║██║   ██║╚════██║   ██║   ██║██║╚██╗██║",
        "╚█████╔╝╚██████╔╝███████║   ██║   ██║██║ ╚████║",
        " ╚════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝",
    ]
)

JUSTIN_GRADIENT_LOGO_LINES = [
    "      ██╗██╗   ██╗███████╗████████╗██╗███╗   ██╗",
    "      ██║██║   ██║██╔════╝╚══██╔══╝██║████╗  ██║",
    "      ██║██║   ██║███████╗   ██║   ██║██╔██╗ ██║",
    " ██   ██║██║   ██║╚════██║   ██║   ██║██║╚██╗██║",
    " ╚█████╔╝╚██████╔╝███████║   ██║   ██║██║ ╚████║",
    "  ╚════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝",
]


@dataclass(slots=True)
class CliMetrics:
    turns: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0

    @property
    def avg_latency_seconds(self) -> float:
        if self.turns <= 0:
            return 0.0
        return self.total_latency_seconds / self.turns


class JustinCliRenderer:
    def __init__(self, interactive: bool) -> None:
        self.interactive = interactive
        self.metrics = CliMetrics()
        if RICH_AVAILABLE:
            self.console = Console(soft_wrap=True)
            self.err_console = Console(stderr=True, soft_wrap=True)
        else:
            self.console = None
            self.err_console = None

    def render_banner(self, config: AgentConfig, session_id: str | None) -> None:
        if not self.interactive:
            return

        if not RICH_AVAILABLE:
            line = "=" * 72
            print(line)
            print("JUSTIN CLI")
            print(f"provider: {_provider_title(config)}")
            print(f"model:    {config.model_name}")
            print(f"session:  {session_id or 'new'}")
            print(line)
            return

        panel_width = self._panel_width()
        logo = self._build_logo()
        meta = Table.grid(padding=(0, 2))
        meta.add_row("Provider", f"[bold]{_provider_title(config)}[/bold]")
        meta.add_row("Model", config.model_name)
        meta.add_row("Session", session_id or "new")
        if config.api_base:
            meta.add_row("API Base", config.api_base)
        meta.add_row("Theme", "Amber / Hermes-like")

        header = Group(
            logo,
            Text("-" * max(18, panel_width - 8), style="#7f4d00"),
            meta,
        )
        self.console.print(
            Panel(
                header,
                title="[bold #ffb000]JUSTIN CLI[/bold #ffb000]",
                border_style="#ffb000",
                padding=(1, 2),
                width=panel_width,
            )
        )

    def show_help(self) -> None:
        if not self.interactive:
            return
        if not RICH_AVAILABLE:
            _print_cli_help()
            return

        panel_width = self._panel_width()
        table = Table(show_header=True, header_style="bold #ffb000", box=None, expand=True)
        table.add_column("Command", style="bold")
        table.add_column("Description", style="#d8b37a")
        rows = [
            ("/help", "Show command list"),
            ("/session", "Show active session id"),
            ("/provider", "Show current provider config"),
            ("/stats", "Show current chat metrics"),
            ("/theme", "Show active CLI theme"),
            ("/new", "Start a new chat session"),
            ("/setup", "Re-run setup wizard"),
            ("/candidates", "List pending memory candidates"),
            ("/approve <id>", "Approve candidate"),
            ("/reject <id> [note]", "Reject candidate"),
            ("/memories [query]", "List/search approved memories"),
            ("/clear", "Clear terminal"),
            ("/exit", "Quit"),
        ]
        for command, description in rows:
            table.add_row(command, description)
        self.console.print(
            Panel(
                table,
                title="[bold #ffb000]Commands[/bold #ffb000]",
                border_style="#d28d00",
                padding=(0, 1),
                width=panel_width,
            )
        )

    def thinking(self) -> ContextManager[None]:
        if self.interactive and RICH_AVAILABLE:
            return self.console.status("[bold #ffb000]JUSTIN is thinking...[/bold #ffb000]", spinner="dots")

        print("Justin is thinking...", file=sys.stderr, flush=True)
        return nullcontext()

    def show_assistant_message(self, content: str) -> None:
        if not self.interactive:
            return
        if not RICH_AVAILABLE:
            print(f"\nJustin> {content}\n")
            return

        self.console.print(
            Panel(
                Text(content),
                title="[bold #ffb000]Justin[/bold #ffb000]",
                border_style="#ffb000",
                padding=(0, 1),
            )
        )

    def show_tool_events(self, events: list[Any]) -> None:
        if not events or not self.interactive:
            return
        if not RICH_AVAILABLE:
            print("Tool Events:")
            for event in events:
                marker = "ok" if event.ok else "error"
                print(f"  - [{marker}] {event.tool_name}: {event.summary} ({event.latency_ms}ms)")
            return

        table = Table(show_header=True, header_style="bold #ffb000", box=None)
        table.add_column("Status", style="bold")
        table.add_column("Tool", style="#f3c88d")
        table.add_column("Summary", style="white")
        table.add_column("Latency", style="dim")
        for event in events:
            marker = "[green]ok[/green]" if event.ok else "[red]error[/red]"
            table.add_row(marker, event.tool_name, event.summary, f"{event.latency_ms}ms")
        self.console.print(
            Panel(table, title="[bold dim]Tool Events[/bold dim]", border_style="dim")
        )

    def show_citations(self, citations: list[Any]) -> None:
        if not citations or not self.interactive:
            return
        if not RICH_AVAILABLE:
            print("Citations:")
            for c in citations:
                print(f"  - [{c.label}] {c.title}: {c.url}")
            return

        table = Table(show_header=False, box=None)
        table.add_column("Label", style="bold #f3c88d")
        table.add_column("Details", style="white")
        for c in citations:
            table.add_row(f"[{c.label}]", f"{c.title}\n[dim]{c.url}[/dim]")
        self.console.print(
            Panel(table, title="[bold dim]Citations[/bold dim]", border_style="dim")
        )

    def show_activated_skills(self, skills: list[Any]) -> None:
        if not skills or not self.interactive:
            return
        if not RICH_AVAILABLE:
            print("Activated Skills:")
            for s in skills:
                print(f"  - {s.name} ({s.version}): {s.summary}")
            return

        table = Table(show_header=False, box=None)
        table.add_column("Skill", style="bold #f3c88d")
        table.add_column("Summary", style="white")
        for s in skills:
            table.add_row(f"{s.name} v{s.version}", s.summary)
        self.console.print(
            Panel(table, title="[bold dim]Activated Skills[/bold dim]", border_style="dim")
        )

    def show_context_telemetry(self, telemetry: Any) -> None:
        if not telemetry or not self.interactive:
            return
        if not RICH_AVAILABLE:
            print(f"Telemetry: Context {telemetry.context_tokens_after} tokens "
                  f"(saved {telemetry.saved_tokens}).")
            return

        self.console.print(
            f"[dim]telemetry: context {telemetry.context_tokens_after} tokens "
            f"| saved {telemetry.saved_tokens} tokens by compression[/dim]"
        )

    def show_candidates(self, candidates: list[Any]) -> None:
        if not candidates or not self.interactive:
            return
        if not RICH_AVAILABLE:
            print("candidate memories:")
            for candidate in candidates:
                print(f"  - {candidate.id} [{candidate.kind}] {candidate.content}")
            return

        table = Table(show_header=True, header_style="bold #ffb000", box=None)
        table.add_column("ID", style="bold")
        table.add_column("Kind", style="#f3c88d")
        table.add_column("Content", style="white")
        for candidate in candidates:
            table.add_row(candidate.id, candidate.kind, candidate.content)
        self.console.print(
            Panel(table, title="[bold #ffb000]Pending Candidate Memories[/bold #ffb000]", border_style="#d28d00")
        )

    def show_info(self, message: str) -> None:
        if self.interactive and RICH_AVAILABLE:
            self.console.print(f"[#f3c88d]{message}[/#f3c88d]")
        else:
            print(message)

    def show_latency(self, elapsed_seconds: float) -> None:
        if self.interactive and RICH_AVAILABLE:
            self.console.print(
                f"[dim]status: responded in {elapsed_seconds:.1f}s | turns={self.metrics.turns} "
                f"| failures={self.metrics.failures}[/dim]"
            )
        else:
            print(f"Justin responded in {elapsed_seconds:.1f}s.", file=sys.stderr, flush=True)

    def show_error(self, exc: Exception, elapsed_seconds: float) -> None:
        detail = str(exc).strip() or exc.__class__.__name__
        lowered = detail.lower()

        if isinstance(exc, TimeoutError) or "timeout" in lowered or "timed out" in lowered:
            message = (
                f"Justin timed out after {elapsed_seconds:.1f}s waiting for the model. "
                "Try: 1) /new for a fresh session, 2) increase JUSTIN_MODEL_TIMEOUT_SECONDS, "
                "3) lower JUSTIN_MODEL_MAX_TOKENS, or run `Justin setup` to switch providers."
            )
        elif "remote end closed connection" in lowered or "connection closed by remote server" in lowered:
            message = (
                f"Justin request failed after {elapsed_seconds:.1f}s: remote server closed the connection. "
                "Please retry. If this repeats, reduce JUSTIN_MODEL_MAX_TOKENS or switch model/provider."
            )
        else:
            message = f"Justin request failed after {elapsed_seconds:.1f}s: {detail}"

        if self.interactive and RICH_AVAILABLE:
            self.err_console.print(
                Panel(
                    message,
                    title="[bold red]Request Error[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )
        else:
            print(message, file=sys.stderr, flush=True)

    def on_turn_success(self, elapsed_seconds: float) -> None:
        self.metrics.turns += 1
        self.metrics.last_latency_seconds = elapsed_seconds
        self.metrics.total_latency_seconds += elapsed_seconds

    def on_turn_failure(self) -> None:
        self.metrics.failures += 1

    def show_stats(self, config: AgentConfig, session_id: str | None) -> None:
        avg = self.metrics.avg_latency_seconds
        if self.interactive and RICH_AVAILABLE:
            table = Table(show_header=False, box=None, expand=True)
            table.add_row("Session", session_id or "new")
            table.add_row("Provider", _provider_title(config))
            table.add_row("Model", config.model_name)
            table.add_row("Turns", str(self.metrics.turns))
            table.add_row("Failures", str(self.metrics.failures))
            table.add_row("Last latency", f"{self.metrics.last_latency_seconds:.1f}s")
            table.add_row("Avg latency", f"{avg:.1f}s")
            self.console.print(
                Panel(
                    table,
                    title="[bold #ffb000]Session Stats[/bold #ffb000]",
                    border_style="#d28d00",
                    width=self._panel_width(),
                )
            )
            return

        print(f"session: {session_id or 'new'}")
        print(f"provider: {_provider_title(config)}")
        print(f"model: {config.model_name}")
        print(f"turns: {self.metrics.turns}")
        print(f"failures: {self.metrics.failures}")
        print(f"last latency: {self.metrics.last_latency_seconds:.1f}s")
        print(f"avg latency: {avg:.1f}s")

    def show_theme(self) -> None:
        self.show_info("Theme: Amber/Black with 3-band gradient JUSTIN logo and aligned Hermes-like panels.")

    def _build_logo(self):
        if not RICH_AVAILABLE:
            return Text("\n".join(JUSTIN_GRADIENT_LOGO_LINES))

        logo = Text()
        palette = (
            "#ffe18f",
            "#ffd56f",
            "#ffc24c",
            "#ffaf30",
            "#ea8b15",
            "#c8680f",
        )
        for index, line in enumerate(JUSTIN_GRADIENT_LOGO_LINES):
            logo.append(line, style=f"bold {palette[index]}")
            if index < len(JUSTIN_GRADIENT_LOGO_LINES) - 1:
                logo.append("\n")

        shadow_width = max(18, len(JUSTIN_GRADIENT_LOGO_LINES[-1].strip()) - 2)
        shadow = Text("   " + ("▄" * shadow_width), style="dim #4a2500")
        return Group(logo, shadow)

    def _panel_width(self) -> int:
        if not RICH_AVAILABLE:
            return 72
        available = max(self.console.width - 2, 40)
        return min(available, 88)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Justin: a local-first agent.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("setup", help="Run first-time provider/API setup wizard.")
    subparsers.add_parser("wechat", help="Start the agent as a WeChat bot.")

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

    if args.command in {"chat", "wechat"} and getattr(args, "message", None) is None:
        config = _maybe_prompt_first_run_setup(config)

    bundle = build_runtime_bundle(config)
    runtime = JustinRuntime(bundle)
    try:
        match args.command:
            case "chat":
                _run_chat(runtime, args.session, args.message)
            case "wechat":
                from .wechat import start_wechat_bot
                start_wechat_bot(runtime, config)
            case "candidate":
                _run_candidate_commands(runtime, args)
            case "memory":
                _run_memory_commands(runtime, args)
            case "session":
                _print_json([to_plain_dict(item) for item in runtime.list_sessions()])
    finally:
        runtime.close()


def _get_prompt_session(runtime: JustinRuntime) -> Any:
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    history_file = runtime.config.settings_path.parent / ".cli_history"
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    @bindings.add("c-j") # Some terminals send ctrl-j for alt-enter
    def _(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "." * (width - 1) + " ",
    )

def _run_chat(runtime: JustinRuntime, session_id: str | None, message: str | None) -> None:
    renderer = JustinCliRenderer(interactive=message is None)

    if message:
        result = _send_with_feedback(runtime, message, session_id, renderer)
        if result is None:
            return
        _print_json(to_plain_dict(result))
        return

    renderer.render_banner(runtime.config, session_id)
    renderer.show_help()
    active_session_id = session_id

    pt_session = _get_prompt_session(runtime)
    if pt_session:
        print("Tip: Press Alt+Enter or Ctrl+J to insert a newline. Press Enter to submit.")

    while True:
        try:
            if pt_session:
                prompt = pt_session.prompt(HTML("<b><ansigreen>you></ansigreen></b> ")).strip()
            else:
                prompt = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not prompt:
            continue
        if prompt.startswith("/"):
            active_session_id, should_exit = _handle_slash_command(prompt, runtime, active_session_id, renderer)
            if should_exit:
                return
            continue
        if prompt in {"/exit", "/quit"}:
            return
        result = _send_with_feedback(runtime, prompt, active_session_id, renderer)
        if result is None:
            continue
        active_session_id = result.session.id
        renderer.show_tool_events(result.tool_events)
        renderer.show_citations(result.citations)
        renderer.show_activated_skills(result.activated_skills)
        renderer.show_assistant_message(result.assistant_message.content)
        renderer.show_candidates(result.candidates)
        renderer.show_context_telemetry(result.context_telemetry)


def _send_with_feedback(
    runtime: JustinRuntime,
    content: str,
    session_id: str | None,
    renderer: JustinCliRenderer,
):
    started_at = time.perf_counter()
    with renderer.thinking():
        try:
            result = runtime.send_message(content=content, session_id=session_id)
        except Exception as exc:  # keep CLI stable and show user-facing errors.
            elapsed = time.perf_counter() - started_at
            renderer.on_turn_failure()
            renderer.show_error(exc, elapsed)
            return None

    elapsed = time.perf_counter() - started_at
    renderer.on_turn_success(elapsed)
    renderer.show_latency(elapsed)
    return result


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
    os.environ["JUSTIN_MODEL_TEMPERATURE"] = str(config.model_temperature)
    os.environ["JUSTIN_MODEL_TOP_P"] = str(config.model_top_p)
    os.environ["JUSTIN_MODEL_MAX_TOKENS"] = str(config.model_max_tokens)
    os.environ["JUSTIN_MODEL_TIMEOUT_SECONDS"] = str(config.model_timeout_seconds)
    os.environ["JUSTIN_MODEL_RETRY_MAX_TOKENS"] = str(config.model_retry_max_tokens)
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


def _print_cli_help() -> None:
    print("Commands:")
    print("  /help                  Show command list")
    print("  /session               Show active session id")
    print("  /provider              Show current provider config")
    print("  /stats                 Show current chat metrics")
    print("  /theme                 Show current CLI theme")
    print("  /new                   Start a new session")
    print("  /setup                 Re-run setup wizard")
    print("  /wechat                Start the agent in WeChat mode")
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
    renderer: JustinCliRenderer,
) -> tuple[str | None, bool]:
    tokens = raw_command.split()
    command = tokens[0].lower()

    if command in {"/exit", "/quit"}:
        return active_session_id, True
    if command == "/help":
        renderer.show_help()
        return active_session_id, False
    if command == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        renderer.render_banner(runtime.config, active_session_id)
        return active_session_id, False
    if command == "/session":
        renderer.show_info(f"active session: {active_session_id or '(new session on next message)'}")
        return active_session_id, False
    if command == "/provider":
        renderer.show_info(f"Provider: {_provider_title(runtime.config)}")
        renderer.show_info(f"Model: {runtime.config.model_name}")
        if runtime.config.api_base:
            renderer.show_info(f"API base: {runtime.config.api_base}")
        return active_session_id, False
    if command == "/stats":
        renderer.show_stats(runtime.config, active_session_id)
        return active_session_id, False
    if command == "/theme":
        renderer.show_theme()
        return active_session_id, False
    if command == "/new":
        renderer.show_info("Started a new session.")
        return None, False
    if command == "/setup":
        updated = run_setup_wizard(runtime.config)
        runtime.apply_config(updated)
        renderer.show_info("Runtime provider config reloaded.")
        return active_session_id, False
    if command == "/wechat":
        from .wechat import start_wechat_bot
        start_wechat_bot(runtime, runtime.config)
        return active_session_id, False
    if command == "/candidates":
        items = runtime.list_candidates(status="pending")
        if not items:
            renderer.show_info("No pending candidates.")
            return active_session_id, False
        for item in items[:20]:
            renderer.show_info(f"- {item.id} [{item.kind}] {item.content}")
        return active_session_id, False
    if command == "/approve":
        if len(tokens) < 2:
            renderer.show_info("usage: /approve <candidate-id>")
            return active_session_id, False
        try:
            memory = runtime.confirm_candidate(tokens[1])
            renderer.show_info(f"approved -> memory {memory.id}")
        except KeyError as exc:
            renderer.show_info(str(exc))
        return active_session_id, False
    if command == "/reject":
        if len(tokens) < 2:
            renderer.show_info("usage: /reject <candidate-id> [note]")
            return active_session_id, False
        note = " ".join(tokens[2:]) if len(tokens) > 2 else None
        try:
            runtime.reject_candidate(tokens[1], note)
            renderer.show_info("candidate rejected.")
        except KeyError as exc:
            renderer.show_info(str(exc))
        return active_session_id, False
    if command == "/memories":
        query = " ".join(tokens[1:]).strip()
        items = runtime.search_memories(query) if query else runtime.list_memories()
        if not items:
            renderer.show_info("No memories found.")
            return active_session_id, False
        for item in items[:20]:
            score = f" score={item.score:.2f}" if hasattr(item, "score") else ""
            renderer.show_info(f"- {item.id} [{item.kind}]{score} {item.content}")
        return active_session_id, False

    renderer.show_info(f"Unknown command: {raw_command}. Type /help.")
    return active_session_id, False
