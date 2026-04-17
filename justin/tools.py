from __future__ import annotations

import subprocess
import time
import os
import signal
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib import parse, request


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, object]
    cost_hint: str = "low"
    timeout_sec: int = 20


@dataclass(slots=True)
class ToolContext:
    session_id: str | None
    workspace_root: Path
    home_dir: Path
    cwd: Path


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: object
    summary: str
    error: str | None = None
    source: str = "builtin"
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionPolicy:
    workspace_root: Path
    home_dir: Path
    allowed_tools: set[str] = field(default_factory=set)
    allowed_programs: set[str] = field(default_factory=lambda: {"git", "rg", "where"})
    allow_harness: bool = True
    command_timeout_sec: int = 60
    network_enabled: bool = True
    max_output_chars: int = 4000

    def allows_tool(self, name: str) -> bool:
        return not self.allowed_tools or name in self.allowed_tools

    def resolve_path(self, raw_path: str | Path | None, *, must_exist: bool = True) -> Path:
        candidate = Path(raw_path or self.workspace_root)
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        allowed_roots = [self.workspace_root.resolve(), self.home_dir.resolve()]
        if not any(_is_relative_to(candidate, root) for root in allowed_roots):
            raise PermissionError(f"Path is outside allowed roots: {candidate}")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Path not found: {candidate}")
        return candidate

    def validate_program(self, program: str) -> None:
        normalized = Path(program).name.lower()
        if normalized.endswith(".exe"):
            normalized = normalized[:-4]
        if normalized not in {item.lower() for item in self.allowed_programs}:
            raise PermissionError(
                f"Command '{program}' is not allowed. Allowed programs: {sorted(self.allowed_programs)}"
            )


class ToolRegistry:
    def __init__(self, policy: ExecutionPolicy) -> None:
        self.policy = policy
        self._specs: dict[str, ToolSpec] = {}
        self._executors: dict[str, object] = {}

    def register(self, spec: ToolSpec, executor) -> None:
        self._specs[spec.name] = spec
        self._executors[spec.name] = executor

    def list_specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def execute(self, name: str, arguments: dict[str, object], context: ToolContext) -> ToolResult:
        if not self.policy.allows_tool(name):
            raise PermissionError(f"Tool '{name}' is disabled by policy.")
        executor = self._executors.get(name)
        if executor is None:
            raise KeyError(f"Unknown tool: {name}")
        started_at = time.perf_counter()
        try:
            result: ToolResult = executor(arguments, context, self.policy)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            return ToolResult(
                ok=False,
                output={},
                summary=f"{name} failed",
                error=str(exc),
                meta={"latency_ms": latency_ms},
            )

        result.meta = dict(result.meta)
        result.meta.setdefault("latency_ms", int((time.perf_counter() - started_at) * 1000))
        return result


def build_default_tool_registry(policy: ExecutionPolicy, search_service=None) -> ToolRegistry:
    registry = ToolRegistry(policy=policy)
    registry.register(
        ToolSpec(
            name="fs_list",
            description="List files and folders inside the allowed workspace.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        _execute_fs_list,
    )
    registry.register(
        ToolSpec(
            name="fs_read",
            description="Read a text file inside the allowed workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        ),
        _execute_fs_read,
    )
    registry.register(
        ToolSpec(
            name="command_run",
            description="Run an allowlisted command without using a shell.",
            input_schema={
                "type": "object",
                "properties": {
                    "program": {"type": "string"},
                    "args": {"type": "array"},
                    "cwd": {"type": "string"},
                },
                "required": ["program"],
            },
        ),
        _execute_command_run,
    )
    registry.register(
        ToolSpec(
            name="harness_bash",
            description="Run a shell command (bash/cmd) in the workspace. Use this for testing, building, or executing code. Long-running processes will be terminated after timeout.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Relative to workspace root. Defaults to root."},
                },
                "required": ["command"],
            },
            timeout_sec=60,
        ),
        _execute_harness_bash,
    )
    registry.register(
        ToolSpec(
            name="http_fetch",
            description="Fetch raw text content from an HTTP or HTTPS URL.",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        ),
        _execute_http_fetch,
    )
    registry.register(
        ToolSpec(
            name="page_extract",
            description="Fetch and extract readable text from a web page.",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        ),
        _execute_page_extract,
    )
    if search_service is not None:
        registry.register(
            ToolSpec(
                name="search_web",
                description="Search the public web using free providers and return ranked results.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
            ),
            lambda arguments, context, current_policy: _execute_search_web(
                arguments, context, current_policy, search_service
            ),
        )
    return registry


def _execute_fs_list(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy) -> ToolResult:
    root = policy.resolve_path(arguments.get("path"))
    limit = max(1, min(int(arguments.get("limit", 40)), 200))
    items = []
    for entry in sorted(root.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))[:limit]:
        items.append(
            {
                "name": entry.name,
                "path": str(entry),
                "type": "file" if entry.is_file() else "dir",
            }
        )
    return ToolResult(
        ok=True,
        output={"path": str(root), "items": items},
        summary=f"Listed {len(items)} entries from {root.name or str(root)}.",
        meta={"count": len(items)},
    )


def _execute_fs_read(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy) -> ToolResult:
    path = policy.resolve_path(arguments.get("path"))
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got directory: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    start_line = max(int(arguments.get("start_line", 1)), 1)
    end_line = min(int(arguments.get("end_line", start_line + 79)), len(lines) or 1)
    excerpt = "\n".join(lines[start_line - 1 : end_line])
    excerpt = excerpt[: policy.max_output_chars]
    return ToolResult(
        ok=True,
        output={
            "path": str(path),
            "start_line": start_line,
            "end_line": end_line,
            "content": excerpt,
        },
        summary=f"Read {path.name} lines {start_line}-{end_line}.",
        meta={"chars": len(excerpt)},
    )


def _execute_command_run(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy) -> ToolResult:
    program = str(arguments.get("program", "")).strip()
    if not program:
        raise ValueError("program is required")
    policy.validate_program(program)
    args = [str(item) for item in arguments.get("args", [])]
    cwd = policy.resolve_path(arguments.get("cwd"), must_exist=True)
    completed = subprocess.run(
        [program, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=policy.command_timeout_sec,
        shell=False,
        check=False,
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    combined = "\n".join(part for part in [stdout, stderr] if part)
    combined = combined[: policy.max_output_chars]
    return ToolResult(
        ok=completed.returncode == 0,
        output={
            "program": program,
            "args": args,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout[: policy.max_output_chars],
            "stderr": stderr[: policy.max_output_chars],
        },
        summary=f"Ran {program} {' '.join(args)} (exit={completed.returncode}).".strip(),
        error=None if completed.returncode == 0 else combined or f"Command exited with {completed.returncode}",
        meta={"exit_code": completed.returncode},
    )


def _execute_harness_bash(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy) -> ToolResult:
    if not policy.allow_harness:
        raise PermissionError("Harness (shell) execution is disabled by policy.")

    command = str(arguments.get("command", "")).strip()
    if not command:
        raise ValueError("command is required")

    cwd_arg = arguments.get("cwd")
    cwd = policy.resolve_path(cwd_arg, must_exist=True) if cwd_arg else policy.workspace_root

    # Use psutil to clean up process tree on timeout
    import psutil

    def kill_proc_tree(pid, include_parent=True):
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if include_parent:
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

    # Ensure stdout/stderr uses UTF-8 to prevent Windows encoding issues
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["LANG"] = "en_US.UTF-8"

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    stdout_data = ""
    try:
        stdout_data, _ = proc.communicate(timeout=policy.command_timeout_sec)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        kill_proc_tree(proc.pid)
        stdout_data, _ = proc.communicate()
        returncode = proc.returncode
        stdout_data += "\n\n[Timeout] Command terminated after {} seconds.".format(policy.command_timeout_sec)

    stdout_data = stdout_data.strip()
    excerpt = stdout_data[: policy.max_output_chars]

    return ToolResult(
        ok=returncode == 0,
        output={
            "command": command,
            "cwd": str(cwd),
            "exit_code": returncode,
            "output": excerpt,
        },
        summary=f"Ran bash command '{command[:20]}...' (exit={returncode}).",
        error=None if returncode == 0 else (excerpt or f"Command exited with {returncode}"),
        meta={"exit_code": returncode},
    )

def _execute_http_fetch(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy) -> ToolResult:
    if not policy.network_enabled:
        raise PermissionError("HTTP fetch is disabled by policy.")
    url = _normalize_url(str(arguments.get("url", "")).strip())
    req = request.Request(url, headers={"User-Agent": "Justin-Agent/0.2"})
    with request.urlopen(req, timeout=20) as response:
        raw = response.read(policy.max_output_chars)
        charset = response.headers.get_content_charset() or "utf-8"
        content = raw.decode(charset, errors="replace")
        content_type = response.headers.get_content_type()
    return ToolResult(
        ok=True,
        output={"url": url, "content_type": content_type, "content": content},
        summary=f"Fetched {url}.",
        meta={"content_type": content_type},
    )


def _execute_page_extract(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy) -> ToolResult:
    fetched = _execute_http_fetch(arguments, context, policy)
    content = str(fetched.output.get("content", "")) if isinstance(fetched.output, dict) else ""
    parser = _HTMLTextExtractor()
    parser.feed(content)
    text = parser.get_text()[: policy.max_output_chars]
    return ToolResult(
        ok=True,
        output={
            "url": fetched.output.get("url"),
            "title": parser.title,
            "text": text,
        },
        summary=f"Extracted page text from {fetched.output.get('url')}.",
        meta={"title": parser.title},
    )


def _execute_search_web(arguments: dict[str, object], context: ToolContext, policy: ExecutionPolicy, search_service) -> ToolResult:
    if not policy.network_enabled:
        raise PermissionError("Web search is disabled by policy.")
    query = str(arguments.get("query", "")).strip()
    top_k = max(1, min(int(arguments.get("top_k", 5)), 10))
    locale = str(arguments.get("locale", "en-US"))
    results = search_service.search(query=query, top_k=top_k, locale=locale)
    payload = [
        {
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
            "source": item.source,
            "fetched_at": item.fetched_at,
            "confidence": item.confidence,
        }
        for item in results
    ]
    return ToolResult(
        ok=True,
        output={"query": query, "results": payload},
        summary=f"Found {len(payload)} search results for '{query}'.",
        meta={"count": len(payload)},
    )


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []
        self._in_title = False
        self._title_parts: list[str] = []

    @property
    def title(self) -> str:
        return " ".join(part.strip() for part in self._title_parts if part.strip()).strip()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "li", "section", "article", "h1", "h2", "h3", "h4", "br"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "section", "article", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
        cleaned = " ".join(data.split())
        if cleaned:
            self._chunks.append(cleaned)

    def get_text(self) -> str:
        text = " ".join(self._chunks)
        text = text.replace(" \n ", "\n").replace("\n ", "\n").replace(" \n", "\n")
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def _normalize_url(url: str) -> str:
    parsed = parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected a full HTTP/HTTPS URL, got: {url}")
    return url


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
