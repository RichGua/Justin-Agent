from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretPattern:
    name: str
    regex: re.Pattern[str]
    has_group_value: bool = False


PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    SecretPattern("nvidia_key", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b")),
    SecretPattern(
        "private_key_block",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ),
    SecretPattern(
        "api_key_assignment",
        re.compile(
            r"(?i)\b(?:JUSTIN_API_KEY|OPENAI_API_KEY|NVIDIA_API_KEY|api_key)\b\s*[:=]\s*[\"']([^\"'\n]{8,})[\"']"
        ),
        has_group_value=True,
    ),
)


def _run_git(args: list[str]) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip() or "git command failed")
    return proc.stdout


def _parse_null_delimited(raw: bytes) -> list[str]:
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\x00") if item]


def _read_staged_file(path: str) -> bytes:
    return _run_git(["show", f":{path}"])


def _read_worktree_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    placeholder_markers = (
        "your_",
        "your-",
        "example",
        "replace",
        "changeme",
        "xxxxx",
        "nvidia_api_key",
        "justin_api_key",
        "openai_api_key",
        "${",
        "$",
        "<",
    )
    return any(marker in lowered for marker in placeholder_markers)


def _scan_text(path: str, text: str) -> list[str]:
    findings: list[str] = []
    for pattern in PATTERNS:
        for match in pattern.regex.finditer(text):
            if pattern.has_group_value:
                value = match.group(1).strip()
                if _looks_like_placeholder(value):
                    continue
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append(f"{path}:{line_no} [{pattern.name}]")
    return findings


def _target_files(staged_only: bool) -> list[str]:
    if staged_only:
        raw = _run_git(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT"])
        return _parse_null_delimited(raw)
    raw = _run_git(["ls-files", "-z"])
    return _parse_null_delimited(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan files for likely secrets.")
    parser.add_argument("--staged", action="store_true", help="Scan staged files only.")
    args = parser.parse_args()

    try:
        files = _target_files(staged_only=args.staged)
    except RuntimeError as exc:
        print(f"secret-check: {exc}", file=sys.stderr)
        return 2

    findings: list[str] = []
    for path in files:
        if path.startswith(".venv/"):
            continue
        try:
            raw = _read_staged_file(path) if args.staged else _read_worktree_file(path)
        except (OSError, RuntimeError):
            continue
        if _is_binary(raw):
            continue
        text = _decode_text(raw)
        findings.extend(_scan_text(path, text))

    if findings:
        print("secret-check: potential secrets found. Commit blocked.", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        print(
            "If this is a false positive, replace with placeholders (e.g. your_api_key_here) before commit.",
            file=sys.stderr,
        )
        return 1

    print("secret-check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
