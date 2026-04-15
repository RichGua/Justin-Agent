from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .types import InstalledSkill


MANIFEST_FILENAME = "justin.skill.json"


@dataclass(slots=True)
class SkillManifest:
    name: str
    version: str = "0.1.0"
    summary: str = ""
    description: str = ""
    entry: str = "SKILL.md"
    triggers: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = ""
    activation_prompt: str | None = None

    def to_record(self, install_path: Path) -> InstalledSkill:
        return InstalledSkill(
            name=self.name,
            version=self.version,
            source=self.source,
            install_path=str(install_path),
            summary=self.summary,
            description=self.description,
            entry=self.entry,
            triggers=self.triggers,
            required_tools=self.required_tools,
            tags=self.tags,
        )


class SkillManager:
    def __init__(self, skills_dir: Path, store) -> None:
        self.skills_dir = skills_dir
        self.store = store
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def list_installed(self) -> list[InstalledSkill]:
        return self.store.list_installed_skills()

    def install(self, source: str, name: str | None = None) -> InstalledSkill:
        if _looks_like_github_source(source):
            return self.install_from_github(source, name=name)
        return self.install_from_local(Path(source), name=name)

    def install_from_local(self, source_path: Path, name: str | None = None) -> InstalledSkill:
        source_root = source_path.expanduser().resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Skill path does not exist: {source_root}")
        manifest = load_skill_manifest(source_root, source=f"local:{source_root}")
        install_name = _safe_skill_name(name or manifest.name)
        target_dir = self.skills_dir / install_name
        _copy_skill_tree(source_root, target_dir)
        installed = manifest.to_record(target_dir)
        if name:
            installed.name = install_name
        self.store.save_installed_skill(installed)
        return installed

    def install_from_github(self, source: str, name: str | None = None) -> InstalledSkill:
        git_url, requested_subdir = _normalize_github_source(source)
        with tempfile.TemporaryDirectory(prefix="justin_skill_") as temp_dir:
            temp_root = Path(temp_dir) / "repo"
            result = subprocess.run(
                ["git", "clone", "--depth", "1", git_url, str(temp_root)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
                raise RuntimeError(f"GitHub skill install failed: {detail}")
            source_root = _discover_skill_root(temp_root, requested_subdir=requested_subdir)
            manifest = load_skill_manifest(source_root, source=source)
            install_name = _safe_skill_name(name or manifest.name)
            target_dir = self.skills_dir / install_name
            _copy_skill_tree(source_root, target_dir)
        installed = manifest.to_record(target_dir)
        if name:
            installed.name = install_name
        self.store.save_installed_skill(installed)
        return installed

    def update(self, name: str) -> InstalledSkill:
        current = self.store.get_installed_skill(name)
        if current is None:
            raise KeyError(f"Unknown skill: {name}")
        self.remove(name)
        return self.install(current.source, name=name)

    def remove(self, name: str) -> None:
        current = self.store.get_installed_skill(name)
        if current is None:
            raise KeyError(f"Unknown skill: {name}")
        path = Path(current.install_path)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        self.store.remove_installed_skill(name)

    def match_for_query(self, query: str, limit: int = 3) -> list[InstalledSkill]:
        normalized = query.lower()
        ranked: list[tuple[int, InstalledSkill]] = []
        for skill in self.list_installed():
            score = 0
            for trigger in skill.triggers:
                if trigger and trigger.lower() in normalized:
                    score += 3
            for tag in skill.tags:
                if tag and tag.lower() in normalized:
                    score += 1
            if skill.name.lower() in normalized:
                score += 2
            if score > 0:
                ranked.append((score, skill))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [pair[1] for pair in ranked[:limit]]

    def build_activation_block(self, skills: list[InstalledSkill]) -> str:
        blocks: list[str] = []
        for skill in skills:
            path = Path(skill.install_path) / skill.entry
            prompt = ""
            if path.exists():
                prompt = path.read_text(encoding="utf-8").strip()
            prompt = prompt[:800].strip()
            summary = skill.summary or skill.description or skill.name
            header = f"- {skill.name} ({skill.version}): {summary}"
            if prompt:
                blocks.append(f"{header}\n  {prompt.replace(chr(10), ' ')}")
            else:
                blocks.append(header)
        return "\n".join(blocks)


def load_skill_manifest(path: Path, source: str) -> SkillManifest:
    manifest_path = path / MANIFEST_FILENAME
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = SkillManifest(
            name=str(payload.get("name") or path.name),
            version=str(payload.get("version") or "0.1.0"),
            summary=str(payload.get("summary") or ""),
            description=str(payload.get("description") or ""),
            entry=str(payload.get("entry") or "SKILL.md"),
            triggers=[str(item) for item in payload.get("triggers", [])],
            required_tools=[str(item) for item in payload.get("required_tools", [])],
            tags=[str(item) for item in payload.get("tags", [])],
            activation_prompt=payload.get("activation_prompt"),
            source=source,
        )
    else:
        entry_path = path / "SKILL.md"
        if not entry_path.exists():
            raise FileNotFoundError(f"Skill manifest missing: expected {manifest_path} or {entry_path}")
        summary = _derive_summary_from_skill_md(entry_path)
        manifest = SkillManifest(
            name=path.name,
            summary=summary,
            description=summary,
            entry="SKILL.md",
            source=source,
        )

    entry_file = path / manifest.entry
    if not entry_file.exists():
        raise FileNotFoundError(f"Skill entry file not found: {entry_file}")
    manifest.name = _safe_skill_name(manifest.name)
    return manifest


def _derive_summary_from_skill_md(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#") and not cleaned.startswith("---"):
            return cleaned[:160]
    return path.stem


def _copy_skill_tree(source_root: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(
        source_root,
        target_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )


def _normalize_github_source(source: str) -> tuple[str, str | None]:
    base, _, fragment = source.partition("#")
    if base.startswith("https://github.com/") or base.startswith("http://github.com/"):
        git_url = base if base.endswith(".git") else f"{base}.git"
    else:
        cleaned = base.strip().strip("/")
        git_url = f"https://github.com/{cleaned}.git"
    return git_url, fragment or None


def _discover_skill_root(repo_root: Path, requested_subdir: str | None = None) -> Path:
    if requested_subdir:
        candidate = (repo_root / requested_subdir).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Skill subdirectory not found in repo: {requested_subdir}")
        return candidate

    if _is_skill_root(repo_root):
        return repo_root

    candidates = [path for path in repo_root.iterdir() if path.is_dir() and _is_skill_root(path)]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "Could not determine skill root. Provide a repo where SKILL.md or justin.skill.json is at the root, "
        "or append '#subdir' to select one skill directory."
    )


def _is_skill_root(path: Path) -> bool:
    return (path / MANIFEST_FILENAME).exists() or (path / "SKILL.md").exists()


def _looks_like_github_source(source: str) -> bool:
    normalized = source.strip().lower()
    return normalized.startswith("https://github.com/") or normalized.count("/") >= 1


def _safe_skill_name(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "skill"
