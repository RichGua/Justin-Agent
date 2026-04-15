from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_FILENAME = "settings.json"
PROVIDER_LOCAL = "local"
PROVIDER_OPENAI = "openai"
PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"
PROVIDER_OLLAMA = "ollama"
PROVIDER_NVIDIA_NIM = "nvidia-nim"
SUPPORTED_PROVIDERS = {
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
    PROVIDER_OLLAMA,
    PROVIDER_NVIDIA_NIM,
}


@dataclass(slots=True)
class AgentConfig:
    home_dir: Path
    database_path: Path
    settings_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    model_provider: str = PROVIDER_LOCAL
    model_name: str = "local-fallback"
    api_base: str | None = None
    api_key: str | None = None
    retrieval_top_k: int = 5
    context_messages: int = 8
    embedding_dimensions: int = 96

    def __post_init__(self) -> None:
        if self.settings_path is None:
            self.settings_path = self.home_dir / DEFAULT_SETTINGS_FILENAME

    @classmethod
    def from_env(cls) -> "AgentConfig":
        home_dir = Path(os.getenv("JUSTIN_HOME", Path.cwd() / ".justin"))
        settings_path = home_dir / DEFAULT_SETTINGS_FILENAME
        persisted = cls._load_settings(settings_path)

        def _pick(env_name: str, persisted_key: str, default: Any = None) -> Any:
            return os.getenv(env_name, persisted.get(persisted_key, default))

        model_provider = _pick("JUSTIN_MODEL_PROVIDER", "model_provider", PROVIDER_LOCAL)
        if model_provider not in SUPPORTED_PROVIDERS:
            model_provider = PROVIDER_LOCAL

        return cls(
            home_dir=home_dir,
            database_path=home_dir / "agent.db",
            settings_path=settings_path,
            host=_pick("JUSTIN_HOST", "host", "127.0.0.1"),
            port=int(_pick("JUSTIN_PORT", "port", "8765")),
            model_provider=model_provider,
            model_name=_pick("JUSTIN_MODEL_NAME", "model_name", "local-fallback"),
            api_base=_pick("JUSTIN_API_BASE", "api_base"),
            api_key=_pick("JUSTIN_API_KEY", "api_key"),
            retrieval_top_k=int(_pick("JUSTIN_RETRIEVAL_TOP_K", "retrieval_top_k", "5")),
            context_messages=int(_pick("JUSTIN_CONTEXT_MESSAGES", "context_messages", "8")),
            embedding_dimensions=int(_pick("JUSTIN_EMBEDDING_DIMENSIONS", "embedding_dimensions", "96")),
        )

    def ensure_directories(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)

    def has_user_settings(self) -> bool:
        return self.settings_path.exists()

    def to_settings(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "retrieval_top_k": self.retrieval_top_k,
            "context_messages": self.context_messages,
            "embedding_dimensions": self.embedding_dimensions,
        }

    def save_settings(self) -> None:
        self.ensure_directories()
        payload = self.to_settings()
        self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_settings(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            # `utf-8-sig` keeps compatibility with files saved by PowerShell UTF-8 BOM.
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return {}
