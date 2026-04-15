from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AgentConfig:
    home_dir: Path
    database_path: Path
    host: str = "127.0.0.1"
    port: int = 8765
    model_provider: str = "local"
    model_name: str = "local-fallback"
    api_base: str | None = None
    api_key: str | None = None
    retrieval_top_k: int = 5
    context_messages: int = 8
    embedding_dimensions: int = 96

    @classmethod
    def from_env(cls) -> "AgentConfig":
        home_dir = Path(os.getenv("PERSONAL_AGENT_HOME", Path.cwd() / ".personal_agent"))
        return cls(
            home_dir=home_dir,
            database_path=home_dir / "agent.db",
            host=os.getenv("PERSONAL_AGENT_HOST", "127.0.0.1"),
            port=int(os.getenv("PERSONAL_AGENT_PORT", "8765")),
            model_provider=os.getenv("PERSONAL_AGENT_MODEL_PROVIDER", "local"),
            model_name=os.getenv("PERSONAL_AGENT_MODEL_NAME", "local-fallback"),
            api_base=os.getenv("PERSONAL_AGENT_API_BASE"),
            api_key=os.getenv("PERSONAL_AGENT_API_KEY"),
            retrieval_top_k=int(os.getenv("PERSONAL_AGENT_RETRIEVAL_TOP_K", "5")),
            context_messages=int(os.getenv("PERSONAL_AGENT_CONTEXT_MESSAGES", "8")),
            embedding_dimensions=int(os.getenv("PERSONAL_AGENT_EMBEDDING_DIMENSIONS", "96")),
        )

    def ensure_directories(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
