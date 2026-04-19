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
WECHAT_POLICY_OPEN = "open"
WECHAT_POLICY_ALLOWLIST = "allowlist"
WECHAT_POLICY_DISABLED = "disabled"
WECHAT_POLICY_PAIRING = "pairing"
WECHAT_DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
WECHAT_ACCESS_OPEN = WECHAT_POLICY_OPEN
WECHAT_ACCESS_ALLOWLIST = WECHAT_POLICY_ALLOWLIST
WECHAT_ACCESS_DISABLED = WECHAT_POLICY_DISABLED
SUPPORTED_PROVIDERS = {
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
    PROVIDER_OLLAMA,
    PROVIDER_NVIDIA_NIM,
}
SUPPORTED_WECHAT_ACCESS_POLICIES = {
    WECHAT_POLICY_OPEN,
    WECHAT_POLICY_ALLOWLIST,
    WECHAT_POLICY_DISABLED,
}
SUPPORTED_WECHAT_DM_POLICIES = {
    WECHAT_POLICY_OPEN,
    WECHAT_POLICY_ALLOWLIST,
    WECHAT_POLICY_DISABLED,
    WECHAT_POLICY_PAIRING,
}
SUPPORTED_WECHAT_GROUP_POLICIES = {
    WECHAT_POLICY_OPEN,
    WECHAT_POLICY_ALLOWLIST,
    WECHAT_POLICY_DISABLED,
}


@dataclass(slots=True)
class AgentConfig:
    home_dir: Path
    database_path: Path
    settings_path: Path | None = None
    skills_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    model_provider: str = PROVIDER_LOCAL
    model_name: str = "local-fallback"
    api_base: str | None = None
    api_key: str | None = None
    model_temperature: float = 0.3
    model_top_p: float = 0.95
    model_max_tokens: int = 1024
    model_timeout_seconds: int = 60
    model_retry_max_tokens: int = 8192
    retrieval_top_k: int = 5
    context_messages: int = 8
    embedding_dimensions: int = 96
    context_max_input_tokens: int = 3200
    context_reserved_output_tokens: int = 800
    context_summary_trigger_messages: int = 10
    context_tool_fact_limit: int = 4
    context_policy: str = "balanced"
    search_provider_order: str = "duckduckgo,wikipedia"
    search_top_k: int = 5
    search_cache_ttl_hours: int = 12
    search_locale: str = "en-US"
    network_tools_enabled: bool = True
    shell_allowed_programs: str = "git,rg,where"
    tool_max_output_chars: int = 4000
    
    # WeChat settings
    wechat_enabled: bool = False
    wechat_account_id: str = ""
    wechat_token: str = ""
    wechat_base_url: str = WECHAT_DEFAULT_BASE_URL
    wechat_auto_reply_prefix: str = "[Justin] "
    wechat_dm_policy: str = WECHAT_POLICY_OPEN
    wechat_allowed_users: str = ""
    wechat_group_policy: str = WECHAT_POLICY_DISABLED
    wechat_group_allowed_users: str = ""
    
    # Agent Identity
    agent_name: str = "Justin"
    system_prompt_prefix: str = "You are {agent_name}, a practical local-first agent."

    def __post_init__(self) -> None:
        if self.settings_path is None:
            self.settings_path = self.home_dir / DEFAULT_SETTINGS_FILENAME
        if self.skills_dir is None:
            self.skills_dir = self.home_dir / "skills"
        if self.wechat_dm_policy not in SUPPORTED_WECHAT_DM_POLICIES:
            self.wechat_dm_policy = WECHAT_POLICY_OPEN
        if self.wechat_group_policy not in SUPPORTED_WECHAT_GROUP_POLICIES:
            self.wechat_group_policy = WECHAT_POLICY_DISABLED

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
        def _pick_any(env_names: tuple[str, ...], persisted_keys: tuple[str, ...], default: Any = None) -> Any:
            for env_name in env_names:
                value = os.getenv(env_name)
                if value not in (None, ""):
                    return value
            for persisted_key in persisted_keys:
                if persisted_key in persisted and persisted.get(persisted_key) not in (None, ""):
                    return persisted.get(persisted_key)
            return default

        dm_policy = _pick_any(
            ("JUSTIN_WECHAT_DM_POLICY", "WEIXIN_DM_POLICY", "JUSTIN_WECHAT_ACCESS_POLICY"),
            ("wechat_dm_policy", "wechat_access_policy"),
            WECHAT_POLICY_OPEN,
        )
        if dm_policy not in SUPPORTED_WECHAT_DM_POLICIES:
            dm_policy = WECHAT_POLICY_OPEN

        group_policy = _pick_any(
            ("JUSTIN_WECHAT_GROUP_POLICY", "WEIXIN_GROUP_POLICY"),
            ("wechat_group_policy",),
            WECHAT_POLICY_DISABLED,
        )
        if group_policy not in SUPPORTED_WECHAT_GROUP_POLICIES:
            group_policy = WECHAT_POLICY_DISABLED

        return cls(
            home_dir=home_dir,
            database_path=home_dir / "agent.db",
            settings_path=settings_path,
            skills_dir=Path(_pick("JUSTIN_SKILLS_DIR", "skills_dir", home_dir / "skills")),
            host=_pick("JUSTIN_HOST", "host", "127.0.0.1"),
            port=int(_pick("JUSTIN_PORT", "port", "8765")),
            model_provider=model_provider,
            model_name=_pick("JUSTIN_MODEL_NAME", "model_name", "local-fallback"),
            api_base=_pick("JUSTIN_API_BASE", "api_base"),
            api_key=_pick("JUSTIN_API_KEY", "api_key"),
            model_temperature=float(_pick("JUSTIN_MODEL_TEMPERATURE", "model_temperature", "0.3")),
            model_top_p=float(_pick("JUSTIN_MODEL_TOP_P", "model_top_p", "0.95")),
            model_max_tokens=int(_pick("JUSTIN_MODEL_MAX_TOKENS", "model_max_tokens", "1024")),
            model_timeout_seconds=int(_pick("JUSTIN_MODEL_TIMEOUT_SECONDS", "model_timeout_seconds", "60")),
            model_retry_max_tokens=int(
                _pick("JUSTIN_MODEL_RETRY_MAX_TOKENS", "model_retry_max_tokens", "8192")
            ),
            retrieval_top_k=int(_pick("JUSTIN_RETRIEVAL_TOP_K", "retrieval_top_k", "5")),
            context_messages=int(_pick("JUSTIN_CONTEXT_MESSAGES", "context_messages", "8")),
            embedding_dimensions=int(_pick("JUSTIN_EMBEDDING_DIMENSIONS", "embedding_dimensions", "96")),
            context_max_input_tokens=int(
                _pick("JUSTIN_CONTEXT_MAX_INPUT_TOKENS", "context_max_input_tokens", "3200")
            ),
            context_reserved_output_tokens=int(
                _pick("JUSTIN_CONTEXT_RESERVED_OUTPUT_TOKENS", "context_reserved_output_tokens", "800")
            ),
            context_summary_trigger_messages=int(
                _pick("JUSTIN_CONTEXT_SUMMARY_TRIGGER_MESSAGES", "context_summary_trigger_messages", "10")
            ),
            context_tool_fact_limit=int(
                _pick("JUSTIN_CONTEXT_TOOL_FACT_LIMIT", "context_tool_fact_limit", "4")
            ),
            context_policy=_pick("JUSTIN_CONTEXT_POLICY", "context_policy", "balanced"),
            search_provider_order=_pick("JUSTIN_SEARCH_PROVIDER_ORDER", "search_provider_order", "duckduckgo,wikipedia"),
            search_top_k=int(_pick("JUSTIN_SEARCH_TOP_K", "search_top_k", "5")),
            search_cache_ttl_hours=int(_pick("JUSTIN_SEARCH_CACHE_TTL_HOURS", "search_cache_ttl_hours", "12")),
            search_locale=_pick("JUSTIN_SEARCH_LOCALE", "search_locale", "en-US"),
            network_tools_enabled=_pick("JUSTIN_NETWORK_TOOLS_ENABLED", "network_tools_enabled", "true")
            not in {"0", "false", "False"},
            shell_allowed_programs=_pick("JUSTIN_SHELL_ALLOWED_PROGRAMS", "shell_allowed_programs", "git,rg,where"),
            tool_max_output_chars=int(_pick("JUSTIN_TOOL_MAX_OUTPUT_CHARS", "tool_max_output_chars", "4000")),
            wechat_enabled=str(_pick("JUSTIN_WECHAT_ENABLED", "wechat_enabled", "false")).lower() in {"1", "true", "yes"},
            wechat_account_id=_pick_any(("JUSTIN_WECHAT_ACCOUNT_ID", "WEIXIN_ACCOUNT_ID"), ("wechat_account_id",), ""),
            wechat_token=_pick_any(("JUSTIN_WECHAT_TOKEN", "WEIXIN_TOKEN"), ("wechat_token",), ""),
            wechat_base_url=_pick_any(("JUSTIN_WECHAT_BASE_URL", "WEIXIN_BASE_URL"), ("wechat_base_url",), WECHAT_DEFAULT_BASE_URL),
            wechat_auto_reply_prefix=_pick("JUSTIN_WECHAT_AUTO_REPLY_PREFIX", "wechat_auto_reply_prefix", "[Justin] "),
            wechat_dm_policy=dm_policy,
            wechat_allowed_users=_pick_any(
                ("JUSTIN_WECHAT_ALLOWED_USERS", "WEIXIN_ALLOWED_USERS"),
                ("wechat_allowed_users",),
                "",
            ),
            wechat_group_policy=group_policy,
            wechat_group_allowed_users=_pick_any(
                ("JUSTIN_WECHAT_GROUP_ALLOWED_USERS", "WEIXIN_GROUP_ALLOWED_USERS"),
                ("wechat_group_allowed_users",),
                "",
            ),
            agent_name=_pick("JUSTIN_AGENT_NAME", "agent_name", "Justin"),
            system_prompt_prefix=_pick("JUSTIN_SYSTEM_PROMPT_PREFIX", "system_prompt_prefix", "You are {agent_name}, a practical local-first agent."),
        )

    def ensure_directories(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.wechat_accounts_dir().mkdir(parents=True, exist_ok=True)

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
            "model_temperature": self.model_temperature,
            "model_top_p": self.model_top_p,
            "model_max_tokens": self.model_max_tokens,
            "model_timeout_seconds": self.model_timeout_seconds,
            "model_retry_max_tokens": self.model_retry_max_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "context_messages": self.context_messages,
            "embedding_dimensions": self.embedding_dimensions,
            "skills_dir": str(self.skills_dir),
            "context_max_input_tokens": self.context_max_input_tokens,
            "context_reserved_output_tokens": self.context_reserved_output_tokens,
            "context_summary_trigger_messages": self.context_summary_trigger_messages,
            "context_tool_fact_limit": self.context_tool_fact_limit,
            "context_policy": self.context_policy,
            "search_provider_order": self.search_provider_order,
            "search_top_k": self.search_top_k,
            "search_cache_ttl_hours": self.search_cache_ttl_hours,
            "search_locale": self.search_locale,
            "network_tools_enabled": self.network_tools_enabled,
            "shell_allowed_programs": self.shell_allowed_programs,
            "tool_max_output_chars": self.tool_max_output_chars,
            "wechat_enabled": self.wechat_enabled,
            "wechat_account_id": self.wechat_account_id,
            "wechat_token": self.wechat_token,
            "wechat_base_url": self.wechat_base_url,
            "wechat_auto_reply_prefix": self.wechat_auto_reply_prefix,
            "wechat_dm_policy": self.wechat_dm_policy,
            "wechat_allowed_users": self.wechat_allowed_users,
            "wechat_group_policy": self.wechat_group_policy,
            "wechat_group_allowed_users": self.wechat_group_allowed_users,
            "agent_name": self.agent_name,
            "system_prompt_prefix": self.system_prompt_prefix,
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
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return {}

    def gateway_state_dir(self) -> Path:
        return self.home_dir / "gateway"

    def wechat_state_dir(self) -> Path:
        return self.gateway_state_dir() / "wechat"

    def wechat_accounts_dir(self) -> Path:
        return self.wechat_state_dir() / "accounts"

    def wechat_active_account_path(self) -> Path:
        return self.wechat_state_dir() / "active_account.json"

    def parse_wechat_allowed_users(self) -> list[str]:
        seen: set[str] = set()
        users: list[str] = []
        raw_values = (self.wechat_allowed_users or "").replace("\n", ",").split(",")
        for item in raw_values:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                users.append(normalized)
        return users

    def parse_wechat_group_allowed_users(self) -> list[str]:
        seen: set[str] = set()
        groups: list[str] = []
        raw_values = (self.wechat_group_allowed_users or "").replace("\n", ",").split(",")
        for item in raw_values:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                groups.append(normalized)
        return groups
