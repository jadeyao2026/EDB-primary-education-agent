"""Environment-driven configuration.

The model connection is three variables — key, base URL, model name — so the agent can point at
any endpoint that speaks the Anthropic Messages API, not just api.anthropic.com.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED_URL = "https://www.edb.gov.hk/tc/edu-system/primary-secondary/primary.html"

# Only these hosts are ever fetched, regardless of what links a page contains.
ALLOWED_HOSTS = frozenset({"www.edb.gov.hk", "edb.gov.hk"})

# A linked page is only followed if its path contains one of these fragments.
# Keeps the crawl on primary-education material instead of the whole site.
PATH_KEYWORDS = (
    "primary",
    "edu-system",
    "curriculum-development",
    "student-parents",
    "sch-admission",
)


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    # Empty means "use the SDK default" (https://api.anthropic.com).
    base_url: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    )
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    summary_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_SUMMARY_MODEL", "claude-haiku-4-5-20251001")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    webhook_url: str = field(default_factory=lambda: os.getenv("WEBHOOK_URL", "").strip())
    webhook_secret: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", "").strip())
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "EDB_USER_AGENT",
            "EDB-primary-education-agent/0.1 (+contact: you@example.com)",
        )
    )
    crawl_delay: float = field(default_factory=lambda: float(os.getenv("EDB_CRAWL_DELAY", "1.0")))
    cache_ttl_hours: float = field(
        default_factory=lambda: float(os.getenv("EDB_CACHE_TTL_HOURS", "24"))
    )
    max_pages: int = field(default_factory=lambda: int(os.getenv("EDB_MAX_PAGES", "25")))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("EDB_TIMEOUT", "20")))
    offline: bool = field(default_factory=lambda: _flag("EDB_OFFLINE"))
    # Off by default: httpx also reads the Windows registry proxy, so a local VPN client can
    # break edb.gov.hk with an SSL EOF even when no HTTP_PROXY is set. Turn on behind a
    # corporate proxy that is the only route out.
    trust_env_proxy: bool = field(default_factory=lambda: _flag("EDB_TRUST_ENV_PROXY"))

    @property
    def db_path(self) -> Path:
        raw = os.getenv("EDB_DB_PATH", "data/edb.db")
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def notifications_enabled(self) -> bool:
        """No webhook configured means dry-run: the message is printed instead."""
        return bool(self.webhook_url)

    @property
    def endpoint(self) -> str:
        """Where model calls actually go. For display and health checks."""
        return self.base_url or "https://api.anthropic.com (SDK default)"

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in. "
                "For a non-Anthropic endpoint, set ANTHROPIC_BASE_URL and ANTHROPIC_MODEL too."
            )
        return self.anthropic_api_key


def get_config() -> Config:
    return Config()
