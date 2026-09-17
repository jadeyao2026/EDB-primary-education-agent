"""Polite fetching: robots.txt, conditional GET, on-disk cache, one request at a time."""

from __future__ import annotations

import hashlib
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urlparse, urlunparse

import httpx

from .config import ALLOWED_HOSTS, Config


@dataclass
class FetchResult:
    url: str
    status: str  # ok | not_modified | cached | blocked | offline | error
    html: str = ""
    etag: str | None = None
    last_modified: str | None = None
    detail: str = ""

    @property
    def has_html(self) -> bool:
        return bool(self.html)


def canonical_url(url: str) -> str:
    """Drop the fragment and normalise host casing so one page maps to one row."""
    url, _ = urldefrag(url.strip())
    parts = urlparse(url)
    netloc = parts.netloc.lower()
    if netloc.endswith(":443"):
        netloc = netloc[:-4]
    if netloc.endswith(":80"):
        netloc = netloc[:-3]
    scheme = parts.scheme.lower() or "https"
    # The site links itself over both http and https; collapse to one so a page maps to one row.
    if scheme == "http" and netloc in ALLOWED_HOSTS:
        scheme = "https"
    return urlunparse(parts._replace(netloc=netloc, scheme=scheme))


def is_allowed_host(url: str) -> bool:
    return urlparse(url).netloc.lower() in ALLOWED_HOSTS


class PoliteFetcher:
    """Single-threaded, rate-limited, cache-first HTTP client for edb.gov.hk."""

    def __init__(self, cfg: Config, cache_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir or (cfg.db_path.parent / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            headers={
                "User-Agent": cfg.user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-HK,zh-TW,zh;q=0.8",
            },
            timeout=cfg.request_timeout,
            follow_redirects=True,
            trust_env=cfg.trust_env_proxy,
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request = 0.0
        self.warnings: list[str] = []

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PoliteFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- politeness ---

    def _sleep_if_needed(self, extra_delay: float = 0.0) -> None:
        delay = max(self.cfg.crawl_delay, extra_delay)
        elapsed = time.monotonic() - self._last_request
        if self._last_request and elapsed < delay:
            time.sleep(delay - elapsed)

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        host = urlparse(url).netloc.lower()
        if host in self._robots:
            return self._robots[host]
        parser = urllib.robotparser.RobotFileParser()
        try:
            self._sleep_if_needed()
            resp = self._client.get(f"https://{host}/robots.txt")
            self._last_request = time.monotonic()
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
            else:
                parser.parse([])
        except httpx.HTTPError as exc:
            # Cannot read the policy. We allow but record it, and the README says so.
            self.warnings.append(f"robots.txt unreachable for {host}: {exc}")
            parser.parse([])
        self._robots[host] = parser
        return parser

    def _may_fetch(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(self.cfg.user_agent, url)

    # --- cache ---

    def _cache_file(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.html"

    def _read_cache(self, url: str, respect_ttl: bool = True) -> str | None:
        path = self._cache_file(url)
        if not path.exists():
            return None
        if respect_ttl:
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours > self.cfg.cache_ttl_hours:
                return None
        return path.read_text(encoding="utf-8", errors="replace")

    def _write_cache(self, url: str, html: str) -> None:
        self._cache_file(url).write_text(html, encoding="utf-8")

    # --- fetch ---

    def fetch(
        self,
        url: str,
        etag: str | None = None,
        last_modified: str | None = None,
        force: bool = False,
    ) -> FetchResult:
        url = canonical_url(url)
        if not is_allowed_host(url):
            return FetchResult(url, "blocked", detail="host not in allowlist")

        if not force:
            cached = self._read_cache(url)
            if cached is not None:
                return FetchResult(url, "cached", html=cached, etag=etag, last_modified=last_modified)

        if self.cfg.offline:
            stale = self._read_cache(url, respect_ttl=False)
            if stale is not None:
                return FetchResult(url, "cached", html=stale, detail="offline mode, stale cache")
            return FetchResult(url, "offline", detail="offline mode and nothing cached")

        if not self._may_fetch(url):
            return FetchResult(url, "blocked", detail="disallowed by robots.txt")

        # force means "really read the bytes", so conditional headers are dropped too:
        # a 304 would otherwise hide an edited local snapshot from the change check.
        headers: dict[str, str] = {}
        if not force:
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        robots_delay = self._robots_for(url).crawl_delay(self.cfg.user_agent)
        self._sleep_if_needed(float(robots_delay or 0.0))
        try:
            resp = self._client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            return FetchResult(url, "error", detail=str(exc))
        finally:
            self._last_request = time.monotonic()

        if resp.status_code == 304:
            stale = self._read_cache(url, respect_ttl=False)
            return FetchResult(
                url, "not_modified", html=stale or "", etag=etag, last_modified=last_modified
            )
        if resp.status_code >= 400:
            return FetchResult(url, "error", detail=f"HTTP {resp.status_code}")

        html = resp.text
        self._write_cache(url, html)
        return FetchResult(
            url,
            "ok",
            html=html,
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
        )
