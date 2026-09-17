"""Crawl the seed page plus depth-1 primary-education links, snapshot and chunk them.

`refresh_page` is the single place where fetch -> normalize -> hash-compare happens, so the
daily check and the initial ingest cannot drift apart.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .config import SEED_URL, Config
from .db import (
    get_page,
    init_db,
    insert_snapshot,
    latest_snapshot,
    mark_fetched,
    replace_chunks,
    upsert_page,
)
from .fetcher import PoliteFetcher, canonical_url
from .normalize import body_text_for_hash, content_hash, extract_links, split_sections


@dataclass
class PageResult:
    url: str
    status: str  # new | changed | unchanged | blocked | error | offline
    title: str = ""
    page_id: int | None = None
    snapshot_id: int | None = None
    old_hash: str | None = None
    new_hash: str | None = None
    sections: int = 0
    detail: str = ""

    @property
    def is_change(self) -> bool:
        return self.status in {"new", "changed"}


@dataclass
class IngestReport:
    results: list[PageResult] = field(default_factory=list)
    embedded: int = 0
    warnings: list[str] = field(default_factory=list)

    def by_status(self, *statuses: str) -> list[PageResult]:
        return [r for r in self.results if r.status in statuses]

    def summary_line(self) -> str:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"{len(self.results)} page(s): {parts}; {self.embedded} chunk(s) embedded"


def refresh_page(
    cfg: Config,
    conn: sqlite3.Connection,
    fetcher: PoliteFetcher,
    url: str,
    force: bool = False,
) -> PageResult:
    url = canonical_url(url)
    existing = get_page(conn, url)
    result = fetcher.fetch(
        url,
        etag=existing["etag"] if existing else None,
        last_modified=existing["last_modified"] if existing else None,
        force=force,
    )

    if result.status == "not_modified":
        if existing:
            mark_fetched(conn, int(existing["id"]), result.etag, result.last_modified)
            conn.commit()
        return PageResult(
            url,
            "unchanged",
            page_id=int(existing["id"]) if existing else None,
            detail="server reported 304 Not Modified",
        )

    if not result.has_html:
        return PageResult(url, result.status if result.status != "ok" else "error", detail=result.detail)

    title, sections = split_sections(result.html, url)
    body = body_text_for_hash(sections)
    new_hash = content_hash(body)

    page_id = upsert_page(conn, url, title)
    previous = latest_snapshot(conn, page_id)
    # Recomputed from the stored text, never read from the content_hash column. Editing
    # snapshots.body_text by hand must be enough to make the next check fire, which is how a
    # reviewer tests change detection without touching the live government page.
    old_hash = content_hash(previous["body_text"]) if previous else None

    if previous and old_hash == new_hash:
        mark_fetched(conn, page_id, result.etag, result.last_modified)
        conn.commit()
        return PageResult(
            url, "unchanged", title=title, page_id=page_id, old_hash=old_hash,
            new_hash=new_hash, sections=len(sections),
        )

    snapshot_id = insert_snapshot(conn, page_id, new_hash, body, source=result.status)
    replace_chunks(conn, page_id, snapshot_id, sections)
    mark_fetched(conn, page_id, result.etag, result.last_modified)
    conn.commit()
    return PageResult(
        url,
        "changed" if previous else "new",
        title=title,
        page_id=page_id,
        snapshot_id=snapshot_id,
        old_hash=old_hash,
        new_hash=new_hash,
        sections=len(sections),
    )


def discover_urls(cfg: Config, fetcher: PoliteFetcher, seed: str = SEED_URL) -> list[str]:
    """Seed first, then its primary-education links. Order is stable for reproducible runs."""
    seed = canonical_url(seed)
    result = fetcher.fetch(seed)
    urls = [seed]
    if result.has_html:
        for link in extract_links(result.html, seed):
            if link not in urls:
                urls.append(link)
    return urls[: cfg.max_pages]


def ingest(
    cfg: Config,
    conn: sqlite3.Connection,
    seed: str = SEED_URL,
    force: bool = False,
    embed: bool = True,
) -> IngestReport:
    init_db(conn)
    report = IngestReport()
    with PoliteFetcher(cfg) as fetcher:
        for url in discover_urls(cfg, fetcher, seed):
            report.results.append(refresh_page(cfg, conn, fetcher, url, force=force))
        report.warnings.extend(fetcher.warnings)

    if embed:
        from .retrieval import index_pending

        try:
            report.embedded = index_pending(cfg, conn)
        except Exception as exc:
            # The model download is the one step that needs the network beyond edb.gov.hk.
            # Retrieval still answers via BM25, so this must not fail the whole ingest.
            report.warnings.append(
                f"向量索引未完成（{type(exc).__name__}）；檢索將只用 BM25。"
                " 可稍後執行 `python -m app.cli warmup` 再 ingest。"
            )
    return report
