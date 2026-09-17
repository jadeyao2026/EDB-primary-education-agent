"""Change detection: hash compare, unified diff, plain-language summary, webhook push.

The stored hash column is never trusted as the source of truth. It is always recomputed from the
stored snapshot text, so editing `snapshots.body_text` by hand is enough to make the check fire.
That is exactly how a reviewer tests this without touching the live government page.
"""

from __future__ import annotations

import difflib
import sqlite3
from dataclasses import dataclass, field

from . import notify
from .config import Config
from .db import (
    get_page,
    init_db,
    insert_change_event,
    latest_snapshot,
    list_pages,
    recent_change_events,
    set_notify_status,
    utcnow,
)
from .fetcher import PoliteFetcher, canonical_url
from .ingest import refresh_page
from .llm import summarise_diff
from .normalize import content_hash

DIFF_CONTEXT_LINES = 2
MAX_DIFF_LINES = 120


@dataclass
class ChangeReport:
    checked: int = 0
    changes: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notified: int = 0

    @property
    def changed(self) -> int:
        return len(self.changes)

    def summary_line(self) -> str:
        base = f"checked {self.checked} page(s), {self.changed} changed, {self.notified} notified"
        return base + (f", {len(self.errors)} error(s)" if self.errors else "")


def build_diff(old_text: str, new_text: str, page_url: str) -> str:
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"{page_url} (previous snapshot)",
        tofile=f"{page_url} (current)",
        lineterm="",
        n=DIFF_CONTEXT_LINES,
    )
    lines = list(diff)
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... diff truncated, {len(lines) - MAX_DIFF_LINES} more line(s)"]
    return "\n".join(lines)


def first_changed_section(diff_text: str) -> str:
    """Best-effort section name for the notification, taken from the first changed line."""
    for line in diff_text.splitlines():
        if line.startswith("+") and line[1:].strip().startswith("## "):
            return line[1:].strip()[3:]
        if line.startswith("-") and line[1:].strip().startswith("## "):
            return line[1:].strip()[3:]
    for line in diff_text.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            text = line[1:].strip()
            if text:
                return text[:40]
    return ""


def check_page(
    cfg: Config,
    conn: sqlite3.Connection,
    fetcher: PoliteFetcher,
    url: str,
    send_notification: bool = True,
    force: bool = True,
) -> dict[str, object] | None:
    """Compare the stored snapshot against a fresh fetch. Returns a change dict or None."""
    url = canonical_url(url)
    existing = get_page(conn, url)
    previous = latest_snapshot(conn, int(existing["id"])) if existing else None
    old_text = previous["body_text"] if previous else ""
    # Recomputed, not read from the column: a hand-edited snapshot must be detectable.
    old_hash = content_hash(old_text) if previous else None

    result = refresh_page(cfg, conn, fetcher, url, force=force)
    if result.status in {"blocked", "error", "offline"}:
        raise RuntimeError(f"{url}: {result.status} {result.detail}".strip())

    page = get_page(conn, url)
    if page is None:
        return None
    current = latest_snapshot(conn, int(page["id"]))
    if current is None:
        return None
    new_text = current["body_text"]
    new_hash = content_hash(new_text)

    if old_hash is None:
        return None  # first time we have seen this page; nothing to compare against
    if old_hash == new_hash:
        return None

    diff_text = build_diff(old_text, new_text, url)
    section = first_changed_section(diff_text)
    summary = summarise_diff(cfg, diff_text, page["title"] or url, section)
    detected_at = utcnow()
    event_id = insert_change_event(
        conn, int(page["id"]), old_hash, new_hash, diff_text, summary
    )
    conn.commit()

    change: dict[str, object] = {
        "event_id": event_id,
        "page_title": page["title"] or url,
        "page_url": url,
        "section_title": section,
        "summary": summary,
        "diff_text": diff_text,
        "detected_at": detected_at,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "notify_status": "skipped",
    }

    if send_notification:
        payload = notify.build_payload(
            page_title=change["page_title"],
            page_url=url,
            summary=summary,
            detected_at=detected_at,
            section_title=section,
            diff_text=diff_text,
            event_id=event_id,
        )
        outcome = notify.send(cfg, payload)
        change["notify_status"] = outcome.status
        change["notify_detail"] = outcome.detail
        set_notify_status(conn, event_id, outcome.status)
        conn.commit()
    return change


def check_all(
    cfg: Config,
    conn: sqlite3.Connection,
    send_notifications: bool = True,
    only_url: str | None = None,
) -> ChangeReport:
    init_db(conn)
    report = ChangeReport()
    targets = [only_url] if only_url else [row["url"] for row in list_pages(conn, watched_only=True)]
    if not targets:
        report.errors.append("no pages in the database yet; run `ingest` first")
        return report

    with PoliteFetcher(cfg) as fetcher:
        for url in targets:
            report.checked += 1
            try:
                change = check_page(cfg, conn, fetcher, url, send_notification=send_notifications)
            except RuntimeError as exc:
                report.errors.append(str(exc))
                continue
            if change:
                report.changes.append(change)
                if change.get("notify_status") in {"sent", "dry_run"}:
                    report.notified += 1

    if report.changes:
        from .retrieval import index_pending

        try:
            index_pending(cfg, conn)
        except Exception as exc:
            # Detection and notification already happened; a stale vector index must not
            # turn a successful check into a failure.
            report.errors.append(f"變更已偵測並通知，但向量索引未更新（{type(exc).__name__}）。")
    return report


def history(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return recent_change_events(conn, limit)
