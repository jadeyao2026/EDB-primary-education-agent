"""Tool schemas and handlers the model can invoke.

Every handler returns a JSON string. The dispatcher records each call into `tool_trace`, which is
what the Streamlit trace panel and `cli.py trace` read back, so a tool call is provable rather
than merely asserted.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable

from . import notify
from .config import Config
from .db import get_chunk, list_pages, latest_snapshot
from .retrieval import search

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_edb_docs",
        "description": (
            "在已抓取的教育局小學教育頁面中搜尋相關段落。這是回答問題的唯一資料來源。"
            "回傳每個段落的 chunk_id、章節標題、來源網址與內文。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋關鍵詞或問題，可用繁體或簡體中文"},
                "top_k": {"type": "integer", "description": "回傳段落數，預設 5，最多 10"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_section",
        "description": "依 chunk_id 取回某個段落的完整內文，用於確認引用或需要更多上下文時。",
        "input_schema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "integer"}},
            "required": ["chunk_id"],
        },
    },
    {
        "name": "list_watched_pages",
        "description": "列出目前被監控的教育局頁面、標題、最後抓取時間與快照字數。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_for_updates",
        "description": (
            "立即檢查被監控頁面是否有變更（重新抓取並與本地快照比對雜湊）。"
            "回傳變更摘要。只有使用者明確要求檢查更新時才呼叫。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "只檢查單一網址，留空則檢查全部"},
                "notify": {"type": "boolean", "description": "偵測到變更時是否推送通知，預設 false"},
            },
        },
    },
    {
        "name": "diff_page",
        "description": "顯示某頁面最近一次已記錄的變更差異與說明。",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "send_notification",
        "description": (
            "把一段訊息推送到設定好的 webhook。只有使用者明確要求發送通知時才呼叫。"
            "未設定 WEBHOOK_URL 時會改為在主控台印出（dry-run）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "要推送的內容，需為完整可讀的句子"},
                "page_url": {"type": "string"},
                "page_title": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
]


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=None)


def handle_search(cfg: Config, conn: sqlite3.Connection, args: dict[str, Any]) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return _json({"error": "query is required"})
    top_k = max(1, min(int(args.get("top_k") or 5), 10))
    hits = search(cfg, conn, query, top_k=top_k)
    if not hits:
        return _json({"results": [], "note": "沒有找到相關段落。資料庫可能尚未 ingest。"})
    return _json(
        {
            "results": [
                {
                    "chunk_id": h.chunk_id,
                    "section_title": h.section_title,
                    "page_title": h.page_title,
                    "source_url": h.citation_url,
                    "text": h.text,
                    "score": h.score,
                }
                for h in hits
            ]
        }
    )


def handle_get_section(cfg: Config, conn: sqlite3.Connection, args: dict[str, Any]) -> str:
    row = get_chunk(conn, int(args.get("chunk_id", 0)))
    if row is None:
        return _json({"error": f"chunk_id {args.get('chunk_id')} not found"})
    anchor = row["anchor"] or ""
    return _json(
        {
            "chunk_id": int(row["id"]),
            "section_title": row["section_title"],
            "page_title": row["page_title"],
            "source_url": f"{row['url']}#{anchor}" if anchor else row["url"],
            "text": row["text_display"],
        }
    )


def handle_list_pages(cfg: Config, conn: sqlite3.Connection, args: dict[str, Any]) -> str:
    pages = []
    for row in list_pages(conn, watched_only=True):
        snap = latest_snapshot(conn, int(row["id"]))
        pages.append(
            {
                "url": row["url"],
                "title": row["title"],
                "last_fetched": row["last_fetched"],
                "snapshot_chars": len(snap["body_text"]) if snap else 0,
            }
        )
    return _json({"watched_pages": pages, "count": len(pages)})


def handle_check_updates(cfg: Config, conn: sqlite3.Connection, args: dict[str, Any]) -> str:
    from .watcher import check_all

    report = check_all(
        cfg,
        conn,
        send_notifications=bool(args.get("notify", False)),
        only_url=(args.get("url") or None),
    )
    return _json(
        {
            "checked": report.checked,
            "changed": report.changed,
            "notified": report.notified,
            "errors": report.errors,
            "changes": [
                {
                    "page_title": c["page_title"],
                    "page_url": c["page_url"],
                    "section_title": c["section_title"],
                    "summary": c["summary"],
                    "notify_status": c.get("notify_status"),
                }
                for c in report.changes
            ],
        }
    )


def handle_diff_page(cfg: Config, conn: sqlite3.Connection, args: dict[str, Any]) -> str:
    url = (args.get("url") or "").strip()
    row = conn.execute(
        "SELECT ce.*, p.url, p.title FROM change_events ce JOIN pages p ON p.id = ce.page_id"
        " WHERE p.url = ? ORDER BY ce.id DESC LIMIT 1",
        (url,),
    ).fetchone()
    if row is None:
        return _json({"url": url, "note": "這個頁面目前沒有已記錄的變更。"})
    return _json(
        {
            "url": row["url"],
            "page_title": row["title"],
            "detected_at": row["detected_at"],
            "summary": row["summary"],
            "diff_excerpt": row["diff_text"][:1500],
            "notify_status": row["notify_status"],
        }
    )


def handle_send_notification(cfg: Config, conn: sqlite3.Connection, args: dict[str, Any]) -> str:
    summary = (args.get("summary") or "").strip()
    if not summary:
        return _json({"error": "summary is required"})
    payload = notify.build_payload(
        page_title=args.get("page_title") or "教育局小學教育頁面",
        page_url=args.get("page_url") or "",
        summary=summary,
        detected_at="manual",
    )
    outcome = notify.send(cfg, payload)
    return _json({"status": outcome.status, "detail": outcome.detail})


HANDLERS: dict[str, Callable[[Config, sqlite3.Connection, dict[str, Any]], str]] = {
    "search_edb_docs": handle_search,
    "get_section": handle_get_section,
    "list_watched_pages": handle_list_pages,
    "check_for_updates": handle_check_updates,
    "diff_page": handle_diff_page,
    "send_notification": handle_send_notification,
}
