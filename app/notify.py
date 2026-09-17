"""Generic webhook notifier.

The payload carries a ready-to-read `text` sentence, not raw HTML, so webhook.site, Slack
incoming webhooks and n8n all render something a human understands. With no WEBHOOK_URL set we
fall back to dry-run and print the identical message, so the project is demoable with zero setup.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config

SIGNATURE_HEADER = "X-EDB-Signature"


@dataclass
class NotifyResult:
    status: str  # sent | dry_run | failed
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"sent", "dry_run"}


def build_payload(
    *,
    page_title: str,
    page_url: str,
    summary: str,
    detected_at: str,
    section_title: str = "",
    diff_text: str = "",
    event_id: int | None = None,
) -> dict[str, Any]:
    lines = [f"教育局頁面有更新：{page_title or page_url}", "", summary.strip()]
    if section_title:
        lines.append(f"（相關章節：{section_title}）")
    lines += ["", f"來源：{page_url}", f"偵測時間：{detected_at}"]
    return {
        "text": "\n".join(lines),
        "event_id": event_id,
        "page_title": page_title,
        "page_url": page_url,
        "section_title": section_title,
        "summary": summary.strip(),
        "detected_at": detected_at,
        # Truncated: a webhook body is not the place for a full page dump.
        "diff_excerpt": diff_text[:2000],
    }


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def send(cfg: Config, payload: dict[str, Any]) -> NotifyResult:
    if not cfg.notifications_enabled:
        print("\n[dry-run webhook] WEBHOOK_URL is not set. Message that would be sent:\n")
        print(payload["text"])
        print()
        return NotifyResult("dry_run", "printed to console")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if cfg.webhook_secret:
        headers[SIGNATURE_HEADER] = _sign(cfg.webhook_secret, body)

    try:
        resp = httpx.post(cfg.webhook_url, content=body, headers=headers, timeout=15)
    except httpx.HTTPError as exc:
        return NotifyResult("failed", str(exc))
    if resp.status_code >= 400:
        return NotifyResult("failed", f"HTTP {resp.status_code}: {resp.text[:200]}")
    return NotifyResult("sent", f"HTTP {resp.status_code}")


def send_test(cfg: Config) -> NotifyResult:
    payload = build_payload(
        page_title="小學教育（測試訊息）",
        page_url="https://www.edb.gov.hk/tc/edu-system/primary-secondary/primary.html",
        summary="這是一則測試通知，用來確認 webhook 設定正確。並非真實的頁面變更。",
        detected_at="test",
        section_title="",
    )
    return send(cfg, payload)
