"""Anthropic client plus the diff-to-plain-language summariser.

Change detection deliberately works without an API key: `mechanical_summary` produces a readable
sentence from the diff alone, so the daily check and the webhook still function if the key is
missing or the model call fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .config import Config

SUMMARY_SYSTEM = (
    "你是香港教育局網頁的變更說明助手。使用者會給你一段 unified diff。"
    "請用繁體中文、1 至 3 句平實的話說明這次頁面改了什麼，讓家長或老師看得懂。"
    "只描述 diff 中真實出現的改動，不要推測原因或影響，不要輸出 HTML、標記或 diff 符號。"
    "如果只是排版或次序調整，就直接說明是排版調整。"
)


@lru_cache(maxsize=4)
def get_client(api_key: str, base_url: str = ""):
    # Keyed on both so switching endpoint mid-process gets a fresh client.
    from anthropic import Anthropic

    return Anthropic(api_key=api_key, base_url=base_url or None)


@dataclass
class ProbeResult:
    model: str
    ok: bool
    latency_ms: int
    detail: str


def probe_model(cfg: Config, model: str) -> ProbeResult:
    """One minimal round trip, to prove the key, endpoint and model name all work together."""
    started = time.monotonic()
    try:
        resp = get_client(cfg.require_api_key(), cfg.base_url).messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return ProbeResult(model, False, elapsed, f"{type(exc).__name__}: {str(exc)[:300]}")

    elapsed = int((time.monotonic() - started) * 1000)
    reply = "".join(b.text for b in resp.content if b.type == "text").strip()
    served = getattr(resp, "model", model) or model
    detail = f"replied {reply[:40]!r}, served by {served}"
    if served != model:
        # Gateways often silently route to whatever they have; worth surfacing, not failing.
        detail += " (endpoint substituted the model)"
    return ProbeResult(model, True, elapsed, detail)


def _diff_stats(diff_text: str) -> tuple[list[str], list[str]]:
    added, removed = [], []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") and line[1:].strip():
            added.append(line[1:].strip())
        elif line.startswith("-") and line[1:].strip():
            removed.append(line[1:].strip())
    return added, removed


def mechanical_summary(diff_text: str, page_title: str = "") -> str:
    """Readable summary built without the model. Used as fallback and in tests."""
    added, removed = _diff_stats(diff_text)
    if not added and not removed:
        return "偵測到頁面內容有改動，但無法解析具體差異。"

    parts: list[str] = []
    if removed and added:
        parts.append(f"有 {len(removed)} 處內容被刪除或改寫，{len(added)} 處內容為新增或改後版本。")
    elif added:
        parts.append(f"新增了 {len(added)} 處內容。")
    else:
        parts.append(f"刪除了 {len(removed)} 處內容。")

    if removed:
        parts.append(f"原文例如：「{removed[0][:80]}」")
    if added:
        parts.append(f"現在例如：「{added[0][:80]}」")
    return " ".join(parts)


def summarise_diff(cfg: Config, diff_text: str, page_title: str, section_hint: str = "") -> str:
    """Ask Haiku to turn a diff into plain language; degrade to the mechanical summary."""
    if not cfg.anthropic_api_key or not diff_text.strip():
        return mechanical_summary(diff_text, page_title)

    prompt = f"頁面標題：{page_title}\n"
    if section_hint:
        prompt += f"可能相關的章節：{section_hint}\n"
    prompt += f"\nunified diff：\n```\n{diff_text[:6000]}\n```"

    try:
        resp = get_client(cfg.anthropic_api_key, cfg.base_url).messages.create(
            model=cfg.summary_model,
            max_tokens=400,
            system=[{"type": "text", "text": SUMMARY_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # network, auth, rate limit
        return f"{mechanical_summary(diff_text, page_title)}（自動摘要不可用：{type(exc).__name__}）"

    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    return text or mechanical_summary(diff_text, page_title)
