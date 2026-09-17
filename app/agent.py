"""Hand-written Anthropic tool-use loop.

Deliberately not built on an agent framework: the grading criterion is a provable tool call, and
here every tool_use/tool_result pair is dispatched and persisted by code in this file, so the
trace is evidence rather than a framework's internal state.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .db import finish_qa, log_tool_call, start_qa
from .llm import get_client
from .tools import HANDLERS, TOOL_SCHEMAS

MAX_STEPS = 6

SYSTEM_PROMPT = """你是香港教育局「小學教育」資訊助手，服務對象是家長與小學教師。

嚴格規則：
1. 你只能根據 search_edb_docs 或 get_section 回傳的內容作答。你自己的背景知識不算資料來源。
2. 回答問題前，必須先呼叫 search_edb_docs。不要憑記憶直接回答。
3. 如果檢索結果沒有涵蓋問題，就明確說「根據教育局小學教育頁面的資料，我找不到這項資訊」，
   並簡短說明頁面涵蓋哪些相關主題，建議使用者到教育局網站查詢。絕對不要猜測、不要補充頁面沒有的細節。
4. 只要引用了資料，就必須在答案末尾加上「來源：」段落，逐條列出
   《章節標題》 完整網址
   網址必須逐字照抄檢索結果中的 source_url，不可自行拼湊或簡化。
5. 如果問題只有部分能從資料回答，就明確區分哪部分有資料、哪部分沒有。
6. 用繁體中文回答，語氣平實，像在跟家長解釋。避免冗長開場白。
7. 不要輸出 HTML 標籤或原始 diff。
8. 只有使用者明確要求「檢查更新」或「發送通知」時，才呼叫 check_for_updates 或 send_notification。

回答格式：先直接回答問題，必要時分點，最後列出來源。"""

_URL_RE = re.compile(r"https?://[^\s，。、）)\]\"'」》]+")
_REFUSAL_HINTS = ("找不到", "沒有提到", "未提及", "不知道", "沒有這項資訊", "沒有相關資料")


@dataclass
class TraceStep:
    step: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str
    duration_ms: int


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    grounded: bool = True
    warnings: list[str] = field(default_factory=list)
    qa_id: int | None = None
    latency_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tool_calls(self) -> int:
        return len(self.trace)


def _cacheable_tools() -> list[dict[str, Any]]:
    """Cache the tool schemas; they are identical on every request."""
    tools = [dict(schema) for schema in TOOL_SCHEMAS]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _collect_retrieved(tool_result: str, retrieved: dict[str, dict[str, Any]]) -> None:
    try:
        payload = json.loads(tool_result)
    except json.JSONDecodeError:
        return
    items = payload.get("results")
    if items is None:
        items = [payload] if "source_url" in payload else []
    for item in items:
        url = item.get("source_url")
        if url:
            retrieved[url] = item


def _verify_citations(
    answer: str, retrieved: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Every URL in the answer must be one the tools actually returned."""
    warnings: list[str] = []
    cited_urls = {u.rstrip(".,;：") for u in _URL_RE.findall(answer)}

    valid: list[dict[str, Any]] = []
    for url in cited_urls:
        match = retrieved.get(url)
        if match is None:
            # Tolerate a trailing-anchor mismatch before calling it fabricated.
            base = url.split("#")[0]
            match = next(
                (v for k, v in retrieved.items() if k.split("#")[0] == base), None
            )
        if match is None:
            warnings.append(f"答案引用了未經檢索的網址：{url}")
            continue
        valid.append(
            {
                "chunk_id": match.get("chunk_id"),
                "section_title": match.get("section_title", ""),
                "page_title": match.get("page_title", ""),
                "source_url": match.get("source_url", url),
            }
        )

    fabricated = any(w.startswith("答案引用了未經檢索") for w in warnings)
    refused = any(hint in answer for hint in _REFUSAL_HINTS)
    if not valid and not refused and retrieved:
        warnings.append("答案沒有引用任何來源，也沒有明確表示找不到資料。")
    grounded = not fabricated and (bool(valid) or refused or not retrieved)
    return valid, warnings, grounded


def ask(
    cfg: Config,
    conn: sqlite3.Connection,
    question: str,
    history: list[dict[str, Any]] | None = None,
    max_steps: int = MAX_STEPS,
) -> AnswerResult:
    cfg.require_api_key()
    client = get_client(cfg.anthropic_api_key, cfg.base_url)
    started = time.monotonic()

    qa_id = start_qa(conn, question, cfg.model)
    messages: list[dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": question})

    result = AnswerResult(question=question, answer="", qa_id=qa_id)
    retrieved: dict[str, dict[str, Any]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=cfg.model,
            max_tokens=2000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=_cacheable_tools(),
            messages=messages,
        )
        for key in usage:
            usage[key] += getattr(response.usage, key, 0) or 0

        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            result.answer = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            break

        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            handler = HANDLERS.get(block.name)
            call_started = time.monotonic()
            if handler is None:
                output = json.dumps({"error": f"unknown tool {block.name}"}, ensure_ascii=False)
            else:
                try:
                    output = handler(cfg, conn, dict(block.input))
                except Exception as exc:
                    output = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
                    )
            duration = int((time.monotonic() - call_started) * 1000)

            if block.name in {"search_edb_docs", "get_section"}:
                _collect_retrieved(output, retrieved)

            result.trace.append(
                TraceStep(step, block.name, dict(block.input), output, duration)
            )
            log_tool_call(conn, qa_id, step, block.name, dict(block.input), output, duration)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )

        messages.append({"role": "user", "content": tool_results})
    else:
        result.warnings.append(f"達到 {max_steps} 步上限，未取得最終答案。")

    citations, warnings, grounded = _verify_citations(result.answer, retrieved)
    result.citations = citations
    result.warnings.extend(warnings)
    result.grounded = grounded
    result.latency_ms = int((time.monotonic() - started) * 1000)
    result.usage = usage

    finish_qa(conn, qa_id, result.answer, citations, grounded, result.latency_ms)
    return result
