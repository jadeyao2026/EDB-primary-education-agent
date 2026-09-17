"""Streamlit UI: chat with citations, the tool-call trace, and a refresh button.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import streamlit as st

from app.config import get_config
from app.db import connect, init_db, latest_snapshot, list_pages
from app.watcher import history

st.set_page_config(page_title="EDB 小學教育資訊助手", page_icon="📘", layout="wide")


@st.cache_resource
def _bootstrap():
    cfg = get_config()
    conn = connect(cfg.db_path)
    init_db(conn)
    return cfg, conn


cfg, conn = _bootstrap()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "results" not in st.session_state:
    st.session_state.results = {}

st.title("📘 教育局小學教育資訊助手")
st.caption(
    "答案只來自已抓取的教育局小學教育頁面。頁面沒寫的，助手會直接說找不到。"
)

with st.sidebar:
    st.header("監控狀態")
    pages = list_pages(conn, watched_only=True)
    st.metric("監控頁面", len(pages))
    st.metric("已索引段落", conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
    st.caption(
        f"通知：{'webhook 已設定' if cfg.notifications_enabled else 'dry-run（未設定 WEBHOOK_URL）'}"
    )

    if st.button("🔄 立即檢查更新", use_container_width=True):
        from app.watcher import check_all

        with st.spinner("重新抓取並比對雜湊…"):
            report = check_all(cfg, conn, send_notifications=True)
        st.session_state["last_check"] = report
        st.rerun()

    report = st.session_state.get("last_check")
    if report is not None:
        if report.changes:
            st.success(f"偵測到 {report.changed} 項變更，已推送 {report.notified} 則")
            for change in report.changes:
                st.write(f"**{change['page_title']}**")
                st.write(change["summary"])
                st.caption(f"通知狀態：{change.get('notify_status')}")
        else:
            st.info(report.summary_line())
        for error in report.errors:
            st.warning(error)

    with st.expander("被監控的頁面"):
        for row in pages:
            snap = latest_snapshot(conn, int(row["id"]))
            chars = len(snap["body_text"]) if snap else 0
            st.markdown(f"- [{row['title'] or row['url']}]({row['url']}) · {chars} 字")

    with st.expander("變更記錄"):
        events = history(conn, 10)
        if not events:
            st.caption("目前沒有變更記錄。")
        for event in events:
            st.markdown(f"**{event['detected_at'][:16]}** · {event['page_title']}")
            st.caption(event["summary"][:300])

if not cfg.anthropic_api_key:
    st.error(
        "尚未設定 ANTHROPIC_API_KEY。複製 .env.example 為 .env 並填入金鑰後重新啟動。"
        "（左側的更新檢查與通知不需要金鑰，可以直接使用。）"
    )

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        result = st.session_state.results.get(index)
        if result is None:
            continue

        if result["citations"]:
            st.caption("來源")
            for citation in result["citations"]:
                title = citation.get("section_title") or citation.get("page_title") or "來源"
                st.caption(f"· 《{title}》 {citation['source_url']}")

        for warning in result["warnings"]:
            st.warning(warning)

        with st.expander(
            f"🔧 工具呼叫軌跡（{len(result['trace'])} 次呼叫 · {result['latency_ms']}ms · "
            f"grounded={result['grounded']}）"
        ):
            for step in result["trace"]:
                st.markdown(f"**step {step['step']} · `{step['tool_name']}`** · {step['duration_ms']}ms")
                st.code(json.dumps(step["tool_input"], ensure_ascii=False, indent=2), language="json")
                st.text(step["tool_result"][:1500])
            if result.get("cache_read"):
                st.caption(f"prompt cache 讀取：{result['cache_read']} tokens")

question = st.chat_input("例如：小學全日制有什麼好處？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    if not cfg.anthropic_api_key:
        st.session_state.messages.append(
            {"role": "assistant", "content": "需要 ANTHROPIC_API_KEY 才能回答問題。"}
        )
        st.rerun()

    from app.agent import ask

    with st.spinner("檢索教育局頁面…"):
        try:
            result = ask(cfg, conn, question)
        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"出錯了：{type(exc).__name__}: {exc}"}
            )
            st.rerun()

    index = len(st.session_state.messages)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer or "(沒有取得答案)"}
    )
    st.session_state.results[index] = {
        "citations": result.citations,
        "warnings": result.warnings,
        "grounded": result.grounded,
        "latency_ms": result.latency_ms,
        "cache_read": result.usage.get("cache_read_input_tokens", 0),
        "trace": [
            {
                "step": step.step,
                "tool_name": step.tool_name,
                "tool_input": step.tool_input,
                "tool_result": step.tool_result,
                "duration_ms": step.duration_ms,
            }
            for step in result.trace
        ],
    }
    st.rerun()
