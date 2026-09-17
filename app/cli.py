"""Command line entry point.

`check` is the cron-facing command. `edit-snapshot` exists so a reviewer can change one sentence
in the local snapshot and watch the check plus the push react, without touching the live site.
"""

from __future__ import annotations

import argparse
import sys

from .config import SEED_URL, get_config
from .db import connect, init_db


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp936 here and mangle Traditional Chinese output."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import ingest

    cfg = get_config()
    conn = connect(cfg.db_path)
    report = ingest(cfg, conn, seed=args.seed, force=args.force, embed=not args.no_embed)
    print(report.summary_line())
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for result in report.results:
        print(f"  {result.status:9s} sections={result.sections:3d}  {result.title[:24]:26s} {result.url}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .watcher import check_all

    cfg = get_config()
    conn = connect(cfg.db_path)
    report = check_all(cfg, conn, send_notifications=not args.no_notify, only_url=args.url)
    print(report.summary_line())
    for error in report.errors:
        print(f"  error: {error}")
    for change in report.changes:
        print(f"\n--- 變更：{change['page_title']}")
        print(f"    {change['page_url']}")
        if change["section_title"]:
            print(f"    章節：{change['section_title']}")
        print(f"    說明：{change['summary']}")
        print(f"    通知：{change.get('notify_status')} {change.get('notify_detail', '')}".rstrip())
    if not report.changes and not report.errors:
        print("沒有偵測到變更。")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .agent import ask

    cfg = get_config()
    conn = connect(cfg.db_path)
    result = ask(cfg, conn, args.question)

    if args.show_trace:
        print("=== 工具呼叫軌跡 ===")
        for step in result.trace:
            print(f"[step {step.step}] {step.tool_name}({step.tool_input}) {step.duration_ms}ms")
            print(f"    -> {step.tool_result[:220]}")
        print()

    print(result.answer or "(沒有取得答案)")
    print()
    print(f"[tool calls: {result.tool_calls} | grounded: {result.grounded} | "
          f"{result.latency_ms}ms | qa_id={result.qa_id}]")
    if result.usage.get("cache_read_input_tokens"):
        print(f"[prompt cache read: {result.usage['cache_read_input_tokens']} tokens]")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_pages(args: argparse.Namespace) -> int:
    from .db import latest_snapshot, list_pages

    cfg = get_config()
    conn = connect(cfg.db_path)
    init_db(conn)
    rows = list_pages(conn, watched_only=True)
    if not rows:
        print("資料庫還沒有頁面，請先執行 ingest。")
        return 1
    for row in rows:
        snapshot = latest_snapshot(conn, int(row["id"]))
        chars = len(snapshot["body_text"]) if snapshot else 0
        print(f"  [{row['id']:2d}] {row['title'][:24]:26s} chars={chars:6d}  {row['url']}")
    print(f"\n{len(rows)} watched page(s)")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from .watcher import history

    cfg = get_config()
    conn = connect(cfg.db_path)
    init_db(conn)
    rows = history(conn, args.limit)
    if not rows:
        print("目前沒有變更記錄。")
        return 0
    for row in rows:
        print(f"[{row['id']:3d}] {row['detected_at']}  {row['page_title'][:22]:24s} "
              f"notify={row['notify_status']}")
        print(f"      {row['summary'][:150]}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .db import trace_for_qa

    cfg = get_config()
    conn = connect(cfg.db_path)
    init_db(conn)
    qa_id = args.qa_id
    if qa_id is None:
        row = conn.execute("SELECT id FROM qa_log ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            print("目前沒有問答記錄。")
            return 1
        qa_id = int(row["id"])
    qa = conn.execute("SELECT * FROM qa_log WHERE id = ?", (qa_id,)).fetchone()
    if qa is None:
        print(f"找不到 qa_id={qa_id}")
        return 1
    print(f"Q [{qa_id}] {qa['question']}")
    print(f"grounded={bool(qa['grounded'])} latency={qa['latency_ms']}ms model={qa['model']}")
    for step in trace_for_qa(conn, qa_id):
        print(f"\n[step {step['step']}] {step['tool_name']}  {step['duration_ms']}ms")
        print(f"  input : {step['tool_input']}")
        print(f"  result: {step['tool_result'][:500]}")
    print(f"\nA: {qa['answer'][:800]}")
    return 0


def cmd_notify_test(args: argparse.Namespace) -> int:
    from . import notify

    cfg = get_config()
    outcome = notify.send_test(cfg)
    print(f"status={outcome.status} detail={outcome.detail}")
    return 0 if outcome.ok else 1


def cmd_health(args: argparse.Namespace) -> int:
    """Prove the key, endpoint and model names work before anything else is run."""
    from .llm import probe_model

    cfg = get_config()
    print(f"endpoint : {cfg.endpoint}")
    # Length only, never a prefix: enough to spot a truncated paste without echoing the secret.
    key_state = f"set ({len(cfg.anthropic_api_key)} chars)" if cfg.anthropic_api_key else "MISSING"
    print(f"api key  : {key_state}")
    if not cfg.anthropic_api_key:
        print("\n請把 ANTHROPIC_API_KEY 填進 .env。若使用其他平台，另外設定 ANTHROPIC_BASE_URL 與 ANTHROPIC_MODEL。")
        return 1

    # dict.fromkeys dedupes when both env vars name the same model.
    models = [args.model] if args.model else list(dict.fromkeys([cfg.model, cfg.summary_model]))

    failures = 0
    for model in models:
        result = probe_model(cfg, model)
        print(f"\n[{'OK  ' if result.ok else 'FAIL'}] {model}  {result.latency_ms}ms")
        print(f"       {result.detail}")
        failures += not result.ok

    print(f"\n{len(models) - failures}/{len(models)} model(s) reachable")
    return 1 if failures else 0


def cmd_warmup(args: argparse.Namespace) -> int:
    from .retrieval import warmup

    print(warmup(get_config()))
    return 0


def cmd_edit_snapshot(args: argparse.Namespace) -> int:
    """Simulate an upstream edit by rewriting the stored snapshot text."""
    from .db import get_page, latest_snapshot
    from .fetcher import canonical_url

    cfg = get_config()
    conn = connect(cfg.db_path)
    init_db(conn)
    url = canonical_url(args.url or SEED_URL)
    page = get_page(conn, url)
    if page is None:
        print(f"資料庫中沒有這個頁面：{url}")
        return 1
    snapshot = latest_snapshot(conn, int(page["id"]))
    if snapshot is None:
        print("這個頁面還沒有快照。")
        return 1

    body = snapshot["body_text"]
    if args.find:
        if args.find not in body:
            print(f"快照中找不到這段文字：{args.find!r}")
            return 1
        updated = body.replace(args.find, args.replace or "", 1)
    else:
        lines = body.splitlines()
        target = next((i for i, line in enumerate(lines) if len(line) > 15), None)
        if target is None:
            print("快照太短，無法自動改動。")
            return 1
        original = lines[target]
        lines[target] = (args.replace or "（本行由 edit-snapshot 改寫，用於測試變更偵測）")
        print(f"改寫第 {target + 1} 行：\n  原文：{original[:90]}\n  改為：{lines[target][:90]}")
        updated = "\n".join(lines)

    conn.execute("UPDATE snapshots SET body_text = ? WHERE id = ?", (updated, snapshot["id"]))
    conn.commit()
    print(f"已改寫快照 id={snapshot['id']}（{url}）")
    print("接著執行：python -m app.cli check   （應該偵測到變更並推送）")
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    from apscheduler.schedulers.blocking import BlockingScheduler

    from .watcher import check_all

    cfg = get_config()
    hour, _, minute = args.at.partition(":")

    def job() -> None:
        conn = connect(cfg.db_path)
        try:
            report = check_all(cfg, conn, send_notifications=True)
            print(f"[daily check] {report.summary_line()}", flush=True)
        finally:
            conn.close()

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "cron", hour=int(hour), minute=int(minute or 0))
    print(f"每日 {args.at} 執行檢查。Ctrl+C 結束。")
    if args.run_now:
        job()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n已停止。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="EDB 小學教育資訊 agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="抓取並索引教育局小學教育頁面")
    p.add_argument("--seed", default=SEED_URL)
    p.add_argument("--force", action="store_true", help="忽略快取重新抓取")
    p.add_argument("--no-embed", action="store_true", help="跳過向量化")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("check", help="檢查頁面是否有變更（cron 用這個）")
    p.add_argument("--url", default=None, help="只檢查單一網址")
    p.add_argument("--no-notify", action="store_true", help="偵測但不推送")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("ask", help="向 agent 提問")
    p.add_argument("question")
    p.add_argument("--show-trace", action="store_true", help="印出工具呼叫軌跡")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("pages", help="列出被監控的頁面")
    p.set_defaults(func=cmd_pages)

    p = sub.add_parser("history", help="列出變更記錄")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("trace", help="印出某次問答的工具軌跡")
    p.add_argument("--qa-id", type=int, default=None)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("notify-test", help="發送一則測試通知")
    p.set_defaults(func=cmd_notify_test)

    p = sub.add_parser("health", help="檢查 API key、endpoint 與模型是否可用")
    p.add_argument("--model", default=None, help="只測試指定的模型名稱")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("warmup", help="預先下載本地 embedding 模型")
    p.set_defaults(func=cmd_warmup)

    p = sub.add_parser("edit-snapshot", help="改寫本地快照，用於測試變更偵測")
    p.add_argument("--url", default=None)
    p.add_argument("--find", default=None, help="要被取代的文字")
    p.add_argument("--replace", default=None, help="取代後的文字")
    p.set_defaults(func=cmd_edit_snapshot)

    p = sub.add_parser("schedule", help="常駐執行每日檢查")
    p.add_argument("--at", default="09:07", help="每日執行時間 HH:MM，預設 09:07")
    p.add_argument("--run-now", action="store_true", help="啟動時先跑一次")
    p.set_defaults(func=cmd_schedule)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
