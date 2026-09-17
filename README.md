# EDB 小學教育 Agent

A grounded Q&A agent over the Hong Kong Education Bureau primary-education pages, with change
detection and a webhook push.

Source of truth:
[小學教育](https://www.edb.gov.hk/tc/edu-system/primary-secondary/primary.html) plus the 11
primary-education pages it links.

- Answers come only from fetched pages. If the pages do not say, the agent says it does not know.
- Every answer carries a citation (section title + URL), and cited URLs are verified against what
  the tools actually returned — a fabricated URL is flagged, not shown as a source.
- The model calls real tools. Every `tool_use` / `tool_result` pair is persisted to SQLite and
  viewable in the UI or via `python -m app.cli trace`.
- A daily check (or the Refresh button) hashes each page's cleaned body text, diffs any change,
  turns it into plain language, and POSTs it to a webhook.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # then put your ANTHROPIC_API_KEY in it
python -m app.cli health        # confirm the key, endpoint and models before anything else

python -m app.cli warmup        # pre-download the local embedding model (~45MB, one time)
python -m app.cli ingest        # crawl + index (a prebuilt data/edb.db already ships with this repo)
streamlit run streamlit_app.py  # UI at http://localhost:8501
```

Ask from the CLI instead:

```bash
python -m app.cli ask "小學全日制有什麼好處？" --show-trace
python -m app.cli ask "幼稚園學券的金額是多少？"   # expect an explicit "not found"
```

If `warmup` is slow or blocked, ingest still works — retrieval falls back to BM25 keyword search
and prints a warning. In mainland China, `HF_ENDPOINT=https://hf-mirror.com` makes the model
download usable.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **yes** | — | The only third-party key needed. |
| `ANTHROPIC_BASE_URL` | no | *(empty)* | Endpoint for model calls. Empty means `https://api.anthropic.com`. Set it to use a gateway or another platform. |
| `WEBHOOK_URL` | no | *(empty)* | Where change alerts are POSTed. Empty means dry-run: the identical message prints to the console. |
| `WEBHOOK_SECRET` | no | *(empty)* | If set, requests carry `X-EDB-Signature: sha256=<hmac>`. |
| `EDB_DB_PATH` | no | `data/edb.db` | SQLite file. |
| `EDB_USER_AGENT` | no | project UA | Sent to edb.gov.hk. Put a real contact address here. |
| `EMBEDDING_MODEL` | no | `BAAI/bge-small-zh-v1.5` | Local embedding model, runs on CPU via onnxruntime. |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-4-6` | Main agent loop. |
| `ANTHROPIC_SUMMARY_MODEL` | no | `claude-haiku-4-5-20251001` | Diff-to-plain-language summaries. |
| `EDB_CRAWL_DELAY` | no | `1.0` | Seconds between requests. |
| `EDB_CACHE_TTL_HOURS` | no | `24` | How long a cached page counts as fresh. |
| `EDB_OFFLINE` | no | `false` | Answer from cache only, make no network requests. |
| `EDB_TRUST_ENV_PROXY` | no | `false` | Let the fetcher use the system proxy. Off by default — see below. |

`WEBHOOK_URL` is deliberately optional so the project is demoable with zero setup.
Get a throwaway endpoint at [webhook.site](https://webhook.site) to see a real POST.

### If every page fails with an SSL error

```
checked 12 page(s), 0 changed, 0 notified, 12 error(s)
  error: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
```

That is a local proxy, not the government site. On Windows, httpx resolves proxies from the
registry (`Internet Settings\ProxyServer`) as well as `HTTP_PROXY`/`HTTPS_PROXY`, so a running VPN
client routes the crawl through a tunnel that edb.gov.hk connections die in — with no proxy
environment variable anywhere to hint at it. The tell is that `curl https://www.edb.gov.hk/...`
succeeds while the fetcher does not.

So the fetcher passes `trust_env=False` by default and reaches the site directly.
`EDB_TRUST_ENV_PROXY=true` restores the old behaviour for a deployment where a corporate proxy is
the only route out. Note this covers page fetching only; the Anthropic SDK keeps its own proxy
handling, which is usually what you want, since the model endpoint and the HK government site
rarely want the same route.

## Pointing at a different endpoint or model

The model connection is three independent variables, so nothing about the agent assumes
`api.anthropic.com`. Any endpoint that speaks the Anthropic Messages API works — a corporate
gateway, a self-hosted proxy, or a third-party platform that exposes an Anthropic-compatible route.

```bash
ANTHROPIC_API_KEY=<that platform's key>
ANTHROPIC_BASE_URL=https://your-gateway.example.com
ANTHROPIC_MODEL=<a model that endpoint actually serves>
ANTHROPIC_SUMMARY_MODEL=<usually a cheaper/faster one>
```

`ANTHROPIC_BASE_URL` is named after the variable the Anthropic SDK already honours, so the two
never disagree. Empty means "use the SDK default", passed through as `base_url=None` rather than an
empty string. `Config.endpoint` (`app/config.py`) resolves the display form, and `get_client()`
(`app/llm.py`) is `lru_cache`d on **both** key and base URL, so changing endpoint mid-process gets a
fresh client instead of a stale one.

Model names are not validated against a list — a hardcoded allow-list would be the thing that
breaks first on a platform with its own naming. If the name is wrong, `health` tells you.

## Health check

```bash
python -m app.cli health                    # probe both configured models
python -m app.cli health --model <name>     # probe one specific model name
```

It prints the resolved endpoint and the key's length (length only — never a prefix, so the output is
safe to paste into a bug report), then sends one 16-token round trip per model. That single call is
what makes it useful: it proves the key, the endpoint and the model name work *together*, which is
the combination that actually fails. Exit code is non-zero if any probe fails, so it works as a CI or
container preflight step.

```
endpoint : https://api.anthropic.com (SDK default)
api key  : set (108 chars)

[OK  ] claude-sonnet-4-6  742ms
       replied 'ok', served by claude-sonnet-4-6

2/2 model(s) reachable
```

Two failure modes worth knowing:

- **`AuthenticationError: 401 Invalid token`** — the key is wrong *for that endpoint*. A gateway will
  not accept an `sk-ant-` key and vice versa. Check the key length in the output first; a 3-character
  key means the `.env` placeholder was never replaced.
- **`(endpoint substituted the model)`** — the probe succeeded but `response.model` came back
  different from what was requested. Gateways silently route to whatever they have. This is reported,
  not treated as a failure, because the agent will still work — just not on the model you named.

`probe_model` deliberately catches every exception and folds it into `ProbeResult.detail` instead of
raising, so one dead model does not hide the status of the others.

## Triggering the check

```bash
python -m app.cli check                 # check every watched page, push on change
python -m app.cli check --no-notify     # detect only
python -m app.cli check --url <page>    # single page
python -m app.cli notify-test           # verify webhook wiring
python -m app.cli history               # past change events
```

The Streamlit sidebar has a **🔄 立即檢查更新** button that does the same thing.

Schedule it either way:

```bash
python -m app.cli schedule --at 09:07 --run-now    # in-process APScheduler daemon
```

```cron
7 9 * * *  cd /path/to/repo && .venv/bin/python -m app.cli check >> logs/check.log 2>&1
```

On Windows, Task Scheduler running
`D:\...\.venv\Scripts\python.exe -m app.cli check` daily achieves the same.

## Verifying change detection without touching the government site

The stored hash column is never trusted — the comparison hash is always recomputed from the stored
snapshot text. So editing a snapshot by hand is enough to make the check fire:

```bash
python -m app.cli edit-snapshot \
  --url "https://www.edb.gov.hk/tc/edu-system/primary-secondary/applicable-to-primary/whole-day-schooling/index.html" \
  --find "好處" --replace "優點"
python -m app.cli check
```

Expected output — a plain-language message naming the section and quoting before/after, followed by
`checked 1 page(s), 1 changed, 1 notified`. Omit `--find`/`--replace` to have it rewrite one line
automatically.

## Architecture

```
edb.gov.hk ──robots.txt, ETag/If-Modified-Since, 1s delay, 24h cache──┐
                                                                      ▼
   pick content container → strip nav/menu → split on h1 + pseudo-headings → 1200-char cap
                                                                      ▼
   SQLite: pages | snapshots | chunks | embeddings | change_events | qa_log | tool_trace
                          │                                    │
        hybrid retrieval (fastembed + jieba BM25, RRF)     watcher: recomputed SHA-256
                          │                                → difflib → Haiku summary
                          ▼                                          │
   agent loop (Anthropic tool use)                                    ▼
   search_edb_docs · get_section · list_watched_pages           webhook (HMAC optional)
   check_for_updates · diff_page · send_notification            or dry-run to console
                          │
              Streamlit UI  ·  CLI (cron target)
```

Design choices worth naming:

- **No agent framework.** The loop is ~60 lines in `app/agent.py`. The grading criterion is a
  provable tool call, and a hand-written loop makes the trace first-party evidence.
- **No vector database.** 29 chunks; a numpy dot product beats standing up infrastructure.
- **Traditional for display, Simplified for indexing.** `bge-small-zh` and jieba are trained on
  simplified text, so chunks are indexed via OpenCC t2s while citations quote the page's own
  traditional wording.
- **Content container selected before chrome is stripped.** The EDB template puts the entire
  mega-menu inside `<header>`; stripping first empties the page.

## Tests

```bash
python -m pytest tests/test_pipeline.py -q   # 15 tests, no API key or network needed
python -m pytest tests/test_evals.py -v -s   # 17 eval cases, needs a key; skips without one
```

`evals/questions.yaml` covers the three probes named in the task: questions the pages answer,
questions they do not, and edge cases (ambiguous, partially covered, prompt injection).

## AI usage note

[`AI_USAGE.md`](AI_USAGE.md) — what I built, what the model drafted, the suggestion I rejected
(`sentence-transformers` for `fastembed`) and why, and the one thing I would do next.

## Limits, honestly

Not production-ready. What I know is weak:

- **Coverage.** Depth-1 crawl from the seed page, 12 pages, 29 chunks. PDF attachments and
  JS-rendered content are skipped, so questions whose answers live in a circular PDF get a
  truthful "not found" rather than an answer.
- **Refusal depends on the prompt, not the retriever.** RRF always ranks something, so an
  out-of-scope query still returns low-relevance chunks. Grounding is enforced by the system
  prompt plus post-hoc citation verification — there is no relevance threshold yet. A score floor
  would make refusal structural instead of instructed.
- **Evals are thin.** 17 cases, assertions are keyword-based. Enough to catch regressions, not
  enough to claim an accuracy number.
- **20 schools using it** would break at, in order: the Anthropic per-key rate limit (one shared
  key, no per-user quota); SQLite's single-writer lock once a scheduled check overlaps live
  questions; and the absence of any auth or per-tenant isolation — every user shares one corpus
  and one question log.
- **PII.** `qa_log` stores raw question text indefinitely with no retention policy. A parent
  typing a child's name into the chat would have it persisted. Needs a TTL and a redaction pass.
- **Change detection is body-text only.** Volatile counters are filtered out, but a page whose
  content moves into an attachment, or a pure reordering, will read as a change with a slightly
  awkward summary.
- **Politeness caveat.** If `robots.txt` is unreachable, the fetcher proceeds and records a
  warning rather than blocking. A stricter deployment should fail closed.
