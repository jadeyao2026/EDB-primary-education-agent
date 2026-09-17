# AI usage note

## What I built

A grounded Q&A agent over the EDB primary-education pages (12 pages, 29 chunks), plus a change
watcher that pushes plain-language alerts to a webhook.

Six real tools the model calls: `search_edb_docs`, `get_section`, `list_watched_pages`,
`check_for_updates`, `diff_page`, `send_notification`. Every `tool_use` / `tool_result` pair lands
in SQLite (`tool_trace`) and is readable via `python -m app.cli trace` or the UI expander.

The parts I consider mine rather than the model's: the EDB template reverse-engineering in
`app/normalize.py`, the recompute-the-hash-from-stored-text decision in `app/ingest.py`, and the
BM25 token-overlap fix in `app/retrieval.py`. Each came from a failure I had to diagnose, not from
a first draft.

## What the model drafted

Most of the mechanical surface: the SQLite schema and helper functions, the CLI argparse wiring,
the Streamlit layout, the webhook payload shape and HMAC signing, the docstrings, and the first
pass of the test fixtures. That work is real but low-risk — it is either obviously correct or
obviously broken.

It also drafted the first version of `split_sections`, which was wrong in a way that looked right:
it stripped `<header>`/`<nav>`/`<footer>` before selecting a content container. On the EDB template
the entire mega-menu lives inside `<header class="header_container">`, and stripping it first
emptied the page — 5,731 of 5,860 characters gone, leaving 170. I only caught it because the chunk
count was implausible, not because anything raised.

## One suggestion I rejected

The model proposed `sentence-transformers` for embeddings, which is the conventional pick. I
rejected it for `fastembed`.

`sentence-transformers` pulls torch — 200MB+ on a slow connection, and on this machine PyPI was
running at 2.9 KB/s before I switched to a Chinese mirror. The whole reason we chose local
embeddings over an embedding API was to keep the grader down to one key and one short install.
Trading a 30-second API signup for a multi-hundred-megabyte dependency loses that trade. `fastembed`
runs the same `bge-small-zh-v1.5` weights through onnxruntime at ~45MB, on CPU, no torch.

The cost is real: fastembed has a smaller model catalogue and less community tooling. For 29 chunks
I do not need either.

## One thing I would do next

Add a relevance floor to retrieval so refusal becomes structural instead of instructed.

Right now RRF always ranks something. Ask about kindergarten vouchers and the retriever cheerfully
returns the five least-irrelevant primary-school chunks; the only thing standing between that and a
confident wrong answer is the system prompt plus post-hoc citation verification. That works in my
17 eval cases, but it is a behaviour I am asking the model to have, not a property the system has.
A score threshold below which `search_edb_docs` returns an empty result set would make "I don't
know" the mechanical outcome rather than the well-behaved one.

I would want that before trusting this in front of parents, and I would want the eval set to grow
past keyword assertions before quoting an accuracy number.
