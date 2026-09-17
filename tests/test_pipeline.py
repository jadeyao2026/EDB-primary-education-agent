"""Tests that need no API key and no network: normalization, hashing, diff, notify, retrieval."""

from __future__ import annotations

import json

import pytest

from app import notify
from app.config import Config
from app.db import connect, init_db, insert_snapshot, replace_chunks, upsert_page
from app.fetcher import canonical_url, is_allowed_host
from app.llm import mechanical_summary
from app.normalize import (
    body_text_for_hash,
    content_hash,
    extract_links,
    split_sections,
    to_simplified,
)
from app.watcher import build_diff, first_changed_section

SEED = "https://www.edb.gov.hk/tc/edu-system/primary-secondary/primary.html"

# Mirrors the real EDB template: content lives in .inner_page_content_container, the mega-menu
# sits inside <header>, and subheadings are bold paragraphs rather than h2/h3.
SAMPLE_HTML = """
<html><body>
<header class="header_container">
  <nav class="menu_bar"><a href="/tc/about-edb/index.html">關於教育局</a>
  <a href="/tc/edu-system/preprimary/index.html">幼稚園教育</a></nav>
</header>
<div class="section_container">
 <div class="inner_page_content_container generic">
  <h1 class="generic_inner_page_paragraph_title_h1">小學全日制</h1>
  <p>教育局推行小學全日制，目標是提升學習效能。</p>
  <p><strong>小學全日制的好處</strong></p>
  <p>學生有更多時間參與課外活動，家長接送安排也較簡單。</p>
  <p><a href="/tc/edu-system/primary-secondary/applicable-to-primary/small-class-teaching/index.html">小班教學</a></p>
  <p><a href="/tc/curriculum-development/report.pdf">報告 PDF</a></p>
  <p><a href="https://example.com/outside">外部連結</a></p>
 </div>
</div>
<footer><p>版權所有</p></footer>
</body></html>
"""


def test_split_sections_finds_pseudo_headings():
    title, sections = split_sections(SAMPLE_HTML, SEED)
    assert title == "小學全日制"
    titles = [s["section_title"] for s in sections]
    assert "小學全日制的好處" in titles, "bold paragraph must be treated as a heading"
    assert len(sections) >= 2


def test_split_sections_drops_chrome_but_keeps_content():
    _, sections = split_sections(SAMPLE_HTML, SEED)
    body = " ".join(s["text_display"] for s in sections)
    assert "提升學習效能" in body
    assert "關於教育局" not in body, "mega-menu must not leak into content"
    assert "版權所有" not in body


def test_index_text_is_simplified_display_stays_traditional():
    _, sections = split_sections(SAMPLE_HTML, SEED)
    target = next(s for s in sections if "好處" in s["section_title"])
    assert "好處" in target["text_display"], "citations must show the page's own wording"
    assert "好处" in target["text_index"], "index text should be simplified for jieba/BM25"


def test_to_simplified_roundtrip():
    assert to_simplified("學習") == "学习"


def test_extract_links_filters_scope():
    links = extract_links(SAMPLE_HTML, SEED)
    assert any("small-class-teaching" in link for link in links)
    assert not any(link.endswith(".pdf") for link in links), "attachments are skipped"
    assert not any("example.com" in link for link in links), "off-site links are skipped"
    assert not any("preprimary" in link for link in links), "menu links must not be followed"


def test_canonical_url_normalises_scheme_and_fragment():
    assert canonical_url("http://www.edb.gov.hk/tc/x.html#frag") == "https://www.edb.gov.hk/tc/x.html"
    assert is_allowed_host(SEED)
    assert not is_allowed_host("https://example.com/x")


def test_hash_ignores_volatile_noise():
    base = [{"section_title": "測試", "anchor": "", "text_display": "內容 A", "text_index": "内容 A"}]
    noisy = [
        {
            "section_title": "測試",
            "anchor": "",
            "text_display": "內容 A 瀏覽次數：1234",
            "text_index": "内容 A",
        }
    ]
    assert content_hash(body_text_for_hash(base)) == content_hash(body_text_for_hash(noisy))


def test_hash_changes_on_real_edit():
    a = [{"section_title": "測試", "anchor": "", "text_display": "申請日期為九月", "text_index": ""}]
    b = [{"section_title": "測試", "anchor": "", "text_display": "申請日期為十月", "text_index": ""}]
    assert content_hash(body_text_for_hash(a)) != content_hash(body_text_for_hash(b))


def test_diff_and_section_detection():
    old = "## 小一入學\n申請日期為九月"
    new = "## 小一入學\n申請日期為十月"
    diff = build_diff(old, new, SEED)
    assert "申請日期為九月" in diff and "申請日期為十月" in diff
    assert first_changed_section(diff)


def test_mechanical_summary_is_human_readable():
    diff = build_diff("申請日期為九月", "申請日期為十月", SEED)
    summary = mechanical_summary(diff, "小學教育")
    assert "九月" in summary and "十月" in summary
    assert "<" not in summary and "@@" not in summary, "no raw HTML or diff markers"


def test_webhook_payload_carries_plain_language_text():
    payload = notify.build_payload(
        page_title="小學教育",
        page_url=SEED,
        summary="申請日期由九月改為十月。",
        detected_at="2026-09-15T09:00:00+00:00",
        section_title="小一入學",
        diff_text="@@ -1 +1 @@\n-九月\n+十月",
    )
    assert "申請日期由九月改為十月。" in payload["text"]
    assert SEED in payload["text"]
    assert "@@" not in payload["text"], "diff markers stay out of the readable message"
    assert json.dumps(payload, ensure_ascii=False)


def test_notify_dry_run_without_webhook_url(capsys):
    cfg = Config(webhook_url="", anthropic_api_key="")
    result = notify.send(cfg, notify.build_payload(
        page_title="t", page_url=SEED, summary="測試摘要", detected_at="now",
    ))
    assert result.status == "dry_run"
    assert "測試摘要" in capsys.readouterr().out


def test_webhook_signature_is_stable():
    body = b'{"a":1}'
    first = notify._sign("secret", body)
    assert first == notify._sign("secret", body)
    assert first != notify._sign("other", body)
    assert first.startswith("sha256=")


@pytest.fixture()
def seeded_db(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    page_id = upsert_page(conn, SEED, "小學全日制")
    _, sections = split_sections(SAMPLE_HTML, SEED)
    snapshot_id = insert_snapshot(conn, page_id, "h", body_text_for_hash(sections))
    replace_chunks(conn, page_id, snapshot_id, sections)
    conn.commit()
    return conn


def test_search_works_without_vectors(seeded_db):
    """BM25 must still answer if the embedding model was never downloaded."""
    from app.retrieval import search

    cfg = Config(embedding_model="definitely-not-a-real-model")
    hits = search(cfg, seeded_db, "全日制的好處", top_k=3)
    assert hits, "keyword retrieval should return results with no embeddings present"
    assert any("全日制" in hit.text for hit in hits)
    assert hits[0].citation_url.startswith("https://www.edb.gov.hk")


def test_search_tool_returns_citable_fields(seeded_db):
    from app.tools import handle_search

    cfg = Config(embedding_model="definitely-not-a-real-model")
    payload = json.loads(handle_search(cfg, seeded_db, {"query": "全日制", "top_k": 2}))
    assert payload["results"]
    for item in payload["results"]:
        assert item["source_url"].startswith("https://")
        assert item["chunk_id"] and "text" in item
