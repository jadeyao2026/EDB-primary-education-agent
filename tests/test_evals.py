"""Eval runner for the three cases the task says will be probed.

Needs ANTHROPIC_API_KEY and an ingested database, so it skips by default rather than failing.
Run explicitly:  pytest tests/test_evals.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.agent import ask
from app.config import get_config
from app.db import connect

QUESTIONS = yaml.safe_load((Path(__file__).parent.parent / "evals" / "questions.yaml").read_text("utf-8"))

REFUSAL_HINTS = ("找不到", "沒有提到", "未提及", "不知道", "沒有相關資料", "沒有這項資訊", "無法確定")


def _cfg_or_skip():
    cfg = get_config()
    if not cfg.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    if not cfg.db_path.exists():
        pytest.skip("database not ingested yet; run `python -m app.cli ingest`")
    return cfg


@pytest.fixture(scope="module")
def conn():
    cfg = _cfg_or_skip()
    connection = connect(cfg.db_path)
    if connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
        pytest.skip("no chunks indexed yet")
    return connection


@pytest.mark.parametrize("spec", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_eval_case(spec, conn):
    cfg = _cfg_or_skip()
    result = ask(cfg, conn, spec["question"])
    answer = result.answer

    print(f"\n[{spec['id']}] {spec['question']}\n{answer[:400]}\n")

    assert answer, "agent returned no answer"
    assert result.tool_calls >= 1, "agent must call at least one tool before answering"
    assert not any(
        w.startswith("答案引用了未經檢索") for w in result.warnings
    ), f"fabricated citation: {result.warnings}"

    if spec.get("expect_refusal"):
        assert any(hint in answer for hint in REFUSAL_HINTS), (
            "expected an explicit 'not found' style answer"
        )

    if spec.get("expect_citation") is True:
        assert result.citations, "answer should carry at least one verified citation"
    elif spec.get("expect_citation") is False:
        assert not result.citations, "should not cite sources for an uncovered question"

    for phrase in spec.get("expect_phrases", []):
        assert phrase in answer, f"expected {phrase!r} in answer"

    assert "<" not in answer or "</" not in answer, "answer must not contain HTML"
