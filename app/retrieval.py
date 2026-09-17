"""Hybrid retrieval: local fastembed vectors + jieba BM25, fused by reciprocal rank.

No embedding API key is needed. Vectors are float32 blobs in SQLite; with a corpus of tens of
pages a numpy dot product is faster than standing up a vector database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .config import Config
from .db import all_chunks_with_vectors, chunks_missing_vectors, store_embeddings
from .normalize import to_simplified

RRF_K = 60  # reciprocal-rank-fusion damping constant


@dataclass
class Hit:
    chunk_id: int
    url: str
    page_title: str
    section_title: str
    anchor: str
    text: str
    score: float

    @property
    def citation_url(self) -> str:
        return f"{self.url}#{self.anchor}" if self.anchor else self.url

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation_url"] = self.citation_url
        return data


@lru_cache(maxsize=2)
def _embedder(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_texts(cfg: Config, texts: list[str]) -> np.ndarray:
    vectors = list(_embedder(cfg.embedding_model).embed(texts))
    return _l2_normalise(np.asarray(vectors, dtype=np.float32))


def warmup(cfg: Config) -> str:
    """Force the model download up front so the first question is not a 60-second wait."""
    vector = embed_texts(cfg, ["小一入學統籌辦法"])
    return f"{cfg.embedding_model} ready, dim={vector.shape[1]}"


def index_pending(cfg: Config, conn: sqlite3.Connection, batch: int = 32) -> int:
    """Embed any chunk that has no vector yet. Safe to call repeatedly."""
    rows = chunks_missing_vectors(conn)
    if not rows:
        return 0
    done = 0
    for start in range(0, len(rows), batch):
        window = rows[start : start + batch]
        vectors = embed_texts(cfg, [r["text_index"] for r in window])
        store_embeddings(
            conn,
            [(int(r["id"]), int(v.shape[0]), v.tobytes()) for r, v in zip(window, vectors)],
        )
        conn.commit()
        done += len(window)
    return done


def tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.lcut(to_simplified(text).lower()) if t.strip()]


def _bm25_order(query: str, corpus: list[list[str]], limit: int) -> list[int]:
    """Rank by BM25, keeping any document that shares a token with the query.

    Filtering on `score > 0` looks reasonable but is wrong here: BM25Okapi's IDF goes negative
    for a term present in more than half the corpus, so on a small corpus the most obviously
    relevant chunks score below zero and vanish. Token overlap is the honest test of "matched".
    """
    from rank_bm25 import BM25Okapi

    tokens = tokenize(query)
    if not tokens:
        return []
    scores = BM25Okapi(corpus).get_scores(tokens)
    wanted = set(tokens)
    matched = [i for i, doc in enumerate(corpus) if wanted & set(doc)]
    matched.sort(key=lambda i: scores[i], reverse=True)
    return matched[:limit]


def _vector_order(cfg: Config, rows: list[sqlite3.Row], query: str, limit: int) -> list[int]:
    positions = [i for i, r in enumerate(rows) if r["vector"]]
    if not positions:
        return []
    try:
        query_vector = embed_texts(cfg, [query])[0]
    except Exception:
        # Model unavailable (no download, offline). BM25 alone still answers.
        return []
    matrix = np.vstack([np.frombuffer(rows[i]["vector"], dtype=np.float32) for i in positions])
    sims = matrix @ query_vector
    return [positions[int(t)] for t in np.argsort(sims)[::-1][:limit]]


def search(
    cfg: Config,
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    candidates: int = 20,
) -> list[Hit]:
    rows = all_chunks_with_vectors(conn)
    if not rows:
        return []

    corpus = [tokenize(r["text_index"]) for r in rows]
    fused: dict[int, float] = {}
    for order in (_vector_order(cfg, rows, query, candidates), _bm25_order(query, corpus, candidates)):
        for rank, position in enumerate(order):
            fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        Hit(
            chunk_id=int(rows[pos]["id"]),
            url=rows[pos]["url"],
            page_title=rows[pos]["page_title"] or "",
            section_title=rows[pos]["section_title"] or "",
            anchor=rows[pos]["anchor"] or "",
            text=rows[pos]["text_display"],
            score=round(score, 6),
        )
        for pos, score in ranked
    ]
