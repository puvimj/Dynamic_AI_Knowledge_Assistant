#!/usr/bin/env python3
"""
Phase 7 - Retrieval Pipeline (basic similarity search)

    question -> embed (same model as the index) -> top-k nearest chunks -> Hit objects
                                                  -> confidence score, context string, source list

Python:
    from kb_retrieval import Retriever, format_context
    r = Retriever.load("vectordb/recursive__all-MiniLM-L6-v2__faiss")
    hits = r.search("What is a convolutional neural network?", k=5)
    print(r.confidence(hits), format_context(hits))

CLI (quick check of the retrieval step on its own):
    python kb_retrieval.py --db vectordb/recursive__all-MiniLM-L6-v2__faiss "what is overfitting?" -k 5
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from kb_embed import load_vector_db


_STOP = set("a an and are as at be by do does for from how in is it of on or that the this to was what when where which "
            "who why with about between compare explain give example examples summarize summary tell me define "
            "difference differences vs versus".split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _title_tokens(title: str) -> set[str]:
    return {t for t in _tokens(re.sub(r"\(.*?\)", " ", title)) if t not in _STOP}


@dataclass
class Hit:
    rank: int            # 1 = best
    score: float         # cosine similarity (higher = more similar)
    chunk_id: str
    article: str
    url: str
    source: str          # filename in the knowledge base
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    def __init__(self, store, embedder, chunks: list[dict], stats: dict | None = None):
        self.store, self.embedder, self.chunks, self.stats = store, embedder, chunks, stats or {}
        self.last_latency_ms = 0.0
        self._bm25 = None

    # -- lexical side of the hybrid search ------------------------------------------------------- #
    def _bm25_index(self):
        if self._bm25 is None:
            docs = [Counter(_tokens(c["text"])) for c in self.chunks]
            df = Counter(t for d in docs for t in d)
            lens = [sum(d.values()) for d in docs]
            self._bm25 = (docs, df, lens, sum(lens) / max(len(lens), 1))
        return self._bm25

    def _bm25_scores(self, terms: list[str]) -> np.ndarray:
        docs, df, lens, avg = self._bm25_index()
        n, out = len(docs), np.zeros(len(docs))
        for t in set(terms):
            if t not in df:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            for i, d in enumerate(docs):
                f = d.get(t, 0)
                if f:
                    out[i] += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * lens[i] / avg))
        return out

    @classmethod
    def load(cls, db_dir: str) -> "Retriever":
        store, embedder, chunks, stats = load_vector_db(db_dir)
        return cls(store, embedder, chunks, stats)

    def search(self, query: str, k: int = 5, min_score: float = 0.0, hybrid: bool = True) -> list[Hit]:
        """Top-k chunks. Hit.score is always the embedding cosine similarity (used for confidence).

        hybrid=True re-ranks the embedding candidates with two cheap signals that embeddings alone miss:
          * keyword match (BM25) between the question and the passage, and
          * a title match: if the article's title is what the question asks about ("What is FAISS?" ->
            the article "FAISS"), its passages are lifted above passages that merely mention the word.
        """
        t0 = time.perf_counter()
        qvec = self.embedder.encode([query], query=True)
        pool = min(len(self.chunks), max(k * 6, 40)) if hybrid else k
        raw = self.store.search(qvec, pool)[0]
        if hybrid and raw:
            terms = [t for t in _tokens(query) if t not in _STOP]
            defining = bool(re.match(r"\s*(what|who)\b.*\b(is|are|was|were)\b|\s*define\b", query, re.I))
            dense = np.array([sc for _, sc in raw])
            lex = self._bm25_scores(terms)[[i for i, _ in raw]] if terms else np.zeros(len(raw))
            lex = lex / lex.max() if lex.max() > 0 else lex
            boost = np.zeros(len(raw))
            for j, (idx, _) in enumerate(raw):
                title = _title_tokens(self.chunks[idx]["article"])
                if title and terms:
                    boost[j] = 0.30 if title <= set(terms) else 0.12 * len(title & set(terms)) / len(title)
                    if defining and boost[j] >= 0.30 and self.chunks[idx].get("start", 1) == 0:
                        boost[j] += 0.15           # "What is X?" -> the opening passage of article X defines it
            order = np.argsort(-(dense + 0.25 * lex + boost), kind="stable")[:k]
            raw = [raw[j] for j in order]
        self.last_latency_ms = 1000 * (time.perf_counter() - t0)
        hits = []
        for idx, score in raw:
            if score < min_score:
                continue
            c = self.chunks[idx]
            hits.append(Hit(len(hits) + 1, round(float(score), 4), c["chunk_id"], c["article"], c["url"],
                            c["source"], c["text"], c["start"], c["end"]))
        return hits

    def confidence(self, hits: list[Hit]) -> float:
        """0-1 heuristic: 60% best score + 40% mean of top-3, each rescaled between a 'no match' floor
        and a 'strong match' ceiling for the embedding model. It is a retrieval-confidence signal (is the
        knowledge base likely to contain the answer?), not a probability. Calibrate floor/ceil
        (Embedder.score_floor / score_ceil) on your own evaluation set."""
        if not hits:
            return 0.0
        lo, hi = self.embedder.score_floor, self.embedder.score_ceil

        def norm(x: float) -> float:
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

        top3 = [h.score for h in hits[:3]]
        return round(0.6 * norm(hits[0].score) + 0.4 * norm(float(np.mean(top3))), 3)


def confidence_label(conf: float) -> str:
    return "High" if conf >= 0.66 else "Medium" if conf >= 0.33 else "Low"


def pack_context(hits: list[Hit], max_chars: int | None = None) -> tuple[str, list[Hit]]:
    """Numbered passages for the prompt, stopping before `max_chars`. Returns (text, hits_actually_used).
    Numbers match the [n] citations the LLM is asked to use."""
    parts, used_hits, used = [], [], 0
    for h in hits:
        block = f"[{h.rank}] {h.article}\n{h.text}"
        if max_chars and used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used_hits.append(h)
        used += len(block)
    return "\n\n".join(parts), used_hits


def format_context(hits: list[Hit], max_chars: int | None = None) -> str:
    return pack_context(hits, max_chars)[0]


def unique_sources(hits: list[Hit]) -> list[dict]:
    """One entry per article (best score first) with the passage numbers that came from it."""
    by_article: dict[str, dict] = {}
    for h in hits:
        s = by_article.setdefault(h.source, {"article": h.article, "url": h.url,
                                             "best_score": h.score, "passages": []})
        s["passages"].append(h.rank)
        s["best_score"] = max(s["best_score"], h.score)
    return sorted(by_article.values(), key=lambda s: -s["best_score"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 7: similarity search over the vector DB")
    ap.add_argument("query")
    ap.add_argument("--db", required=True, help="vector DB folder, e.g. vectordb/recursive__all-MiniLM-L6-v2__faiss")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    r = Retriever.load(a.db)
    hits = r.search(a.query, a.k)
    print(f"{len(hits)} results in {r.last_latency_ms:.1f} ms | confidence {r.confidence(hits)} "
          f"({confidence_label(r.confidence(hits))})\n")
    for h in hits:
        print(f"[{h.rank}] {h.score:.3f}  {h.article}  ({h.chunk_id})\n    {h.text[:160].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
