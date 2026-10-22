#!/usr/bin/env python3
"""
Phase 4 - Text Chunking: compare Fixed vs Recursive vs Semantic chunking.

Reads the cleaned knowledge base (kb_clean.py output) and, for each strategy, measures:
    * number of chunks (+ chunk length stats)
    * processing time (chunking, indexing, per-query latency)
    * retrieval accuracy (Hit@1 / Hit@3 / Hit@5 / MRR) on an auto-generated question set

Outputs:
    chunks/chunks_fixed.jsonl | chunks_recursive.jsonl | chunks_semantic.jsonl
    chunks/chunk_comparison.csv

CLI:
    python kb_chunking.py --kb knowledge_base_clean --out chunks
    python kb_chunking.py --size 1000 --overlap 200 --queries 300 --embedder minilm

Retrieval accuracy - how it is measured (and its limits)
    Every "## Section" of every article is a ground-truth passage. The query is
    "<Article title> <Section heading>"; a retrieved chunk counts as correct if it is from that
    article AND overlaps the section by >= 50% of min(chunk length, section length).
    The same retriever is used for all three strategies, so the comparison is fair, but these
    are structural proxy questions. For the final RAG evaluation report, replace them with a
    hand-written or LLM-generated question/answer set.

Requires: numpy, scipy, scikit-learn   (optional: sentence-transformers for --embedder minilm)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

Span = tuple[int, int]

# =========================================================================== #
# 1. Fixed-size chunking
# =========================================================================== #
def fixed_chunks(text: str, size: int = 1000, overlap: int = 200) -> list[Span]:
    """Sliding character window. Fast and predictable, but cuts mid-word / mid-sentence."""
    step = max(size - overlap, 1)
    spans = []
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
    return spans


# =========================================================================== #
# 2. Recursive chunking (same idea as LangChain's RecursiveCharacterTextSplitter)
# =========================================================================== #
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _atomic_spans(text: str, s: int, e: int, seps: list[str], size: int) -> list[Span]:
    """Split [s, e) on the coarsest separator present until every piece is <= size."""
    if e - s <= size:
        return [(s, e)]
    for i, sep in enumerate(seps):
        pieces, pos = [], s
        while True:
            idx = text.find(sep, pos, e)
            if idx == -1:
                pieces.append((pos, e))
                break
            pieces.append((pos, idx + len(sep)))        # separator stays with the left piece
            pos = idx + len(sep)
        pieces = [p for p in pieces if p[1] > p[0]]
        if len(pieces) > 1:
            out = []
            for ps, pe in pieces:
                out.extend(_atomic_spans(text, ps, pe, seps[i + 1:], size))
            return out
    return [(p, min(p + size, e)) for p in range(s, e, size)]       # no separator at all: hard cut


def recursive_chunks(text: str, size: int = 1000, overlap: int = 200) -> list[Span]:
    """Paragraphs -> lines -> sentences -> clauses -> words; pieces merged up to `size`."""
    atoms = _atomic_spans(text, 0, len(text), SEPARATORS, size)
    chunks, cur = [], []
    for span in atoms:
        if cur and span[1] - cur[0][0] > size:
            chunks.append((cur[0][0], cur[-1][1]))
            while cur and cur[-1][1] - cur[0][0] > overlap:         # keep a tail of <= overlap chars
                cur.pop(0)
            while cur and span[1] - cur[0][0] > size:
                cur.pop(0)
        cur.append(span)
    if cur and (not chunks or cur[-1][1] > chunks[-1][1]):
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


# =========================================================================== #
# 3. Semantic chunking (split where the topic changes between adjacent sentences)
# =========================================================================== #
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')


def _sentence_units(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, is_heading) for each sentence; '## Heading' lines are their own unit."""
    units, pos = [], 0
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        if not line.strip():
            continue
        if line.startswith("#"):
            units.append((start, end, True))
            continue
        last = start
        for m in _SENT_SPLIT.finditer(line):
            units.append((last, start + m.start(), False))
            last = start + m.end()
        units.append((last, end, False))
    return [u for u in units if text[u[0]:u[1]].strip()]


class Embedder:
    """TF-IDF (no downloads) or a sentence-transformers model. One instance is shared by the
    semantic chunker and the retriever so the comparison is consistent."""

    def __init__(self, kind: str = "auto", model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        if kind in ("auto", "minilm"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.kind = f"sentence-transformers:{model_name}"
            except Exception as e:
                if kind == "minilm":
                    raise RuntimeError(f"sentence-transformers unavailable: {e}")
        if self.model is None:
            self.kind = "tfidf"
        self.tfidf: TfidfVectorizer | None = None

    def fit(self, corpus: list[str]) -> None:
        if self.model is None:
            self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
            self.tfidf.fit(corpus)

    def encode(self, texts: list[str]):
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                                     show_progress_bar=False)
        return normalize(self.tfidf.transform(texts))


def _adjacent_similarity(V) -> np.ndarray:
    """Cosine similarity between each (smoothed) sentence window and the next one."""
    n = V.shape[0]
    band = sparse.diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")   # window of 3 sentences
    W = normalize(band @ V)
    if sparse.issparse(W):
        return np.asarray(W[:-1].multiply(W[1:]).sum(axis=1)).ravel()
    return (W[:-1] * W[1:]).sum(axis=1)


def semantic_chunks(text: str, embedder: Embedder, size: int = 1000, min_chars: int = 300,
                    percentile: float = 25, units=None, vectors=None) -> list[Span]:
    """Break where adjacent-sentence similarity falls below the given percentile of this document's
    similarities. Always breaks at headings; never exceeds `size` (hard cap); merges tiny chunks
    until `min_chars`. No overlap."""
    units = units if units is not None else _sentence_units(text)
    if not units:
        return []
    if len(units) == 1:
        return [(units[0][0], units[0][1])]
    sims = _adjacent_similarity(vectors if vectors is not None
                                else embedder.encode([text[s:e] for s, e, _ in units]))
    thr = np.percentile(sims, percentile)

    chunks, c_start, c_end = [], units[0][0], units[0][1]
    for i in range(1, len(units)):
        s, e, is_head = units[i]
        cur_len = c_end - c_start
        topic_break = sims[i - 1] < thr and cur_len >= min_chars
        too_big = e - c_start > size
        if is_head or topic_break or too_big:
            chunks.append((c_start, c_end))
            c_start = s
        c_end = e
    chunks.append((c_start, c_end))

    out = []                                          # oversize sentence runs -> recursive fallback
    for s, e in chunks:
        if e - s > size:
            out.extend((s + a, s + b) for a, b in recursive_chunks(text[s:e], size, 0))
        else:
            out.append((s, e))
    return out


# =========================================================================== #
# Data loading + chunk records
# =========================================================================== #
def load_docs(kb_dir: Path) -> list[dict]:
    meta = {}
    mp = kb_dir / "metadata.csv"
    if mp.exists():
        with mp.open(newline="", encoding="utf-8") as f:
            meta = {r["file"]: r for r in csv.DictReader(f)}
    docs = []
    for p in sorted(kb_dir.glob("*.txt")):
        m = meta.get(p.name, {})
        docs.append({"file": p.name, "article": m.get("article_name") or p.stem.replace("_", " "),
                     "url": m.get("url", ""), "text": p.read_text(encoding="utf-8")})
    return docs


def make_records(docs, per_doc_spans: dict[str, list[Span]], strategy: str) -> list[dict]:
    recs = []
    for d in docs:
        for i, (s, e) in enumerate(per_doc_spans[d["file"]]):
            recs.append({"chunk_id": f"{d['file']}::{strategy}::{i:04d}", "source": d["file"],
                         "article": d["article"], "url": d["url"], "start": s, "end": e,
                         "text": d["text"][s:e].strip()})
    return [r for r in recs if r["text"]]


# =========================================================================== #
# Evaluation
# =========================================================================== #
def build_eval_set(docs, n_queries: int, seed: int = 42, min_section_chars: int = 300) -> list[dict]:
    """One query per '## Section' (title + heading); ground truth = that section's character span."""
    items = []
    for d in docs:
        text = d["text"]
        heads = [(m.start(), m.end(), m.group(1).strip())
                 for m in re.finditer(r"^#{2,6}\s+(.+)$", text, re.M)]
        for i, (hs, he, title) in enumerate(heads):
            end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
            if end - he >= min_section_chars:
                items.append({"query": f"{d['article']} {title}", "file": d["file"],
                              "start": hs, "end": end})
    random.Random(seed).shuffle(items)
    return items[:n_queries]


def _is_hit(rec: dict, q: dict) -> bool:
    if rec["source"] != q["file"]:
        return False
    overlap = min(rec["end"], q["end"]) - max(rec["start"], q["start"])
    need = 0.5 * min(rec["end"] - rec["start"], q["end"] - q["start"])
    return overlap > 0 and overlap >= need


def evaluate_retrieval(records: list[dict], queries: list[dict], embedder: Embedder, ks=(1, 3, 5)) -> dict:
    t0 = time.perf_counter()
    C = embedder.encode([r["text"] for r in records])
    index_time = time.perf_counter() - t0
    if not queries:
        return {"index_time_s": index_time, **{f"hit@{k}": float("nan") for k in ks}, "mrr": float("nan"),
                "avg_query_ms": float("nan"), "n_queries": 0}

    t0 = time.perf_counter()
    Q = embedder.encode([q["query"] for q in queries])
    scores = Q @ C.T
    scores = scores.toarray() if sparse.issparse(scores) else np.asarray(scores)
    top_k = max(ks) * 4                                           # MRR looks a little deeper than Hit@5
    top = np.argsort(-scores, axis=1)[:, :top_k]
    query_time = time.perf_counter() - t0

    hits = {k: 0 for k in ks}
    rr = 0.0
    for q, row in zip(queries, top):
        rank = next((r for r, idx in enumerate(row, 1) if _is_hit(records[idx], q)), None)
        if rank:
            rr += 1 / rank
            for k in ks:
                hits[k] += rank <= k
    n = len(queries)
    return {"index_time_s": index_time, **{f"hit@{k}": hits[k] / n for k in ks}, "mrr": rr / n,
            "avg_query_ms": 1000 * query_time / n, "n_queries": n}


# =========================================================================== #
# Single-strategy chunking (used by the app / pipeline)
# =========================================================================== #
def chunk_corpus(kb_dir: str, out_dir: str, strategy: str = "recursive", size: int = 1000,
                 overlap: int = 200, embedder_kind: str = "auto") -> dict:
    """Chunk every cleaned article with ONE strategy and write chunks/chunks_<strategy>.jsonl."""
    if strategy not in ("fixed", "recursive", "semantic"):
        raise ValueError(f"unknown strategy {strategy!r}")
    docs = load_docs(Path(kb_dir))
    if not docs:
        raise ValueError(f"No .txt files in {kb_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if strategy == "fixed":
        spans = {d["file"]: fixed_chunks(d["text"], size, overlap) for d in docs}
    elif strategy == "recursive":
        spans = {d["file"]: recursive_chunks(d["text"], size, overlap) for d in docs}
    else:
        emb = Embedder(embedder_kind)
        emb.fit([d["text"] for d in docs] if emb.model is None else [])
        spans = {d["file"]: semantic_chunks(d["text"], emb, size) for d in docs}
    elapsed = time.perf_counter() - t0

    recs = make_records(docs, spans, strategy)
    path = out / f"chunks_{strategy}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": str(path), "n_chunks": len(recs), "n_articles": len(docs), "chunk_time_s": round(elapsed, 3)}


# =========================================================================== #
# Orchestration
# =========================================================================== #
def compare_strategies(kb_dir="knowledge_base_clean", out_dir="chunks", *, size=1000, overlap=200,
                       n_queries=200, embedder_kind="auto", seed=42, on_progress=print) -> list[dict]:
    docs = load_docs(Path(kb_dir))
    if not docs:
        raise SystemExit(f"No .txt files in {kb_dir}. Run kb_clean.py first.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(embedder_kind)
    embedder.fit([d["text"] for d in docs] if embedder.model is None else [])
    on_progress(f"{len(docs)} documents | embedder: {embedder.kind} | size={size} overlap={overlap}")
    queries = build_eval_set(docs, n_queries, seed)
    on_progress(f"{len(queries)} evaluation questions (one per article section)")

    results = []
    for name in ("fixed", "recursive", "semantic"):
        t0 = time.perf_counter()
        if name == "fixed":
            spans = {d["file"]: fixed_chunks(d["text"], size, overlap) for d in docs}
        elif name == "recursive":
            spans = {d["file"]: recursive_chunks(d["text"], size, overlap) for d in docs}
        else:
            spans = {d["file"]: semantic_chunks(d["text"], embedder, size) for d in docs}
        chunk_time = time.perf_counter() - t0

        recs = make_records(docs, spans, name)
        with (out / f"chunks_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        lens = [len(r["text"]) for r in recs]
        ev = evaluate_retrieval(recs, queries, embedder)
        row = {"strategy": name, "num_chunks": len(recs),
               "avg_chunk_chars": round(statistics.mean(lens)), "median_chunk_chars": round(statistics.median(lens)),
               "min_chunk_chars": min(lens), "max_chunk_chars": max(lens),
               "chunking_time_s": round(chunk_time, 3), "indexing_time_s": round(ev["index_time_s"], 3),
               "avg_query_ms": round(ev["avg_query_ms"], 2),
               "hit@1": round(ev["hit@1"], 3), "hit@3": round(ev["hit@3"], 3),
               "hit@5": round(ev["hit@5"], 3), "mrr": round(ev["mrr"], 3)}
        results.append(row)
        on_progress(f"  {name:<10} chunks={row['num_chunks']:<6} avg={row['avg_chunk_chars']:<5} "
                    f"chunk_time={row['chunking_time_s']}s  hit@1={row['hit@1']}  hit@5={row['hit@5']}  mrr={row['mrr']}")

    with (out / "chunk_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    on_progress(f"Saved chunks + chunk_comparison.csv to {out}/")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: compare fixed / recursive / semantic chunking")
    ap.add_argument("--kb", default="knowledge_base_clean", help="cleaned knowledge base folder")
    ap.add_argument("--out", default="chunks")
    ap.add_argument("--size", type=int, default=1000, help="max chunk size in characters (~250 tokens)")
    ap.add_argument("--overlap", type=int, default=200, help="overlap in characters (fixed & recursive)")
    ap.add_argument("--queries", type=int, default=200, help="number of evaluation questions")
    ap.add_argument("--embedder", default="auto", choices=["auto", "tfidf", "minilm"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    compare_strategies(a.kb, a.out, size=a.size, overlap=a.overlap, n_queries=a.queries,
                       embedder_kind=a.embedder, seed=a.seed)


if __name__ == "__main__":
    main()
