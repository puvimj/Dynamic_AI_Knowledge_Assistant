#!/usr/bin/env python3
"""
Phase 5 (Embedding Generation) + Phase 6 (Vector Database: FAISS and ChromaDB)

    chunks/chunks_recursive.jsonl   (from kb_chunking.py)
        -> embed every chunk                      (Phase 5)
        -> store in FAISS or ChromaDB             (Phase 6)
    vectordb/<strategy>__<model>__<backend>/
        store.json        backend, model, dim, #chunks, timings, size   (feeds the Analytics page)
        chunks.jsonl      chunk text + article + url + offsets
        faiss.index | chroma/
        embedder.joblib   only for the offline 'tfidf-lsa' embedder
    embeddings/<model>__<hash>.npy   cached vectors (so FAISS and Chroma don't re-embed)

CLI:
    python kb_embed.py build --chunks chunks/chunks_recursive.jsonl --db faiss
    python kb_embed.py build --chunks chunks/chunks_recursive.jsonl --db chroma --model BAAI/bge-small-en-v1.5
    python kb_embed.py benchmark --chunks chunks/chunks_recursive.jsonl --kb knowledge_base_clean \
        --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 --dbs faiss chroma

Embedding models
    all-MiniLM-L6-v2        384-d, fast, good default
    BAAI/bge-small-en-v1.5  384-d, usually a bit better at retrieval (uses a query prefix - handled)
    tfidf-lsa               offline fallback (no download, no GPU); weaker, for tests only

Requires: numpy scikit-learn joblib faiss-cpu chromadb sentence-transformers
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"
OFFLINE_MODEL = "tfidf-lsa"


def _slug(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s).strip("_")


def read_chunks(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# =========================================================================== #
# Phase 5 - embedders
# =========================================================================== #
class Embedder:
    """Wraps a sentence-transformers model (or the offline TF-IDF+LSA fallback).
    encode() always returns L2-normalised float32 vectors, so inner product == cosine."""

    def __init__(self, name: str = DEFAULT_MODEL, lsa_dim: int = 256):
        self.name, self.model, self.pipeline, self._lsa_dim = name, None, None, lsa_dim
        low = name.lower()
        self.query_prefix = ("Represent this sentence for searching relevant passages: " if "bge" in low
                             else "query: " if "e5" in low else "")
        self.doc_prefix = "passage: " if "e5" in low else ""
        if name == OFFLINE_MODEL:
            self.dim = None
            self.score_floor, self.score_ceil = 0.05, 0.50      # LSA cosines run lower than neural ones
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers  (or use --model tfidf-lsa)") from e
            self.model = SentenceTransformer(name)
            get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
            self.dim = get_dim()
            self.score_floor, self.score_ceil = 0.20, 0.70      # used for the confidence score

    @property
    def cacheable(self) -> bool:
        return self.model is not None

    # -- offline model must be fitted on the corpus ------------------------- #
    def fit(self, texts: list[str]) -> "Embedder":
        if self.model is None:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import Normalizer
            tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), max_features=50000)
            n_feat = tfidf.fit_transform(texts).shape[1]
            dim = max(2, min(self._lsa_dim, len(texts) - 1, n_feat - 1))
            self.pipeline = make_pipeline(tfidf, TruncatedSVD(n_components=dim, random_state=0),
                                          Normalizer(copy=False))
            self.pipeline.fit(texts)
            self.dim = dim
        return self

    def encode(self, texts: list[str], *, query: bool = False, batch_size: int = 64) -> np.ndarray:
        if self.model is not None:
            prefix = self.query_prefix if query else self.doc_prefix
            vecs = self.model.encode([prefix + t for t in texts], batch_size=batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=len(texts) > 500)
        else:
            vecs = self.pipeline.transform(texts)
        return np.ascontiguousarray(vecs, dtype="float32")

    def save(self, directory: Path) -> None:
        if self.pipeline is not None:
            import joblib
            joblib.dump(self.pipeline, directory / "embedder.joblib")

    @classmethod
    def load(cls, name: str, directory: Path) -> "Embedder":
        emb = cls(name)
        if name == OFFLINE_MODEL:
            import joblib
            emb.pipeline = joblib.load(Path(directory) / "embedder.joblib")
            emb.dim = emb.pipeline[-2].n_components
        return emb


def embed_chunks(chunks: list[dict], embedder: Embedder, cache_dir: str | None = "embeddings",
                 batch_size: int = 64, on_progress=print) -> tuple[np.ndarray, float, bool]:
    """Returns (vectors, seconds, loaded_from_cache). Vectors are cached on disk for neural models."""
    texts = [c["text"] for c in chunks]
    cache = None
    if cache_dir and embedder.cacheable:
        key = hashlib.md5("\x00".join(c["chunk_id"] + t for c, t in zip(chunks, texts)).encode()).hexdigest()[:10]
        cache = Path(cache_dir) / f"{_slug(embedder.name)}__{key}.npy"
        if cache.exists():
            on_progress(f"  embeddings loaded from cache: {cache}")
            return np.load(cache), 0.0, True
    t0 = time.perf_counter()
    vectors = embedder.encode(texts, batch_size=batch_size)
    dt = time.perf_counter() - t0
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return vectors, dt, False


# =========================================================================== #
# Phase 6 - vector stores (same interface; both do exact cosine search)
# =========================================================================== #
class FaissStore:
    backend = "faiss"

    def build(self, vectors: np.ndarray, chunks: list[dict], path: Path) -> None:
        import faiss
        self.index = faiss.IndexFlatIP(vectors.shape[1])      # exact search; fine up to ~1M chunks
        self.index.add(vectors)
        faiss.write_index(self.index, str(path / "faiss.index"))

    def load(self, path: Path) -> None:
        import faiss
        self.index = faiss.read_index(str(Path(path) / "faiss.index"))

    def search(self, qvecs: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        scores, ids = self.index.search(qvecs, min(k, self.index.ntotal))
        return [[(int(i), float(s)) for i, s in zip(r_ids, r_s) if i >= 0] for r_ids, r_s in zip(ids, scores)]

    def vectors(self) -> np.ndarray:
        """All stored vectors in chunk order (for embedding statistics / projections)."""
        return self.index.reconstruct_n(0, self.index.ntotal)

    @staticmethod
    def size_bytes(path: Path) -> int:
        return (Path(path) / "faiss.index").stat().st_size


def reset_chroma_cache() -> None:
    """Chroma keeps one client per folder for the life of the process. Rebuilding an index in the same folder
    (app 'Build' pressed twice, benchmark re-running) would otherwise hit the stale client
    ('Collection already exists') or keep deleted files open."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


class ChromaStore:
    backend = "chroma"
    _BATCH = 4000                                              # Chroma limits the size of one add()

    def build(self, vectors: np.ndarray, chunks: list[dict], path: Path) -> None:
        import chromadb
        reset_chroma_cache()
        client = chromadb.PersistentClient(path=str(path / "chroma"))
        self.col = client.create_collection("knowledge_base", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), self._BATCH):
            part = chunks[i:i + self._BATCH]
            self.col.add(ids=[str(i + j) for j in range(len(part))],
                         embeddings=vectors[i:i + self._BATCH].tolist(),
                         documents=[c["text"] for c in part],
                         metadatas=[{"source": c["source"], "article": c["article"], "url": c["url"],
                                     "start": c["start"], "end": c["end"]} for c in part])

    def load(self, path: Path) -> None:
        import chromadb
        self.col = chromadb.PersistentClient(path=str(Path(path) / "chroma")).get_collection("knowledge_base")

    def search(self, qvecs: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        res = self.col.query(query_embeddings=qvecs.tolist(), n_results=min(k, self.col.count()))
        return [[(int(i), 1.0 - float(d)) for i, d in zip(ids, dists)]        # cosine distance -> similarity
                for ids, dists in zip(res["ids"], res["distances"])]

    def vectors(self) -> np.ndarray:
        n = self.col.count()
        got = self.col.get(ids=[str(i) for i in range(n)], include=["embeddings"])
        by_id = {i: e for i, e in zip(got["ids"], got["embeddings"])}      # restore chunk order
        return np.asarray([by_id[str(i)] for i in range(n)], dtype="float32")

    @staticmethod
    def size_bytes(path: Path) -> int:
        return sum(f.stat().st_size for f in (Path(path) / "chroma").rglob("*") if f.is_file())


STORES = {"faiss": FaissStore, "chroma": ChromaStore}


def build_vector_db(chunks_path: str, out_root: str = "vectordb", model: str = DEFAULT_MODEL,
                    backend: str = "faiss", *, emb_dir: str | None = "embeddings", batch_size: int = 64,
                    prebuilt=None, on_progress=print) -> dict:
    """Embed the chunks and persist them in a vector DB. Returns the stats dict (also saved as store.json).
    `prebuilt` = (embedder, vectors, embed_seconds) lets the benchmark share one embedding run."""
    chunks = read_chunks(chunks_path)
    if not chunks:
        raise SystemExit(f"{chunks_path} is empty")
    strategy = Path(chunks_path).stem.replace("chunks_", "")
    path = Path(out_root) / f"{strategy}__{_slug(model)}__{backend}"
    if path.exists():
        if backend == "chroma":
            reset_chroma_cache()                               # release handles before deleting the old index
        shutil.rmtree(path)
    path.mkdir(parents=True)

    if prebuilt:
        embedder, vectors, embed_s = prebuilt
    else:
        embedder = Embedder(model)
        embedder.fit([c["text"] for c in chunks])
        on_progress(f"  embedding {len(chunks)} chunks with {model} ...")
        vectors, embed_s, _ = embed_chunks(chunks, embedder, emb_dir, batch_size, on_progress)

    t0 = time.perf_counter()
    store = STORES[backend]()
    store.build(vectors, chunks, path)
    index_s = time.perf_counter() - t0

    embedder.save(path)
    with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    stats = {"backend": backend, "model": model, "strategy": strategy, "dim": int(vectors.shape[1]),
             "n_chunks": len(chunks), "n_articles": len({c["source"] for c in chunks}),
             "embed_time_s": round(embed_s, 3), "index_time_s": round(index_s, 3),
             "index_size_mb": round(STORES[backend].size_bytes(path) / 1e6, 3),
             "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "path": str(path)}
    (path / "store.json").write_text(json.dumps(stats, indent=2))
    on_progress(f"  {backend}: {stats['n_chunks']} vectors x {stats['dim']}d | embed {stats['embed_time_s']}s | "
                f"index {stats['index_time_s']}s | {stats['index_size_mb']} MB -> {path}")
    return stats


def load_vector_db(path: str):
    """-> (store, embedder, chunks, stats). Rebuilds the exact embedder the DB was created with."""
    path = Path(path)
    stats = json.loads((path / "store.json").read_text())
    store = STORES[stats["backend"]]()
    store.load(path)
    return store, Embedder.load(stats["model"], path), read_chunks(path / "chunks.jsonl"), stats


# =========================================================================== #
# Benchmark: >= 2 embedding models x >= 2 vector DBs (project guideline)
# =========================================================================== #
def benchmark(chunks_path: str, kb_dir: str, models: list[str], backends: list[str], *,
              out_root: str = "vectordb", n_queries: int = 200, emb_dir: str | None = "embeddings",
              on_progress=print) -> list[dict]:
    from kb_chunking import _is_hit, build_eval_set, load_docs          # same eval questions as Phase 4
    queries = build_eval_set(load_docs(Path(kb_dir)), n_queries)
    chunks = read_chunks(chunks_path)
    on_progress(f"{len(chunks)} chunks | {len(queries)} evaluation questions")
    rows, quiet = [], (lambda *_: None)

    for model in models:
        embedder = Embedder(model)
        embedder.fit([c["text"] for c in chunks])
        vectors, embed_s, _ = embed_chunks(chunks, embedder, emb_dir, on_progress=quiet)
        for backend in backends:
            st = build_vector_db(chunks_path, out_root, model, backend, prebuilt=(embedder, vectors, embed_s),
                                 on_progress=quiet)
            store, emb, chs, _ = load_vector_db(st["path"])

            t0 = time.perf_counter()
            Q = emb.encode([q["query"] for q in queries], query=True)
            enc_ms = 1000 * (time.perf_counter() - t0) / len(queries)
            t0 = time.perf_counter()
            results = store.search(Q, 20)
            search_ms = 1000 * (time.perf_counter() - t0) / len(queries)

            hits, rr = {1: 0, 3: 0, 5: 0}, 0.0
            for q, res in zip(queries, results):
                rank = next((r for r, (idx, _) in enumerate(res, 1) if _is_hit(chs[idx], q)), None)
                if rank:
                    rr += 1 / rank
                    for k in hits:
                        hits[k] += rank <= k
            n = len(queries)
            row = {"model": model, "backend": backend, "dim": st["dim"], "n_chunks": st["n_chunks"],
                   "embed_time_s": st["embed_time_s"], "index_time_s": st["index_time_s"],
                   "index_size_mb": st["index_size_mb"], "query_embed_ms": round(enc_ms, 2),
                   "search_ms": round(search_ms, 3), "hit@1": round(hits[1] / n, 3),
                   "hit@3": round(hits[3] / n, 3), "hit@5": round(hits[5] / n, 3), "mrr": round(rr / n, 3)}
            rows.append(row)
            on_progress(f"  {model:<26} {backend:<7} hit@1={row['hit@1']} hit@5={row['hit@5']} mrr={row['mrr']} "
                        f"search={row['search_ms']}ms size={row['index_size_mb']}MB")

    out = Path(out_root) / "vectordb_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    on_progress(f"Saved {out}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Phases 5+6: embeddings and vector databases")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed chunks and store them in a vector DB")
    b.add_argument("--chunks", default="chunks/chunks_recursive.jsonl")
    b.add_argument("--out", default="vectordb")
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--db", default="faiss", choices=list(STORES))
    b.add_argument("--batch-size", type=int, default=64)
    c = sub.add_parser("benchmark", help="compare embedding models x vector DBs")
    c.add_argument("--chunks", default="chunks/chunks_recursive.jsonl")
    c.add_argument("--kb", default="knowledge_base_clean", help="cleaned KB (used to build eval questions)")
    c.add_argument("--out", default="vectordb")
    c.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, "BAAI/bge-small-en-v1.5"])
    c.add_argument("--dbs", nargs="+", default=list(STORES), choices=list(STORES))
    c.add_argument("--queries", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "build":
        build_vector_db(a.chunks, a.out, a.model, a.db, batch_size=a.batch_size)
    else:
        benchmark(a.chunks, a.kb, a.models, a.dbs, out_root=a.out, n_queries=a.queries)


if __name__ == "__main__":
    main()
