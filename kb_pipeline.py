"""
End-to-end pipeline + workspaces (used by the Streamlit app; also usable from a notebook).

A *workspace* is one knowledge base for one domain - rebuild or switch domains without touching code:

    workspaces/<name>/
        raw/        downloaded (or imported) articles            Phase 1
        clean/      cleaned articles + metadata.csv              Phases 2-3
        chunks/     chunks_<strategy>.jsonl                      Phase 4
        vectordb/   <strategy>__<model>__<backend>/              Phases 5-6
        embeddings/ cached vectors
        config.json          active vector DB + settings
        pipeline_stats.json  stage timings and counts (Analytics page)
        query_log.jsonl      one line per chat question (Analytics / Evaluation pages)

    from kb_pipeline import Workspace, build_pipeline
    stats = build_pipeline(Workspace("ml_basics"),
                           keywords=["Machine Learning", "Deep Learning", "Computer Vision",
                                     "Natural Language Processing", "Transformer deep learning architecture"],
                           top_n=5, on_progress=lambda frac, label: print(f"{frac:.0%} {label}"))
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("KB_WORKSPACES", "workspaces"))

# (label, share of the progress bar)
STAGES = [("Downloading articles", 0.45), ("Cleaning text", 0.05), ("Chunking", 0.10),
          ("Embedding + building vector database", 0.40)]


class PipelineError(RuntimeError):
    """A problem the user can fix (bad input, nothing downloaded...). Message is safe to show."""


def safe_name(name: str) -> str:
    return re.sub(r"[^\w-]+", "_", name.strip()).strip("_") or "default"


def list_workspaces() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir()) if ROOT.exists() else []


class Workspace:
    def __init__(self, name: str):
        self.name = safe_name(name)
        self.root = ROOT / self.name
        self.raw, self.clean = self.root / "raw", self.root / "clean"
        self.chunks, self.vectordb, self.embeddings = (self.root / "chunks", self.root / "vectordb",
                                                       self.root / "embeddings")
        self.config_path, self.stats_path = self.root / "config.json", self.root / "pipeline_stats.json"
        self.log_path = self.root / "query_log.jsonl"
        self.feedback_path = self.root / "feedback.jsonl"

    def ensure(self) -> "Workspace":
        for d in (self.raw, self.clean, self.chunks, self.vectordb):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def delete(self) -> bool:
        """Remove this workspace folder (articles, vector DBs, logs). Returns False if files stayed behind."""
        if self.root.resolve().parent != ROOT.resolve():          # never delete anything outside workspaces/
            raise PipelineError("Refusing to delete a folder outside the workspaces directory.")
        shutil.rmtree(self.root, ignore_errors=True)
        return not self.root.exists()

    # -- json helpers ------------------------------------------------------- #
    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def stats(self) -> dict:
        return self._read(self.stats_path)

    def config(self) -> dict:
        return self._read(self.config_path)

    # -- vector DBs ----------------------------------------------------------- #
    def list_dbs(self) -> list[Path]:
        """Built vector DBs, newest first."""
        if not self.vectordb.exists():
            return []
        dbs = [p for p in self.vectordb.iterdir() if (p / "store.json").exists()]
        return sorted(dbs, key=lambda p: (p / "store.json").stat().st_mtime, reverse=True)

    def active_db(self) -> Path | None:
        chosen = self.config().get("active_db")
        if chosen and (Path(chosen) / "store.json").exists():
            return Path(chosen)
        dbs = self.list_dbs()
        return dbs[0] if dbs else None

    def set_active_db(self, path: str | Path) -> None:
        cfg = self.config()
        cfg["active_db"] = str(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(cfg, indent=2))

    # -- query log ------------------------------------------------------------ #
    def log_query(self, ans: dict) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "question": ans["question"], "task": ans["task"], "model": ans["model"],
               "retrieval_ms": ans["retrieval_ms"], "generation_ms": ans["generation_ms"],
               "confidence": ans["confidence"], "refused": ans["refused"],
               "cited": bool(re.search(r"\[\d+\]", ans["answer"])), "n_sources": len(ans["sources"])}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_query_log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with self.log_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # -- thumbs up/down on answers ------------------------------------------------ #
    def log_feedback(self, question: str, helpful: bool, model: str = "") -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": question,
               "helpful": bool(helpful), "model": model}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_feedback(self) -> list[dict]:
        if not self.feedback_path.exists():
            return []
        with self.feedback_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def preview_articles(keywords: list[str], top_n: int = 5, contact_email: str = "") -> list[dict]:
    """Search Wikipedia only (no download): the articles each keyword would bring in.
    Returns rows {keyword, article, keep}; the first `top_n` per keyword are pre-selected."""
    from wikipedia_kb import USER_AGENT, WikipediaClient, parse_keywords
    ua = USER_AGENT if not contact_email else f"DynamicAIKnowledgeAssistant/1.0 (contact: {contact_email})"
    client, rows = WikipediaClient(ua), []
    for kw in parse_keywords(keywords):
        try:
            titles = client.search(kw, limit=top_n * 2)
        except Exception as e:
            raise PipelineError(f"Could not search Wikipedia for '{kw}': {e}") from e
        titles = [t for t in titles if not t.lower().startswith(("list of", "index of", "outline of"))]
        rows += [{"keep": i < top_n, "keyword": kw, "article": t} for i, t in enumerate(titles)]
    return rows


def build_pipeline(ws: Workspace, *, keywords: list[str] | None = None, source: str = "wikipedia",
                   append: bool = False, selected: dict[str, list[str]] | None = None,
                   import_dir: str | None = None, top_n: int = 5, strategy: str = "recursive",
                   size: int = 1000, overlap: int = 200, embed_model: str = "all-MiniLM-L6-v2",
                   backend: str = "faiss", min_words: int = 100, contact_email: str = "",
                   on_progress=None, on_log=print) -> dict:
    """Run the whole pipeline and return the stats dict (also saved to pipeline_stats.json).

    source = "wikipedia"  search + download `keywords` (replaces the workspace's articles)
           = "import"     copy .txt articles from `import_dir` (e.g. your Dataset_Builder folder)
           = "reuse"      keep the existing cleaned articles; only re-chunk / re-embed / re-index

    append=True adds the new articles to the workspace's existing ones instead of replacing them.
    selected={keyword: [titles]} downloads exactly those articles (from preview_articles) instead of the top-N.
    """
    from kb_chunking import chunk_corpus
    from kb_clean import clean_knowledge_base
    from kb_embed import build_vector_db
    from wikipedia_kb import USER_AGENT, build_knowledge_base

    ws.ensure()
    prev = ws.stats() if source == "reuse" else {}
    progress = on_progress or (lambda frac, label: None)
    starts = [sum(w for _, w in STAGES[:i]) for i in range(len(STAGES))]
    stage = lambda i, inner=0.0: progress(starts[i] + STAGES[i][1] * inner, STAGES[i][0])
    stats: dict = {"source": source, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    t_all = time.perf_counter()

    # ---- 1. download / import --------------------------------------------- #
    if source == "reuse":
        if not any(ws.clean.glob("*.txt")):
            raise PipelineError("This workspace has no cleaned articles yet - build or import first.")
        keep = ("keywords", "top_n", "download_time_s", "clean_time_s", "n_raw", "n_articles", "n_removed")
        stats.update({k: prev[k] for k in keep if k in prev})
    else:
        stage(0)
        t0 = time.perf_counter()
        if source == "wikipedia":
            keywords = [k.strip() for k in (list(selected) if selected else keywords or []) if k.strip()]
            if not keywords:
                raise PipelineError("Enter at least one keyword.")
            ua = USER_AGENT if not contact_email else f"DynamicAIKnowledgeAssistant/1.0 (contact: {contact_email})"
            seen = {"n": 0}

            def log(msg: str) -> None:
                on_log(msg)
                if msg.startswith("[search]"):
                    seen["n"] += 1
                    stage(0, (seen["n"] - 1) / len(keywords))

            before = ws.stats() if append else {}
            build_knowledge_base(keywords, top_n, str(ws.raw), fresh=not append, selected=selected, user_agent=ua,
                                 on_progress=log)
            old_kw = before.get("keywords", []) if append else []
            stats.update(keywords=list(dict.fromkeys([*old_kw, *keywords])), top_n=top_n)
        elif source == "import":
            src = Path(import_dir or "")
            files = sorted(src.glob("*.txt")) if src.is_dir() else []
            if not files:
                raise PipelineError(f"No .txt files found in '{import_dir}'.")
            shutil.rmtree(ws.raw, ignore_errors=True)
            ws.raw.mkdir(parents=True)
            for p in files:
                shutil.copy2(p, ws.raw / p.name)
            if (src / "metadata.csv").exists():
                shutil.copy2(src / "metadata.csv", ws.raw / "metadata.csv")
            on_log(f"Imported {len(files)} articles from {src}")
            stats.update(keywords=[], top_n=None)
        else:
            raise PipelineError(f"Unknown source {source!r}")
        stats["download_time_s"] = round(time.perf_counter() - t0, 2)
        stats["n_raw"] = len(list(ws.raw.glob("*.txt")))
        if stats["n_raw"] == 0:
            raise PipelineError("No articles were downloaded. Check your keywords and internet connection "
                                "(the log lists any errors).")

        # ---- 2. clean + metadata ------------------------------------------ #
        stage(1)
        t0 = time.perf_counter()
        rows, report = clean_knowledge_base(str(ws.raw), str(ws.clean), min_words=min_words, on_progress=on_log)
        stats.update(clean_time_s=round(time.perf_counter() - t0, 2), n_articles=len(rows),
                     n_removed=sum(r["status"] == "removed" for r in report))
        if not rows:
            raise PipelineError("Every article was removed during cleaning (too short or duplicates).")

    # ---- 3. chunk ----------------------------------------------------------- #
    stage(2)
    ch = chunk_corpus(str(ws.clean), str(ws.chunks), strategy, size, overlap)
    on_log(f"Chunked {ch['n_articles']} articles into {ch['n_chunks']} chunks ({strategy})")
    stats.update(chunk_time_s=ch["chunk_time_s"], n_chunks=ch["n_chunks"])

    # ---- 4. embed + index --------------------------------------------------- #
    stage(3)
    db = build_vector_db(ch["path"], str(ws.vectordb), embed_model, backend, emb_dir=str(ws.embeddings),
                         on_progress=on_log)
    stats.update(embed_time_s=db["embed_time_s"], index_time_s=db["index_time_s"],
                 total_time_s=round(time.perf_counter() - t_all, 2),
                 config={"strategy": strategy, "size": size, "overlap": overlap,
                         "embed_model": embed_model, "backend": backend, "min_words": min_words},
                 active_db=db["path"])
    ws.stats_path.write_text(json.dumps(stats, indent=2))
    ws.set_active_db(db["path"])
    progress(1.0, "Done")
    return stats
