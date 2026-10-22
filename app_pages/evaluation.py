"""Page 5 - Evaluation: retrieval quality, chunking comparison, model x vector-DB benchmark, answer quality."""

from __future__ import annotations

import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from . import ui
from .common import current_ws, load_csv, load_retriever, require_index

MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "tfidf-lsa"]


def _retrieval_eval(ws, db, n: int) -> dict | None:
    from kb_chunking import _is_hit, build_eval_set, load_docs
    retr = load_retriever(db)
    queries = build_eval_set(load_docs(ws.clean), n)
    if not queries:
        return None
    t0 = time.perf_counter()
    Q = retr.embedder.encode([q["query"] for q in queries], query=True)
    results = retr.store.search(Q, 20)
    ms = 1000 * (time.perf_counter() - t0) / len(queries)
    hits, rr, misses = {1: 0, 3: 0, 5: 0}, 0.0, []
    for q, res in zip(queries, results):
        rank = next((r for r, (idx, _) in enumerate(res, 1) if _is_hit(retr.chunks[idx], q)), None)
        if rank:
            rr += 1 / rank
            for k in hits:
                hits[k] += rank <= k
        if not rank or rank > 5:
            misses.append({"question": q["query"], "expected_file": q["file"]})
    nq = len(queries)
    out = {"db": db.name, "n_queries": nq, "hit@1": hits[1] / nq, "hit@3": hits[3] / nq, "hit@5": hits[5] / nq,
           "mrr": rr / nq, "avg_query_ms": ms, "misses": misses[:50],
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ws.root / "eval_results.json").write_text(json.dumps(out, indent=2))
    return out


def _log_box():
    box, lines = st.empty(), []

    def log(msg: str) -> None:
        lines.append(msg)
        box.code("\n".join(lines[-12:]), language=None)
    return log


def _grouped(df: pd.DataFrame, label: str, cols: list[str], key: str, names: dict) -> None:
    long = df.melt(id_vars=label, value_vars=cols, var_name="metric", value_name="score")
    long["metric"] = long["metric"].map(names)
    fig = px.bar(long, x=label, y="score", color="metric", barmode="group", color_discrete_sequence=ui.SERIES,
                 labels={"score": "Score (higher is better)", label: ""})
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>")
    fig.update_yaxes(range=[0, 1])
    ui.show_chart(ui.style_fig(fig, height=340, legend=True), key)


NAMES = {"hit@1": "Right answer ranked 1st", "hit@5": "Right answer in top 5", "mrr": "Average rank score"}


def render() -> None:
    ws = current_ws()
    ui.hero("Evaluation", "Check how well the assistant finds the right passages, and compare different settings.",
            compact=True)
    db = require_index(ws)
    if db is None:
        return
    cfg = ws.stats().get("config", {})
    tab1, tab2, tab3, tab4 = st.tabs(["Search quality", "Chunking", "Models and databases", "Answers"])

    with tab1:
        st.write("We turn each article section into a question, then check whether the search brings back a passage "
                 "from that section. These automatic questions are a quick test; add your own for the final report.")
        n = st.slider("Number of test questions", 20, 300, 100, 20)
        if st.button("Run retrieval evaluation", type="primary"):
            with st.spinner("Testing the search..."):
                res = _retrieval_eval(ws, db, n)
            if res is None:
                st.warning("Could not make questions. Articles need '## Section' headings with 300+ characters of text.")
        saved = ws.root / "eval_results.json"
        if saved.exists():
            r = json.loads(saved.read_text())
            st.caption(f"Last run {r['ts']} on {r['db']} ({r['n_queries']} questions)")
            ui.stat_cards([
                {"label": "Right passage was first (Hit@1)", "value": f"{r['hit@1']:.0%}", "tone": "good" if r["hit@1"] >= .5 else "warn"},
                {"label": "In the top 3 (Hit@3)", "value": f"{r['hit@3']:.0%}", "tone": "cool"},
                {"label": "In the top 5 (Hit@5)", "value": f"{r['hit@5']:.0%}", "tone": "cool"},
                {"label": "Average rank score (MRR)", "value": f"{r['mrr']:.2f}", "note": "1.00 is perfect", "tone": "cool"},
                {"label": "Time per question", "value": f"{r['avg_query_ms']:.1f} ms"}])
            if r["misses"]:
                with st.expander(f"Questions it missed ({len(r['misses'])} shown)"):
                    st.dataframe(pd.DataFrame(r["misses"]), hide_index=True)

    with tab2:
        st.write("Chunking cuts articles into passages. Compare the three ways of cutting: fixed size, recursive "
                 "(splits at paragraphs and sentences) and semantic (splits when the topic changes).")
        if st.button("Run fixed vs recursive vs semantic", type="primary"):
            from kb_chunking import compare_strategies
            with st.status("Comparing chunking strategies...", expanded=True):
                compare_strategies(str(ws.clean), str(ws.chunks), size=cfg.get("size", 1000),
                                   overlap=cfg.get("overlap", 200), n_queries=150, on_progress=_log_box())
        comp = load_csv(ws.chunks / "chunk_comparison.csv")
        if comp is not None:
            _grouped(comp, "strategy", ["hit@1", "hit@5", "mrr"], "ev_chunk", NAMES)
            with st.expander("See the full table"):
                st.dataframe(comp, hide_index=True)

    with tab3:
        st.write("Try different embedding models (how text becomes numbers) and vector databases (where those numbers "
                 "are stored), then compare accuracy and speed.")
        c1, c2 = st.columns(2)
        models = c1.multiselect("Embedding models", MODELS, default=[cfg.get("embed_model", MODELS[0])])
        backends = c2.multiselect("Vector databases", ["faiss", "chroma"], default=["faiss", "chroma"])
        strategy = cfg.get("strategy", "recursive")
        chunk_file = ws.chunks / f"chunks_{strategy}.jsonl"
        if st.button("Run benchmark", type="primary", disabled=not (models and backends and chunk_file.exists())):
            from kb_embed import benchmark
            st.cache_resource.clear()                                   # indexes are rebuilt; drop open handles
            with st.status("Benchmarking (downloads models on first use)...", expanded=True):
                benchmark(str(chunk_file), str(ws.clean), models, backends, out_root=str(ws.vectordb),
                          n_queries=150, emb_dir=str(ws.embeddings), on_progress=_log_box())
            st.cache_data.clear()
        bench = load_csv(ws.vectordb / "vectordb_comparison.csv")
        if bench is not None:
            bench = bench.assign(combo=bench["model"].str.split("/").str[-1] + " on " + bench["backend"].str.upper())
            left, right = st.columns(2)
            with left:
                ui.section("How accurate")
                _grouped(bench, "combo", ["hit@1", "hit@5", "mrr"], "ev_acc", NAMES)
            with right:
                ui.section("How fast", "Time to answer one search. Lower is better.")
                fig = px.bar(bench, x="combo", y="search_ms", color_discrete_sequence=[ui.SERIES[1]],
                             labels={"combo": "", "search_ms": "Milliseconds"})
                fig.update_traces(hovertemplate="%{x}: %{y:.2f} ms<extra></extra>")
                ui.show_chart(ui.style_fig(fig, height=340), "ev_speed")
            with st.expander("See the full table"):
                st.dataframe(bench.drop(columns="combo"), hide_index=True)

    with tab4:
        log = pd.DataFrame(ws.read_query_log())
        if log.empty:
            ui.empty_state("No questions asked yet", "Ask a few questions in the Chat Assistant, then come back.")
            st.page_link("pages/3_Chat_Assistant.py", label="Open Chat Assistant", icon=":material/forum:")
        else:
            answered = log[~log["refused"]]
            ui.stat_cards([
                {"label": "Questions asked", "value": len(log)},
                {"label": "Answers with sources [n]", "value": f"{answered['cited'].mean():.0%}" if len(answered) else "-",
                 "note": "Answers that point to a passage", "tone": "good"},
                {"label": "Not answered (not sure)", "value": f"{log['refused'].mean():.0%}", "tone": "warn"},
                {"label": "Average confidence", "value": f"{log['confidence'].mean():.0%}", "tone": "cool"}])
            st.caption("Next step for the final report: score faithfulness and relevance of answers with a judge model (for example Ragas).")
