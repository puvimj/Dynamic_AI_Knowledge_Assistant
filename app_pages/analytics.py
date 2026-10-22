"""Analytics: how the library is built, how the embeddings look, and how the assistant performs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from . import ui
from .common import current_ws, load_chunks_df, load_retriever, require_index


@st.cache_data(show_spinner="Computing embedding statistics...")
def _embedding_stats(db_path: str, mtime: float, max_points: int = 1500) -> dict:
    retr = load_retriever(Path(db_path))
    V = retr.store.vectors()
    articles = np.array([c["article"] for c in retr.chunks])
    n = len(V)
    rng = np.random.default_rng(0)
    norms = np.linalg.norm(V, axis=1)
    i, j = rng.integers(0, n, 5000), rng.integers(0, n, 5000)
    keep = i != j
    cos = np.einsum("ij,ij->i", V[i[keep]], V[j[keep]]) if keep.any() else np.array([])
    out = {"n": n, "dim": int(V.shape[1]), "norm_mean": float(norms.mean()),
           "mean_cosine": float(cos.mean()) if len(cos) else float("nan"), "cosines": cos, "projection": None}
    if n >= 3:
        idx = rng.choice(n, min(n, max_points), replace=False)
        X = V[idx] - V[idx].mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ vt[:2].T
        snippets = [retr.chunks[k]["text"].replace("\n", " ")[:110] for k in idx]
        out["projection"] = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "article": articles[idx],
                                          "snippet": snippets})
    return out


def _overview(ws, store: dict, stats: dict, log: pd.DataFrame) -> None:
    ui.stat_cards([
        {"label": "Articles", "value": store["n_articles"]},
        {"label": "Chunks", "value": f"{store['n_chunks']:,}", "tone": "cool"},
        {"label": "Vector database size", "value": f"{store['index_size_mb']:.2f} MB", "note": store["backend"], "tone": "cool"},
        {"label": "Embedding dimension", "value": store["dim"], "tone": "cool"},
        {"label": "Average retrieval", "value": f"{log['retrieval_ms'].mean():.0f} ms" if len(log) else "no chats yet", "tone": "good"},
        {"label": "Average response", "value": f"{(log['retrieval_ms'] + log['generation_ms']).mean() / 1000:.1f} s"
         if len(log) else "no chats yet", "tone": "good"}])
    ui.section("Where the build time went", "Hover a bar for the exact time.")
    stages = {"Download": stats.get("download_time_s"), "Cleaning": stats.get("clean_time_s"),
              "Chunking": stats.get("chunk_time_s"), "Embedding": store["embed_time_s"], "Indexing": store["index_time_s"]}
    perf = pd.DataFrame([{"stage": k, "seconds": v} for k, v in stages.items() if v is not None])
    left, right = st.columns([2, 1])
    with left:
        fig = px.bar(perf, x="seconds", y="stage", orientation="h", color_discrete_sequence=[ui.SERIES[0]])
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_traces(hovertemplate="%{y}: %{x:.2f} s<extra></extra>")
        ui.show_chart(ui.style_fig(fig, height=270), "an_stages")
    with right:
        st.dataframe(perf.round({"seconds": 2}), hide_index=True)
    ui.section("Vector databases in this workspace", "Each chunking, model and database combination you built.")
    rows = [json.loads((p / "store.json").read_text()) for p in ws.list_dbs()]
    st.dataframe(pd.DataFrame(rows)[["strategy", "model", "backend", "n_chunks", "dim", "index_size_mb",
                                     "embed_time_s", "index_time_s", "built_at"]], hide_index=True)


def _chunks(chunks: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        ui.section("Chunk length")
        bins = st.slider("Number of bins", 5, 40, min(20, max(5, len(chunks) // 10)))
        fig = px.histogram(chunks, x="chars", nbins=bins, color_discrete_sequence=[ui.SERIES[0]],
                           labels={"chars": "Characters per chunk"})
        fig.update_traces(hovertemplate="%{x} characters: %{y} chunks<extra></extra>")
        fig.update_layout(yaxis_title="Chunks")
        ui.show_chart(ui.style_fig(fig, height=320), "an_hist")
        st.caption(f"Mean {chunks['chars'].mean():.0f}, median {chunks['chars'].median():.0f}, "
                   f"min {chunks['chars'].min()}, max {chunks['chars'].max()} characters.")
    with right:
        ui.section("Chunks per article", "Click a bar to see that article's chunks.")
        per = chunks["article"].value_counts().head(25).rename_axis("article").reset_index(name="chunks")
        fig = px.bar(per, x="chunks", y="article", orientation="h", color_discrete_sequence=[ui.SERIES[2]])
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_traces(hovertemplate="%{y}: %{x} chunks<extra></extra>")
        event = ui.show_chart(ui.style_fig(fig, height=max(320, 22 * len(per) + 40)), "an_per_article", select=True)
    points = event.selection.points if event is not None else []
    if points:
        name = points[0].get("y")
        ui.section(f"Chunks from {name}")
        st.dataframe(chunks[chunks["article"] == name][["chunk_id", "words", "chars", "preview"]], hide_index=True)


def _embeddings(db: Path, store: dict, chunks: pd.DataFrame) -> None:
    emb = _embedding_stats(str(db), (db / "store.json").stat().st_mtime)
    ui.stat_cards([
        {"label": "Model", "value": store["model"].split("/")[-1]},
        {"label": "Vectors by dimension", "value": f"{emb['n']:,} x {emb['dim']}", "tone": "cool"},
        {"label": "Vector length (mean)", "value": f"{emb['norm_mean']:.3f}", "note": "1.0 means normalised", "tone": "cool"},
        {"label": "Mean similarity of random pairs", "value": f"{emb['mean_cosine']:.3f}",
         "note": "Low means chunks are well spread out", "tone": "cool"}])
    proj = emb["projection"]
    if proj is None:
        st.caption("Too few chunks to draw a map.")
        return
    ui.section("Map of the chunks", "Each dot is a chunk. Chunks about similar things land close together. "
               "Choose up to three articles to highlight.")
    top = chunks["article"].value_counts().index.tolist()
    chosen = st.multiselect("Highlight articles", sorted(chunks["article"].unique()), default=top[:3],
                            max_selections=3, key="an_highlight")
    proj = proj.assign(group=np.where(proj["article"].isin(chosen), proj["article"], "Other"))
    colors = {a: ui.SERIES[i] for i, a in enumerate(chosen)} | {"Other": ui.OTHER}
    fig = px.scatter(proj, x="PC1", y="PC2", color="group", color_discrete_map=colors, custom_data=["article", "snippet"],
                     category_orders={"group": [*chosen, "Other"]})
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color=ui.SURFACE)),
                      hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}...<extra></extra>")
    for t in fig.data:
        if t.name == "Other":
            t.marker.opacity = 0.35
    ui.show_chart(ui.style_fig(fig, height=430, legend=True), "an_pca")
    ui.section("Similarity of random chunk pairs", "A narrow peak near zero means the embeddings separate topics well.")
    fig = px.histogram(pd.DataFrame({"cosine": emb["cosines"]}), x="cosine", nbins=40, color_discrete_sequence=[ui.SERIES[4]])
    fig.update_traces(hovertemplate="similarity %{x:.2f}: %{y} pairs<extra></extra>")
    fig.update_layout(yaxis_title="Pairs")
    ui.show_chart(ui.style_fig(fig, height=260), "an_cos")


def _performance(ws, log: pd.DataFrame) -> None:
    if log.empty:
        ui.empty_state("No questions asked yet", "Use the Chat Assistant and this page fills in.")
        st.page_link("pages/3_Chat_Assistant.py", label="Open Chat Assistant", icon=":material/forum:")
        return
    models = sorted(log["model"].unique())
    pick = st.pills("Model", models, selection_mode="multi", default=models) or models
    view = log[log["model"].isin(pick)].reset_index(drop=True)
    total = view["retrieval_ms"] + view["generation_ms"]
    ui.stat_cards([
        {"label": "Questions", "value": len(view)},
        {"label": "Retrieval, median and slowest 5%", "value": f"{view['retrieval_ms'].median():.0f} / {view['retrieval_ms'].quantile(.95):.0f} ms", "tone": "cool"},
        {"label": "Response, median and slowest 5%", "value": f"{total.median() / 1000:.1f} / {total.quantile(.95) / 1000:.1f} s", "tone": "cool"},
        {"label": "Average confidence", "value": f"{view['confidence'].mean():.0%}", "tone": "good"}])
    left, right = st.columns(2)
    with left:
        ui.section("Retrieval time per question")
        fig = px.line(view.assign(question_number=view.index + 1), x="question_number", y="retrieval_ms", markers=True,
                      color_discrete_sequence=[ui.SERIES[0]], labels={"question_number": "Question", "retrieval_ms": "Milliseconds"},
                      hover_data={"question": True})
        fig.update_traces(line=dict(width=2), marker=dict(size=8), hovertemplate="Question %{x}: %{y:.0f} ms<br>%{customdata[0]}<extra></extra>")
        ui.show_chart(ui.style_fig(fig, height=280), "an_ret")
    with right:
        ui.section("Generation time per question")
        fig = px.line(view.assign(question_number=view.index + 1), x="question_number", y="generation_ms", markers=True,
                      color_discrete_sequence=[ui.SERIES[1]], labels={"question_number": "Question", "generation_ms": "Milliseconds"},
                      hover_data={"question": True})
        fig.update_traces(line=dict(width=2), marker=dict(size=8), hovertemplate="Question %{x}: %{y:.0f} ms<br>%{customdata[0]}<extra></extra>")
        ui.show_chart(ui.style_fig(fig, height=280), "an_gen")
    ui.section("How confident were the answers?")
    fig = px.histogram(view, x="confidence", nbins=10, range_x=[0, 1], color_discrete_sequence=[ui.SERIES[2]])
    fig.update_traces(hovertemplate="confidence %{x}: %{y} questions<extra></extra>")
    fig.update_layout(yaxis_title="Questions", xaxis_tickformat=".0%")
    ui.show_chart(ui.style_fig(fig, height=240), "an_conf")
    fb = pd.DataFrame(ws.read_feedback())
    if len(fb):
        helpful = int(fb["helpful"].sum())
        ui.section("Your ratings")
        ui.stat_cards([{"label": "Marked helpful", "value": helpful, "tone": "good"},
                       {"label": "Marked not helpful", "value": len(fb) - helpful, "tone": "warn"}])
    st.dataframe(view.iloc[::-1][["ts", "question", "task", "model", "retrieval_ms", "generation_ms", "confidence", "refused"]],
                 hide_index=True, height=260)


def render() -> None:
    ws = current_ws()
    ui.hero("Analytics", "How the library was built, how the embeddings are spread, and how the assistant performs.",
            compact=True)
    db = require_index(ws)
    if db is None:
        return
    store = json.loads((db / "store.json").read_text())
    stats = ws.stats()
    chunks = load_chunks_df(db)
    log = pd.DataFrame(ws.read_query_log())

    tab_o, tab_c, tab_e, tab_p = st.tabs(["Overview", "Chunks", "Embeddings", "Chat performance"])
    with tab_o:
        _overview(ws, store, stats, log)
        snapshot = {"workspace": ws.name, "store": store, "build": stats, "questions": len(log)}
        st.download_button("Download snapshot (JSON)", json.dumps(snapshot, indent=2, default=str), "analytics.json",
                           "application/json")
    with tab_c:
        _chunks(chunks)
    with tab_e:
        _embeddings(db, store, chunks)
    with tab_p:
        _performance(ws, log)
