"""Home: where the library stands and the next thing to do."""

from __future__ import annotations

import streamlit as st

from . import ui
from .common import current_ws, pipeline_status

STATIONS = [("Search", "Wikipedia finds the best articles for each keyword"),
            ("Download", "Top articles saved as plain text, duplicates skipped"),
            ("Clean", "HTML, odd characters and empty pages removed"),
            ("Chunk", "Articles split into passages"),
            ("Index", "Passages embedded and stored in FAISS or Chroma"),
            ("Answer", "The best passages go to the LLM; every answer cites them")]


def render() -> None:
    ws = current_ws()
    s = pipeline_status(ws.name)
    m = s["metrics"]
    cfg = m.get("config", {})

    if s["has_db"]:
        topics = m.get("keywords") or []
        about = f" on {', '.join(topics[:3])}{' and more' if len(topics) > 3 else ''}" if topics else ""
        ui.hero("Your library is ready",
                f"{m.get('n_articles', 'Several')} Wikipedia articles{about}, indexed and waiting. Ask a question "
                "and the answer comes with the passages and articles it was built from.")
        a, b, _ = st.columns([1.2, 1.2, 3])
        a.page_link("pages/3_Chat_Assistant.py", label="Ask a question", icon=":material/forum:")
        b.page_link("pages/2_Knowledge_Base.py", label="Browse articles", icon=":material/menu_book:")
        ui.stat_cards([
            {"label": "Articles", "value": m.get("n_articles", "-")},
            {"label": "Passages (chunks)", "value": f"{m.get('n_chunks', 0):,}", "tone": "cool"},
            {"label": "Embedding model", "value": str(cfg.get("embedding_model", "-")).split("/")[-1], "tone": "cool"},
            {"label": "Vector database", "value": str(cfg.get("vector_db", "-")).upper(), "tone": "cool"}])
        if topics:
            ui.section("Topics in this library")
            ui.pills(topics)
    else:
        ui.hero("Build a library on any topic, then ask it questions",
                "Type a few keywords. The assistant downloads the best matching Wikipedia articles, indexes them, "
                "and answers from them, showing its sources every time.")
        st.page_link("pages/1_Dataset_Builder.py", label="Build your library", icon=":material/library_add:")

    ui.section("How it works", "Each step runs for you when you press Build.")
    done = sum([s["has_raw"] * 2, s["has_clean"], s["has_chunks"], s["has_db"] * 2])
    done = len(STATIONS) if s["has_db"] else min(done, len(STATIONS) - 1)
    ui.html_block(ui.stage_tracker([n for n, _ in STATIONS], done, [d for _, d in STATIONS]))

    ui.section("Where to go")
    cols = st.columns(4)
    tiles = [("pages/1_Dataset_Builder.py", "Dataset Builder", "Pick keywords and build or rebuild the library.", ":material/library_add:"),
             ("pages/2_Knowledge_Base.py", "Knowledge Base", "Read the articles and check what was kept.", ":material/menu_book:"),
             ("pages/4_Analytics.py", "Analytics", "Explore chunks, embeddings and response times.", ":material/monitoring:"),
             ("pages/5_Evaluation.py", "Evaluation", "Test retrieval and compare settings.", ":material/fact_check:")]
    for col, (path, label, note, icon) in zip(cols, tiles):
        with col:
            st.page_link(path, label=label, icon=icon)
            st.caption(note)
