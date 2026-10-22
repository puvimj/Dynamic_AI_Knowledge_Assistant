"""Knowledge Base: browse, search and read what the library contains."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from . import ui
from .common import current_ws, load_csv


def _bar(series: pd.Series, key: str, label: str) -> None:
    df = series.rename_axis(label).reset_index(name="Articles").sort_values("Articles")
    fig = px.bar(df, x="Articles", y=label, orientation="h", color_discrete_sequence=[ui.SERIES[0]])
    fig.update_traces(hovertemplate="%{y}: %{x} articles<extra></extra>")
    fig.update_xaxes(dtick=1)
    ui.show_chart(ui.style_fig(fig, height=max(200, 34 * len(df) + 60)), key)


def render() -> None:
    ws = current_ws()
    meta = load_csv(ws.clean / "metadata.csv")
    if meta is None or meta.empty:
        ui.hero("Knowledge Base", "Everything the assistant can read and cite.", compact=True)
        ui.empty_state(f"Workspace {ws.name} has no articles yet",
                       "Build a library first, then come back to browse it.")
        st.page_link("pages/1_Dataset_Builder.py", label="Open Dataset Builder", icon=":material/library_add:")
        return

    db = ws.active_db()
    n_chunks = json.loads((db / "store.json").read_text())["n_chunks"] if db else None
    ui.hero("Knowledge Base", "Everything the assistant can read and cite. Search the articles, open one to read it, "
            "or look at how the library is made up.", compact=True)
    ui.stat_cards([
        {"label": "Articles", "value": len(meta)},
        {"label": "Words", "value": f"{int(meta['word_count'].sum()):,}", "tone": "cool"},
        {"label": "Chunks", "value": f"{n_chunks:,}" if n_chunks else "not indexed", "tone": "cool"},
        {"label": "Categories", "value": meta["category"].nunique(), "tone": "cool"},
        {"label": "Average length", "value": f"{int(meta['word_count'].mean()):,} words", "tone": "cool"}])

    tab_articles, tab_overview, tab_files = st.tabs(["Articles", "Overview", "Files"])

    with tab_articles:
        has_kw = "keywords" in meta.columns and meta["keywords"].notna().any()
        picked: list[str] = []
        if has_kw:
            kws = sorted(meta["keywords"].dropna().str.split("|").explode().str.strip().replace("", pd.NA).dropna().unique())
            picked = st.pills("Found by keyword", kws, selection_mode="multi") or []
        query = st.text_input("Search titles, categories and keywords", placeholder="for example: neural",
                              icon=":material/search:")
        view = meta
        if picked:
            hay = view["keywords"].fillna("")
            view = view[hay.apply(lambda s: any(p in [x.strip() for x in s.split("|")] for p in picked))]
        if query:
            text = view[[c for c in ["article_name", "category", "keywords"] if c in view.columns]]
            view = view[text.fillna("").agg(" ".join, axis=1).str.lower().str.contains(query.lower(), regex=False)]
        view = view.reset_index(drop=True)
        if "keywords" in view.columns:
            view = view.assign(keywords=view["keywords"].fillna("").str.replace("|", ", ", regex=False))
        st.caption(f"{len(view)} of {len(meta)} articles. Select a row to read it.")
        if view.empty:
            ui.empty_state("No articles match", "Clear the keyword filter or try a different search.")
            return
        cols = [c for c in ["article_name", "category", "keywords", "word_count", "url"] if c in view.columns]
        cfg = {"article_name": "Article", "category": "Category", "keywords": "Found by keyword",
               "word_count": st.column_config.ProgressColumn("Words", format="%d", min_value=0,
                                                             max_value=int(meta["word_count"].max())),
               "url": st.column_config.LinkColumn("Wikipedia", display_text="Open")}
        event = st.dataframe(view[cols], hide_index=True, on_select="rerun", selection_mode="single-row",
                             key="kb_table", column_config={k: v for k, v in cfg.items() if k in cols})
        rows = event.selection.rows if event is not None else []
        row = view.iloc[rows[0] if rows and rows[0] < len(view) else 0]
        ui.section(row["article_name"], f"{int(row['word_count']):,} words, category {row['category']}")
        reader = (ws.clean / row["file"]).read_text(encoding="utf-8")
        with st.container(height=520, border=True):
            ui.html_block(ui.article_html(reader, query))
        st.download_button("Download metadata.csv", meta.to_csv(index=False), "metadata.csv", "text/csv")

    with tab_overview:
        left, right = st.columns(2)
        with left:
            ui.section("Articles per keyword")
            if has_kw:
                kw = meta["keywords"].dropna().str.split("|").explode().str.strip()
                _bar(kw[kw != ""].value_counts(), "kb_kw", "Keyword")
            else:
                st.caption("No keyword information (articles were imported without metadata).")
        with right:
            ui.section("Categories")
            _bar(meta["category"].value_counts().head(12), "kb_cat", "Category")
        ui.section("Article length", "Hover a bar for the exact word count.")
        fig = px.bar(meta.sort_values("word_count"), x="article_name", y="word_count",
                     color_discrete_sequence=[ui.SERIES[2]], labels={"article_name": "Article", "word_count": "Words"})
        fig.update_xaxes(showticklabels=False)
        fig.update_traces(hovertemplate="%{x}<br>%{y:,} words<extra></extra>")
        ui.show_chart(ui.style_fig(fig, height=280), "kb_len")

    with tab_files:
        report = load_csv(ws.clean / "cleaning_report.csv")
        if report is None:
            st.caption("No cleaning report for this workspace.")
        else:
            kept = int((report["status"] != "removed").sum())
            removed = report[report["status"] == "removed"]
            ui.stat_cards([{"label": "Files kept", "value": kept, "tone": "good"},
                           {"label": "Files removed", "value": len(removed), "tone": "warn" if len(removed) else "cool"}])
            if len(removed):
                st.dataframe(removed[["file", "reason", "detail"]], hide_index=True)
            with st.expander("Full cleaning report"):
                st.dataframe(report, hide_index=True)
        errors = ws.raw / "error_log.csv"
        if errors.exists():
            with st.expander("Download errors"):
                st.dataframe(load_csv(errors), hide_index=True)
