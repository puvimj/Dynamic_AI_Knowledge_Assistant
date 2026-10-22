"""Dataset Builder: keywords -> download -> clean -> chunk -> embed -> index."""

from __future__ import annotations

import importlib.util

import pandas as pd
import streamlit as st

from kb_pipeline import STAGES, build_pipeline, preview_articles

from . import ui
from .common import clear_caches_and_chat, current_ws, load_csv

EXAMPLES = {
    "Machine learning": "Machine learning\nDeep learning\nComputer vision\nNatural language processing\n"
                        "Transformer (deep learning architecture)\nConvolutional neural network",
    "Space science": "Solar System\nBlack hole\nExoplanet\nGalaxy\nSpace exploration",
    "Human health": "Immune system\nDiabetes\nCardiovascular disease\nVaccination\nNutrition",
}
MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "tfidf-lsa"]
STAGE_NAMES = ["Download", "Clean", "Chunk", "Index"]
STAGE_NOTES = [s[0] for s in STAGES]


def _fill_example(name: str) -> None:
    st.session_state["keywords_text"] = EXAMPLES[name]


def _neural_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _index_settings() -> dict:
    default = 0 if _neural_available() else MODELS.index("tfidf-lsa")
    with st.expander("Advanced settings (optional)"):
        c1, c2, c3 = st.columns(3)
        strategy = c1.selectbox("Chunking strategy", ["recursive", "fixed", "semantic"],
                                help="Compare them on the Evaluation page.")
        size = c2.slider("Chunk size (characters)", 300, 2000, 1000, 100)
        overlap = c3.slider("Overlap (characters)", 0, 500, 200, 50, help="Ignored by semantic chunking.")
        c4, c5, c6 = st.columns(3)
        model = c4.selectbox("Embedding model", MODELS, index=default,
                             help="'tfidf-lsa' works offline but retrieves worse than the neural models.")
        backend = c5.selectbox("Vector database", ["faiss", "chroma"])
        min_words = c6.number_input("Drop articles shorter than (words)", 20, 1000, 100, 10)
        if not _neural_available():
            st.caption("The sentence-transformers package is not installed, so the offline tfidf-lsa model is "
                       "preselected. Install it (pip install sentence-transformers) for better answers.")
    return dict(strategy=strategy, size=size, overlap=min(overlap, size - 100), embed_model=model,
                backend=backend, min_words=int(min_words))


def _run_build(ws, **kwargs) -> None:
    tracker = st.empty()
    bar = st.progress(0.0, text="Starting")
    lines: list[str] = []

    def draw(active: int) -> None:
        with tracker.container():
            ui.html_block(ui.stage_tracker(STAGE_NAMES, active, ["" for _ in STAGE_NAMES]))

    draw(0)
    with st.status("Building your library", expanded=False) as status:
        box = st.empty()

        def on_log(msg: str) -> None:
            lines.append(msg)
            box.code("\n".join(lines[-14:]), language=None)

        def on_progress(frac: float, label: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=label)
            idx = STAGE_NOTES.index(label) if label in STAGE_NOTES else len(STAGE_NAMES)
            draw(idx)

        try:
            stats = build_pipeline(ws, on_progress=on_progress, on_log=on_log, **kwargs)
        except RuntimeError as e:        # PipelineError, missing packages, Wikipedia unreachable: all fixable
            status.update(label="Build stopped", state="error", expanded=True)
            st.error(str(e))
            return
        except Exception as e:
            status.update(label="Build failed", state="error", expanded=True)
            st.exception(e)
            return
        status.update(label="Library ready", state="complete", expanded=False)
    clear_caches_and_chat()
    st.balloons()
    st.success(f"Built {stats['n_articles']} articles and {stats['n_chunks']} chunks in "
               f"{stats['total_time_s']:.1f} seconds.")
    st.page_link("pages/3_Chat_Assistant.py", label="Ask your first question", icon=":material/forum:")


def _last_build(ws) -> None:
    stats = ws.stats()
    if not stats:
        ui.empty_state("Nothing built in this workspace yet",
                       "Pick an example or type your own keywords above, then press Build knowledge base.")
        return
    ui.section("Last build", "What the most recent run produced.")
    ui.stat_cards([
        {"label": "Articles kept", "value": stats.get("n_articles", "-"),
         "note": f"{stats.get('n_removed', 0)} removed while cleaning"},
        {"label": "Chunks", "value": f"{stats.get('n_chunks', 0):,}", "tone": "cool"},
        {"label": "Download time", "value": f"{stats.get('download_time_s', 0):.1f}s", "tone": "cool"},
        {"label": "Total build time", "value": f"{stats.get('total_time_s', 0):.1f}s", "tone": "cool"}])
    ui.pills(stats.get("keywords", []), "brass")

    meta = load_csv(ws.clean / "metadata.csv")
    if meta is not None:
        cols = [c for c in ["article_name", "category", "keywords", "word_count", "character_count",
                            "download_date", "url"] if c in meta.columns]
        cfg = {"article_name": "Article", "category": "Category", "keywords": "Found by keyword",
               "word_count": st.column_config.ProgressColumn("Words", format="%d", min_value=0,
                                                             max_value=int(meta["word_count"].max())),
               "character_count": "Characters", "download_date": "Downloaded",
               "url": st.column_config.LinkColumn("Wikipedia", display_text="Open")}
        meta = meta.assign(keywords=meta["keywords"].fillna("").str.replace("|", ", ", regex=False)) if "keywords" in meta.columns else meta
        st.dataframe(meta[cols], hide_index=True, column_config={k: v for k, v in cfg.items() if k in cols})
        st.download_button("Download metadata.csv", meta.to_csv(index=False), "metadata.csv", "text/csv")
    report = load_csv(ws.clean / "cleaning_report.csv")
    if report is not None and (report["status"] == "removed").any():
        with st.expander(f"{(report['status'] == 'removed').sum()} files removed during cleaning"):
            st.dataframe(report[report["status"] == "removed"][["file", "reason", "detail"]], hide_index=True)
    errors = ws.raw / "error_log.csv"
    if errors.exists():
        with st.expander("Download errors"):
            st.dataframe(load_csv(errors), hide_index=True)


def render() -> None:
    ws = current_ws()
    ui.hero("Build a library", f"Workspace {ws.name}. Choose a topic, and the builder downloads the best Wikipedia "
            "articles, cleans them and makes them searchable. Use a new workspace for a different topic.",
            compact=True)
    settings = _index_settings()
    tab_wiki, tab_import, tab_reindex = st.tabs(["From Wikipedia", "Import articles", "Re-index"])

    with tab_wiki:
        st.caption("Start from an example, or type your own keywords.")
        cols = st.columns(len(EXAMPLES) + 2)
        for col, name in zip(cols, EXAMPLES):
            col.button(name, key=f"ex_{name}", on_click=_fill_example, args=(name,), width="stretch")
        keywords_text = st.text_area("Keywords (one per line, or comma-separated)", key="keywords_text", height=150,
                                     placeholder="Machine learning\nDeep learning\nComputer vision\n...",
                                     help="After typing, click outside the box or press Ctrl+Enter so the app sees your keywords.")
        keywords = [k.strip() for line in keywords_text.splitlines() for k in line.split(",") if k.strip()]
        ui.pills(keywords, "brass")
        c1, c2 = st.columns(2)
        top_n = c1.slider("Articles per keyword", 1, 15, 3,
                          help="The closest matches Wikipedia finds for each keyword. Lower means fewer off-topic articles.")
        email = c2.text_input("Contact email for Wikipedia's API (recommended)", placeholder="you@example.com",
                              help="Wikimedia asks API clients to identify themselves; anonymous clients get "
                                   "rate-limited (HTTP 429) sooner.")
        has_library = any(ws.clean.glob("*.txt"))
        append = False
        if has_library:
            append = st.checkbox("Add to the existing library (keep the articles already here)", value=True, key="add_mode",
                                 help="Untick to replace everything in this workspace with the new download.")
            st.caption(("New articles are added to the current ones." if append else
                        "The current articles in this workspace will be replaced.") + " For a separate topic, "
                       "create a new workspace in the sidebar instead.")

        b1, b2, _ = st.columns([1.3, 1.5, 3])
        if b1.button("Preview articles", icon=":material/search:",
                     help="Search Wikipedia and list the articles before downloading, so you can untick off-topic ones."):
            if not keywords:
                st.warning("Type at least one keyword first.")
            else:
                try:
                    with st.spinner("Searching Wikipedia..."):
                        st.session_state["preview"] = {"sig": (tuple(keywords), top_n),
                                                       "rows": preview_articles(keywords, top_n, email)}
                except RuntimeError as e:
                    st.error(str(e))
        build_clicked = b2.button("Build knowledge base", type="primary")

        selected = None
        prev = st.session_state.get("preview")
        if prev and prev["sig"] == (tuple(keywords), top_n):
            ui.section("Choose the articles to download", "Untick anything that is off-topic. Only ticked articles are downloaded.")
            edited = st.data_editor(pd.DataFrame(prev["rows"]), hide_index=True, key="preview_editor", height=min(420, 38 * len(prev["rows"]) + 40),
                                    disabled=["keyword", "article"],
                                    column_config={"keep": st.column_config.CheckboxColumn("Download", width="small"),
                                                   "keyword": "Keyword", "article": "Wikipedia article"})
            selected = {}
            for kw, art in edited[edited["keep"]][["keyword", "article"]].itertuples(index=False):
                selected.setdefault(kw, []).append(art)
            st.caption(f"{sum(len(v) for v in selected.values())} of {len(edited)} articles ticked.")
        elif prev:
            st.info("The keywords or the article count changed. Press Preview articles again, or build directly.")
        elif keywords:
            est = len(keywords) * top_n
            st.caption(f"Without a preview, up to {est} articles are downloaded (about {max(1, round(est * 1.2 / 60))} minute(s)).")

        if build_clicked:
            if not keywords:
                st.warning("Type at least one keyword (or press an example above) first. If you just typed, click outside the box once.")
            elif selected is not None and not selected:
                st.warning("Tick at least one article to download.")
            else:
                _run_build(ws, source="wikipedia", keywords=keywords, top_n=top_n, contact_email=email,
                           append=append, selected=selected, **settings)

    with tab_import:
        st.write("Already have articles, for example the AI_Knowledge_Base folder from your notebook? "
                 "Import them instead of downloading again.")
        folder = st.text_input("Folder on this machine containing .txt articles", placeholder="AI_Knowledge_Base")
        uploads = st.file_uploader("Or upload .txt files", type="txt", accept_multiple_files=True)
        if st.button("Import and build index", disabled=not (folder or uploads)):
            import_dir = folder
            if uploads:
                up = ws.root / "_upload"
                up.mkdir(parents=True, exist_ok=True)
                for old in up.glob("*"):
                    old.unlink()
                for f in uploads:
                    (up / f.name).write_bytes(f.getvalue())
                import_dir = str(up)
            _run_build(ws, source="import", import_dir=import_dir, **settings)

    with tab_reindex:
        st.write("Keep the downloaded articles and change only how they are chunked, embedded or stored. "
                 "Every combination is kept, so you can compare them on the Evaluation page.")
        if st.button("Re-chunk, embed and index", disabled=not any(ws.clean.glob("*.txt"))):
            _run_build(ws, source="reuse", **settings)

    _last_build(ws)
