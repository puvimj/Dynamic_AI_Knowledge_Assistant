"""Shared helpers for the Streamlit pages: workspace access, cached loaders, the sidebar."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from kb_pipeline import Workspace, list_workspaces, safe_name

from . import ui

ASSETS = Path(__file__).parent / "assets"


def current_ws() -> Workspace:
    return Workspace(st.session_state.get("ws_name", "default"))


def pipeline_status(ws_name: str) -> dict:
    """Which pipeline stages exist for a workspace (used by the home page and the sidebar)."""
    ws = Workspace(ws_name)
    has = lambda d, pat: d.exists() and any(d.glob(pat))
    metrics = dict(ws.stats())
    cfg = dict(metrics.get("config", {}))
    cfg.setdefault("embedding_model", cfg.get("embed_model", "-"))
    cfg.setdefault("vector_db", cfg.get("backend", "-"))
    metrics["config"] = cfg
    db = ws.active_db()
    return {"has_raw": has(ws.raw, "*.txt"), "has_clean": has(ws.clean, "*.txt"),
            "has_chunks": has(ws.chunks, "*.jsonl"), "has_db": db is not None,
            "db_path": str(db) if db else None, "metrics": metrics}


def sidebar_workspace() -> str:
    """Workspace picker + status card. Every workspace is a separate knowledge base, so a new
    domain never needs code changes. Returns the active workspace name (st.session_state['ws_name'])."""
    names = sorted({*list_workspaces(), "default"})
    current = st.session_state.get("ws_name")
    if current not in names:
        current = "default"
    with st.sidebar:
        current = st.selectbox("Workspace", names, index=names.index(current),
                               help="A workspace is one knowledge base with its own articles, vector database "
                                    "and chat history. Use a new one for a different topic.")
        with st.expander("New workspace"):
            new = st.text_input("Name", placeholder="for example: cybersecurity", key="new_ws_name")
            if st.button("Create workspace", disabled=not safe_name(new or "")):
                ws = Workspace(new).ensure()
                st.session_state["ws_name"] = ws.name
                st.rerun()
        with st.expander("Delete this workspace"):
            st.caption(f"Permanently removes everything in {current}: articles, vector databases and chat history.")
            sure = st.checkbox("Yes, delete it", key=f"confirm_delete_{current}")
            if st.button("Delete workspace", icon=":material/delete:", disabled=not sure):
                clear_caches_and_chat()                      # release open index files first (needed on Windows)
                if Workspace(current).delete():
                    st.session_state["ws_name"] = "default"
                    st.rerun()
                st.error("Some files are still in use. Stop the app, delete the folder by hand, and start again.")
        s = pipeline_status(current)
        m = s["metrics"]
        if s["has_db"]:
            body = (f'<span class="dot on"></span><b>Ready</b><br>{m.get("n_articles", "?")} articles, '
                    f'{m.get("n_chunks", "?")} chunks')
        else:
            body = '<span class="dot off"></span><b>Nothing built yet</b><br>Start in Dataset Builder'
        ui.html_block(f'<div class="kb-status">{body}</div>')
    st.session_state["ws_name"] = current
    return current


# --- cached loaders (keyed by file mtime so a rebuild invalidates them) ---------------------------- #
@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv(path: Path) -> pd.DataFrame | None:
    return _read_csv(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_resource(show_spinner="Loading vector database...")
def _retriever(db_path: str, mtime: float):
    from kb_retrieval import Retriever
    return Retriever.load(db_path)


def load_retriever(db: Path):
    return _retriever(str(db), (db / "store.json").stat().st_mtime)


@st.cache_data(show_spinner=False)
def _chunks_df(db_path: str, mtime: float) -> pd.DataFrame:
    rows = []
    with open(Path(db_path) / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            text = c["text"].replace("\n", " ")
            rows.append({"chunk_id": c["chunk_id"], "source": c["source"], "article": c["article"],
                         "chars": len(c["text"]), "words": len(c["text"].split()),
                         "preview": text[:170] + ("..." if len(text) > 170 else "")})
    return pd.DataFrame(rows)


def load_chunks_df(db: Path) -> pd.DataFrame:
    return _chunks_df(str(db), (db / "chunks.jsonl").stat().st_mtime)


def db_label(db: Path) -> str:
    s = json.loads((db / "store.json").read_text())
    return f"{s['model']} on {s['backend'].upper()}, {s['strategy']} chunks ({s['n_chunks']})"


def require_index(ws: Workspace) -> Path | None:
    """The active vector DB, or show a hint and return None."""
    db = ws.active_db()
    if db is None:
        ui.empty_state("No knowledge base in this workspace yet",
                       "Open Dataset Builder, enter a few keywords, and build one. It takes about a minute.")
        st.page_link("pages/1_Dataset_Builder.py", label="Open Dataset Builder", icon=":material/library_add:")
    return db


def clear_caches_and_chat() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()
    for k in ("bot", "bot_sig", "messages", "chat_db"):
        st.session_state.pop(k, None)
