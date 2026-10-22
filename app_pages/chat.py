"""Chat Assistant: multi-turn RAG chat with streaming answers, sources, highlighted context and feedback."""

from __future__ import annotations

import re
from pathlib import Path

import requests
import streamlit as st

from kb_generate import LLMError, RAGAssistant, make_llm
from kb_prompts import TASKS

from . import ui
from .common import current_ws, db_label, load_csv, load_retriever, require_index

LLMS = ["llama3", "gemma", "mistral", "gpt", "extractive"]
OPEN_MODELS = {"llama3", "gemma", "mistral"}
STOP = {"what", "is", "are", "the", "of", "and", "how", "does", "do", "a", "an", "to", "in", "for", "with", "between",
        "compare", "explain", "give", "example", "summarize", "difference"}


@st.cache_data(ttl=30, show_spinner=False)
def _ollama_up() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=0.7).ok
    except requests.RequestException:
        return False


def _terms(question: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", question) if w.lower() not in STOP]


def render_details(ans: dict, question: str = "") -> None:
    """Confidence, source articles and retrieved context for one answer."""
    if ans["refused"]:
        st.warning("The library probably doesn't cover this. Try different wording, or build a library with "
                   "related keywords.")
    ui.html_block(f'<div class="chips">{ui.confidence_badge(ans["confidence"], ans["confidence_label"])}'
                  f'<span class="chip">{esc(TASKS[ans["task"]][0])}</span><span class="chip">{esc(ans["model"])}</span>'
                  f'<span class="chip">retrieval {ans["retrieval_ms"]:.0f} ms</span>'
                  f'<span class="chip">generation {ans["generation_ms"]:.0f} ms</span></div>')
    if ans["sources"]:
        st.caption("Sources. The numbers match the [n] citations in the answer.")
        ui.html_block(ui.source_pills_html(ans["sources"]))
    with st.expander(f"Retrieved passages ({len(ans['context'])})"):
        terms = _terms(question)
        for h in ans["context"]:
            body = h["text"][:900] + ("..." if len(h["text"]) > 900 else "")
            ui.html_block(f'<div class="ctx"><small>[{h["rank"]}] {esc(h["article"])}, similarity {h["score"]:.2f}'
                          f'</small>{ui.highlight(body, terms)}</div>'.replace("\n", " "))


esc = ui.esc


def _settings(ws) -> dict:
    with st.popover("Settings", icon=":material/tune:"):
        st.selectbox("LLM", LLMS, key="llm_name",
                     help="llama3, gemma and mistral run through Ollama on this machine; gpt needs OPENAI_API_KEY; "
                          "extractive quotes the best passage (no LLM).")
        st.slider("Passages to retrieve (k)", 1, 10, 5, key="k")
        st.selectbox("Answer style", ["default", "concise", "detailed"], key="style")
        st.selectbox("Task", ["auto", *TASKS], key="task",
                     format_func=lambda t: "Auto-detect" if t == "auto" else TASKS[t][0])
        dbs = ws.list_dbs()
        active = ws.active_db()
        paths = [str(d) for d in dbs]
        picked = st.selectbox("Vector database", paths, index=paths.index(str(active)) if active else 0,
                              format_func=lambda p: db_label(Path(p)))
        if active is None or str(active) != picked:
            ws.set_active_db(picked)
        with st.expander("Connect another LLM server"):
            base_url = st.text_input("Base URL (any OpenAI-compatible server)", key="base_url",
                                     placeholder="http://localhost:11434/v1")
            model = st.text_input("Model name", key="model_name", placeholder="llama3.1:8b")
            api_key = st.text_input("API key", key="api_key", type="password")
    return {"db": Path(picked), "base_url": base_url or None, "model": model or None, "api_key": api_key or None}


def _get_bot(opts: dict) -> RAGAssistant | None:
    retriever = load_retriever(opts["db"])
    sig = (str(opts["db"]), st.session_state["llm_name"], opts["base_url"], opts["model"], opts["api_key"])
    bot = st.session_state.get("bot")
    if bot is None or st.session_state.get("bot_sig") != sig:
        try:
            llm = make_llm(st.session_state["llm_name"], base_url=opts["base_url"], model=opts["model"],
                           api_key=opts["api_key"])
        except (LLMError, ValueError) as e:
            st.error(str(e))
            return None
        new = RAGAssistant(retriever, llm)
        if bot is not None and st.session_state.get("chat_db") == str(opts["db"]):
            new.history = bot.history                           # keep the conversation across LLM switches
        st.session_state["bot"], st.session_state["bot_sig"] = new, sig
        bot = new
    return bot


def _ask(question: str) -> None:
    st.session_state["pending"] = question


def _feedback(ws, i: int) -> None:
    value = st.session_state.get(f"fb_{i}")
    msgs = st.session_state["messages"]
    if value is not None and i > 0:
        ws.log_feedback(msgs[i - 1]["content"], value == 1, msgs[i]["answer"]["model"])


def _suggestions(ws) -> list[str]:
    meta = load_csv(ws.clean / "metadata.csv")
    if meta is None or meta.empty:
        return []
    names = [re.sub(r"\s*\(.*?\)", "", n) for n in meta["article_name"].tolist()]
    out = [f"What is {names[0]}?"]
    if len(names) > 1:
        out.append(f"Compare {names[0]} and {names[1]}")
    if len(names) > 2:
        out.append(f"Summarize {names[2]}")
    if len(names) > 3:
        out.append(f"Give an example of {names[3]}")
    return out


def _transcript() -> str:
    out = []
    for m in st.session_state.get("messages", []):
        out.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}\n")
        if m.get("answer"):
            out.append("Sources: " + ", ".join(s["article"] for s in m["answer"]["sources"]) + "\n")
    return "\n".join(out)


def render() -> None:
    ws = current_ws()
    ui.hero("Ask your library", "Answers come from your articles, with the passages and sources shown underneath.",
            compact=True)
    if require_index(ws) is None:
        return
    st.session_state.setdefault("llm_name", "llama3" if _ollama_up() else "extractive")

    top = st.columns([3.2, 1.4, 1.3, 1.3])
    with top[1]:
        opts = _settings(ws)
    if st.session_state.get("chat_db") != str(opts["db"]):          # different knowledge base -> fresh chat
        st.session_state["messages"] = []
        st.session_state.pop("bot", None)
        st.session_state["chat_db"] = str(opts["db"])
    st.session_state.setdefault("messages", [])
    with top[2]:
        if st.button("Clear chat", icon=":material/delete:", help="Remove all messages and start a new conversation", disabled=not st.session_state["messages"]):
            st.session_state["messages"] = []
            if st.session_state.get("bot"):
                st.session_state["bot"].reset()
            st.rerun()
    with top[3]:
        st.download_button("Export", _transcript(), "chat.md", "text/markdown", icon=":material/download:",
                           disabled=not st.session_state["messages"])
    llm = st.session_state["llm_name"]
    with top[0]:
        st.caption(f"Answering from {db_label(opts['db'])}. Model: {llm}.")
    if llm in OPEN_MODELS and not opts["base_url"] and not _ollama_up():
        st.warning(f"Ollama isn't running at localhost:11434, so {llm} can't answer. Start it "
                   "(ollama serve, then ollama pull), or choose extractive in Settings.")

    for i, m in enumerate(st.session_state["messages"]):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("answer"):
                render_details(m["answer"], st.session_state["messages"][i - 1]["content"])
                st.feedback("thumbs", key=f"fb_{i}", on_change=_feedback, args=(ws, i))

    prompt = st.chat_input("Ask a question about your library")
    prompt = prompt or st.session_state.pop("pending", None)
    if not prompt:
        if not st.session_state["messages"]:
            ui.section("Not sure what to ask?", "Pick one, or type your own below.")
            sug = _suggestions(ws)
            cols = st.columns(max(len(sug), 1))
            for col, q in zip(cols, sug):
                col.button(q, key=f"sug_{q}", on_click=_ask, args=(q,), width="stretch")
        return

    bot = _get_bot(opts)
    if bot is None:
        return
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            text = st.write_stream(bot.ask_stream(prompt, task=st.session_state["task"], k=st.session_state["k"],
                                                  style=st.session_state["style"]))
        except LLMError as e:
            st.error(str(e))
            st.session_state["messages"].pop()                      # don't keep a question that got no answer
            return
        ans = bot.last_answer.to_dict()
    ws.log_query(ans)
    st.session_state["messages"].append({"role": "assistant", "content": text, "answer": ans})
    st.rerun()
