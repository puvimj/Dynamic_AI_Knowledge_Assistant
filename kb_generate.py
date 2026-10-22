#!/usr/bin/env python3
"""
Phase 9 - Response Generation (RAG assistant)

    question -> retrieve (Phase 7) -> prompt (Phase 8) -> LLM -> Answer
    Answer shows: Final Answer, Retrieved Context, Source Articles, Confidence Score

LLM options (all stream tokens):
    llama3 | gemma | mistral   open models served locally by Ollama (http://localhost:11434, OpenAI-style API)
    gpt                        OpenAI (optional) - needs OPENAI_API_KEY
    extractive                 no LLM at all: quotes the best passage (offline demo / pipeline test)
    --backend hf               load the model in-process with transformers (GPU; needs HF access to gated repos)
    --base-url/--model         any other OpenAI-compatible server (Groq, Together, vLLM, LM Studio, ...)

Setup for the open models (once, on the machine running the app):
    ollama pull llama3.1:8b      # or: gemma2:9b   mistral:7b      (edit PRESETS if tags change)

CLI:
    python kb_generate.py --db vectordb/recursive__all-MiniLM-L6-v2__faiss --llm llama3 "What is a CNN?"
    python kb_generate.py --db vectordb/... --llm mistral --chat          # multi-turn chat in the terminal

Python / Streamlit:
    from kb_retrieval import Retriever
    from kb_generate import RAGAssistant, make_llm
    bot = RAGAssistant(Retriever.load(db), make_llm("llama3"))
    ans = bot.ask("Compare CNNs and RNNs", on_token=print)        # ans.answer, ans.context, ans.sources, ans.confidence
    # token streaming in Streamlit: st.write_stream(bot.ask_stream(q)); then use bot.last_answer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

import requests

from kb_prompts import PROMPT_VERSION, TASKS, build_messages, detect_task
from kb_retrieval import Hit, Retriever, confidence_label, pack_context, unique_sources

OLLAMA_URL = "http://localhost:11434/v1"
PRESETS = {
    "llama3":  {"base_url": OLLAMA_URL, "model": "llama3.1:8b", "api_key": "ollama",
                "hf": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    "gemma":   {"base_url": OLLAMA_URL, "model": "gemma2:9b", "api_key": "ollama", "hf": "google/gemma-2-9b-it"},
    "mistral": {"base_url": OLLAMA_URL, "model": "mistral:7b", "api_key": "ollama",
                "hf": "mistralai/Mistral-7B-Instruct-v0.3"},
    "gpt":     {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
}

REFUSAL = ("I couldn't find enough relevant information in the knowledge base to answer that. "
           "Try rephrasing the question, or rebuild the knowledge base with keywords related to this topic.")


_FOLLOW_UP = re.compile(r"^\s*(and|but|so|also|then|what about|how about)\b|"
                        r"\b(it|its|they|them|their|these|those|this|he|she|his|her|former|latter)\b", re.I)


class LLMError(RuntimeError):
    pass


def _merge_system(messages: list[dict], model_name: str, families: tuple[str, ...]) -> list[dict]:
    """Gemma's (and older Mistral) chat templates reject a 'system' role: fold it into the first user turn."""
    if not any(f in model_name.lower() for f in families) or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{sys_text}\n\n{m['content']}"
            break
    return rest


# =========================================================================== #
# LLM backends (each exposes stream(messages) -> iterator of text pieces)
# =========================================================================== #
class OpenAICompatLLM:
    """Any server that speaks POST /chat/completions with SSE streaming (Ollama, OpenAI, Groq, vLLM ...)."""

    def __init__(self, base_url: str, model: str, api_key: str = "", *, temperature: float = 0.2,
                 max_tokens: int = 800, timeout: int = 180):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.name = model

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {"model": self.model, "stream": True, "temperature": self.temperature,
                   "max_tokens": self.max_tokens,
                   "messages": _merge_system(messages, self.model, ("gemma",))}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        hint = (f" Is the server running at {self.base_url}? For Ollama: `ollama serve` and "
                f"`ollama pull {self.model}`.") if "11434" in self.base_url else ""
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers,
                              stream=True, timeout=(10, self.timeout))
        except requests.RequestException as e:
            raise LLMError(f"Cannot reach {self.base_url}: {e}.{hint}") from e
        if r.status_code != 200:
            raise LLMError(f"LLM server returned HTTP {r.status_code}: {r.text[:300]}{hint}")
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                piece = json.loads(data)["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, ValueError):
                continue
            if piece:
                yield piece


class HFLocalLLM:
    """Runs the model inside this Python process with transformers (needs a GPU for 7-9B models;
    4-bit loading via bitsandbytes fits a free Colab T4). Gated models need `huggingface-cli login`."""

    def __init__(self, repo_id: str, *, load_in_4bit: bool = True, temperature: float = 0.2,
                 max_new_tokens: int = 800):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.name, self.temperature, self.max_new_tokens = repo_id, temperature, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(repo_id)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        from threading import Thread
        from transformers import TextIteratorStreamer
        msgs = _merge_system(messages, self.name, ("gemma", "mistral"))
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen = dict(**inputs, streamer=streamer, max_new_tokens=self.max_new_tokens,
                   do_sample=self.temperature > 0, temperature=self.temperature or None)
        t = Thread(target=self.model.generate, kwargs=gen)
        t.start()
        yield from streamer
        t.join()


class ExtractiveLLM:
    """No-LLM fallback: picks the 2 sentences that best answer the question from the retrieved passages and
    cites them [n]. Prefers definitions ("X is a ...") for "what is X" questions. Lets you demo and test the
    whole pipeline without a model server; a real LLM writes much better answers."""
    name = "extractive (no LLM)"
    _STOP = set("a an and are as at be by do does for from how in is it of on or that the this to was what when "
                "where which who why with about between compare explain give example examples summarize summary "
                "tell me define difference differences vs versus".split())
    _DEFINES = re.compile(r"\b(is|are|was|were|refers to|means|stands for|denotes)\b", re.I)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        text = self.answer(messages[-1]["content"])
        for word in text.split(" "):
            yield word + " "

    def answer(self, prompt: str) -> str:
        q = re.search(r"Question:\s*(.*)\s*$", prompt, re.S)
        terms = {t for t in re.findall(r"[a-z0-9]+", (q.group(1) if q else "").lower()) if t not in self._STOP}
        defining = bool(re.match(r"\s*(what|who)\b.*\b(is|are|was|were)\b|\s*define\b", q.group(1), re.I)) if q else False
        cands = []
        for m in re.finditer(r"\[(\d+)\] ([^\n]*)\n(.*?)(?=\n\n\[\d+\] |\n\nTask:|\Z)", prompt.split("Task:")[0], re.S):
            n, article, passage = int(m.group(1)), m.group(2), m.group(3)
            title = {t for t in re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", " ", article).lower()) if t not in self._STOP}
            for pos, sent in enumerate(re.split(r"(?<=[.!?])\s+", passage.strip())):
                sent = re.sub(r"\s+", " ", sent).strip()
                if len(sent) < 30 or any(sent == c[3] for c in cands) or "#" in sent:                       # skip fragments and run-in headings
                    continue
                words = set(re.findall(r"[a-z0-9]+", sent.lower()))
                score = len(terms & words) / max(len(terms), 1)
                if defining and self._DEFINES.search(sent[:120]) and terms & set(re.findall(r"[a-z0-9]+", sent[:60].lower())):
                    score += 0.6
                if title and title <= terms:
                    score += 0.2
                score += 0.15 / (1 + pos) + 0.1 / n
                cands.append((score, n, pos, sent))
        if not cands:
            return "No passage available. [1]"
        best = sorted(cands, key=lambda c: -c[0])[:2]
        return " ".join(f"{sent} [{n}]" for _, n, _, sent in sorted(best, key=lambda c: (c[1], c[2])))


def make_llm(name: str = "llama3", *, backend: str = "api", base_url: str | None = None,
             model: str | None = None, api_key: str | None = None, **kwargs):
    """name: llama3 | gemma | mistral | gpt | extractive  (or any label when --base-url/--model are given)."""
    if name == "extractive":
        return ExtractiveLLM()
    preset = PRESETS.get(name, {})
    if backend == "hf":
        repo = model or preset.get("hf")
        if not repo:
            raise ValueError(f"No Hugging Face repo known for {name!r}; pass model=")
        return HFLocalLLM(repo, **kwargs)
    url, mdl = base_url or preset.get("base_url"), model or preset.get("model")
    if not (url and mdl):
        raise ValueError(f"Unknown LLM {name!r}: choose {list(PRESETS)} or pass base_url and model")
    key = api_key or preset.get("api_key") or os.environ.get(preset.get("api_key_env", ""), "")
    if preset.get("api_key_env") and not key:
        raise LLMError(f"Set the {preset['api_key_env']} environment variable to use {name!r}")
    return OpenAICompatLLM(url, mdl, key, **kwargs)


# =========================================================================== #
# RAG assistant
# =========================================================================== #
@dataclass
class Prepared:
    task: str
    hits: list[Hit]
    confidence: float
    messages: list[dict]
    refused: bool


@dataclass
class Answer:
    question: str
    task: str
    answer: str
    context: list[Hit]                 # passages actually placed in the prompt
    sources: list[dict]                # unique articles with URL, best score, passage numbers
    confidence: float                  # 0-1 retrieval confidence
    confidence_label: str
    model: str
    retrieval_ms: float
    generation_ms: float
    refused: bool = False
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["context"] = [h.to_dict() for h in self.context]
        return d


class RAGAssistant:
    def __init__(self, retriever: Retriever, llm, *, k: int = 5, min_confidence: float = 0.10,
                 max_context_chars: int = 6000, history_turns: int = 3):
        self.retriever, self.llm = retriever, llm
        self.k, self.min_confidence = k, min_confidence
        self.max_context_chars, self.history_turns = max_context_chars, history_turns
        self.history: list[tuple[str, str]] = []
        self.last_answer: Answer | None = None

    def reset(self) -> None:
        self.history.clear()

    def _retrieval_query(self, question: str) -> str:
        """Follow-ups ('what about its limits?', 'and the finally clause?') carry no topic of their own, so
        the previous question is prepended. Only questions that LOOK like follow-ups (pronouns, 'and...',
        'what about...', or very short) are expanded - a fresh short question must not inherit the old topic.
        (A fuller solution rewrites the question with the LLM - an Advanced-RAG upgrade.)"""
        n_words = len(question.split())
        if self.history and n_words < 12 and (n_words < 4 or _FOLLOW_UP.search(question)):
            return f"{self.history[-1][0]} {question}"
        return question

    def prepare(self, question: str, task: str = "auto", *, k: int | None = None,
                style: str = "default") -> Prepared:
        task = detect_task(question) if task == "auto" else task
        k = max(k or self.k, 6) if task == "compare" else (k or self.k)   # comparisons need both sides
        hits = self.retriever.search(self._retrieval_query(question), k)
        conf = self.retriever.confidence(hits)
        context, used = pack_context(hits, self.max_context_chars)
        refused = not hits or conf < self.min_confidence
        messages = build_messages(task, question, context, self.history, style, self.history_turns)
        return Prepared(task, used, conf, messages, refused)

    def ask_stream(self, question: str, task: str = "auto", *, k: int | None = None,
                   style: str = "default", remember: bool = True) -> Iterator[str]:
        """Yields the answer in pieces as the LLM produces them (pass it to st.write_stream).
        Once exhausted, `self.last_answer` holds the full Answer (sources, confidence, timings)."""
        self.last_answer = None
        t0 = time.perf_counter()
        p = self.prepare(question, task, k=k, style=style)
        retrieval_ms = 1000 * (time.perf_counter() - t0)

        t0 = time.perf_counter()
        if p.refused:                                  # don't let the LLM improvise without evidence
            text = REFUSAL
            yield text
        else:
            pieces = []
            for tok in self.llm.stream(p.messages):
                pieces.append(tok)
                yield tok
            text = "".join(pieces).strip()
        gen_ms = 1000 * (time.perf_counter() - t0)

        if remember and not p.refused:
            self.history.append((question, text))
        self.last_answer = Answer(question, p.task, text, p.hits, unique_sources(p.hits), p.confidence,
                                  confidence_label(p.confidence), getattr(self.llm, "name", "llm"),
                                  round(retrieval_ms, 1), round(gen_ms, 1), p.refused)

    def ask(self, question: str, task: str = "auto", *, k: int | None = None, style: str = "default",
            on_token: Callable[[str], None] | None = None, remember: bool = True) -> Answer:
        for tok in self.ask_stream(question, task, k=k, style=style, remember=remember):
            if on_token:
                on_token(tok)
        return self.last_answer


def format_answer(ans: Answer, *, include_answer: bool = True, context_chars: int = 300) -> str:
    """The four things the chatbot must display: answer, retrieved context, sources, confidence."""
    out = []
    if include_answer:
        out += [f"=== Final Answer ({TASKS[ans.task][0]}) ===", ans.answer, ""]
    out += [f"=== Confidence Score ===",
            f"{ans.confidence:.2f} ({ans.confidence_label})  |  model: {ans.model}  |  "
            f"retrieval {ans.retrieval_ms:.0f} ms, generation {ans.generation_ms:.0f} ms", "",
            "=== Source Articles ==="]
    out += [f"{i}. {s['article']} - {s['url'] or 'no url'}  (passages {', '.join(f'[{n}]' for n in s['passages'])}, "
            f"score {s['best_score']:.2f})" for i, s in enumerate(ans.sources, 1)] or ["(none)"]
    out += ["", "=== Retrieved Context ==="]
    for h in ans.context:
        snippet = h.text.replace("\n", " ")
        out.append(f"[{h.rank}] {h.article} (score {h.score:.2f}): {snippet[:context_chars]}"
                   f"{'...' if len(snippet) > context_chars else ''}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9: ask the knowledge base")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--db", required=True, help="vector DB folder from kb_embed.py")
    ap.add_argument("--llm", default="llama3", help="llama3 | gemma | mistral | gpt | extractive")
    ap.add_argument("--backend", default="api", choices=["api", "hf"])
    ap.add_argument("--base-url"), ap.add_argument("--model"), ap.add_argument("--api-key")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--task", default="auto", choices=["auto", *TASKS])
    ap.add_argument("--style", default="default", choices=["default", "concise", "detailed"])
    ap.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
    ap.add_argument("--json", action="store_true", help="print the Answer as JSON")
    a = ap.parse_args()

    bot = RAGAssistant(Retriever.load(a.db),
                       make_llm(a.llm, backend=a.backend, base_url=a.base_url, model=a.model, api_key=a.api_key),
                       k=a.k)

    def run(q: str) -> None:
        if a.json:
            print(json.dumps(bot.ask(q, a.task, style=a.style).to_dict(), indent=2, ensure_ascii=False))
            return
        print("\n=== Final Answer ===")
        ans = bot.ask(q, a.task, style=a.style, on_token=lambda t: print(t, end="", flush=True))
        print("\n\n" + format_answer(ans, include_answer=False))

    if a.chat:
        print("Multi-turn chat. Commands: /reset  /quit")
        while True:
            q = input("\nYou> ").strip()
            if q in ("/quit", "/exit"):
                break
            if q == "/reset":
                bot.reset()
                print("(history cleared)")
            elif q:
                run(q)
    elif a.question:
        run(a.question)
    else:
        ap.error("give a question or use --chat")


if __name__ == "__main__":
    main()
