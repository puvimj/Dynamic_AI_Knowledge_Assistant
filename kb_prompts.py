"""
Phase 8 - Prompt Engineering

Five task templates (QA, Summarization, Concept Explanation, Comparison, Example Generation) that all
share one grounding rule: answer ONLY from the numbered context and cite it as [n].

    from kb_prompts import detect_task, build_messages
    task = detect_task("Compare CNNs and transformers")            # -> "compare"
    messages = build_messages(task, question, context, history=[("prev q", "prev a")])

Prompt-engineering experiments (a project deliverable): vary one thing at a time and re-run the same
question set - e.g. `style` ("default" / "concise" / "detailed"), the grounding rule wording, number of
retrieved passages, or few-shot examples - and compare answers and citation accuracy. Edit TASKS / STYLES
below; PROMPT_VERSION is logged so results stay traceable.
"""

from __future__ import annotations

import re

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a knowledge assistant that answers from a retrieved knowledge base.\n"
    "Rules:\n"
    "1. Use ONLY the numbered context passages provided. Do not use outside knowledge for facts.\n"
    "2. Cite the passages you rely on inline as [1], [2], ... after the relevant sentence.\n"
    "3. If the context does not contain enough information, say \"I couldn't find that in the knowledge "
    "base\" and state what is missing. Never guess.\n"
    "4. If passages disagree, say so and cite both."
)

# task -> (display name, instruction appended to the user message)
TASKS: dict[str, tuple[str, str]] = {
    "qa": ("Question Answering",
           "Answer the question directly in 2-5 sentences. Lead with the answer, then give the key supporting "
           "detail."),
    "summarize": ("Summarization",
                  "Summarize what the context says about the topic: a 2-sentence overview followed by 4-6 bullet "
                  "points of the most important facts."),
    "explain": ("Concept Explanation",
                "Explain the concept to a beginner using the context: (1) a one-sentence definition, (2) how it "
                "works in plain language, (3) why it matters. You may add one short analogy if you label it "
                "'Analogy' - everything else must come from the context."),
    "compare": ("Comparison",
                "Compare the items in the question using the context. Give a markdown table (rows = aspects, "
                "columns = items) followed by a 2-sentence takeaway. Write 'not covered' for any cell the context "
                "does not support."),
    "example": ("Example Generation",
                "Give one concrete example (a short code snippet or real-world scenario) that illustrates the "
                "concept. Base it on the context; if the exact example is not in the context, label it "
                "'Illustrative example' and keep it consistent with the cited facts."),
}

STYLES = {
    "default": "",
    "concise": "Keep the whole answer under 80 words.",
    "detailed": "Be thorough: use short headings and cover edge cases the context mentions.",
}

# Order matters: the first matching rule wins.
_RULES = [
    ("compare", r"\b(compare|comparison|difference[s]? between|differ(s|ence)?|versus|vs\.?|better than|"
                r"distinguish|similarities)\b"),
    ("example", r"\b(example[s]?|for instance|show me how|demonstrate|sample code|code (for|to)|use case[s]?)\b"),
    ("summarize", r"\b(summari[sz]e|summary|overview of|tl;?dr|key points|main points|in brief|briefly describe)\b"),
    ("explain", r"\b(explain|how (does|do|is|are)\b.*\bwork|why (does|do|is|are)|intuition|in simple terms|"
                r"eli5|walk me through|break (it )?down)\b"),
]


def detect_task(question: str) -> str:
    """Rule-based intent routing. Returns a key of TASKS ('qa' when nothing else matches)."""
    q = question.lower()
    for task, pattern in _RULES:
        if re.search(pattern, q):
            return task
    return "qa"


def build_messages(task: str, question: str, context: str, history: list[tuple[str, str]] | None = None,
                   style: str = "default", max_history_turns: int = 3) -> list[dict]:
    """Chat messages (system + prior turns + user). History is a list of (question, answer) pairs;
    only the most recent turns are kept so the context window is spent on retrieved passages."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {list(TASKS)}")
    instruction = TASKS[task][1]
    if STYLES.get(style):
        instruction += " " + STYLES[style]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in (history or [])[-max_history_turns:]:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content":
                     f"Context:\n{context}\n\nTask: {instruction}\n\nQuestion: {question}"})
    return messages
