#!/usr/bin/env python3
"""
Phase 2 (Data Cleaning) + Phase 3 (Metadata Generation)

Reads the raw Phase 1 output and writes a cleaned copy (raw files are never modified):

    knowledge_base/            (input, from wikipedia_kb.py or your Dataset_Builder notebook)
    knowledge_base_clean/
        Machine_Learning.txt   cleaned text, sections kept as "## Heading"
        ...
        metadata.csv           Phase 3: article_name, url, category, word_count,
                               character_count, download_date (+ file, keywords, raw_word_count)
        cleaning_report.csv    one row per input file: kept / removed + reason
        stopword_analysis.csv  (only with --stopwords)

CLI:
    python kb_clean.py --raw knowledge_base --out knowledge_base_clean
    python kb_clean.py --raw AI_Knowledge_Base --out knowledge_base_clean --stopwords

Notebook / Streamlit:
    from kb_clean import clean_knowledge_base
    rows, report = clean_knowledge_base("knowledge_base", "knowledge_base_clean")

Requires: pip install beautifulsoup4   (optional - a regex fallback is used without it)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:                      # regex fallback keeps the script dependency-free
    BeautifulSoup = None

# Sections that are lists of links/citations rather than knowledge.
NOISE_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes",
    "footnotes", "bibliography", "sources", "citations", "works cited",
    "general references", "notes and references",
}

# Typographic characters -> plain ASCII equivalents (done before NFKC).
CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ",
    "•": "-", "·": "-",
}
_INVISIBLE = re.compile("[​‌‍⁠﻿­]")      # zero-width, BOM, soft hyphen
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")             # keeps \t and \n
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")                    # == Title ==
_CITATION = re.compile(r"\[(?:\d{1,3}|[a-z]|citation needed|clarification needed|when\?|who\?|"
                       r"dubious[^\]]*|according to whom\?)\]", re.I)
_TAG = re.compile(r"<[^>]+>")

METADATA_FIELDS = ["file", "article_name", "url", "category", "keywords",
                   "word_count", "character_count", "raw_word_count", "download_date"]


# --------------------------------------------------------------------------- #
# Individual cleaning steps (each is small and independently testable)
# --------------------------------------------------------------------------- #
def remove_html(text: str) -> str:
    """Strip tags and decode entities (&amp; &nbsp; ...). Drops <script>/<style> content."""
    if "<" in text and ">" in text:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(text, "html.parser")
            for t in soup(["script", "style"]):
                t.decompose()
            text = soup.get_text()
        else:
            text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
            text = _TAG.sub(" ", text)
    return html.unescape(text)


_MATH_OPEN = re.compile(r"\{\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)")


def remove_math_markup(text: str) -> str:
    """Wikipedia plain text leaves {\\displaystyle ...} / {\\textstyle ...} blobs for formulas; drop them
    (nested braces ok)."""
    out, i = [], 0
    while True:
        m = _MATH_OPEN.search(text, i)
        j = m.start() if m else -1
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            depth += (text[k] == "{") - (text[k] == "}")
            k += 1
            if depth == 0:
                break
        i = k
    return "".join(out)


def normalize_unicode(text: str) -> str:
    for src, dst in CHAR_MAP.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)      # ligatures, full-width forms, superscripts...
    text = _INVISIBLE.sub("", text)
    return _CONTROL.sub("", text)


def clean_special_chars(text: str) -> str:
    """Remove emoji / private-use / symbol junk but KEEP letters of any language, digits,
    and technical punctuation (+ # % = < > / \\ ^ _ ~ | $ @ & * -) that ML text needs."""
    text = _CITATION.sub("", text)
    keep = []
    for ch in text:
        if ch in "\n\t" or unicodedata.category(ch)[0] in "LNPZ":
            keep.append(ch)
        elif ch in "+=<>^~|$%&*\\":                 # math/code symbols (category S) worth keeping
            keep.append(ch)
        else:
            keep.append(" ")                        # So / Co / Cn (emoji, icons) -> space
    return "".join(keep)


def restructure_sections(text: str) -> str:
    """'== Heading ==' -> '## Heading' and drop References/See also/... sections."""
    out, skip_level = [], None
    for line in text.split("\n"):
        m = _HEADING.match(line.strip())
        if m:
            level, title = len(m.group(1)) - 1, m.group(2).strip()   # '==' is level 1
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and title.lower() in NOISE_SECTIONS:
                skip_level = level
            if skip_level is None:
                out.append(f"\n{'#' * (level + 1)} {title}\n")
            continue
        if skip_level is None:
            out.append(line)
    return "\n".join(out)


_MATH_WORDS = {"min", "max", "argmax", "argmin", "s.t.", "sup", "inf", "log", "exp", "sign", "subject to"}


def tidy_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and (ln.startswith("#") or not re.fullmatch(r"[\W_]*", ln))]
    # formula residue: Wikipedia plain text puts every symbol of an equation on its own line ("x", "1", "ε", "min")
    lines = [ln for ln in lines if ln.startswith("#") or len(ln) > 2 and ln.lower() not in _MATH_WORDS]
    text = "\n\n".join(lines)          # Wikipedia uses single \n between paragraphs; make them explicit blank lines
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)            # "word ." -> "word."
    text = re.sub(r"\(\s*\)", "", text)                      # empty parens left by removed math/citations
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip() + "\n"


def clean_text(raw: str) -> str:
    text = remove_html(raw)
    text = remove_math_markup(text)
    text = normalize_unicode(text)
    text = restructure_sections(text)
    text = clean_special_chars(text)
    return tidy_whitespace(text)


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #
def _fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def _shingles(text: str, k: int = 5, mod: int = 4) -> set[int]:
    """Subsampled word k-shingle hashes (keeps ~1/mod of them) for fast near-duplicate checks."""
    words = re.findall(r"\w+", text.lower())
    hs = (hash(" ".join(words[i:i + k])) for i in range(max(len(words) - k + 1, 0)))
    return {h for h in hs if h % mod == 0}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# --------------------------------------------------------------------------- #
# Stopword analysis (optional)
# --------------------------------------------------------------------------- #
def stopword_analysis(texts: list[str], out_csv: Path, top: int = 25) -> dict:
    """Corpus-level stopword share + top words with and without stopwords.

    NOTE: this is analysis only. Do NOT strip stopwords from the text you embed - modern
    embedding models and LLMs use them. It matters for keyword search (BM25/TF-IDF).
    """
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP
    except ImportError:
        STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the "
                         "to was were will with this which not but also can their they been than "
                         "into such these other more most".split())
    words = [w for t in texts for w in re.findall(r"[a-z][a-z'-]+", t.lower())]
    counts = Counter(words)
    stop_total = sum(c for w, c in counts.items() if w in STOP)
    content = Counter({w: c for w, c in counts.items() if w not in STOP})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "top_word_all", "count_all", "top_word_no_stopwords", "count_no_stopwords"])
        for i, ((a, ca), (b, cb)) in enumerate(zip(counts.most_common(top), content.most_common(top)), 1):
            w.writerow([i, a, ca, b, cb])
    return {"total_words": len(words), "unique_words": len(counts),
            "stopword_share": round(stop_total / max(len(words), 1), 3),
            "top_content_words": [w for w, _ in content.most_common(10)]}


# --------------------------------------------------------------------------- #
# Phase 1 metadata reader (supports my wikipedia_kb.py AND your notebook's format)
# --------------------------------------------------------------------------- #
def _read_phase1_metadata(raw_dir: Path) -> dict[str, dict]:
    path = raw_dir / "metadata.csv"
    if not path.exists():
        return {}
    rename = {"File": "file", "Title": "article_name", "URL": "url",
              "Words": "word_count", "Characters": "character_count"}
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r = {rename.get(k, k): v for k, v in r.items()}
            if r.get("file"):
                rows[r["file"]] = r
    return rows


def _file_date(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).date().isoformat()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def clean_knowledge_base(raw_dir: str = "knowledge_base", out_dir: str = "knowledge_base_clean", *,
                         min_words: int = 100, near_dup_threshold: float = 0.9,
                         stopwords: bool = False, on_progress=print):
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):                       # rebuild from scratch each run
        old.unlink()

    p1 = _read_phase1_metadata(raw)
    files = sorted(p for p in raw.glob("*.txt"))
    report, kept = [], []           # kept: (file, cleaned_text, meta_row)
    seen_exact: dict[str, str] = {}
    seen_sketch: list[tuple[str, set]] = []

    for p in files:
        def drop(reason, detail=""):
            report.append({"file": p.name, "status": "removed", "reason": reason, "detail": detail,
                           "raw_words": raw_words, "clean_words": clean_words})
        try:
            raw_text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raw_words = clean_words = 0
            drop("unreadable", str(e))
            continue
        raw_words = len(raw_text.split())
        text = clean_text(raw_text)
        clean_words = len(text.split())

        if clean_words < min_words:                                       # empty / stub page
            drop("empty_or_too_short", f"<{min_words} words after cleaning")
            continue

        fp = _fingerprint(text)
        if fp in seen_exact:                                              # exact duplicate
            drop("exact_duplicate", f"same as {seen_exact[fp]}")
            continue
        sk = _shingles(text)
        dup_of = next((f for f, s in seen_sketch if _jaccard(sk, s) >= near_dup_threshold), None)
        if dup_of:                                                        # near duplicate
            drop("near_duplicate", f"~same as {dup_of}")
            continue
        seen_exact[fp] = p.name
        seen_sketch.append((p.name, sk))

        m = p1.get(p.name, {})
        kept.append((p.name, text, {
            "file": p.name,
            "article_name": m.get("article_name") or p.stem.replace("_", " "),
            "url": m.get("url", ""),
            "category": m.get("category") or "Uncategorized",
            "keywords": m.get("keywords", ""),
            "word_count": clean_words,
            "character_count": len(text),
            "raw_word_count": raw_words,
            "download_date": m.get("download_date") or _file_date(p),
        }))
        report.append({"file": p.name, "status": "kept", "reason": "", "detail": "",
                       "raw_words": raw_words, "clean_words": clean_words})

    for name, text, _ in kept:
        (out / name).write_text(text, encoding="utf-8")

    meta_rows = [m for _, _, m in kept]
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:           # Phase 3
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        w.writerows(meta_rows)
    with (out / "cleaning_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "status", "reason", "detail", "raw_words", "clean_words"])
        w.writeheader()
        w.writerows(report)

    removed = Counter(r["reason"] for r in report if r["status"] == "removed")
    on_progress(f"Cleaned {len(files)} files -> kept {len(kept)}, removed {sum(removed.values())} "
                f"({dict(removed) or 'none'})")
    if stopwords and kept:
        s = stopword_analysis([t for _, t, _ in kept], out / "stopword_analysis.csv")
        on_progress(f"Stopwords: {s['stopword_share']:.1%} of {s['total_words']:,} words; "
                    f"top content words: {', '.join(s['top_content_words'])}")
    return meta_rows, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 + 3: clean the raw knowledge base and write metadata.csv")
    ap.add_argument("--raw", default="knowledge_base", help="raw Phase 1 folder")
    ap.add_argument("--out", default="knowledge_base_clean", help="cleaned output folder")
    ap.add_argument("--min-words", type=int, default=100, help="drop pages shorter than this after cleaning")
    ap.add_argument("--near-dup", type=float, default=0.9, help="near-duplicate Jaccard threshold")
    ap.add_argument("--stopwords", action="store_true", help="also write stopword_analysis.csv")
    a = ap.parse_args()
    clean_knowledge_base(a.raw, a.out, min_words=a.min_words, near_dup_threshold=a.near_dup,
                         stopwords=a.stopwords)


if __name__ == "__main__":
    main()
