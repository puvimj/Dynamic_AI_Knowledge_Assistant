#!/usr/bin/env python3
"""
Phase 1 - Knowledge Base Creation (Dynamic AI Knowledge Assistant)

keywords -> search Wikipedia -> top-N relevant articles per keyword
         -> de-duplicate -> download -> store as .txt -> metadata.csv

CLI:
    python wikipedia_kb.py "Machine Learning" "Deep Learning" "Computer Vision" \
        "Natural Language Processing" "Large Language Model" --top-n 5

Notebook / Streamlit:
    from wikipedia_kb import build_knowledge_base
    rows = build_knowledge_base(["Machine Learning", "Deep Learning"], top_n=5,
                                on_progress=st.write)

Requires: pip install requests
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia asks API clients to send a descriptive User-Agent with contact info.
# A generic one is a common cause of 429 (Too Many Requests). Put YOUR email here.
USER_AGENT = "DynamicAIKnowledgeAssistant/1.0 (student RAG project; contact: your-email@example.com)"

METADATA_FIELDS = [
    "file", "article_name", "url", "category", "keywords",
    "word_count", "character_count", "download_date",
]

# Friendly filenames for well-known titles (keys are lowercase Wikipedia titles).
ALIASES = {
    "convolutional neural network": "CNN",
    "recurrent neural network": "RNN",
    "generative adversarial network": "GAN",
    "transformer (deep learning architecture)": "Transformer",
    "transformer (machine learning model)": "Transformer",
}

_SKIP_PREFIXES = ("list of ", "outline of ", "index of ", "timeline of ")
_NOISE_CATEGORY = re.compile(
    r"^(Articles|All articles|All Wikipedia|CS1|Webarchive|Short description|Use .* dates|"
    r"Wikipedia|Commons category|Pages )", re.I)


# --------------------------------------------------------------------------- #
# Wikipedia API client (serial, throttled, honours Retry-After)
# --------------------------------------------------------------------------- #
class WikipediaClient:
    def __init__(self, user_agent: str = USER_AGENT, delay: float = 1.0, max_retries: int = 6):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay
        self.max_retries = max_retries
        self._last_call = 0.0

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
        for attempt in range(self.max_retries):
            # throttle: at most one request every `delay` seconds
            wait = self.delay - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

            try:
                r = self.session.get(API_URL, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue

            if r.status_code in (429, 503):            # rate limited / overloaded
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** attempt * 2)
                continue

            r.raise_for_status()
            data = r.json()
            if data.get("error", {}).get("code") == "maxlag":   # servers busy
                time.sleep(5)
                continue
            return data
        raise RuntimeError("Wikipedia API still failing after retries")

    def search(self, keyword: str, limit: int) -> list[str]:
        """Titles of the most relevant articles for `keyword`, best match first."""
        data = self._get({
            "action": "query", "list": "search", "srsearch": keyword,
            "srlimit": min(limit, 50), "srnamespace": 0, "srprop": "",
        })
        return [hit["title"] for hit in data["query"]["search"]]

    def fetch_article(self, title: str) -> dict | None:
        """Full plain-text article (section headings kept as == Heading ==)."""
        data = self._get({
            "action": "query", "titles": title, "redirects": 1,
            "prop": "extracts|info|categories",
            "explaintext": 1, "exsectionformat": "wiki",
            "inprop": "url", "clshow": "!hidden", "cllimit": "max",
        })
        page = data["query"]["pages"][0]
        if page.get("missing"):
            return None
        cats = [c["title"].removeprefix("Category:") for c in page.get("categories", [])]
        return {
            "title": page["title"],
            "text": (page.get("extract") or "").strip(),
            "url": page.get("fullurl", ""),
            "categories": [c for c in cats if not _NOISE_CATEGORY.match(c)],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_keywords(raw: list[str]) -> list[str]:
    """Accept separate items or comma-separated strings; drop blanks and duplicates."""
    out, seen = [], set()
    for item in raw:
        for kw in item.split(","):
            kw = kw.strip()
            if kw and kw.lower() not in seen:
                seen.add(kw.lower())
                out.append(kw)
    return out


def safe_filename(title: str) -> str:
    """'Machine learning' -> 'Machine_Learning'; aliases like CNN apply."""
    name = ALIASES.get(title.lower(), title)
    name = re.sub(r"[^\w\s-]", "", name)
    name = " ".join(w[:1].upper() + w[1:] for w in name.split())   # keeps acronyms intact
    return re.sub(r"\s+", "_", name)[:120] or "untitled"


def _unique_filename(title: str, used: set[str]) -> str:
    base, n = safe_filename(title), 1
    name = f"{base}.txt"
    while name in used:
        n += 1
        name = f"{base}_{n}.txt"
    return name


def _is_disambiguation(text: str) -> bool:
    return "may refer to" in text[:400].lower()


def _load_metadata(out_dir: Path) -> dict[str, dict]:
    """Existing rows (cache), keyed by lowercase title; only if the .txt still exists."""
    path = out_dir / "metadata.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["article_name"].lower(): r
                for r in csv.DictReader(f) if (out_dir / r["file"]).exists()}


def _add_keyword(row: dict, keyword: str) -> None:
    kws = [k for k in row.get("keywords", "").split("|") if k]
    if keyword not in kws:
        kws.append(keyword)
    row["keywords"] = "|".join(kws)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def build_knowledge_base(
    keywords: list[str],
    top_n: int = 5,
    out_dir: str = "knowledge_base",
    *,
    delay: float = 1.0,
    min_words: int = 200,
    skip_lists: bool = True,
    fresh: bool = False,
    selected: dict[str, list[str]] | None = None,
    user_agent: str = USER_AGENT,
    on_progress: Callable[[str], None] = print,
) -> list[dict]:
    """Build the knowledge base and return the metadata rows.

    - Top-N is applied PER keyword; an article matching several keywords is
      downloaded once and lists all of them in `keywords`.
    - Re-running is cheap: articles already in metadata.csv are not re-downloaded.
    - fresh=True deletes existing .txt files + metadata.csv in out_dir first.
    - selected={keyword: [article titles]} downloads exactly those articles (no search, no top-N).
    """
    keywords = list(selected) if selected is not None else parse_keywords(keywords)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if fresh:
        for f in out.glob("*.txt"):
            f.unlink()
        (out / "metadata.csv").unlink(missing_ok=True)

    client = WikipediaClient(user_agent, delay)
    rows = _load_metadata(out)          # cache of earlier downloads
    errors: list[dict] = []
    today = datetime.now(timezone.utc).date().isoformat()

    for keyword in keywords:
        on_progress(f"[search] {keyword}")
        if selected is not None:
            titles, top_n = list(selected[keyword]), len(selected[keyword])     # user-picked articles
        else:
            try:
                titles = client.search(keyword, limit=top_n * 3)   # extras survive filtering
            except Exception as e:
                errors.append({"keyword": keyword, "title": "", "error": str(e)})
                on_progress(f"  ! search failed: {e}")
                continue

        kept = 0
        for title in titles:
            if kept >= top_n:
                break
            if skip_lists and title.lower().startswith(_SKIP_PREFIXES):
                continue

            if title.lower() in rows:                           # cache / duplicate
                _add_keyword(rows[title.lower()], keyword)
                kept += 1
                on_progress(f"  = {title} (already have it)")
                continue

            try:
                art = client.fetch_article(title)
            except Exception as e:
                errors.append({"keyword": keyword, "title": title, "error": str(e)})
                on_progress(f"  ! {title}: {e}")
                continue

            if art is None:
                continue
            if art["title"].lower() in rows:                    # redirect to known article
                _add_keyword(rows[art["title"].lower()], keyword)
                kept += 1
                continue
            words = len(art["text"].split())
            if words < min_words or _is_disambiguation(art["text"]):
                on_progress(f"  - skipped (stub/disambiguation): {art['title']}")
                continue

            filename = _unique_filename(art["title"], {r["file"] for r in rows.values()})
            (out / filename).write_text(art["text"], encoding="utf-8")
            row = {
                "file": filename,
                "article_name": art["title"],
                "url": art["url"],
                "category": art["categories"][0] if art["categories"] else "Uncategorized",
                "keywords": "",
                "word_count": words,
                "character_count": len(art["text"]),
                "download_date": today,
            }
            _add_keyword(row, keyword)
            rows[art["title"].lower()] = row
            kept += 1
            on_progress(f"  + {filename} ({words} words)")

        if kept < top_n:
            on_progress(f"  ! only {kept}/{top_n} usable articles for '{keyword}'")

    result = sorted(rows.values(), key=lambda r: r["file"].lower())
    _write_csv(out / "metadata.csv", METADATA_FIELDS, result)
    if errors:
        _write_csv(out / "error_log.csv", ["keyword", "title", "error"], errors)

    on_progress(f"\nDone: {len(result)} articles in {out}/  |  {len(errors)} errors")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Build a Wikipedia knowledge base from keywords.")
    p.add_argument("keywords", nargs="*", help="keywords (separate args or comma-separated)")
    p.add_argument("--top-n", type=int, default=5, help="articles per keyword (default 5)")
    p.add_argument("--out", default="knowledge_base", help="output folder")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between API calls")
    p.add_argument("--min-words", type=int, default=200, help="skip articles shorter than this")
    p.add_argument("--include-lists", action="store_true", help="keep 'List of ...' pages")
    p.add_argument("--fresh", action="store_true", help="wipe existing KB in --out first")
    p.add_argument("--user-agent", default=USER_AGENT)
    args = p.parse_args()

    kws = parse_keywords(args.keywords or [input("Keywords (comma-separated): ")])
    if not kws:
        sys.exit("No keywords provided.")
    if len(kws) < 5:
        print(f"Note: project guidelines ask for at least 5 keywords (got {len(kws)}).")

    build_knowledge_base(kws, args.top_n, args.out, delay=args.delay, min_words=args.min_words,
                         skip_lists=not args.include_lists, fresh=args.fresh,
                         user_agent=args.user_agent)


if __name__ == "__main__":
    main()
