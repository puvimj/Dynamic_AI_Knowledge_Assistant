DYNAMIC AI KNOWLEDGE ASSISTANT
==============================

Type a few keywords, and the app downloads the best matching Wikipedia articles,
cleans them, makes them searchable, and answers your questions from them. Every
answer shows its confidence, the source articles and the passages it used.

It works for any topic. Create a new workspace in the sidebar to build a library
for a different subject (machine learning, space, health, cybersecurity...).


WHAT IT DOES
------------
1. Search and download   Wikipedia articles for each keyword (top-N, or ones you pick in Preview)
2. Clean                 removes HTML, odd characters, duplicates, empty pages and formula junk
3. Metadata              article name, URL, category, word and character counts, download date
4. Chunk                 cuts articles into passages (fixed, recursive or semantic)
5. Embed                 turns passages into vectors (MiniLM, BGE or offline tfidf-lsa)
6. Index                 stores vectors in FAISS or ChromaDB
7. Retrieve              hybrid search: meaning + keyword match + title match
8. Answer                Llama 3, Gemma, Mistral, GPT, or "extractive" (no LLM needed)


PAGES
-----
Home              status of your library and a guide to the steps
Dataset Builder   build from Wikipedia (with article preview), import your own .txt files, re-index
Knowledge Base    browse and read the articles, filter by keyword, see how the library is made up
Chat Assistant    multi-turn chat with sources, confidence, retrieved passages, thumbs up/down, export
Analytics         build timings, chunk sizes, embedding map, chat speed and confidence
Evaluation        search accuracy, chunking comparison, model x vector-database comparison


REQUIREMENTS
------------
- Python 3.10 or newer (tested on 3.11 and 3.13)
- Internet access to download Wikipedia articles and the embedding model (first run only)
- Optional: Ollama (https://ollama.com) to run Llama 3, Gemma or Mistral locally


INSTALL
-------
1. Open a terminal in the project folder (the one that contains app.py).
2. Create and activate a virtual environment (recommended):
       python -m venv .venv
       .venv\Scripts\activate          (Windows)
       source .venv/bin/activate       (macOS / Linux)
3. Install the packages:
       pip install -r requirements.txt


RUN
---
    streamlit run app.py

The app opens at http://localhost:8501


QUICK START
-----------
1. Open Dataset Builder, pick an example topic or type your own keywords (one per line).
   After typing, click outside the box or press Ctrl+Enter.
2. Optional: press "Preview articles", untick anything off-topic.
3. Press "Build knowledge base". It takes about a minute for a few dozen articles.
4. Open Chat Assistant and ask a question.

Tips for good keywords
- Use exact Wikipedia article titles ("Vector database", not "vector db").
- Keep "Articles per keyword" at 2 or 3 to avoid off-topic articles.
- Keep "Add to the existing library" ticked to build up one library in several steps.
- Use a separate workspace for a separate topic.


CHOOSING AN ANSWER MODEL (Chat Assistant > Settings)
----------------------------------------------------
extractive   no LLM; quotes the best sentences from your articles (works with nothing installed)
llama3, gemma, mistral
             run locally through Ollama:
                 ollama serve
                 ollama pull llama3.1:8b
             (use "Connect another LLM server" in Settings to set a model name or URL)
gpt          set the OPENAI_API_KEY environment variable first

Confidence (High / Medium / Low) shows how likely your library is to contain the
answer. If it is Low, the assistant says it is not sure instead of guessing.


PROJECT STRUCTURE
-----------------
app.py                  entry point (navigation and theme)
requirements.txt
wikipedia_kb.py         download articles from Wikipedia
kb_clean.py             cleaning and metadata
kb_chunking.py          chunking strategies and comparison
kb_embed.py             embeddings, FAISS and Chroma stores
kb_retrieval.py         hybrid search and confidence score
kb_prompts.py           prompt templates (QA, summary, explain, compare, example)
kb_generate.py          LLM backends and the answer pipeline
kb_pipeline.py          runs all steps; manages workspaces
pages/                  one thin file per page
app_pages/              page logic (home, dataset_builder, knowledge_base, chat, analytics,
                        evaluation), design system (ui.py) and shared helpers (common.py)
.streamlit/config.toml  theme and settings
static/                 optional: your own background photos (bg.jpg)
workspaces/             created by the app; one folder per library (not committed to Git)


WHERE YOUR DATA IS SAVED
------------------------
workspaces/<name>/
    raw/                articles as downloaded
    clean/              cleaned articles, metadata.csv, cleaning_report.csv
    chunks/             passages
    embeddings/         cached vectors
    vectordb/           search indexes (FAISS / Chroma)
    pipeline_stats.json, query_log.jsonl, feedback.jsonl, config.json

Delete a library from the sidebar ("Delete this workspace") or by deleting its folder.


TROUBLESHOOTING
---------------
Build button seems to do nothing   click outside the keywords box first (or Ctrl+Enter)
HTTP 429 from Wikipedia            enter a contact email in Dataset Builder and retry later
"No module named torchvision" log  harmless; the supplied config turns off the file watcher
Answer looks unrelated             check the article titles on the Knowledge Base page;
                                   use exact titles and the Preview step
Ollama not running                 start "ollama serve" or choose "extractive"
Formulas look broken               Wikipedia plain text has no real formulas; they are removed


LIMITATIONS
-----------
- Answers are only as good as the downloaded articles.
- Wikipedia plain text drops mathematical formulas.
- The "extractive" model copies sentences; use an LLM for written answers.
- Evaluation uses automatic questions built from article sections; add your own for a final report.
- Faithfulness / relevance scoring of generated answers (for example with Ragas) is not included.