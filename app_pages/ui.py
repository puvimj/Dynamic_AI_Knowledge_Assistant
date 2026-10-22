"""
Design system for the Streamlit app: light, friendly and readable (works for kids and adults).

* Large rounded type (Nunito), dark ink on white cards, one blue accent, big buttons.
* inject_theme(page_key)   CSS + a soft generated header picture per page (no image downloads).
                           Drop your own photo into  static/bg_<page>.jpg  (or static/bg.jpg) to replace it;
                           page keys: home, builder, kb, chat, analytics, eval.
* HTML components          hero, stat_cards, pills, source_pills, confidence_badge, stage_tracker, section,
                           empty_state, article_html
* Plotly helpers           style_fig, show_chart, SERIES (colour-blind-checked categorical palette)

HTML is built as single-line strings on purpose: Streamlit's Markdown parser ends an HTML block at a blank line
and turns indented lines into code blocks.
"""

from __future__ import annotations

import base64
import html
import random
import re
from pathlib import Path

import streamlit as st

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
BG, SURFACE = "#F4F8FF", "#FFFFFF"
INK, MUTED = "#1B2433", "#445068"
PRIMARY, SOFT = "#2563EB", "#E3ECFF"
GOOD, WARN, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"     # status palette (fixed)

# Categorical chart palette (light-surface step of the dataviz reference palette), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#9aa3b2"
GRID, AXIS = "#E3E9F4", "#B7C2D6"

esc = html.escape

# --------------------------------------------------------------------------- #
# Header picture (SVG, generated): soft overlapping shapes in a different colour family per page
# --------------------------------------------------------------------------- #
W, H = 1600, 520
PALETTES = {"home": ("#BFD6FF", "#FFE2A8", "#C9F0DD"), "builder": ("#C9F0DD", "#BFD6FF", "#FFE2A8"),
            "kb": ("#FFE2A8", "#FFD0D8", "#BFD6FF"), "chat": ("#D9CCFF", "#BFD6FF", "#FFD0D8"),
            "analytics": ("#BFD6FF", "#C9F0DD", "#D9CCFF"), "eval": ("#FFD0D8", "#FFE2A8", "#C9F0DD")}


def _art(page_key: str) -> str:
    rnd = random.Random(sum(map(ord, page_key)))
    cols = PALETTES.get(page_key, PALETTES["home"])
    shapes = []
    for i in range(9):
        c = cols[i % 3]
        x, y, r = rnd.randint(150, W - 100), rnd.randint(20, 330), rnd.randint(70, 190)
        if i % 3 == 2:
            shapes.append(f'<rect x="{x}" y="{y}" width="{r * 1.5:.0f}" height="{r:.0f}" rx="{r // 2}" fill="{c}" opacity=".7" '
                          f'transform="rotate({rnd.randint(-25, 25)} {x} {y})"/>')
        else:
            shapes.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" opacity=".62"/>')
    dots = "".join(f'<circle cx="{rnd.randint(0, W)}" cy="{rnd.randint(0, 400)}" r="{rnd.randint(3, 7)}" fill="#FFFFFF" opacity=".9"/>'
                   for _ in range(26))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f'<rect width="{W}" height="{H}" fill="{BG}"/>{"".join(shapes)}{dots}</svg>')


@st.cache_data(show_spinner=False)
def _art_uri(page_key: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(_art(page_key).encode()).decode()


def _user_image(page_key: str) -> str | None:
    """static/bg_<page>.* or static/bg.* (served by Streamlit's static file server), if the user added one."""
    root = Path("static")
    for stem in (f"bg_{page_key}", "bg"):
        for ext in ("jpg", "jpeg", "png", "webp"):
            if (root / f"{stem}.{ext}").exists():
                return f"app/static/{stem}.{ext}"
    return None


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
:root{--bg:#F4F8FF;--card:#FFFFFF;--ink:#1B2433;--muted:#445068;--blue:#2563EB;--soft:#E3ECFF;--line:#D5DEEE;
--good:#0a7a0a;--warn:#8a5a00;--bad:#b42318;--font:'Nunito',system-ui,-apple-system,'Segoe UI',sans-serif;}
html,body,.stApp,[data-testid="stAppViewContainer"],button,input,textarea,select{font-family:var(--font)!important;}
.stApp{background-color:var(--bg);color:var(--ink);font-size:1.06rem;}
.stApp p,.stApp li,.stApp label,.stApp span{line-height:1.55;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"],.block-container{max-width:1120px;padding-top:1.2rem;padding-bottom:4rem;}
h1,h2,h3,h4{font-family:var(--font)!important;font-weight:800;color:var(--ink);letter-spacing:-.01em;}
h3{font-size:1.5rem;} h4{font-size:1.2rem;}
a{color:var(--blue);font-weight:600;}
:focus-visible{outline:3px solid #F59E0B!important;outline-offset:2px;}
[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--line);}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{border-radius:12px;font-size:1.05rem;padding:.35rem .6rem;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span{color:var(--ink)!important;font-weight:600;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{background:var(--soft);box-shadow:inset 4px 0 0 var(--blue);}
[data-testid="stSidebarNavSeparator"]{display:none;}
.kb-status{border:1px solid var(--line);border-radius:14px;padding:.7rem .9rem;font-size:.98rem;color:var(--muted);background:#F8FAFF;}
.kb-status b{color:var(--ink);font-weight:800;}
.dot{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:.45rem;background:var(--muted);}
.dot.on{background:#0ca30c;} .dot.off{background:#ec835a;}
.hero{padding:1.7rem 0 1rem;max-width:780px;}
.hero.compact{padding:.9rem 0 .5rem;}
.hero h1{font-size:2.7rem;line-height:1.12;margin:0 0 .4rem;padding:0;font-weight:800;}
.hero.compact h1{font-size:2.15rem;margin-bottom:.2rem;padding:0;}
.hero p{font-size:1.2rem;line-height:1.6;color:var(--muted);margin:0;max-width:60ch;}
.sec{margin:1.8rem 0 .6rem;} .sec h3{margin:0;} .sec p{margin:.25rem 0 0;color:var(--muted);font-size:1.02rem;}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.9rem;margin:.6rem 0 1.1rem;}
.stat{background:var(--card);border:1px solid var(--line);border-top:5px solid var(--blue);border-radius:16px;padding:.9rem 1.1rem 1rem;box-shadow:0 2px 8px rgba(37,99,235,.07);}
.stat.cool{border-top-color:#1baf7a;} .stat.good{border-top-color:#0ca30c;} .stat.warn{border-top-color:#eda100;}
.stat-label{font-size:1rem;color:var(--muted);font-weight:600;}
.stat-value{font-size:2.1rem;line-height:1.2;color:var(--ink);font-weight:800;margin-top:.1rem;overflow-wrap:anywhere;}
.stat-note{font-size:.92rem;color:var(--muted);margin-top:.1rem;}
.pills{display:flex;flex-wrap:wrap;gap:.5rem;margin:.4rem 0 .7rem;}
.pill{display:inline-flex;align-items:center;gap:.45rem;border:1.5px solid var(--line);border-radius:999px;padding:.3rem .9rem;font-size:1rem;font-weight:600;color:var(--ink);background:#fff;text-decoration:none;}
a.pill:hover{border-color:var(--blue);background:var(--soft);}
.pill b{display:inline-flex;align-items:center;justify-content:center;min-width:1.5rem;height:1.5rem;border-radius:999px;background:var(--blue);color:#fff;font-size:.88rem;font-weight:800;}
.pill.brass{background:var(--soft);border-color:#B9CCF5;color:#1D4ED8;}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.5rem 0 .3rem;color:var(--muted);font-size:.95rem;}
.chip{border:1px solid var(--line);border-radius:999px;padding:.15rem .75rem;background:#fff;}
.conf{display:inline-flex;align-items:center;gap:.6rem;font-size:1rem;border-radius:999px;padding:.2rem .85rem;border:1.5px solid var(--line);background:#fff;font-weight:600;}
.conf i{display:inline-block;width:6rem;height:.55rem;border-radius:99px;background:#E3E9F4;overflow:hidden;}
.conf i u{display:block;height:100%;text-decoration:none;}
.conf.good b{color:var(--good);} .conf.warn b{color:var(--warn);} .conf.bad b{color:var(--bad);}
.conf.good i u{background:#0ca30c;} .conf.warn i u{background:#fab219;} .conf.bad i u{background:#d03b3b;}
.track{list-style:none;display:flex;gap:0;margin:1rem 0 1.2rem;padding:0;}
.track li{flex:1;position:relative;padding-top:2.9rem;text-align:left;font-size:.98rem;color:var(--muted);min-width:0;padding-right:.7rem;}
.track li:before{content:attr(data-n);position:absolute;top:0;left:0;width:2.1rem;height:2.1rem;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:800;border:2px solid var(--line);color:var(--muted);background:#fff;z-index:1;}
.track li:after{content:"";position:absolute;top:1rem;left:2.2rem;right:.1rem;border-top:3px solid var(--line);}
.track li:last-child:after{display:none;}
.track li b{display:block;color:var(--ink);font-weight:800;font-size:1.05rem;}
.track li.done:before{content:"✓";background:#0ca30c;border-color:#0ca30c;color:#fff;}
.track li.done:after{border-top-color:#0ca30c;}
.track li.active:before{border-color:var(--blue);background:var(--blue);color:#fff;box-shadow:0 0 0 5px rgba(37,99,235,.2);}
.track li.active b{color:var(--blue);}
.empty{border:2px dashed #B9C6E0;border-radius:16px;padding:1.4rem 1.6rem;background:#fff;margin:.8rem 0;}
.empty b{font-size:1.3rem;font-weight:800;color:var(--ink);} .empty p{margin:.35rem 0 0;color:var(--muted);}
.article{font-size:1.05rem;line-height:1.75;color:var(--ink);max-width:72ch;} .article h4{margin:1.2rem 0 .3rem;} .article p{margin:.5rem 0;}
mark{background:#FFE27A;color:var(--ink);border-radius:4px;padding:0 .15rem;}
.ctx{border-left:5px solid var(--blue);background:#fff;border-radius:0 12px 12px 0;padding:.6rem .9rem;margin:.7rem 0;font-size:1rem;line-height:1.65;color:var(--ink);}
.ctx small{display:block;color:var(--muted);margin-bottom:.25rem;font-size:.9rem;font-weight:700;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:#fff;}
[data-testid="stTabs"] button[role="tab"]{font-size:1.08rem;font-weight:700;padding:.7rem 1.1rem;}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(37,99,235,.06);}
[data-testid="stChatInput"]{border-radius:16px;}
.stButton>button,.stDownloadButton>button,[data-testid="stPopover"]>button,[data-testid="stPageLink-NavLink"]{border-radius:12px;min-height:3rem;font-size:1.05rem;font-weight:700;border:1.5px solid var(--line);}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stPageLink-NavLink"]:hover{border-color:var(--blue);color:var(--blue);}
[data-testid="stPageLink-NavLink"]{background:#fff;}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-top:5px solid var(--blue);border-radius:16px;padding:.8rem 1rem;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-baseweb="select"]>div{font-size:1.05rem;border-radius:12px;}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label{font-size:1.05rem!important;font-weight:700;color:var(--ink);}
[data-testid="stCaptionContainer"] p,.stCaption{font-size:1rem!important;color:var(--muted)!important;}
.stButton>button p,.stDownloadButton>button p,[data-testid="stPageLink-NavLink"] p{font-size:1.05rem!important;font-weight:700;}
.stButton>button[kind="primary"]{background:var(--blue);color:#fff;border-color:var(--blue);}
.stButton>button:disabled{opacity:.55;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] span{font-size:1.05rem;}
[data-testid="stSidebar"] p{font-size:1rem;}
[data-testid="stChatInput"] *:focus-visible,[data-testid="stChatInput"] textarea:focus{outline:none!important;box-shadow:none!important;}
a.pill,a.pill:hover{text-decoration:none;}
[data-testid="stTabs"] button[role="tab"] p,[data-testid="stTab"] p,button[role="tab"] div p{font-size:1.12rem!important;font-weight:700;}
@media (max-width:720px){.hero h1{font-size:2.1rem;}.track{flex-direction:column;gap:1rem;}.track li:after{display:none;}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
"""


def inject_theme(page_key: str = "home") -> None:
    """Call once per run, before the page renders."""
    own = _user_image(page_key)
    art = f'url("{own}")' if own else f'url("{_art_uri(page_key)}")'
    fade = "linear-gradient(to bottom,rgba(244,248,255,.15) 0,rgba(244,248,255,.7) 300px,#F4F8FF 480px)"
    bg = (f".stApp{{background-image:{fade},{art};background-size:100% 520px,{'cover' if own else '100% 520px'};"
          f"background-repeat:no-repeat;background-position:top center;}}")
    st.markdown(f"<style>{CSS}{bg}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML components
# --------------------------------------------------------------------------- #
def html_block(markup: str) -> None:
    st.markdown(re.sub(r"\n\s*", "", markup), unsafe_allow_html=True)


def hero(title: str, lede: str, compact: bool = False) -> None:
    html_block(f'<div class="hero{" compact" if compact else ""}"><h1>{esc(title)}</h1><p>{esc(lede)}</p></div>')


def section(title: str, note: str | None = None) -> None:
    html_block(f'<div class="sec"><h3>{esc(title)}</h3>{f"<p>{esc(note)}</p>" if note else ""}</div>')


def stat_cards(items: list[dict]) -> None:
    """items: {label, value, note?, tone?: brass|cool|good|warn}"""
    cells = []
    for item in items:
        note = f'<div class="stat-note">{esc(str(item["note"]))}</div>' if item.get("note") else ""
        cells.append(f'<div class="stat {item.get("tone", "")}"><div class="stat-label">{esc(str(item["label"]))}</div>'
                     f'<div class="stat-value">{esc(str(item["value"]))}</div>{note}</div>')
    html_block(f'<div class="stats">{"".join(cells)}</div>')


def pills(labels: list[str], tone: str = "") -> None:
    if labels:
        html_block('<div class="pills">' + "".join(f'<span class="pill {tone}">{esc(str(x))}</span>' for x in labels) + "</div>")


def source_pills_html(sources: list[dict]) -> str:
    """One pill per source article; the numbers match the [n] citations inside the answer."""
    out = []
    for s in sources:
        nums = "".join(f"<b>{n}</b>" for n in s["passages"])
        tag = "a" if s.get("url") else "span"
        href = f' href="{esc(s["url"], quote=True)}" target="_blank" rel="noopener"' if s.get("url") else ""
        out.append(f'<{tag} class="pill"{href} title="best match {s["best_score"]:.2f}">{nums}{esc(s["article"])}</{tag}>')
    return '<div class="pills">' + "".join(out) + "</div>"


def confidence_badge(conf: float, label: str) -> str:
    """Status colour + word + percentage, so colour never carries the meaning alone."""
    cls, icon = {"High": ("good", "&#10003;"), "Medium": ("warn", "~")}.get(label, ("bad", "!"))
    return (f'<span class="conf {cls}"><b>{icon} {esc(label)}</b>{conf:.0%}'
            f'<i><u style="width:{max(0, min(conf, 1)) * 100:.0f}%"></u></i></span>')


def stage_tracker(stages: list[str], active: int, notes: list[str] | None = None) -> str:
    """Numbered stations (a true sequence). Stations before `active` are done; active == len(stages) means all done."""
    cells = []
    for n, name in enumerate(stages):
        cls = "done" if n < active else "active" if n == active else ""
        note = f"<span>{esc(notes[n])}</span>" if notes and notes[n] else ""
        cells.append(f'<li class="{cls}" data-n="{n + 1}"><b>{esc(name)}</b>{note}</li>')
    return f'<ol class="track">{"".join(cells)}</ol>'


def empty_state(title: str, body: str) -> None:
    html_block(f'<div class="empty"><b>{esc(title)}</b><p>{esc(body)}</p></div>')


def article_html(text: str, term: str = "") -> str:
    """Cleaned article text ('## Heading' lines) as readable HTML, with the search term highlighted."""
    pat = re.compile(re.escape(esc(term)), re.I) if term.strip() else None
    mark = (lambda s: pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", s)) if pat else (lambda s: s)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            out.append(f'<h4>{mark(esc(line.lstrip("# ").strip()))}</h4>')
        else:
            out.append(f"<p>{mark(esc(line))}</p>")
    return f'<div class="article">{"".join(out)}</div>'


def highlight(text: str, terms: list[str]) -> str:
    """Escape text and wrap any of `terms` (whole words, case-insensitive) in <mark>."""
    safe = esc(text)
    terms = [t for t in dict.fromkeys(t.lower() for t in terms) if len(t) > 2]
    if not terms:
        return safe
    pat = re.compile(r"\b(" + "|".join(re.escape(esc(t)) for t in terms) + r")\b", re.I)
    return pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #
def style_fig(fig, height: int = 340, legend: bool = False):
    """Transparent, quiet chart chrome on the spruce surface; recessive grid; own hover label."""
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito, system-ui, sans-serif", color=MUTED, size=14),
        colorway=SERIES, margin=dict(l=4, r=8, t=12, b=4), showlegend=legend, bargap=0.28,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=AXIS, font=dict(color=INK, size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED, size=14), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickfont=dict(color=MUTED, size=14), title_font=dict(color=MUTED))
    return fig


def show_chart(fig, key: str, select: bool = False):
    """Render a Plotly figure; returns the selection event when select=True."""
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
    if select:
        return st.plotly_chart(fig, width="stretch", config=cfg, key=key, on_select="rerun", selection_mode="points")
    return st.plotly_chart(fig, width="stretch", config=cfg, key=key)
