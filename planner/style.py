"""Look and feel: palette, CSS, masthead and headline-figure row.

Palette (light theme):
  ink      #16211F  text
  slate    #5E6B67  secondary text
  porcelain #F6F8F7 page
  mist     #EDF1EF  sidebar and table headers
  hairline #DDE3E1  rules and borders
  spruce   #0F5C4E  the one strong colour: masthead, primary buttons, active navigation
  brass    #B08A3E  marks forecast months and the masthead rule
Type: Source Serif 4 for titles, IBM Plex Sans for everything else (set in .streamlit/config.toml).
"""
from __future__ import annotations

from html import escape

INK, SLATE, PAPER, MIST, LINE = "#16211F", "#5E6B67", "#F6F8F7", "#EDF1EF", "#DDE3E1"
SPRUCE, BRASS = "#0F5C4E", "#B08A3E"
NEUTRAL, NEUTRAL_LIGHT = "#A3ADA9", "#D5DCD9"
# Categorical series colours, checked for colour-blind separation on the page colour.
SERIES = ["#008A6E", "#D08A1E", "#3A6FC4", "#D0507F", "#7B5BD0", "#2E9BC0", "#B4532E", "#8E9A2C"]
GOOD, WARN, BAD = "#008A6E", "#D08A1E", "#B4532E"
FORECAST_BAND = BRASS

CSS = f"""
<style>
/* fonts ship in static/fonts; declared here as well so headings get the serif even where the theme option is ignored */
@font-face {{ font-family: 'Source Serif 4'; font-weight: 500; font-style: normal; font-display: swap;
              src: url('app/static/fonts/source-serif-4-latin-500-normal.woff2') format('woff2'); }}
@font-face {{ font-family: 'Source Serif 4'; font-weight: 600; font-style: normal; font-display: swap;
              src: url('app/static/fonts/source-serif-4-latin-600-normal.woff2') format('woff2'); }}
h1, h2, h3, .dp-mast h1, .dp-word .n {{ font-family: 'Source Serif 4', Georgia, serif !important; }}
/* page width and rhythm */
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.005em; }}
h2 {{ margin-top: 0.6rem; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {SLATE}; }}

/* masthead (sits on the porcelain page, under the hidden Streamlit header) */
header[data-testid="stHeader"] {{ background: transparent; }}
.dp-mast {{ background: {SPRUCE}; color: #fff; border-radius: 6px; padding: 18px 24px 16px;
            display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap;
            border-bottom: 3px solid {BRASS}; margin-bottom: 18px; }}
.dp-mast h1 {{ color: #fff; font-size: 1.7rem; line-height: 1.15; margin: 0; padding: 0; font-weight: 600; }}
.dp-mast .dp-context {{ text-align: right; font-size: 0.82rem; line-height: 1.45; color: rgba(255,255,255,0.82); }}
.dp-mast .dp-context b {{ color: #fff; font-weight: 500; }}

/* headline figures: one row divided by hairlines */
.dp-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; margin: 4px 0 22px; }}
.dp-kpi {{ padding: 14px 18px 14px 18px; border-left: 1px solid {LINE}; min-width: 0; }}
.dp-kpi:first-child {{ border-left: none; padding-left: 2px; }}
.dp-kpi .l {{ font-size: 0.8rem; color: {SLATE}; margin-bottom: 4px; }}
.dp-kpi .v {{ font-size: 1.6rem; font-weight: 500; color: {INK}; font-variant-numeric: tabular-nums;
              line-height: 1.2; overflow-wrap: anywhere; }}
.dp-kpi .s {{ font-size: 0.8rem; color: {SLATE}; margin-top: 4px; line-height: 1.4; }}

/* sidebar: wordmark and navigation list */
section[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
.dp-word {{ margin: 0 0 18px; }}
.dp-word .n {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.45rem; font-weight: 600; color: {INK}; line-height: 1.1; }}
.dp-word .t {{ font-size: 0.8rem; color: {SLATE}; margin-top: 4px; }}
section[data-testid="stSidebar"] .stElementContainer:has(.stRadio),
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] div[role="radiogroup"] {{ width: 100% !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > div {{ width: 100%; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 7px 12px; border-radius: 5px; width: 100%; box-sizing: border-box;
            margin: 0; border-left: 3px solid transparent; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(15,92,78,0.06); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background: rgba(15,92,78,0.10);
            border-left-color: {SPRUCE}; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color: {SPRUCE}; font-weight: 600; }}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: 0.95rem; }}

/* metrics left as st.metric (Explore, Data) */
[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-size: 0.8rem; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}

/* multiselect chips: quiet tint rather than solid spruce */
[data-tag], [data-baseweb="tag"] {{ background: rgba(15,92,78,0.10) !important; color: {SPRUCE} !important; }}
[data-tag] span, [data-tag] button, [data-tag] svg {{ color: {SPRUCE} !important; fill: {SPRUCE} !important; }}

/* tabs, buttons, focus */
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; }}
.stButton > button, .stDownloadButton > button {{ font-weight: 500; }}
:focus-visible {{ outline: 2px solid {SPRUCE} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def masthead(title: str, context_lines: list[tuple[str, str]]) -> str:
    """Spruce band with the title on the left and short labelled context lines on the right."""
    ctx = "<br>".join(f"{escape(k)} <b>{escape(v)}</b>" for k, v in context_lines)
    return f'<div class="dp-mast"><h1>{escape(title)}</h1><div class="dp-context">{ctx}</div></div>'


def kpis(items: list[tuple[str, str, str]]) -> str:
    """Headline figures: (label, value, note) each."""
    cells = "".join(f'<div class="dp-kpi"><div class="l">{escape(l)}</div><div class="v">{escape(v)}</div>'
                    + (f'<div class="s">{escape(s)}</div>' if s else "") + "</div>" for l, v, s in items)
    return f'<div class="dp-kpis">{cells}</div>'


def wordmark() -> str:
    return '<div class="dp-word"><div class="n">Demand Predictor</div><div class="t">Base SKU demand planning</div></div>'
