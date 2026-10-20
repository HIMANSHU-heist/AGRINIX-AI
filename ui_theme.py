"""Look and feel for AGRINEX AI: one CSS block plus small HTML helpers."""

import html as _html
from datetime import datetime, timedelta, timezone

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');

:root {
  --g900:#14532D; --g800:#1B5E20; --g700:#2E7D32; --g600:#43A047;
  --g100:#E8F5E9; --g50:#F4FAF4; --ink:#1B2B20; --muted:#6B7C70; --line:#E3ECE4;
}
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp li, .stApp input, .stApp textarea {
  font-family: 'Poppins', 'Noto Sans Devanagari', sans-serif;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F4FAF4 0%, #E8F5E9 100%);
  border-right: 1px solid var(--line);
}
.agx-brand { font-size: 22px; font-weight: 800; color: var(--g800); margin-bottom: 2px; }

/* ---------- hero ---------- */
.agx-hero {
  position: relative; overflow: hidden; color: #fff;
  background: linear-gradient(135deg, #14532D 0%, #2E7D32 55%, #66BB6A 100%);
  border-radius: 20px; padding: 28px 32px; margin-bottom: 18px;
  box-shadow: 0 12px 30px rgba(20, 83, 45, .25);
}
.agx-hero::after { content: "🌾"; position: absolute; right: 24px; bottom: -22px; font-size: 130px; opacity: .15; }
.agx-hero .hi { font-size: 14px; opacity: .9; font-weight: 500; }
.agx-hero .title { font-size: 30px; font-weight: 800; margin: 2px 0 12px 0; line-height: 1.2; }
.agx-chip {
  display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
  padding: 5px 12px; border-radius: 999px; font-size: 12.5px; margin: 0 8px 6px 0;
}

/* ---------- stats ---------- */
.agx-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
.agx-stat {
  display: flex; align-items: center; gap: 12px; background: #fff; border: 1px solid var(--line);
  border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 8px rgba(20,83,45,.05);
}
.agx-stat .ico { font-size: 26px; }
.agx-stat .num { font-size: 22px; font-weight: 800; color: var(--g700); line-height: 1.1; }
.agx-stat .lbl { font-size: 12px; color: var(--muted); }

/* ---------- cards / results / badges ---------- */
.agx-card {
  background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 18px;
  margin-bottom: 14px; box-shadow: 0 2px 8px rgba(20,83,45,.04); transition: box-shadow .2s ease;
}
.agx-card:hover { box-shadow: 0 8px 20px rgba(20,83,45,.10); }
.agx-result {
  background: linear-gradient(135deg, #E8F5E9, #F4FAF4); border: 1px solid #A5D6A7;
  border-radius: 16px; padding: 24px; text-align: center;
}
.agx-badge { display: inline-block; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; letter-spacing: .03em; vertical-align: middle; }
.badge-live { background: #C8E6C9; color: #1B5E20; }
.badge-demo { background: #FFE0B2; color: #8D5A00; }
.agx-metric { text-align: center; padding: 14px; }

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: var(--g50); padding: 6px; border-radius: 14px; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 16px; border-radius: 10px; white-space: nowrap; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; color: #33512E; font-size: 14px; }
.stTabs [aria-selected="true"] { background: var(--g700) !important; box-shadow: 0 4px 10px rgba(46,125,50,.3); }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- buttons, inputs, metrics ---------- */
.stButton > button, .stFormSubmitButton > button {
  border-radius: 12px; font-weight: 600; min-height: 44px; border: 1px solid var(--line); transition: all .15s ease;
}
.stButton > button:hover { border-color: var(--g600); color: var(--g700); transform: translateY(-1px); }
.stButton > button[kind^="primary"], .stFormSubmitButton > button[kind^="primary"] {
  background: linear-gradient(135deg, #2E7D32, #43A047); color: #fff; border: none;
  box-shadow: 0 6px 16px rgba(46,125,50,.25);
}
.stButton > button[kind^="primary"]:hover { color: #fff; box-shadow: 0 8px 20px rgba(46,125,50,.35); }
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 10px; }
[data-testid="stMetric"] {
  background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 2px 8px rgba(20,83,45,.05);
}
[data-testid="stExpander"] { border-radius: 12px; border: 1px solid var(--line); }

/* ---------- login page ---------- */
.agx-login-brand { text-align: center; margin: 24px 0 16px 0; }
.agx-login-brand .logo { font-size: 54px; }
.agx-login-brand .name { font-size: 30px; font-weight: 800; color: var(--g800); }
.agx-login-brand .tag { font-size: 14px; color: var(--muted); }

/* ---------- mobile ---------- */
@media (max-width: 768px) {
  .block-container { padding: 1rem .8rem 3rem .8rem; }
  .agx-hero { padding: 20px; border-radius: 16px; }
  .agx-hero .title { font-size: 22px; }
  .agx-stats { grid-template-columns: repeat(2, 1fr); }
  .stTabs [data-baseweb="tab"] { padding: 0 12px; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _greeting():
    hour = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).hour
    if hour < 12:
        return "शुभ प्रभात ☀️"
    if hour < 17:
        return "शुभ दुपार 🌤️"
    return "शुभ संध्याकाळ 🌙"


def hero(name, location, land_area, farmer_id):
    safe_name = _html.escape(name or "शेतकरी")
    chips = "".join(
        f'<span class="agx-chip">{c}</span>'
        for c in (
            f"📍 {_html.escape(location or '-')}",
            f"🌾 {land_area} acres",
            f"🆔 {_html.escape(farmer_id)}",
        )
    )
    st.markdown(
        f'<div class="agx-hero"><div class="hi">{_greeting()}</div>'
        f'<div class="title">नमस्कार, {safe_name}</div><div>{chips}</div></div>',
        unsafe_allow_html=True,
    )


def stat_row(items):
    """items: list of (number, label, icon)"""
    cells = "".join(
        f'<div class="agx-stat"><div class="ico">{ico}</div>'
        f'<div><div class="num">{num}</div><div class="lbl">{lbl}</div></div></div>'
        for num, lbl, ico in items
    )
    st.markdown(f'<div class="agx-stats">{cells}</div>', unsafe_allow_html=True)
