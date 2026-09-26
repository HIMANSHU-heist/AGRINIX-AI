import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

INDIAN_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa",
    "Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland",
    "Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar Islands",
    "Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Delhi",
    "Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"
]

# NOTE: this dataset ("Current Daily Price of Various Commodities from
# Various Markets") is a DAILY SNAPSHOT, not a historical time series.
# We build our own history locally (see log_daily_snapshot below) by
# recording one row per commodity every time the app is used, so a real
# trend accumulates over the days the app stays in use.

# ------------------------------------------------------------------
# Shared session with retries + generous, split connect/read timeouts,
# so a single slow response from data.gov.in doesn't hard-fail the UI.
# ------------------------------------------------------------------
def _get_session():
    session = requests.Session()
    retry = Retry(
        total=1,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

_session = _get_session()
REQUEST_TIMEOUT = (5, 15)  # (connect, read)


def _request(params):
    return _session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)


# ------------------------------------------------------------------
# ONE call that pulls every commodity trading in a state today.
# This replaces fetching commodities + prices separately — the whole
# "stock board" is built from this single response.
# ------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_state_snapshot(api_key, state=None, limit=1500):
    """Returns (df, ok). df has one row per market report, across all
    commodities, for the given state (or all-India if state is None).
    Never raises — on failure returns an empty df and ok=False so the
    UI can quietly fall back instead of showing a traceback."""
    params = {"api-key": api_key, "format": "json", "limit": limit}
    if state:
        params["filters[state]"] = state
    try:
        r = _request(params)
        r.raise_for_status()
        records = r.json().get("records", [])
    except Exception:
        return pd.DataFrame(), False

    if not records:
        return pd.DataFrame(), True

    df = pd.DataFrame(records)
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["modal_price"]) if "modal_price" in df.columns else df
    if "commodity" in df.columns:
        df["commodity"] = df["commodity"].str.strip()
    return df, True


def summarize_by_commodity(df):
    """Stock-board rows: one per commodity, today's price stats across markets."""
    if df.empty or "commodity" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("commodity").agg(
        avg_modal=("modal_price", "mean"),
        min_price=("min_price", "min"),
        max_price=("max_price", "max"),
        markets=("market", "nunique"),
    ).reset_index()
    g = g.sort_values("avg_modal", ascending=False).reset_index(drop=True)
    return g


# ------------------------------------------------------------------
# Local day-over-day history, so the board can show a real trend once
# the app has been used across more than one day. Storage is a plain
# CSV per state on local disk — it persists for as long as the app's
# container/session stays alive, but is not a permanent database.
# ------------------------------------------------------------------
HISTORY_DIR = "price_history"


def _history_path(state):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    key = (state or "all_india").lower().replace(" ", "_")
    return os.path.join(HISTORY_DIR, f"{key}.csv")


def log_daily_snapshot(state, summary_df):
    """Append today's avg price per commodity. Safe to call every run —
    de-duplicates so the same day+commodity is only stored once (latest wins)."""
    if summary_df.empty:
        return
    try:
        path = _history_path(state)
        today = datetime.now().strftime("%Y-%m-%d")
        rows = summary_df[["commodity", "avg_modal"]].copy()
        rows["date"] = today
        if os.path.exists(path):
            existing = pd.read_csv(path)
            combined = pd.concat([existing, rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "commodity"], keep="last")
        else:
            combined = rows
        combined.to_csv(path, index=False)
    except Exception:
        pass  # history is a nice-to-have; never let it break the main view


def get_commodity_history(state, commodity):
    path = _history_path(state)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df = df[df["commodity"] == commodity].sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()


def naive_forecast(history_df, days_ahead=3):
    """Simple linear projection from locally logged history. Returns None
    if there isn't enough history yet to draw a trend from (< 2 days)."""
    if history_df is None or len(history_df) < 2:
        return None
    y = history_df["avg_modal"].values
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + days_ahead)
    future_y = np.polyval(coeffs, future_x)
    return future_y
