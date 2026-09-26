import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mandi_snapshot.json")

INDIAN_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa",
    "Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland",
    "Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar Islands",
    "Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Delhi",
    "Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"
]

# ------------------------------------------------------------------
# PRIMARY PATH: read from data/mandi_snapshot.json, refreshed every
# few hours by .github/workflows/refresh-mandi-data.yml (runs on
# GitHub's runners, which have a far more reliable network path to
# api.data.gov.in than Streamlit Cloud does — Streamlit Cloud's calls
# were observed to reliably ConnectTimeout / ReadTimeout in production).
# FALLBACK PATH: a direct live API call, only used if no snapshot file
# exists yet (e.g. first deploy before the workflow has run once).
# ------------------------------------------------------------------

@st.cache_data(ttl=1800)
def _load_snapshot():
    """Returns (records, fetched_at_str) or (None, None) if no snapshot yet."""
    if not os.path.exists(SNAPSHOT_PATH):
        return None, None
    try:
        with open(SNAPSHOT_PATH) as f:
            data = json.load(f)
        return data.get("records", []), data.get("fetched_at")
    except Exception:
        return None, None


def _get_session():
    session = requests.Session()
    retry = Retry(total=1, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

_session = _get_session()
LIVE_TIMEOUT = (10, 20)


def _live_fetch(api_key, state=None, limit=500):
    params = {"api-key": api_key, "format": "json", "limit": limit}
    if state:
        params["filters[state]"] = state
    try:
        r = _session.get(BASE_URL, params=params, timeout=LIVE_TIMEOUT)
        r.raise_for_status()
        return r.json().get("records", []), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_state_snapshot(api_key, state=None, limit=500):
    """
    Returns (df, status). status is True on success, or a string
    starting with 'ERROR:' if both the local snapshot and the live
    fallback failed.
    """
    records, fetched_at = _load_snapshot()

    if records is not None:
        if state:
            records = [r for r in records if r.get("state", "").strip().lower() == state.strip().lower()]
        source_note = f"snapshot from {fetched_at}" if fetched_at else "local snapshot"
    else:
        # no snapshot on disk yet — try a direct live call as a one-time fallback
        if not api_key:
            return pd.DataFrame(), "ERROR: no snapshot file and no DATA_GOV_API_KEY configured"
        records, err = _live_fetch(api_key, state=state, limit=limit)
        if records is None:
            return pd.DataFrame(), f"ERROR: no local snapshot yet, and live fallback failed: {err}"
        source_note = "live fallback (no snapshot yet)"

    if not records:
        return pd.DataFrame(), True

    df = pd.DataFrame(records)
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["modal_price"]) if "modal_price" in df.columns else df
    if "commodity" in df.columns:
        df["commodity"] = df["commodity"].str.strip()
    df.attrs["source_note"] = source_note
    return df, True


def summarize_by_commodity(df):
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


HISTORY_DIR = "price_history"


def _history_path(state):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    key = (state or "all_india").lower().replace(" ", "_")
    return os.path.join(HISTORY_DIR, f"{key}.csv")


def log_daily_snapshot(state, summary_df):
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
        pass


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
    if history_df is None or len(history_df) < 2:
        return None
    y = history_df["avg_modal"].values
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + days_ahead)
    future_y = np.polyval(coeffs, future_x)
    return future_y
