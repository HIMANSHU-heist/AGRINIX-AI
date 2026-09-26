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
REQUEST_TIMEOUT = (25, 40)  # (connect, read) — kept short so the UI never looks "stuck"


def _request(params):
    return _session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)


@st.cache_data(ttl=900)
def fetch_state_snapshot(api_key, state=None, limit=500):
    """Returns (df, status). status is True on success (even if empty),
    or a string starting with 'ERROR:' describing exactly what failed —
    so the UI can show the real cause instead of a generic message."""
    if not api_key:
        return pd.DataFrame(), "ERROR: no DATA_GOV_API_KEY configured"

    params = {"api-key": api_key, "format": "json", "limit": limit}
    if state:
        params["filters[state]"] = state

    try:
        r = _request(params)
    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), f"ERROR: Timeout contacting data.gov.in: {e}"
    except requests.exceptions.ConnectionError as e:
        return pd.DataFrame(), f"ERROR: ConnectionError reaching data.gov.in: {e}"
    except Exception as e:
        return pd.DataFrame(), f"ERROR: {type(e).__name__}: {e}"

    if r.status_code != 200:
        return pd.DataFrame(), f"ERROR: HTTP {r.status_code} from data.gov.in: {r.text[:300]}"

    try:
        payload = r.json()
    except Exception as e:
        return pd.DataFrame(), f"ERROR: response wasn't valid JSON: {e} — raw start: {r.text[:200]}"

    records = payload.get("records", [])
    if not records:
        return pd.DataFrame(), True  # genuinely no rows for this filter today

    df = pd.DataFrame(records)
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["modal_price"]) if "modal_price" in df.columns else df
    if "commodity" in df.columns:
        df["commodity"] = df["commodity"].str.strip()
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
