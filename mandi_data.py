import os
import json
import requests
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

@st.cache_data(ttl=1800)
def _load_snapshot():
    if not os.path.exists(SNAPSHOT_PATH):
        return None
    try:
        with open(SNAPSHOT_PATH) as f:
            data = json.load(f)
        return data.get("records", [])
    except Exception:
        return None


@st.cache_data(ttl=900)
def _live_fetch(api_key, state=None, limit=300):
    params = {"api-key": api_key, "format": "json", "limit": limit}
    if state:
        params["filters[state]"] = state
    try:
        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("records", [])
    except Exception:
        return None


def fetch_state_snapshot(api_key, state=None, limit=300):
    """Returns a DataFrame of today's mandi records for the given state
    (or all-India). Tries a live API call first; if that's slow/unavailable,
    quietly falls back to the last committed snapshot file, if any."""
    records = _live_fetch(api_key, state=state, limit=limit) if api_key else None

    if not records:
        snapshot_records = _load_snapshot()
        if snapshot_records:
            records = snapshot_records
            if state:
                records = [r for r in records if r.get("state", "").strip().lower() == state.strip().lower()]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["modal_price"]) if "modal_price" in df.columns else df
    if "commodity" in df.columns:
        df["commodity"] = df["commodity"].str.strip()
    return df


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
