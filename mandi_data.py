import requests
import streamlit as st
import pandas as pd

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
# Not every crop trades in every state every day, so an exact
# state+commodity filter often legitimately returns zero rows.
# To stay useful we (1) fetch a generous sample, (2) filter client-side
# with case-insensitive substring matching instead of a brittle exact
# server-side filter, and (3) fall back to all-India if a state has
# no arrivals for that crop today.

@st.cache_data(ttl=1800)
def fetch_commodities(api_key, state=None, limit=2000):
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }
    if state:
        params["filters[state]"] = state
    try:
        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        records = r.json().get("records", [])
        commodities = sorted(set(rec.get("commodity", "").strip() for rec in records if rec.get("commodity")))
        return commodities
    except Exception:
        return []


@st.cache_data(ttl=900)
def fetch_mandi_prices(api_key, commodity, state=None, limit=500):
    """
    Fetches a broad sample and filters client-side (case-insensitive,
    substring match) instead of relying on an exact server-side filter —
    the API's exact-match filtering is picky about casing/spacing and
    this dataset only ever has ~1 day of arrivals per commodity, so a
    strict filter frequently returns nothing even when the crop *is*
    in the dataset under a slightly different label (e.g. "Tomato" vs
    "Tomato Hybrid").
    """
    def _query(state_filter):
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": limit,
        }
        if state_filter:
            params["filters[state]"] = state_filter
        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("records", [])

    try:
        records = _query(state)
    except Exception:
        return pd.DataFrame()

    commodity_lower = commodity.strip().lower()
    matched = [rec for rec in records if commodity_lower in rec.get("commodity", "").lower()]

    # Fallback: if a state filter was given but nothing matched, retry all-India
    if not matched and state:
        try:
            records = _query(None)
            matched = [rec for rec in records if commodity_lower in rec.get("commodity", "").lower()]
        except Exception:
            pass

    if not matched:
        return pd.DataFrame()

    df = pd.DataFrame(matched)
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
        df = df.sort_values("arrival_date")
    return df
