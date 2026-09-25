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

@st.cache_data(ttl=3600)
def fetch_commodities(api_key, state=None, limit=1000):
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }
    if state:
        params["filters[state]"] = state
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        records = r.json().get("records", [])
        commodities = sorted(set(rec.get("commodity", "").strip() for rec in records if rec.get("commodity")))
        return commodities
    except Exception:
        return []

@st.cache_data(ttl=1800)
def fetch_mandi_prices(api_key, commodity, state=None, limit=100):
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
        "filters[commodity]": commodity,
    }
    if state:
        params["filters[state]"] = state
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        records = r.json().get("records", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        for col in ["min_price", "max_price", "modal_price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "arrival_date" in df.columns:
            df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
            df = df.sort_values("arrival_date")
        return df
    except Exception:
        return pd.DataFrame()
