import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
# Not every crop trades in every state every day.

# ------------------------------------------------------------------
# Shared session with retries + generous, split connect/read timeouts.
# data.gov.in is a slow/flaky public API — a single 20s hard timeout
# with no retry means any brief slowdown surfaces as a hard failure.
# ------------------------------------------------------------------
def _get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,  # 0s, 1.5s, 3s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

_session = _get_session()

# (connect_timeout, read_timeout) — give the read side real headroom
REQUEST_TIMEOUT = (10, 45)


def _request(params):
    """Single place all calls go through, so timeout/retry behavior is consistent."""
    return _session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)


@st.cache_data(ttl=1800)
def fetch_commodities(api_key, state=None, limit=500):
    # Smaller default limit than before (was 2000): we only need the
    # *set* of commodity names, and this dataset is a same-day snapshot,
    # so 500 records already gives a representative list while returning
    # far faster and more reliably than pulling the full 2000-row page.
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }
    if state:
        params["filters[state]"] = state
    try:
        r = _request(params)
        debug_info = {"status_code": r.status_code, "url": r.url}
        r.raise_for_status()
        data = r.json()
        records = data.get("records", [])
        commodities = sorted(set(rec.get("commodity", "").strip() for rec in records if rec.get("commodity")))
        return commodities, {"ok": True, "record_count": len(records), **debug_info}
    except requests.exceptions.Timeout as e:
        return [], {"ok": False, "timeout": True, "error": str(e), "url": BASE_URL}
    except Exception as e:
        return [], {"ok": False, "timeout": False, "error": str(e), "url": BASE_URL}


@st.cache_data(ttl=900)
def fetch_mandi_prices(api_key, commodity, state=None, limit=500):
    def _query(state_filter):
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": limit,
        }
        if state_filter:
            params["filters[state]"] = state_filter
        r = _request(params)
        r.raise_for_status()
        return r.json().get("records", [])

    debug = {}
    try:
        records = _query(state)
        debug["first_query_count"] = len(records)
    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), {"ok": False, "timeout": True, "error": str(e)}
    except Exception as e:
        return pd.DataFrame(), {"ok": False, "timeout": False, "error": str(e)}

    commodity_lower = commodity.strip().lower()
    matched = [rec for rec in records if commodity_lower in rec.get("commodity", "").lower()]
    debug["matched_in_first_query"] = len(matched)

    if not matched and state:
        try:
            records = _query(None)
            debug["fallback_query_count"] = len(records)
            matched = [rec for rec in records if commodity_lower in rec.get("commodity", "").lower()]
            debug["matched_in_fallback"] = len(matched)
        except requests.exceptions.Timeout as e:
            debug["fallback_timeout"] = str(e)
        except Exception as e:
            debug["fallback_error"] = str(e)

    if not matched:
        # surface a few sample commodity names actually present, to help debugging
        debug["sample_commodities_seen"] = sorted(set(r.get("commodity", "") for r in records))[:15]
        return pd.DataFrame(), {"ok": True, **debug}

    df = pd.DataFrame(matched)
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y", errors="coerce")
        df = df.sort_values("arrival_date")
    return df, {"ok": True, **debug}
