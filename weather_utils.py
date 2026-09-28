"""
Live weather via Open-Meteo (https://open-meteo.com) — free, no API key
required, and reliably reachable from any cloud host (unlike data.gov.in
in our experience). Good fit for a farmer-facing 7-day forecast.
"""

import requests
import streamlit as st

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌦️",
    61: "🌧️", 63: "🌧️", 65: "🌧️",
    71: "🌨️", 73: "🌨️", 75: "🌨️",
    80: "🌦️", 81: "🌧️", 82: "⛈️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}


@st.cache_data(ttl=3600)
def geocode_location(place_name):
    """Turns a free-text location like 'Nashik, Maharashtra' into lat/lon."""
    try:
        r = requests.get(GEOCODE_URL, params={"name": place_name, "count": 1}, timeout=10)
        r.raise_for_status()
        results = r.json().get("results")
        if not results:
            return None
        top = results[0]
        return {"lat": top["latitude"], "lon": top["longitude"], "resolved_name": f"{top.get('name')}, {top.get('admin1', '')}"}
    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_7day_forecast(lat, lon):
    """Returns a dict with daily arrays: dates, max/min temp, rain probability, weather codes."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": "auto",
        "forecast_days": 7,
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("daily")
    except Exception:
        return None


def icon_for_code(code):
    return WEATHER_CODE_ICONS.get(code, "🌡️")
