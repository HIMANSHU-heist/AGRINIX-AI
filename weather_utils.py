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
    """Returns a dict with daily arrays: dates, max/min temp, rain probability,
    max wind speed (km/h), and weather codes."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max,weathercode",
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


def generate_weather_advisory(groq_client, location, crop, daily, model_name="openai/gpt-oss-20b"):
    """
    Uses the LLM to REASON over the real fetched 7-day forecast (temps, rain
    probability, wind) for a specific crop — never to invent weather data,
    only to translate real numbers into a practical farming decision.
    """
    dates = daily["time"]
    lines = []
    for i in range(len(dates)):
        lines.append(
            f"{dates[i]}: {daily['temperature_2m_max'][i]:.0f}/{daily['temperature_2m_min'][i]:.0f}°C, "
            f"{daily['precipitation_probability_max'][i]}% rain chance, "
            f"wind up to {daily['windspeed_10m_max'][i]:.0f} km/h"
        )
    forecast_text = "\n".join(lines)

    prompt = f"""You are AGRINEX AI's farming weather advisor. Here is the REAL 7-day forecast
for {location} (from Open-Meteo — use only these numbers, never invent different ones):

{forecast_text}

The farmer is growing (or considering): {crop}.

In 3-4 short sentences, plain and practical:
1. Call out the single most important thing this week's weather means for {crop} (irrigation need, disease/fungus risk from humidity+rain, wind damage risk, or a good spraying/harvest window).
2. Recommend the best day(s) this week for a specific field activity (spraying, irrigation, harvesting) based on the actual numbers above.
No markdown headers, no invented numbers beyond what's given above."""

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=220,
    )
    return response.choices[0].message.content
