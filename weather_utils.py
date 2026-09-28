"""
Live weather via Open-Meteo (https://open-meteo.com) — free, no API key
required, and reliably reachable from any cloud host (unlike data.gov.in
in our experience). Good fit for a farmer-facing 7-day forecast.
"""

import json
import re
from datetime import datetime

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


def _clean_text(t):
    """Removes markdown symbols so text looks neat in the UI."""
    t = re.sub(r"[*_`#>]+", "", t or "")
    return re.sub(r"\s+", " ", t).strip()


def generate_weather_advisory(groq_client, location, crop, daily, model_name="openai/gpt-oss-20b"):
    """
    Uses the LLM to REASON over the real fetched 7-day forecast (temps, rain
    probability, wind) for a specific crop — never to invent weather data.

    Returns a dict: {headline, key_risk, best_day, action}
    (or {"raw": text} if the model didn't return valid JSON).
    """
    dates = daily["time"]
    lines = []
    for i in range(len(dates)):
        day_name = datetime.strptime(dates[i], "%Y-%m-%d").strftime("%A")
        rain = daily["precipitation_probability_max"][i]
        wind = daily["windspeed_10m_max"][i]
        lines.append(
            f"{day_name} ({dates[i]}): {daily['temperature_2m_max'][i]:.0f}/{daily['temperature_2m_min'][i]:.0f}°C, "
            f"{rain if rain is not None else 'n/a'}% rain chance, "
            f"wind up to {wind if wind is not None else 0:.0f} km/h"
        )
    forecast_text = "\n".join(lines)

    prompt = f"""You are AGRINEX AI's farming weather advisor. REAL 7-day forecast for {location}
(use only these numbers, never invent others):

{forecast_text}

Crop: {crop}

Reply with ONLY a JSON object with exactly these 4 keys. Each value must be ONE short plain-text
sentence (max 20 words). No markdown, no asterisks, no emojis.
{{
  "headline": "one-line summary of this week's weather for {crop}",
  "key_risk": "the single most important risk (disease, wind, water stress, etc.)",
  "best_day": "best day name(s) for spraying/irrigation/harvest and which activity",
  "action": "one practical thing the farmer should do now"
}}"""

        response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
        response_format={"type": "json_object"},
        extra_body={"reasoning_effort": "low"},   # old SDK madhe pan chalto
    )
    text = response.choices[0].message.content or ""
    try:
        data = json.loads(text)
        return {k: _clean_text(str(data.get(k, ""))) for k in ("headline", "key_risk", "best_day", "action")}
    except Exception:
        return {"raw": _clean_text(text)}
