# 🌾 AGRINEX AI

AGRINEX AI is an intelligent agriculture platform built to help Indian farmers make better decisions — combining trained ML models, a RAG-based advisory agent, live government/weather data, and a multi-agent architecture, all wrapped in a Streamlit app with production-grade CI/CD around it.

Live app: *(add your Streamlit Cloud URL here)*

---

## ✅ What's actually live right now

| Module | Status | How it works |
|---|---|---|
| 🌱 **Crop Recommendation** | 🟢 LIVE | Random Forest model trained on N-P-K, temperature, humidity, pH, rainfall → predicts best-fit crop across 22 classes, with confidence scores and top-3 alternatives. |
| 🤖 **Crop Advisor Agent (RAG)** | 🟢 LIVE | Sits on top of the ML model — retrieves regional crop-calendar knowledge (TF-IDF over a curated knowledge base) and uses Groq (Llama 3.1) to explain *why* a crop fits, and flags regionally-common crops the ML model wasn't even trained on. |
| 🍃 **Disease Detection** | 🟢 LIVE | MobileNetV2 CNN, transfer-learned on the PlantVillage dataset (15 classes across Tomato, Potato, Pepper), with OpenCV preprocessing on uploaded leaf photos. Returns disease, confidence, severity, and a treatment action. |
| ☁️ **Weather Intelligence** | 🟢 LIVE | Real 7-day forecast via Open-Meteo (free, no API key, globally reliable) geocoded from the farmer's own location — temperature, rain probability, and a plain-language spraying/harvest recommendation. |
| 💰 **Market Price Board** | 🟢 LIVE (with graceful fallback) | Pulls today's mandi arrivals from data.gov.in (Agmarknet) for any Indian state, shows a sortable price board across every crop trading that day, drill-down per-crop chart across markets, and a locally-built day-over-day price trend + naive projection that improves the longer the app is used. Falls back to the last known snapshot if the live government API is slow (a known characteristic of this particular dataset). |
| 💬 **Farmer Assistant Chatbot** | 🟢 LIVE | Groq-powered (Llama 3.1) multilingual chat (Marathi / Hindi / English) with reply-to-message threading, scoped to farming topics. |
| 🎭 **Multi-Agent Orchestrator** | 🟢 BUILT | Intent-classification layer that routes a farmer's question to a specialized sub-agent (crop advice / disease / market / general) instead of one giant prompt trying to do everything — see `agents/orchestrator.py`. |
| 🔌 **MCP Server** | 🟢 BUILT | Exposes mandi price lookup, crop-calendar retrieval, and soil analysis as standard MCP tools (`mcp/agrinex_mcp_server.py`), so any MCP-compatible agent can call AGRINEX's data sources directly. |
| 🧪 **CI/CD Pipeline** | 🟢 LIVE | GitHub Actions: lint (ruff) + unit tests + coverage on every push; a scheduled/manual RAG-evaluation pipeline scoring the advisor agent on faithfulness, relevancy, and context precision against a golden Q&A set; Docker build + smoke test on merge to `main`. |
| 📊 **RAG Evaluation Harness** | 🟢 BUILT | `eval/run_rag_eval.py` — LLM-as-judge scoring of the Crop Advisor Agent against 5 hand-written farmer scenarios, gating CI if quality regresses. |
| 🐳 **Containerization** | 🟢 BUILT | `Dockerfile` + `docker-compose.yml` (app + Redis cache + ChromaDB, for future vector search). |
| 🧊 **Caching layer** | 🟢 BUILT | Redis-backed memoization (`cache_utils.py`) ready to wrap expensive LLM calls. |

---

## 🟡 Demo / illustrative UI (not yet wired to real data or models)

| Module | Status | Notes |
|---|---|---|
| 📈 **Yield Prediction** | 🟡 Demo | Rule-based multiplier math on user-entered inputs — no trained yield model yet. |
| 🧪 **Soil Analysis** | 🟡 Demo | Rule-based thresholds on manually entered pH/N/organic carbon — no OCR on uploaded soil reports yet, no real lab-report parsing. |
| 🛒 **Marketplace** (crop listings + farm labour) | 🟡 Demo | UI only — no persistent storage, no real buyer/worker matching yet. |

## 🔴 Planned, not started

Drone analysis · Satellite monitoring · IoT sensor integration · Equipment rental · Agricultural input marketplace · Voice-based interaction · Farm analytics dashboard · AI risk analysis (combined disease/weather/market risk score) · Personalized farmer profile persistence across sessions.

---

## Architecture

```text
                         AGRINEX AI
                              |
                    Streamlit Farmer Interface
                              |
                    Multi-Agent Orchestrator  ──── MCP Server (mandi/soil/calendar tools)
                              |
        +---------------------+----------------------+
        |                     |                       |
        ↓                     ↓                       ↓
  Crop ML Model      Crop Advisor Agent (RAG)    Disease CNN Model
  (Random Forest)     (TF-IDF + Groq/Llama)      (MobileNetV2 + OpenCV)
        |                     |                       |
        ↓                     ↓                       ↓
  Crop Prediction    Regional Explanation      Disease + Treatment
                              |
                    +----------------------+
                    |                      |
                    ↓                      ↓
            Live Weather (Open-Meteo)   Live Mandi Prices (Agmarknet)
```

## Tech stack

Python · Streamlit · scikit-learn · TensorFlow/Keras · OpenCV · Groq (Llama 3.1) · TF-IDF RAG · Open-Meteo · data.gov.in (Agmarknet) · GitHub Actions · Docker · Redis · ChromaDB · MCP

## Repo layout

```
app.py                     Main Streamlit app (all 8 tabs)
crop_agent.py               RAG-based Crop Advisor Agent
mandi_data.py                Live mandi price fetching + local history/trend
weather_utils.py             Open-Meteo live weather
image_utils.py                Testable disease-detection preprocessing
cache_utils.py                 Redis memoization helper
agents/orchestrator.py          Multi-agent intent router
mcp/agrinex_mcp_server.py        MCP tool server
eval/                              RAG evaluation harness + golden Q&A set
tests/                              Unit tests
scripts/fetch_mandi_snapshot.py     Manual mandi data refresh script
.github/workflows/                   CI, RAG-eval, deploy pipelines
Dockerfile / docker-compose.yml        Containerized full stack
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs `.streamlit/secrets.toml` with:
```toml
GROQ_API_KEY = "..."
DATA_GOV_API_KEY = "..."   # optional — Market Price tab falls back gracefully without it
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

---

## A known limitation worth knowing about

The government's mandi-price API (data.gov.in / Agmarknet) has proven unreliable to call **from cloud infrastructure** (both GitHub Actions and Streamlit Cloud observed frequent timeouts / 502s), while working fine from a residential/mobile Indian network. The app is built to degrade gracefully around this — falling back to the last successfully fetched data — rather than pretending the live call always succeeds.

---

*AGRINEX AI — a work in progress, built module by module.*
