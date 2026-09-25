# AGRINEX AI — Advanced Stack (MLOps + Agentic layer)

This folder adds production-grade AI-engineering pieces on top of the existing
`app.py` prototype. Nothing here replaces `app.py` — copy these files into
your project root (keeping the sub-folders) alongside your existing files.

## What's here

```
.github/workflows/
  ci.yml            -> lint + pytest + coverage on every push/PR
  rag-eval.yml       -> runs the golden Q&A set through CropAdvisorAgent on PRs
                        that touch agent/RAG code, comments scores on the PR
  deploy.yml         -> builds Docker image, pushes to GHCR, runs a smoke test

tests/
  test_crop_agent.py     -> unit tests for the TF-IDF RAG retriever
  test_disease_utils.py  -> unit tests for image preprocessing + prediction logic

eval/
  golden_qa.json       -> 5 hand-written farmer scenarios w/ expected keywords
  run_rag_eval.py       -> scores CropAdvisorAgent on context precision,
                           faithfulness, and answer relevancy (LLM-as-judge)

mcp/
  agrinex_mcp_server.py -> exposes mandi price / crop calendar / soil analysis
                           as standard MCP tools any MCP client can call

agents/
  orchestrator.py       -> intent classifier + router between CropAdvisorAgent,
                           DiseaseAgent, MarketAgent, and a general fallback

image_utils.py       -> preprocess_leaf_image() / predict_disease() pulled out
                        of app.py so they're testable without a Streamlit runtime
cache_utils.py        -> Redis-backed memoization for repeated LLM calls
Dockerfile / docker-compose.yml -> full stack: app + Redis + ChromaDB
requirements-dev.txt   -> pytest, ruff, mcp, redis (CI/dev only, not prod)
pytest.ini / ruff.toml -> tool configs
```

## Running things locally

```bash
# tests
pip install -r requirements-dev.txt
pytest tests/ -v

# lint
ruff check .

# RAG eval (needs GROQ_API_KEY set)
python eval/run_rag_eval.py --report eval/eval_report.json --fail-below 0.65

# full stack via Docker
cp .streamlit/secrets.toml .streamlit/secrets.env   # reformat to KEY=VALUE lines first
docker compose up --build
```

## Wiring the orchestrator into app.py's Farmer Assistant tab (Tab 7)

Replace the direct `client.chat.completions.create(...)` call in Tab 7 with:

```python
from agents.orchestrator import Orchestrator

orchestrator = Orchestrator(groq_client, crop_agent=crop_agent)
result = orchestrator.route(user_q, context={
    # pass whatever's relevant/available from other tabs' session_state
})
answer = result["answer"]
```

This gets you automatic intent routing (crop advice vs disease vs market vs
general chit-chat) instead of one giant system prompt trying to do everything.

## GitHub secrets needed for the workflows

- `GROQ_API_KEY_TEST` — a Groq key for CI test runs (separate from your prod key)
- `GITHUB_TOKEN` — provided automatically by GitHub Actions, no setup needed

## Next steps not yet automated here

- Model registry / experiment tracking (MLflow) for `crop_model.pkl` and
  `disease_model.keras` retraining runs
- A `model-training.yml` workflow that reruns `train_disease_model.py` on
  a schedule/dataset-change and opens a PR with the new model artifact
- Drift monitoring dashboard comparing live prediction distributions
  against the training distribution
