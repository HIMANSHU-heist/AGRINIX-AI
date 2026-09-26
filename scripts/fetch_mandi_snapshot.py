"""
Fetches today's full mandi price snapshot from data.gov.in and saves it
as a local JSON file. Meant to run on GitHub Actions (stable network to
Indian government APIs), not on Streamlit Cloud (unreliable/timeout-prone
network path observed in production).

Usage:
    DATA_GOV_API_KEY=xxx python scripts/fetch_mandi_snapshot.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mandi_snapshot.json")

# Pull in pages to get broad coverage without one giant slow request.
PAGE_SIZE = 1000
MAX_PAGES = 10  # up to 10,000 records — plenty for a same-day snapshot


def fetch_all(api_key):
    all_records = []
    for page in range(MAX_PAGES):
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": PAGE_SIZE,
            "offset": page * PAGE_SIZE,
        }
        r = requests.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        records = r.json().get("records", [])
        if not records:
            break
        all_records.extend(records)
        print(f"Fetched page {page + 1}: {len(records)} records (total so far: {len(all_records)})")
        time.sleep(1)  # be polite to the API
    return all_records


def main():
    api_key = os.environ.get("DATA_GOV_API_KEY")
    if not api_key:
        print("ERROR: DATA_GOV_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    records = fetch_all(api_key)
    if not records:
        print("ERROR: fetched zero records, refusing to overwrite existing snapshot", file=sys.stderr)
        sys.exit(1)

    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(snapshot, f)

    print(f"Saved {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
