"""
Fetches today's mandi price snapshot from data.gov.in and saves it as a
local JSON file. Runs on GitHub Actions on a schedule.

Key lesson from production failures: requesting a large `limit` (e.g.
1000) from this particular API frequently times out server-side, even
from GitHub's normally-reliable network. So we fetch many SMALL pages
instead of few large ones, retry each page a couple of times, and keep
whatever we successfully collected rather than failing the whole run
if a handful of pages time out.
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

PAGE_SIZE = 100          # small pages — large ones time out on this API
MAX_PAGES = 40           # 40 x 100 = up to 4000 records
PAGE_TIMEOUT = 30        # seconds per attempt
RETRIES_PER_PAGE = 2
SLEEP_BETWEEN_PAGES = 1  # be polite / avoid rate limiting


def fetch_page(api_key, offset):
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    last_error = None
    for attempt in range(1, RETRIES_PER_PAGE + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=PAGE_TIMEOUT)
            r.raise_for_status()
            return r.json().get("records", [])
        except Exception as e:
            last_error = e
            print(f"  offset={offset} attempt {attempt} failed: {e}")
            time.sleep(2)
    print(f"  offset={offset} gave up after {RETRIES_PER_PAGE} attempts: {last_error}")
    return None  # signal total failure for this page


def fetch_all(api_key):
    all_records = []
    consecutive_failures = 0

    for page in range(MAX_PAGES):
        offset = page * PAGE_SIZE
        records = fetch_page(api_key, offset)

        if records is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("3 consecutive page failures — stopping early, keeping what we have.")
                break
            continue

        consecutive_failures = 0
        if not records:
            print(f"Page at offset {offset} returned 0 records — reached the end.")
            break

        all_records.extend(records)
        print(f"Page {page + 1} (offset {offset}): +{len(records)} records (total: {len(all_records)})")
        time.sleep(SLEEP_BETWEEN_PAGES)

    return all_records


def main():
    api_key = os.environ.get("DATA_GOV_API_KEY")
    if not api_key:
        print("ERROR: DATA_GOV_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    records = fetch_all(api_key)

    if not records:
        print("ERROR: fetched zero records across all attempts, refusing to overwrite existing snapshot", file=sys.stderr)
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
