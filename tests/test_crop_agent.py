import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from crop_agent import CropCalendarRAG


@pytest.fixture
def kb_path(tmp_path):
    kb = [
        {"state": "Maharashtra", "season": "Kharif", "text": "Farmers in Maharashtra commonly grow rice, cotton and soybean during Kharif season due to monsoon rainfall."},
        {"state": "Punjab", "season": "Rabi", "text": "Punjab is well known for wheat cultivation during the Rabi season, supported by canal irrigation."},
        {"state": "Kerala", "season": "Zaid/Summer", "text": "Kerala's humid coastal climate supports coconut and banana plantations year-round."},
    ]
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(kb))
    return str(path)


def test_rag_loads_docs(kb_path):
    rag = CropCalendarRAG(kb_path=kb_path)
    assert len(rag.docs) == 3


def test_rag_retrieves_relevant_state(kb_path):
    rag = CropCalendarRAG(kb_path=kb_path)
    results = rag.retrieve("Maharashtra Kharif crops", top_k=2)
    assert len(results) >= 1
    assert any(r["state"] == "Maharashtra" for r in results)


def test_rag_retrieve_returns_score(kb_path):
    rag = CropCalendarRAG(kb_path=kb_path)
    results = rag.retrieve("Punjab wheat Rabi", top_k=1)
    assert results
    assert "score" in results[0]
    assert 0 <= results[0]["score"] <= 1


def test_rag_empty_query_no_crash(kb_path):
    rag = CropCalendarRAG(kb_path=kb_path)
    results = rag.retrieve("xyzabc nonsense query", top_k=3)
    # low-similarity queries may return nothing — should not raise
    assert isinstance(results, list)
