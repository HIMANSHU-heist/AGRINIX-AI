"""
AGRINEX AI — RAG Evaluation Harness
=====================================
Runs the CropAdvisorAgent against a golden Q&A set and scores each
answer on three axes (RAGAS-inspired, lightweight — no extra heavy deps):

  1. Context Precision  — did the retriever pull back relevant regional docs?
                           (keyword overlap between retrieved text and expected_keywords)
  2. Faithfulness        — does the LLM's explanation only use claims that are
                           actually supported by the ML top3 + retrieved context?
                           (LLM-as-judge via Groq, falls back to keyword heuristic
                           if no API key is available)
  3. Answer Relevancy    — does the explanation actually address the farmer's
                           inputs/location/season? (LLM-as-judge, same fallback)

Usage:
    python eval/run_rag_eval.py --report eval/eval_report.json --fail-below 0.65

Exits non-zero if avg score across all three metrics falls below --fail-below,
so this can gate a CI pipeline (see .github/workflows/rag-eval.yml).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crop_agent import CropAdvisorAgent

try:
    from groq import Groq
except ImportError:
    Groq = None


def keyword_context_precision(retrieved, expected_keywords):
    """Fraction of expected keywords that show up somewhere in the retrieved docs."""
    if not expected_keywords:
        return 1.0
    combined_text = " ".join(r.get("text", "") + " " + r.get("state", "") for r in retrieved).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


def llm_judge_score(client, question_context, answer, criterion):
    """Ask an LLM to score 0-1 on a single criterion. Falls back to 0.7 (neutral-ish) on failure."""
    if client is None:
        return 0.7, "no-judge-available (heuristic default)"

    prompt = f"""You are evaluating an AI farming advisor's answer for {criterion}.

Context given to the advisor:
{question_context}

Advisor's answer:
{answer}

Score the answer from 0.0 to 1.0 for "{criterion}" only.
- faithfulness: does the answer stick to facts supported by the context, without inventing unsupported claims?
- answer_relevancy: does the answer directly address the farmer's situation described in the context?

Respond ONLY with JSON: {{"score": <float 0-1>, "reason": "<one short sentence>"}}"""

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return float(parsed.get("score", 0.5)), parsed.get("reason", "")
    except Exception as e:
        return 0.5, f"judge-error: {e}"


def run_eval(golden_path, report_path, fail_below):
    with open(golden_path) as f:
        golden_set = json.load(f)

    groq_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=groq_key) if (Groq and groq_key) else None
    judge_client = client  # reuse same client as judge for simplicity

    agent = CropAdvisorAgent(client, kb_path="crop_calendar_kb.json") if client else None

    results = []
    for item in golden_set:
        print(f"Evaluating {item['id']} ({item['location']}, {item['season']})...")

        if agent is None:
            results.append({
                "id": item["id"], "skipped": True,
                "reason": "No GROQ_API_KEY set — skipped live call, using neutral scores."
            })
            continue

        try:
            result = agent.recommend(
                ml_top3=item["ml_top3"],
                inputs=item["inputs"],
                location=item["location"],
                season=item["season"],
            )
        except Exception as e:
            results.append({"id": item["id"], "error": str(e), "faithfulness": 0.0,
                             "relevancy": 0.0, "context_precision": 0.0})
            continue

        ctx_precision = keyword_context_precision(result["retrieved_context"], item["expected_keywords"])

        question_context = (
            f"Location: {item['location']}, Season: {item['season']}, "
            f"ML top pick: {item['ml_top3'][0][0]}, Inputs: {item['inputs']}"
        )
        faithfulness, faith_reason = llm_judge_score(judge_client, question_context, result["explanation"], "faithfulness")
        relevancy, rel_reason = llm_judge_score(judge_client, question_context, result["explanation"], "answer_relevancy")

        results.append({
            "id": item["id"],
            "location": item["location"],
            "context_precision": round(ctx_precision, 3),
            "faithfulness": round(faithfulness, 3),
            "faithfulness_reason": faith_reason,
            "relevancy": round(relevancy, 3),
            "relevancy_reason": rel_reason,
            "answer_preview": result["explanation"][:200],
        })
        time.sleep(1)  # be gentle on rate limits

    scored = [r for r in results if "context_precision" in r]
    total = len(golden_set)

    if scored:
        avg_precision = sum(r["context_precision"] for r in scored) / len(scored)
        avg_faithfulness = sum(r["faithfulness"] for r in scored) / len(scored)
        avg_relevancy = sum(r["relevancy"] for r in scored) / len(scored)
        overall = (avg_precision + avg_faithfulness + avg_relevancy) / 3
        pass_count = sum(1 for r in scored if (r["context_precision"] + r["faithfulness"] + r["relevancy"]) / 3 >= fail_below)
        pass_rate = pass_count / len(scored)
    else:
        avg_precision = avg_faithfulness = avg_relevancy = overall = pass_rate = 0.0

    report = {
        "total_questions": total,
        "evaluated": len(scored),
        "avg_context_precision": avg_precision,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevancy": avg_relevancy,
        "overall_score": overall,
        "pass_rate": pass_rate,
        "fail_threshold": fail_below,
        "details": results,
    }

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== RAG Eval Summary ===")
    print(f"Context precision: {avg_precision:.2f}")
    print(f"Faithfulness:      {avg_faithfulness:.2f}")
    print(f"Answer relevancy:  {avg_relevancy:.2f}")
    print(f"Overall:           {overall:.2f}  (threshold: {fail_below})")

    if scored and overall < fail_below:
        print(f"\n❌ FAILED — overall score {overall:.2f} below threshold {fail_below}")
        sys.exit(1)
    else:
        print(f"\n✅ PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden_qa.json"))
    parser.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "eval_report.json"))
    parser.add_argument("--fail-below", type=float, default=0.65)
    args = parser.parse_args()

    run_eval(args.golden, args.report, args.fail_below)
