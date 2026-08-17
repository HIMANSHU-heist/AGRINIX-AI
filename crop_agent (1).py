"""
AGRINEX AI — Crop Advisor Agent
=================================
This is the "Agentic AI" layer that sits ON TOP of the ML model.

What it does:
  1. Takes the ML model's top-3 crop predictions (statistical best-fit
     from the crops it was TRAINED on).
  2. Retrieves relevant regional crop-calendar knowledge (RAG — TF-IDF
     retrieval over crop_calendar_kb.json, no heavy vector-DB dependency
     needed for this size of knowledge base).
  3. Calls an LLM (Groq/Llama) to reason over BOTH sources and produce:
       - a natural-language explanation of the ML pick
       - a flag if the farmer's region/season commonly grows a crop the
         ML model isn't even aware of (covers the ML model's blind spots)
       - a short, farmer-friendly recommendation

This keeps the ML model as the accuracy backbone (it's still what actually
scores crop-fitness from soil/climate numbers) while the agent adds
regional context and explains things in plain language — this is the
"hybrid ML + Agent + RAG" pattern, not a replacement of the ML model.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CropCalendarRAG:
    """Lightweight RAG retriever over the regional crop-calendar knowledge base."""

    def __init__(self, kb_path="crop_calendar_kb.json"):
        with open(kb_path) as f:
            self.docs = json.load(f)
        self.texts = [f"{d['state']} {d['season']} {d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            if sims[i] > 0.05:  # ignore near-zero matches
                results.append({**self.docs[i], "score": float(sims[i])})
        return results


class CropAdvisorAgent:
    """
    Orchestrates: ML prediction (given) + RAG retrieval + LLM reasoning
    → final farmer-facing recommendation.
    """

    def __init__(self, groq_client, kb_path="crop_calendar_kb.json", model_name="llama-3.1-8b-instant"):
        self.client = groq_client
        self.rag = CropCalendarRAG(kb_path)
        self.model_name = model_name

    def recommend(self, ml_top3, inputs, location, season, lang_instruction=""):
        """
        ml_top3: list of (crop_name, confidence_pct) from the ML model, best first
        inputs: dict of N,P,K,temperature,humidity,ph,rainfall the farmer entered
        location: farmer's location string, e.g. "Nashik, Maharashtra"
        season: "Kharif" / "Rabi" / "Zaid/Summer"
        """
        query = f"{location} {season} crops"
        retrieved = self.rag.retrieve(query, top_k=3)
        retrieved_text = "\n".join(f"- ({r['state']}, {r['season']}) {r['text']}" for r in retrieved) or "No specific regional data found."

        ml_crops = {c for c, _ in ml_top3}
        top3_text = "\n".join(f"- {c}: {s:.1f}% model confidence" for c, s in ml_top3)

        system_prompt = f"""You are the AGRINEX AI crop advisory agent. You combine an ML model's
statistical crop recommendation with regional crop-calendar knowledge to give a farmer a clear,
trustworthy recommendation.

Farmer inputs: N={inputs['N']}, P={inputs['P']}, K={inputs['K']}, temperature={inputs['temperature']}C,
humidity={inputs['humidity']}%, pH={inputs['ph']}, rainfall={inputs['rainfall']}mm.
Farmer location: {location}. Season: {season}.

ML model's top-3 recommendations (from soil/climate fit, trained on {len(ml_crops)} known crops):
{top3_text}

Retrieved regional crop-calendar knowledge for this location/season:
{retrieved_text}

{lang_instruction}

Your task:
1. Explain in 2-3 short sentences why the top ML pick suits these soil/climate conditions.
2. If the retrieved regional knowledge mentions a locally common crop for this season that is NOT
   in the ML model's top-3, mention it as an additional option worth considering — but clearly
   label it as "based on regional farming patterns, not the soil-fit model" since the ML model
   wasn't scored on it.
3. Keep the whole answer under 120 words, practical and farmer-friendly. No markdown headers."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": "Give me the recommendation."}],
            temperature=0.4,
            max_tokens=300,
        )
        explanation = response.choices[0].message.content

        # Also surface any regionally-flagged crop the ML never scored, as structured data
        regional_extra_crops = []
        for r in retrieved:
            for word in r["text"].replace(",", " ").replace("(", " ").replace(")", " ").split():
                pass  # kept simple: the LLM does the extraction in prose above

        return {
            "explanation": explanation,
            "retrieved_context": retrieved,
        }
