"""
AGRINEX AI — Multi-Agent Orchestrator
========================================
A lightweight intent-routing layer on top of Groq. Instead of one
monolithic chatbot, the farmer's question is first classified into
an intent, then routed to the right specialized agent:

    crop_advice   -> CropAdvisorAgent (existing ML + RAG agent)
    disease       -> DiseaseAgent (reasons over uploaded disease result + treatment KB)
    market        -> MarketAgent (mandi price trend + selling-timing advice)
    general       -> falls through to a plain farming-assistant reply

This keeps each agent's prompt small and focused (better accuracy,
easier to eval independently — see eval/run_rag_eval.py) instead of
stuffing everything into one giant system prompt.
"""

import json


INTENT_LABELS = ["crop_advice", "disease", "market", "general"]


class DiseaseAgent:
    """Reasons over an already-computed CNN prediction + the disease_info KB."""

    def __init__(self, client, model_name="openai/gpt-oss-20b"):
        self.client = client
        self.model_name = model_name

    def explain(self, disease_label, confidence, severity, action, lang_instruction=""):
        prompt = f"""You are AGRINEX AI's plant disease advisor. A CNN model detected:
Disease: {disease_label} (confidence: {confidence:.1f}%, severity: {severity})
Base treatment note: {action}

{lang_instruction}

In 2-3 short sentences, explain what causes this in plain farmer-friendly language and
reinforce the urgency (or lack of it) based on severity. No markdown headers, under 80 words."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        return response.choices[0].message.content


class MarketAgent:
    """Reasons over recent mandi price records to give selling-timing advice."""

    def __init__(self, client, model_name="openai/gpt-oss-20b"):
        self.client = client
        self.model_name = model_name

    def advise(self, commodity, price_records, lang_instruction=""):
        records_text = "\n".join(
            f"- {r.get('market', '?')}, {r.get('state', '?')}: modal price ₹{r.get('modal_price', '?')} on {r.get('date', '?')}"
            for r in price_records[:10]
        ) or "No recent records available."

        prompt = f"""You are AGRINEX AI's market advisor. Recent mandi prices for {commodity}:
{records_text}

{lang_instruction}

In 2-3 short sentences, tell the farmer whether prices look like they're trending up, down,
or flat, and give one practical suggestion about selling timing. Under 80 words, no markdown."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        return response.choices[0].message.content


class Orchestrator:
    """Classifies farmer intent, then routes to the right specialized agent."""

    def __init__(self, groq_client, crop_agent=None, model_name="openai/gpt-oss-20b"):
        self.client = groq_client
        self.model_name = model_name
        self.crop_agent = crop_agent
        self.disease_agent = DiseaseAgent(groq_client, model_name)
        self.market_agent = MarketAgent(groq_client, model_name)

    def classify_intent(self, user_message):
        prompt = f"""Classify this farmer's message into exactly one intent label.
Labels: crop_advice, disease, market, general

Message: "{user_message}"

Respond ONLY with JSON: {{"intent": "<label>"}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30,
            )
            raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(raw)
            intent = parsed.get("intent", "general")
            return intent if intent in INTENT_LABELS else "general"
        except Exception:
            return "general"

    def route(self, user_message, context=None):
        """
        context: optional dict carrying whatever the calling tab already knows —
                 e.g. {"disease_result": {...}} or {"mandi_records": [...], "commodity": "Tomato"}
                 or {"ml_top3": [...], "inputs": {...}, "location": ..., "season": ...}
        """
        context = context or {}
        intent = self.classify_intent(user_message)

        if intent == "crop_advice" and self.crop_agent and "ml_top3" in context:
            result = self.crop_agent.recommend(
                ml_top3=context["ml_top3"],
                inputs=context["inputs"],
                location=context.get("location", "India"),
                season=context.get("season", "Kharif"),
            )
            return {"intent": intent, "answer": result["explanation"], "sources": result["retrieved_context"]}

        if intent == "disease" and "disease_result" in context:
            d = context["disease_result"]
            answer = self.disease_agent.explain(d["label"], d["confidence"], d["severity"], d["action"])
            return {"intent": intent, "answer": answer, "sources": []}

        if intent == "market" and "mandi_records" in context:
            answer = self.market_agent.advise(context.get("commodity", "this crop"), context["mandi_records"])
            return {"intent": intent, "answer": answer, "sources": context["mandi_records"][:5]}

        # general fallback — no specialized context available, or intent == general
        prompt = f"""You are AGRINEX AI's farming assistant. Answer briefly and practically:
"{user_message}"
Stay strictly on farming topics; if off-topic, politely redirect. Under 80 words."""
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )
        return {"intent": "general", "answer": resp.choices[0].message.content, "sources": []}
