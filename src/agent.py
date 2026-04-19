"""
Agentic AI Lending Decision Support — 3-Step Pipeline

Step 1: Predict  → call existing /predict API
Step 2: Explain  → SHAP-based feature contribution analysis
Step 3: Decide   → LLM (HuggingFace Inference API) → structured JSON report
                   with 3-layer parser and rule-based fallback

This module is the core of Milestone 2's agentic AI system.
"""

import json
import os
import re

import requests
import streamlit as st
import joblib
import pandas as pd

from src.explainer import get_shap_explanation
from src.preprocess import feature_engineering


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HuggingFace Inference API (free tier — stable seq2seq)
HF_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_TIMEOUT = 20  # seconds

# Backend API for predictions (OBOSLETE - Removed)
# BACKEND_URL = "https://credit-risk-score-00ut.onrender.com"


def _get_hf_token() -> str:
    """Retrieve HuggingFace token from Streamlit secrets or environment."""
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Report schema keys (required in every output)
# ---------------------------------------------------------------------------

REPORT_KEYS = {
    "title": str,
    "summary": str,
    "risk_category": str,
    "decision": str,
    "confidence": str,
    "key_findings": list,
    "reasoning": str,
    "conditions": list,
    "recommendation": str,
    "conclusion": str,
    "sources": list,
}


# ---------------------------------------------------------------------------
# Step 1: Predict (calls existing backend API)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_model():
    """Load the machine learning model from disk, cached for performance."""
    return joblib.load("models/final_credit_risk_model.pkl")


def predict_risk(borrower_data: dict) -> float | None:
    """
    Predict risk probability locally using the loaded model.
    Returns float in [0, 1] or None on failure.
    """
    try:
        model = _load_model()
        df = pd.DataFrame([borrower_data])
        df_fe = feature_engineering(df.copy())
        risk = model.predict_proba(df_fe)[0][1]
        return float(risk)
    except Exception as e:
        print(f"[agent] Local prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 2: Explain (SHAP — delegated to explainer module)
# ---------------------------------------------------------------------------

def explain_risk(borrower_data: dict) -> list:
    """
    Get top SHAP-based contributing factors.
    Returns list of factor dicts, or empty list on failure.
    """
    return get_shap_explanation(borrower_data, top_n=3)


# ---------------------------------------------------------------------------
# Step 3a: LLM decision via HuggingFace Inference API
# ---------------------------------------------------------------------------

def _build_prompt(borrower_data: dict, risk_probability: float, shap_factors: list) -> str:
    """Build the structured prompt for the LLM."""

    # Format SHAP factors into readable text
    if shap_factors:
        shap_lines = []
        for i, f in enumerate(shap_factors, 1):
            shap_lines.append(
                f"  {i}. {f['display_name']} (value: {f['actual_value']:.2f}) — "
                f"{f['direction']} (impact: {f['impact']:.4f})"
            )
        shap_summary = "\n".join(shap_lines)
    else:
        shap_summary = "  (Feature-level analysis unavailable)"

    # Determine the correct decision per the rules
    if risk_probability < 0.3:
        expected_decision = "APPROVE"
    elif risk_probability < 0.6:
        expected_decision = "CONDITIONAL"
    else:
        expected_decision = "REJECT"

    prompt = f"""You are a senior credit risk analyst at a financial institution.

TASK: Analyze the borrower profile below and produce a lending decision report.

BORROWER PROFILE:
- Risk Score (default probability): {risk_probability:.1%}
- Age: {borrower_data.get('age', 'N/A')}
- Monthly Income: ${borrower_data.get('monthly_inc', 0):,.0f}
- Dependents: {borrower_data.get('dependents', 0):.0f}
- Revolving Utilization: {borrower_data.get('rev_util', 0):.1%}
- Debt Ratio: {borrower_data.get('debt_ratio', 0):.2f}
- Late Payments (30-59 days): {borrower_data.get('late_30_59', 0):.0f}
- Late Payments (60-89 days): {borrower_data.get('late_60_89', 0):.0f}
- Late Payments (90+ days): {borrower_data.get('late_90', 0):.0f}
- Open Credit Lines: {borrower_data.get('open_credit', 0):.0f}
- Real Estate Loans: {borrower_data.get('real_estate', 0):.0f}

TOP RISK FACTORS (from model explainability analysis):
{shap_summary}

DECISION RULES (you MUST follow these strictly):
- Risk score < 30%: Decision MUST be "{expected_decision if risk_probability < 0.3 else 'APPROVE'}"
- Risk score 30%-60%: Decision MUST be "CONDITIONAL"
- Risk score > 60%: Decision MUST be "REJECT"
- The decision for this borrower MUST be: "{expected_decision}"

ANTI-HALLUCINATION RULES:
- Strictly base your reasoning on the Risk Score and Top Risk Factors provided above.
- Do not introduce new assumptions, external factors, or speculative financial advice beyond the given data.
- Ensure all logic references the numerical impact of the SHAP values directly.

OUTPUT FORMAT: Return ONLY a valid JSON object with NO additional text before or after it. No markdown fencing. No explanation outside the JSON. The JSON must have exactly these keys:

{{
  "title": "Credit Risk Assessment Report",
  "summary": "1-2 sentence overview of the applicant risk profile",
  "risk_category": "Low Risk" or "Medium Risk" or "High Risk",
  "decision": "{expected_decision}",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "key_findings": ["finding 1 referencing a specific factor", "finding 2", "finding 3"],
  "reasoning": "3-5 sentence paragraph explaining the analysis based on the data above",
  "conditions": ["condition 1", "condition 2"] or [] if APPROVE or REJECT,
  "recommendation": "specific actionable lending recommendation",
  "conclusion": "Final 1-sentence verdict",
  "sources": ["ML Model Prediction (Risk Score)", "SHAP Feature Analysis", "Input Data"]
}}

Return ONLY the JSON. No other text."""

    return prompt


def _call_llm(prompt: str) -> str | None:
    """
    Call HuggingFace Inference API.
    Returns raw generation text, or None on failure.
    """
    token = _get_hf_token()
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,  # T5-base limits
            "temperature": 0.1,
            "top_p": 0.9,
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=HF_TIMEOUT,
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return None

        print(f"[agent] HF API returned status {response.status_code}. (HuggingFace Serverless API might be down or token invalid)")
        return None

    except requests.exceptions.Timeout:
        print("[agent] HF API timed out.")
        return None
    except Exception as e:
        print(f"[agent] HF API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3-Layer JSON Parser
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_text: str) -> dict | None:
    """
    Attempt to extract a JSON object from LLM output using 3 strategies.

    Layer 1: Direct JSON parse
    Layer 2: Regex extraction (code fences / embedded JSON)
    Layer 3: Return None → triggers rule-based fallback
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # Layer 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Layer 2: Regex extraction
    patterns = [
        r'```json\s*(.*?)\s*```',                     # ```json ... ```
        r'```\s*(.*?)\s*```',                          # ``` ... ```
        r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',          # outermost { ... }
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Layer 3: Failed
    print(f"[agent] Could not parse JSON from LLM response: {text[:300]}")
    return None


def _validate_report(parsed: dict | None, risk_probability: float) -> dict | None:
    """
    Validate parsed JSON against the required schema.
    Fill missing keys with safe defaults if partially valid.
    Returns None if input is None (signals full fallback needed).
    """
    if parsed is None:
        return None

    # Ensure all required keys exist with correct types
    for key, expected_type in REPORT_KEYS.items():
        if key not in parsed or not isinstance(parsed[key], expected_type):
            if expected_type == str:
                parsed[key] = ""
            elif expected_type == list:
                parsed[key] = []

    # Normalize decision to valid enum
    valid_decisions = {"APPROVE", "CONDITIONAL", "REJECT"}
    parsed["decision"] = parsed["decision"].upper().strip()
    if parsed["decision"] not in valid_decisions:
        parsed["decision"] = _classify_by_score(risk_probability)

    # Normalize risk_category
    valid_categories = {"Low Risk", "Medium Risk", "High Risk"}
    if parsed["risk_category"] not in valid_categories:
        if risk_probability < 0.3:
            parsed["risk_category"] = "Low Risk"
        elif risk_probability < 0.6:
            parsed["risk_category"] = "Medium Risk"
        else:
            parsed["risk_category"] = "High Risk"

    # Ensure title is set
    if not parsed["title"]:
        parsed["title"] = "Credit Risk Assessment Report"

    return parsed


# ---------------------------------------------------------------------------
# Step 3b: Rule-Based Fallback Decision Engine
# ---------------------------------------------------------------------------

def _classify_by_score(risk_probability: float) -> str:
    """Map risk probability to decision string."""
    if risk_probability < 0.3:
        return "APPROVE"
    elif risk_probability < 0.6:
        return "CONDITIONAL"
    else:
        return "REJECT"


def rule_based_decision(
    risk_probability: float,
    shap_factors: list,
    borrower_data: dict,
) -> dict:
    """
    Deterministic rule-based fallback.
    Always returns a complete, schema-valid report — no LLM needed.
    """
    decision = _classify_by_score(risk_probability)

    if risk_probability < 0.3:
        risk_cat, confidence = "Low Risk", "HIGH"
    elif risk_probability < 0.6:
        risk_cat, confidence = "Medium Risk", "MEDIUM"
    else:
        risk_cat, confidence = "High Risk", "HIGH"

    # Build key findings from SHAP factors, or from raw data if SHAP failed
    if shap_factors:
        key_findings = [
            f"{f['display_name']} (value: {f['actual_value']:.2f}) "
            f"{f['direction']} (impact: {f['impact']:.4f})"
            for f in shap_factors[:3]
        ]
        factor_names = ", ".join(f["display_name"] for f in shap_factors[:3])
    else:
        key_findings = [
            f"Risk probability is {risk_probability:.1%}",
            f"Debt ratio: {borrower_data.get('debt_ratio', 'N/A')}",
            f"Monthly income: ${borrower_data.get('monthly_inc', 0):,.0f}",
        ]
        factor_names = "debt ratio, utilization, payment history"

    # Build conditions for CONDITIONAL decisions
    conditions = []
    if decision == "CONDITIONAL":
        if borrower_data.get("debt_ratio", 0) > 1.0:
            conditions.append("Reduce debt-to-income ratio below 1.0")
        if borrower_data.get("rev_util", 0) > 0.7:
            conditions.append("Reduce revolving utilization below 70%")
        if borrower_data.get("late_90", 0) > 0:
            conditions.append("Resolve all 90+ day delinquent accounts")
        if not conditions:
            conditions.append("Provide additional income verification documentation")

    # Recommendation text
    rec_map = {
        "APPROVE": "Proceed with standard loan approval at standard terms.",
        "CONDITIONAL": "Approve with the listed conditions. Re-evaluate after conditions are met.",
        "REJECT": "Decline the application. Recommend credit counseling and re-application after 6-12 months.",
    }

    return {
        "title": "Credit Risk Assessment Report",
        "summary": (
            f"Applicant assessed with a {risk_probability:.1%} probability of default. "
            f"Classification: {risk_cat}."
        ),
        "risk_category": risk_cat,
        "decision": decision,
        "confidence": confidence,
        "key_findings": key_findings,
        "reasoning": (
            f"Based on the model's prediction of {risk_probability:.1%} default probability "
            f"and analysis of key risk factors ({factor_names}), the applicant is classified as "
            f"{risk_cat}. "
            f"{'The risk indicators are within acceptable thresholds for standard approval.' if decision == 'APPROVE' else ''}"
            f"{'Some risk indicators require attention before full approval can be granted.' if decision == 'CONDITIONAL' else ''}"
            f"{'Multiple risk indicators exceed acceptable thresholds, indicating significant default risk.' if decision == 'REJECT' else ''}"
        ),
        "conditions": conditions,
        "recommendation": rec_map[decision],
        "conclusion": (
            f"Decision: {decision}. The applicant's risk profile "
            f"{'supports' if decision == 'APPROVE' else 'conditionally supports' if decision == 'CONDITIONAL' else 'does not support'} "
            f"loan approval."
        ),
        "sources": ["Rule-based Decision Engine", "ML Prediction API", "SHAP Feature Fallback"],
        "_source": "rule_based",  # metadata flag for UI
    }


# ---------------------------------------------------------------------------
# Main Agent Pipeline
# ---------------------------------------------------------------------------

def run_agent(borrower_data: dict, risk_probability: float) -> dict:
    """
    Execute the full 3-step agentic pipeline.

    Parameters
    ----------
    borrower_data : dict
        Raw borrower features (10 input fields).
    risk_probability : float
        Pre-computed risk score from the /predict API.

    Returns
    -------
    dict
        Complete structured lending report with all 10 required keys,
        plus '_source' ("llm" or "rule_based") and '_shap_factors' list.
    """
    if risk_probability is None:
        return {
            "title": "Prediction API Error",
            "summary": "Could not generate decision because the prediction API is unavailable.",
            "risk_category": "Unknown",
            "decision": "REJECT",
            "confidence": "LOW",
            "key_findings": ["Prediction API returned None (timeout or down)"],
            "reasoning": "System cannot formulate a reliable decision without a risk probability score.",
            "conditions": [],
            "recommendation": "Try again later when backend is online.",
            "conclusion": "Evaluation failed.",
            "sources": ["System Diagnostics"],
            "_source": "error",
            "_risk_probability": None,
            "_shap_factors": [],
        }

    # Step 2: Explain
    shap_factors = explain_risk(borrower_data)

    # Step 3: Decide (LLM with fallback)
    prompt = _build_prompt(borrower_data, risk_probability, shap_factors)
    raw_llm_response = _call_llm(prompt)

    report = None
    source = "rule_based"

    if raw_llm_response:
        parsed = _parse_llm_response(raw_llm_response)
        validated = _validate_report(parsed, risk_probability)
        if validated:
            report = validated
            source = "llm"

    # Fallback if LLM failed at any stage
    if report is None:
        report = rule_based_decision(risk_probability, shap_factors, borrower_data)
        source = "rule_based"

    report["_source"] = source
    report["_shap_factors"] = shap_factors
    report["_risk_probability"] = risk_probability

    return report


# ---------------------------------------------------------------------------
# Open-Ended Query Support
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    borrower_data: dict,
    risk_probability: float,
    shap_factors: list,
) -> str:
    """
    Answer a free-text user question about the borrower, grounded in
    the model's prediction and SHAP analysis.

    Returns a plain text response (2-4 sentences), or a fallback message.
    """

    if shap_factors:
        shap_summary = "; ".join(
            f"{f['display_name']} ({f['direction']}, impact {f['impact']:.4f})"
            for f in shap_factors
        )
    else:
        shap_summary = "Feature-level analysis not available."

    prompt = f"""You are a senior credit risk analyst AI assistant.

Context about the current borrower:
- Risk score (default probability): {risk_probability:.1%}
- Age: {borrower_data.get('age', 'N/A')}, Monthly Income: ${borrower_data.get('monthly_inc', 0):,.0f}
- Revolving Utilization: {borrower_data.get('rev_util', 0):.1%}, Debt Ratio: {borrower_data.get('debt_ratio', 0):.2f}
- Late Payments: 30-59d={borrower_data.get('late_30_59', 0):.0f}, 60-89d={borrower_data.get('late_60_89', 0):.0f}, 90+d={borrower_data.get('late_90', 0):.0f}
- Top risk factors: {shap_summary}

User question: {query}

Respond concisely in 2-4 sentences. Reference the specific data above. Do not invent numbers not provided. Be professional."""

    raw = _call_llm(prompt)

    if raw and raw.strip():
        # Clean up any markdown artifacts
        answer = raw.strip()
        answer = re.sub(r'^```.*?\n', '', answer)
        answer = re.sub(r'\n```$', '', answer)
        return answer

    # Fallback for open-ended queries
    if shap_factors:
        top = shap_factors[0]
        return (
            f"Based on the model analysis, this applicant has a {risk_probability:.1%} "
            f"default probability. The most significant factor is "
            f"{top['display_name']} (value: {top['actual_value']:.2f}), which "
            f"{top['direction']}. Please use the 'Analyze Borrower' mode for a "
            f"detailed report."
        )
    return (
        f"This applicant has a {risk_probability:.1%} default probability. "
        f"Please use the 'Analyze Borrower' mode for a complete analysis "
        f"with detailed risk factors and recommendations."
    )
