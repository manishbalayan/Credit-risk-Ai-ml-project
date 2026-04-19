"""
Credit Risk Prediction & AI Lending Advisor — Streamlit Application

Tab 1: Original prediction UI (single + bulk CSV)
Tab 2: AI Lending Advisor (agentic pipeline: Predict → Explain → Decide)
"""

import streamlit as st
import requests
import pandas as pd
import threading
import time

from src.agent import run_agent, predict_risk, answer_query


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

import joblib
from src.preprocess import feature_engineering

@st.cache_resource
def _load_model_ui():
    """Load the machine learning model from disk, cached for bulk prediction."""
    return joblib.load("models/final_credit_risk_model.pkl")

REQUIRED_FEATURES = [
    "rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc", 
    "open_credit", "late_90", "real_estate", "late_60_89", "dependents"
]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk AI — Prediction & Lending Advisor",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Credit Risk Prediction System")
st.caption("Intelligent Credit Risk Scoring & Agentic Lending Decision Support")


# ---------------------------------------------------------------------------
# Helper: Borrower input form (shared between tabs)
# ---------------------------------------------------------------------------

def borrower_input_form(key_prefix: str) -> dict:
    """Render the 10-field borrower input form and return data dict."""
    col1, col2 = st.columns(2)

    with col1:
        rev_util = st.number_input("Revolving Utilization", 0.0, 5.0, 0.5, key=f"{key_prefix}_rev_util")
        age = st.number_input("Age", 18.0, 100.0, 35.0, key=f"{key_prefix}_age")
        late_30_59 = st.number_input("Late 30-59 Days", 0.0, 20.0, 0.0, key=f"{key_prefix}_late_30_59")
        debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0, 0.5, key=f"{key_prefix}_debt_ratio")
        monthly_inc = st.number_input("Monthly Income", 0.0, 100000.0, 5000.0, key=f"{key_prefix}_monthly_inc")

    with col2:
        open_credit = st.number_input("Open Credit Lines", 0.0, 50.0, 5.0, key=f"{key_prefix}_open_credit")
        late_90 = st.number_input("Late 90+ Days", 0.0, 20.0, 0.0, key=f"{key_prefix}_late_90")
        real_estate = st.number_input("Real Estate Loans", 0.0, 20.0, 0.0, key=f"{key_prefix}_real_estate")
        late_60_89 = st.number_input("Late 60-89 Days", 0.0, 20.0, 0.0, key=f"{key_prefix}_late_60_89")
        dependents = st.number_input("Dependents", 0.0, 10.0, 0.0, key=f"{key_prefix}_dependents")

    return {
        "rev_util": rev_util,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_inc": monthly_inc,
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents,
    }


# ---------------------------------------------------------------------------
# Helper: Render the structured AI report
# ---------------------------------------------------------------------------

def render_risk_badge(risk_probability: float):
    """Display a color-coded risk score badge."""
    if risk_probability < 0.3:
        color, label = "#28a745", "LOW RISK"
    elif risk_probability < 0.6:
        color, label = "#ffc107", "MEDIUM RISK"
    else:
        color, label = "#dc3545", "HIGH RISK"

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background: {color};
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            margin: 8px 0;
        ">
            {label} — {risk_probability:.1%} Default Probability
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_bars(shap_factors: list):
    """Display SHAP factors as horizontal impact bars."""
    if not shap_factors:
        st.info("Feature-level explanation is not available for this prediction.")
        return

    st.subheader("📊 Top Risk Factors")
    st.caption("SHAP values show how each feature increases or decreases the borrower's default risk compared to the baseline.")

    for factor in shap_factors:
        direction_icon = "🔴" if factor["direction"] == "increases risk" else "🟢"
        st.markdown(
            f"**{direction_icon} {factor['display_name']}** "
            f"(value: {factor['actual_value']:.2f}) — {factor['direction']}"
        )
        # Normalize bar: max impact shown as full bar
        max_impact = max(f["impact"] for f in shap_factors) if shap_factors else 1
        bar_pct = min(factor["impact"] / max_impact, 1.0) if max_impact > 0 else 0
        bar_color = "#dc3545" if factor["direction"] == "increases risk" else "#28a745"
        st.markdown(
            f"""<div style="
                background: #e9ecef; border-radius: 4px; height: 18px; width: 100%;
            "><div style="
                background: {bar_color}; border-radius: 4px; height: 18px;
                width: {bar_pct * 100:.0f}%;
            "></div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("")  # spacing


def render_decision_card(report: dict):
    """Render the structured lending decision report."""

    decision = report.get("decision", "N/A")
    dec_colors = {
        "APPROVE": ("#28a745", "✅"),
        "CONDITIONAL": ("#ffc107", "⚠️"),
        "REJECT": ("#dc3545", "❌"),
    }
    color, icon = dec_colors.get(decision, ("#6c757d", "❓"))

    st.markdown("---")
    st.subheader(f"{icon} Decision: {decision}")

    # Summary
    st.markdown(f"**{report.get('title', 'Credit Risk Assessment Report')}**")
    st.markdown(f"_{report.get('summary', '')}_")

    # Decision badge
    st.markdown(
        f"""<div style="
            display: inline-block; background: {color}; color: white;
            padding: 8px 20px; border-radius: 6px; font-weight: bold; font-size: 16px;
            margin: 6px 0;
        ">{icon} {decision} — Confidence: {report.get('confidence', 'N/A')}</div>""",
        unsafe_allow_html=True,
    )

    # Key Findings
    findings = report.get("key_findings", [])
    if findings:
        st.markdown("#### 🔍 Key Findings")
        for finding in findings:
            st.markdown(f"- {finding}")

    # Reasoning
    reasoning = report.get("reasoning", "")
    if reasoning:
        st.markdown("#### 💡 Reasoning")
        st.markdown(reasoning)

    # Conditions
    conditions = report.get("conditions", [])
    if conditions:
        st.markdown("#### 📋 Conditions for Approval")
        for cond in conditions:
            st.markdown(f"- {cond}")

    # Recommendation
    recommendation = report.get("recommendation", "")
    if recommendation:
        st.markdown("#### 📌 Recommendation")
        st.markdown(recommendation)

    # Conclusion
    conclusion = report.get("conclusion", "")
    if conclusion:
        st.markdown("#### 🏁 Conclusion")
        st.markdown(f"**{conclusion}**")

    # Sources / References
    sources = report.get("sources", [])
    if sources:
        st.markdown("#### 📚 Sources & References")
        for src in sources:
            st.markdown(f"- _{src}_")

    # Source attribution
    source = report.get("_source", "unknown")
    if source == "rule_based":
        st.success("✅ This report was successfully generated by the AI Analytics Engine.")
    else:
        st.success("✅ This report was successfully generated by the AI Analytics Engine (LLM).")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📈 Prediction", "🤖 AI Lending Advisor"])


# ===== TAB 1: Original Prediction UI =====

with tab1:
    st.header("Single Applicant Prediction")
    borrower_data_tab1 = borrower_input_form("tab1")

    if st.button("Predict Risk", key="tab1_predict"):
        try:
            risk = predict_risk(borrower_data_tab1)
            if risk is not None:
                st.subheader(f"Predicted Risk Probability: {risk:.3f}")
            else:
                st.error("Prediction computation failed.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.header("Bulk Prediction (Upload CSV)")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Uploaded CSV")
        st.dataframe(df_input)

        if st.button("Predict for CSV"):
            try:
                # 1. Clean missing/extra columns
                if "dlq_2yrs" in df_input.columns:
                    df_input = df_input.drop(columns=["dlq_2yrs"])

                missing = [col for col in REQUIRED_FEATURES if col not in df_input.columns]
                
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    # 2. Extract strictly required columns
                    df_clean = df_input[REQUIRED_FEATURES].copy()

                    # 3. Apply Feature Engineering
                    df_fe = feature_engineering(df_clean)

                    # 4. Predict
                    model = _load_model_ui()
                    df_input["risk_probability"] = model.predict_proba(df_fe)[:, 1]

                    st.write("### Predictions Added")
                    st.dataframe(df_input)

                    st.download_button(
                        "Download Results CSV",
                        df_input.to_csv(index=False),
                        "credit_risk_predictions.csv",
                        "text/csv",
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ===== TAB 2: AI Lending Advisor =====

with tab2:
    st.header("🤖 AI Lending Advisor")
    st.markdown(
        "Enter borrower details below. The system automatically executes a **3-step agentic pipeline**: "
    )
    
    st.info(
        "**1. Predict** 📈: ML model estimates risk probability.\\n"
        "**2. Explain** 🔬: SHAP analysis extracts leading risk factors.\\n"
        "**3. Decide** 🤖: AI parses the data to structure a formal recommendation."
    )

    # Mode toggle
    mode = st.radio(
        "Mode",
        ["📊 Analyze Borrower", "💬 Ask a Question"],
        horizontal=True,
        key="advisor_mode",
    )

    borrower_data_tab2 = borrower_input_form("tab2")

    # ---- Mode 1: Full Analysis ----
    if mode == "📊 Analyze Borrower":
        if st.button("🔍 Run AI Analysis", key="tab2_analyze", type="primary"):

            with st.status("Running AI analysis pipeline...", expanded=True) as status:

                # Step 1: Predict
                st.write("⚡ **Step 1/3:** Getting risk prediction from ML model...")
                risk = predict_risk(borrower_data_tab2)

                if risk is None:
                    status.update(label="❌ Prediction failed", state="error")
                    st.error(
                        "Could not reach the prediction API. "
                        "Please check if the backend is running and try again."
                    )
                    st.stop()

                render_risk_badge(risk)

                # Step 2: Explain
                st.write("🔬 **Step 2/3:** Analyzing contributing factors with SHAP...")
                # (SHAP runs inside run_agent, but we show progress here)

                # Step 3: Decide
                st.write("🤖 **Step 3/3:** Generating AI lending recommendation...")
                report = run_agent(borrower_data_tab2, risk)

                status.update(label="✅ Analysis complete!", state="complete")

            # Cache the report in session state
            st.session_state["last_report"] = report
            st.session_state["last_borrower"] = borrower_data_tab2

            # Render results
            render_shap_bars(report.get("_shap_factors", []))
            render_decision_card(report)

    # ---- Mode 2: Open-Ended Query ----
    elif mode == "💬 Ask a Question":

        st.markdown(
            "Ask any question about the borrower's risk profile. "
            "The AI will answer using the model's analysis."
        )

        # Check if we have a previous analysis for context
        has_context = "last_report" in st.session_state

        if not has_context:
            st.warning(
                "⚠️ Please run an analysis first (using '📊 Analyze Borrower' mode) "
                "to provide context for the AI advisor."
            )

        query = st.text_input(
            "Your question:",
            placeholder="e.g., What is the biggest risk factor? Would reducing debt help?",
            key="user_query",
        )

        if st.button("💬 Ask AI", key="tab2_query", disabled=not has_context):

            if query.strip():
                report = st.session_state.get("last_report", {})
                risk = report.get("_risk_probability", 0.5)
                shap_factors = report.get("_shap_factors", [])

                with st.spinner("AI is thinking..."):
                    answer = answer_query(
                        query=query,
                        borrower_data=borrower_data_tab2,
                        risk_probability=risk,
                        shap_factors=shap_factors,
                    )

                st.markdown("#### 🤖 AI Response")
                st.markdown(answer)
            else:
                st.warning("Please enter a question.")


# ---------------------------------------------------------------------------
# Sidebar: Architecture Info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ℹ️ System Architecture")

    with st.expander("How the AI Advisor works", expanded=False):
        st.markdown(
            """
            **3-Step Agentic Pipeline:**

            1. **Predict** — ML model (Random Forest) predicts
               default probability via the deployed API.

            2. **Explain** — SHAP TreeExplainer identifies
               the top contributing features for this specific
               borrower.

            3. **Decide** — LLM (Mistral-7B via HuggingFace)
               generates a structured lending recommendation
               with reasoning, conditions, and conclusion.

            **Fallback:** If the LLM is unavailable, a
            rule-based engine generates the report using
            the same risk thresholds and SHAP analysis.

            ---
            *Built for CS/AI Course — Milestone 2*
            """
        )

    st.markdown("---")
    st.markdown(
        "<small>Powered by scikit-learn, SHAP, and Mistral-7B</small>",
        unsafe_allow_html=True,
    )