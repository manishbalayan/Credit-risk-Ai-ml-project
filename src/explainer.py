"""
SHAP-based model explainer for credit risk predictions.

Provides per-instance feature contribution analysis using TreeExplainer,
with safeguards for feature order consistency and graceful error handling.
"""

import joblib
import shap
import pandas as pd
import numpy as np
import os

from src.preprocess import feature_engineering


# ---------------------------------------------------------------------------
# Module-level model & explainer loading (done once, reused across requests)
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "final_credit_risk_model.pkl")

try:
    _model = joblib.load(_MODEL_PATH)
    _explainer = shap.TreeExplainer(_model)
    _MODEL_LOADED = True
except Exception as e:
    print(f"[explainer] WARNING: Could not load model or create explainer: {e}")
    _model = None
    _explainer = None
    _MODEL_LOADED = False


# Friendly display names for engineered features
FEATURE_DISPLAY_NAMES = {
    "rev_util": "Revolving Utilization",
    "age": "Age",
    "late_30_59": "Late Payments (30-59 days)",
    "debt_ratio": "Debt Ratio",
    "monthly_inc": "Monthly Income",
    "open_credit": "Open Credit Lines",
    "late_90": "Late Payments (90+ days)",
    "real_estate": "Real Estate Loans",
    "late_60_89": "Late Payments (60-89 days)",
    "dependents": "Number of Dependents",
    "total_late": "Total Late Payments",
    "late_weighted": "Weighted Late Severity",
    "util_late_interaction": "Utilization × Delinquency",
    "income_per_dependent": "Income per Dependent",
}


def get_shap_explanation(borrower_data: dict, top_n: int = 3) -> list:
    """
    Compute SHAP values for a single borrower and return top contributing
    features ranked by absolute impact.

    Parameters
    ----------
    borrower_data : dict
        Raw borrower features (the 10 input fields from the UI).
    top_n : int
        Number of top features to return (default 3).

    Returns
    -------
    list[dict]
        Each dict contains:
        - feature       : internal feature name
        - display_name  : human-readable name
        - shap_value    : signed SHAP value
        - impact        : absolute SHAP value
        - direction     : "increases risk" or "decreases risk"
        - actual_value  : the borrower's actual value for that feature

        Returns empty list on any failure (never raises).
    """

    if not _MODEL_LOADED:
        print("[explainer] Model not loaded — returning empty explanation.")
        return []

    try:
        # Build a single-row DataFrame from raw borrower input
        df = pd.DataFrame([borrower_data])

        # Apply the SAME feature engineering pipeline used during training
        df_fe = feature_engineering(df.copy())

        # Ensure column order matches model's expected feature order
        if hasattr(_model, "feature_names_in_"):
            expected = list(_model.feature_names_in_)
            missing = [c for c in expected if c not in df_fe.columns]
            if missing:
                print(f"[explainer] Missing features: {missing} — skipping SHAP.")
                return []
            df_fe = df_fe[expected]  # reorder to exact training order

        # Compute SHAP values
        shap_values = _explainer.shap_values(df_fe)

        # Handle binary classification: shap_values is [array_class0, array_class1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class-1 (default) explanations
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Build ranked factor list
        feature_names = df_fe.columns.tolist()
        factors = []
        for i, fname in enumerate(feature_names):
            val = float(sv[i])
            if np.isnan(val):
                continue
            factors.append({
                "feature": fname,
                "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                "shap_value": val,
                "impact": abs(val),
                "direction": "increases risk" if val > 0 else "decreases risk",
                "actual_value": float(df_fe.iloc[0][fname]),
            })

        # Sort by absolute impact (descending) and return top N
        factors.sort(key=lambda x: x["impact"], reverse=True)
        return factors[:top_n]

    except Exception as e:
        print(f"[explainer] SHAP explanation failed: {e}")
        return []


def get_all_shap_values(borrower_data: dict) -> tuple:
    """
    Return full SHAP values and feature names for detailed visualizations.

    Returns
    -------
    (shap_values_array, feature_names) or (None, None) on failure.
    """

    if not _MODEL_LOADED:
        return None, None

    try:
        df = pd.DataFrame([borrower_data])
        df_fe = feature_engineering(df.copy())

        if hasattr(_model, "feature_names_in_"):
            expected = list(_model.feature_names_in_)
            missing = [c for c in expected if c not in df_fe.columns]
            if missing:
                return None, None
            df_fe = df_fe[expected]

        shap_values = _explainer.shap_values(df_fe)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        return sv, df_fe.columns.tolist()

    except Exception:
        return None, None
