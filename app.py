import streamlit as st
import requests

st.title("Credit Risk Prediction System")

st.write("Enter applicant details:")

rev_util = st.number_input("Revolving Utilization", 0.0, 5.0, 0.5)
age = st.number_input("Age", 18.0, 100.0, 35.0)
late_30_59 = st.number_input("Late 30-59 Days", 0.0, 20.0, 0.0)
debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0, 0.5)
monthly_inc = st.number_input("Monthly Income", 0.0, 100000.0, 5000.0)
open_credit = st.number_input("Open Credit Lines", 0.0, 50.0, 5.0)
late_90 = st.number_input("Late 90+ Days", 0.0, 20.0, 0.0)
real_estate = st.number_input("Real Estate Loans", 0.0, 20.0, 0.0)
late_60_89 = st.number_input("Late 60-89 Days", 0.0, 20.0, 0.0)
dependents = st.number_input("Dependents", 0.0, 10.0, 0.0)

if st.button("Predict Risk"):

    payload = {
        "rev_util": rev_util,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_inc": monthly_inc,
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        risk = result["risk_probability"]

        st.subheader(f"Predicted Risk Probability: {risk:.3f}")

        if risk < 0.3:
            st.success("Low Risk")
        elif risk < 0.6:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")
    else:
        st.error("API Error")
