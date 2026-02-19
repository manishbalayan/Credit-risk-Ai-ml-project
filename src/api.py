from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from src.preprocess import feature_engineering

app = FastAPI()

# Load model once when API starts
model = joblib.load("models/final_credit_risk_model.pkl")


# Request schema
class CreditInput(BaseModel):
    rev_util: float
    age: float
    late_30_59: float
    debt_ratio: float
    monthly_inc: float
    open_credit: float
    late_90: float
    real_estate: float
    late_60_89: float
    dependents: float


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API Running"}


@app.post("/predict")
def predict(data: CreditInput):

    # Convert input to dataframe
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Apply same feature engineering
    df = feature_engineering(df)

    # Predict probability
    risk_score = model.predict_proba(df)[0][1]

    return {
        "risk_probability": float(risk_score)
    }
