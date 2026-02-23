from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd

from src.preprocess import feature_engineering

app = FastAPI()


model = joblib.load("models/final_credit_risk_model.pkl")

REQUIRED_FEATURES = [
    "rev_util",
    "age",
    "late_30_59",
    "debt_ratio",
    "monthly_inc",
    "open_credit",
    "late_90",
    "real_estate",
    "late_60_89",
    "dependents",
]


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
    df = pd.DataFrame([data.dict()])
    df_fe = feature_engineering(df.copy())
    risk = model.predict_proba(df_fe)[0][1]
    return {"risk_probability": float(risk)}



@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    if "dlq_2yrs" in df.columns:
        df = df.drop(columns=["dlq_2yrs"])

    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]

    if missing:
        return {
            "error": "Missing required columns",
            "missing_columns": missing,
            "required_columns": REQUIRED_FEATURES
        }

    df = df[REQUIRED_FEATURES]  

    df_fe = feature_engineering(df.copy())

    df["risk_probability"] = model.predict_proba(df_fe)[:, 1]

    return df.to_dict(orient="records")