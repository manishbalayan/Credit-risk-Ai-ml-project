# Credit Risk Prediction System

## Overview
This project implements an end-to-end Credit Risk Prediction System capable of estimating the probability that an applicant will become delinquent within two years. The system includes:

- A complete machine learning pipeline
- Feature engineering and model tuning
- A FastAPI backend serving predictions
- A Streamlit-based frontend interface
- A fully documented technical journal describing the entire development journey

The solution is designed to be academically rigorous while demonstrating practical engineering standards suitable for real-world deployment.

## Project Structure

```
credit-risk-project/
│
├── data/                       # Dataset used for training
├── models/                     # Exported trained model (.pkl)
├── docs/                       # Technical journal and analysis notes
├── notebooks/                  # Jupyter notebook versions of experiments
├── src/
│   ├── preprocess.py           # Data loading, splitting, and feature engineering
│   ├── train.py                # Full model training pipeline
│   ├── evaluate.py             # Cross-validation & feature importance analysis
│   └── api.py                  # FastAPI backend
│
├── app.py                      # Streamlit frontend interface
├── requirements.txt            # Environment dependencies
└── README.md                   # Project documentation
```

## Key Features

### 1. Machine Learning Pipeline
- Logistic Regression baseline
- Random Forest classifier
- GridSearch-based hyperparameter tuning
- Feature engineering:
  - Credit-to-income ratio
  - Total late payments
  - Revolving utilization buckets
- Cross-validation and stability analysis

### 2. Backend (FastAPI)
- Exposes a `/predict` endpoint
- Accepts structured applicant financial data
- Returns risk probability
- Input validation via Pydantic
- Model served using joblib

### 3. Frontend (Streamlit)
- Clean and intuitive UI
- Numeric input fields for all applicant attributes
- Sends POST requests to FastAPI backend
- Displays predicted risk probability and risk category

## Model Performance

After feature engineering and hyperparameter tuning:

| Metric | Score |
|--------|--------|
| Accuracy | ~0.77 |
| Precision | ~0.79 |
| Recall | ~0.74 |
| F1 Score | ~0.76 |
| ROC-AUC | 0.858 |

Cross-validation mean ROC-AUC: 0.8539  
Standard deviation: 0.0041

## How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd credit-risk-project
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Backend (FastAPI)

### Start the Server
```bash
uvicorn src.api:app --reload
```

API at:
```
http://127.0.0.1:8000
```

Swagger Docs:
```
http://127.0.0.1:8000/docs
```

## Running the Frontend (Streamlit)

Start the UI:
```bash
streamlit run app.py
```

Frontend at:
```
http://localhost:8501
```

## Prediction Workflow

1. User enters applicant details in Streamlit
2. Streamlit sends JSON to FastAPI
3. Backend processes features & loads model
4. Model outputs risk probability
5. UI displays result

## Technical Journal

Full development journey is documented in:
```
docs/technical_journal.md
```

## Deployment Options

Suggested platforms:
- Backend: Railway / Render
- Frontend: Streamlit Cloud

Alternative:
- Docker deployment
- HuggingFace Spaces
- VPS or cloud server

## Future Improvements
- Explainability (SHAP)
- Feature importance plots in UI
- Model drift monitoring
- Authentication layer
- CI/CD workflows

## License
This project is provided for academic and educational use.
