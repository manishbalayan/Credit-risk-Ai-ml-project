# Credit Risk Prediction System

## Overview
This project implements an end-to-end Credit Risk Prediction System capable of estimating the probability that an applicant will become delinquent within two years. The system includes:

- A complete machine learning pipeline
- Feature engineering and model tuning
- A unified Streamlit-based monolithic frontend
- **[NEW]** An Agentic AI Lending Advisor using SHAP and `flan-t5-base`
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
│   └── evaluate.py             # Cross-validation & feature importance analysis
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

### 2. Frontend (Streamlit Monolith)
- Clean and intuitive UI with tabbed navigation
- Model is loaded natively and cached globally via `@st.cache_resource` for zero-latency execution.

### 3. Frontend (Streamlit)
- Clean and intuitive UI with tabbed navigation
- **Tab 1: Single & Bulk Prediction**: Numeric fields for applicant attributes and CSV upload bulk predictions.
- **Tab 2: AI Lending Advisor**: An intelligent agent featuring a 3-step pipeline (Predict → Explain → Decide).
- Displays animated SHAP impact bars and color-coded risk badges.
- **Chat Mode:** Ask the AI open-ended questions about the specific borrower.

### 4. Agentic AI Pipeline (Milestone 2)
- **Predict**: ML Model dynamically scores risk probability *natively in the app*.
- **Explain**: `shap.TreeExplainer` locally extracts top driving risk factors dynamically.
- **Decide**: `flan-t5-base` (via HuggingFace) generates a structured JSON lending report containing reasoning, conditions, and recommendations.
- **100% Robustness**: Contains a deterministic Rule-Based Fallback Engine that automatically takes over if the LLM times out or fails (no crashes).

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

### 4. Setup HuggingFace Token (For AI Agent)
The AI Lending Advisor uses the free HuggingFace Inference API (`google/flan-t5-base`).
1. Create a free account and get an Access Token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Export the token into your terminal before running the app:
   ```bash
   export HF_TOKEN="hf_your_actual_token_here"
   ```
*(Note: If you skip this step, the app will not crash. It automatically detects the missing token and uses the local Rule-Based Engine to generate reports instead.)*

## Running the Application (Streamlit)

> **⚠️ Important:** Ensure your virtual environment is activated before running Streamlit to avoid `ModuleNotFoundError` issues (like missing `shap` or `joblib`)!
> ```bash
> source venv/bin/activate
> ```

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
2. Streamlit natively structures data and applies feature engineering
3. ML Model `.predict_proba()` is executed locally
4. Agent triggers SHAP pipeline based on the output
5. Report renders instantly in the UI

## Technical Journal

Full development journey is documented in:
```
docs/technical_journal.md
```

## Deployment Options

Suggested platforms:
- Streamlit Cloud (Preferred)
- Render (Web Service)

Alternative:
- Docker deployment
- HuggingFace Spaces
- VPS or cloud server

## Future Improvements
- Model drift monitoring
- Authentication layer
- CI/CD workflows

## License
This project is provided for academic and educational use.
