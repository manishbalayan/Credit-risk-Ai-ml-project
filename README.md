# 🏦 Credit Risk Prediction & Agentic AI Lending Advisor

## 📌 Overview

This project presents a **fully self-contained Credit Risk Prediction System** that estimates the probability of a borrower defaulting within two years.

It combines:

* Machine Learning (Random Forest)
* Explainable AI (SHAP)
* Agentic AI Pipeline (Predict → Explain → Decide)
* Dynamic reasoning system (LLM + fallback)

The system is designed to be **robust, fast, and demo-safe**, eliminating external dependencies by running entirely within a Streamlit application.

---

## 🧠 System Architecture

```
User Input
   ↓
Predict (ML Model)
   ↓
Explain (SHAP)
   ↓
Decide (LLM / Pseudo-Agent)
   ↓
Structured Lending Report
```

---

## 🚀 Key Features

### 🔹 1. Machine Learning Pipeline

* Random Forest classifier with hyperparameter tuning
* Feature engineering:

  * Total late payments
  * Weighted delinquencies
  * Credit utilization interaction
  * Income per dependent
* Cross-validation and performance evaluation

---

### 🔹 2. Explainability (SHAP)

* Uses `shap.TreeExplainer`
* Identifies top 3 risk-driving features
* Visual impact bars showing contribution to risk

---

### 🔹 3. Agentic AI Pipeline

* **Predict** → Risk probability using ML
* **Explain** → SHAP feature contributions
* **Decide** → Generates structured lending decision

---

### 🔹 4. Intelligent Decision System

* Uses **HuggingFace LLM (HuggingFaceH4/zephyr-7b-beta)** when available
* Includes **deterministic fallback engine** for reliability
* Ensures **100% uptime (no crashes)**

---

### 🔹 5. Dynamic Query System (Pseudo-Agent)

* Handles user questions like:

  * “Why is this risky?”
  * “How to improve approval chances?”
* Uses:

  * Risk score
  * SHAP explanations
  * Context-aware logic

---

### 🔹 6. Fully Self-Contained Deployment

* No backend required
* No API dependency for core functionality
* Runs entirely on Streamlit

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | ~0.77  |
| Precision | ~0.79  |
| Recall    | ~0.74  |
| F1 Score  | ~0.76  |
| ROC-AUC   | ~0.858 |

---

## 📁 Project Structure

```
credit-risk-project/
│
├── data/
├── models/
├── docs/
├── notebooks/
├── src/
│   ├── agent.py
│   ├── explainer.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Clone Repo

```bash
git clone https://github.com/manishbalayan/Credit-risk-Ai-ml-project.git
cd Credit-risk-Ai-ml-project
```

### 2. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. (Optional) Add HuggingFace Token

```bash
export HF_TOKEN="hf_your_token"
```

👉 If not added:

* System uses fallback
* App still works perfectly

---

### 5. Run App

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 (Add your Streamlit link here)

---

## 🧪 Workflow

1. User inputs borrower details
2. Model predicts risk probability
3. SHAP explains key contributing factors
4. Agent generates structured decision
5. User can ask follow-up questions

---

## 🧠 Key Design Decisions

* Removed backend → reduced latency & failure points
* Added fallback engine → ensures reliability
* Combined ML + Explainability + Reasoning → full pipeline

---

## 🚀 Future Improvements

* Real-time financial API integration
* Multi-agent reasoning
* Model monitoring & drift detection

---

## 📜 License

For academic and educational use only.
