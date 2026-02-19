import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from preprocess import load_data, feature_engineering, split_data


def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":

    print("\nLoading and preprocessing data...")

    # -------- Load + Engineer Data --------
    df = load_data("data/Credit Risk Benchmark Dataset.csv")
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # -------- Logistic Regression --------
    print("\nTraining Logistic Regression...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train_scaled, y_train)

    evaluate_model("Logistic Regression", log_model, X_test_scaled, y_test)

    # -------- Random Forest Baseline --------
    print("\nTraining Random Forest (Baseline)...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    evaluate_model("Random Forest (Baseline)", rf_model, X_test, y_test)

    # -------- Random Forest Tuning --------
    print("\nTuning Random Forest with GridSearchCV...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    print("\nBest Parameters Found:")
    print(grid_search.best_params_)

    evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test)

    # -------- Save Final Model --------
    print("\nSaving final tuned model...")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/final_credit_risk_model.pkl")

    print("Model saved successfully at models/final_credit_risk_model.pkl")
