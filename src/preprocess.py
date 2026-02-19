import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(path)
    return df


def feature_engineering(df):
    """
    Create engineered features based on domain insight.
    """

    # Total number of late payments
    df["total_late"] = (
        df["late_30_59"] +
        df["late_60_89"] +
        df["late_90"]
    )

    # Weighted late payment severity
    df["late_weighted"] = (
        df["late_30_59"] * 1 +
        df["late_60_89"] * 2 +
        df["late_90"] * 3
    )

    # Interaction between utilization and delinquency
    df["util_late_interaction"] = df["rev_util"] * df["total_late"]

    # Income normalized by dependents
    df["income_per_dependent"] = (
        df["monthly_inc"] / (df["dependents"] + 1)
    )

    return df


def split_data(df):
    """
    Split dataset into training and testing sets.
    """

    X = df.drop("dlq_2yrs", axis=1)
    y = df["dlq_2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data("data/Credit Risk Benchmark Dataset.csv")
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
