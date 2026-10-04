"""
BoomBikes Bike-Sharing Demand Prediction (in progress)

Goal: predict daily bike rental demand (`cnt`) for BoomBikes from
weather and calendar features, using multiple linear regression.

This is the hands-on rebuild of an earlier AI-assisted first version,
written step by step to understand the workflow: load data, explore
it, engineer features, fit a model, and check which features matter.

Expects a CSV (commonly named `day.csv`) with columns similar to the
classic bike-sharing dataset:
    season, yr, mnth, holiday, weekday, workingday, weathersit,
    temp, atemp, hum, windspeed, casual, registered, cnt

Run with: python boombikes_regression.py path/to/day.csv
"""

import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
TARGET_COLUMN = "cnt"


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def explore(df):
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nSummary statistics:\n", df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe())
    print("\nCorrelation with target (cnt):")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].sort_values(ascending=False))


def train_model(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"\nR^2 on test set:   {r2:.3f}")
    print(f"RMSE on test set:  {rmse:.1f} rides/day")

    coefficients = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False)
    print("\nStandardized coefficients (feature importance direction):")
    print(coefficients)

    return model, scaler


def main():
    if len(sys.argv) != 2:
        print("Usage: python boombikes_regression.py path/to/day.csv")
        sys.exit(1)

    data_path = sys.argv[1]
    df = load_data(data_path)
    explore(df)
    train_model(df)


if __name__ == "__main__":
    main()
