"""
train_model.py
----------------
Builds a house-price dataset (loads Housing.csv if you drop one in this folder,
otherwise generates a realistic synthetic dataset with the same schema as the
classic Kaggle "Housing Price Prediction" dataset), trains a RandomForest
regression pipeline, and saves it to model.pkl for the Flask API to load.

Run:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "Housing.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
CATEGORICAL_FEATURES = ["furnishingstatus"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def generate_synthetic_dataset(n=2200, seed=RANDOM_STATE):
    """
    Generates a synthetic dataset that mirrors the schema and value ranges of
    the well-known Kaggle 'Housing Price Prediction' dataset (price in INR,
    area in sq ft). Used automatically if no real Housing.csv is supplied.
    """
    rng = np.random.default_rng(seed)

    area = rng.integers(1650, 16200, size=n)
    bedrooms = rng.integers(1, 7, size=n)
    bathrooms = rng.integers(1, 5, size=n)
    stories = rng.integers(1, 5, size=n)
    parking = rng.integers(0, 4, size=n)

    mainroad = rng.choice(["yes", "no"], size=n, p=[0.85, 0.15])
    guestroom = rng.choice(["yes", "no"], size=n, p=[0.3, 0.7])
    basement = rng.choice(["yes", "no"], size=n, p=[0.35, 0.65])
    hotwaterheating = rng.choice(["yes", "no"], size=n, p=[0.15, 0.85])
    airconditioning = rng.choice(["yes", "no"], size=n, p=[0.45, 0.55])
    prefarea = rng.choice(["yes", "no"], size=n, p=[0.3, 0.7])
    furnishingstatus = rng.choice(
        ["furnished", "semi-furnished", "unfurnished"], size=n, p=[0.3, 0.4, 0.3]
    )

    # Base price driven mainly by area, then adjusted by every other feature.
    price = (
        260_000
        + area * 450
        + bedrooms * 180_000
        + bathrooms * 260_000
        + stories * 150_000
        + parking * 90_000
        + (mainroad == "yes") * 300_000
        + (guestroom == "yes") * 140_000
        + (basement == "yes") * 200_000
        + (hotwaterheating == "yes") * 160_000
        + (airconditioning == "yes") * 320_000
        + (prefarea == "yes") * 420_000
        + np.where(
            furnishingstatus == "furnished",
            300_000,
            np.where(furnishingstatus == "semi-furnished", 120_000, 0),
        )
    )

    # Add realistic noise so the model has something to learn, not just a formula.
    noise = rng.normal(0, 350_000, size=n)
    price = np.clip(price + noise, 1_750_000, 13_300_000).round(-3)

    df = pd.DataFrame(
        {
            "price": price,
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }
    )
    return df


def load_dataset():
    if os.path.exists(DATA_PATH):
        print(f"Loading real dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        missing = set(["price"] + ALL_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Housing.csv is missing required columns: {missing}")
        return df
    print("No Housing.csv found -> generating a synthetic training dataset instead.")
    return generate_synthetic_dataset()


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    df = load_dataset()
    X = df[ALL_FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows")
    print(f"MAE: Rs. {mae:,.0f}")
    print(f"R^2: {r2:.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
