"""
src/features.py

Generate feature-engineered dataset for model training.

Usage:
    python -m src.features --sample
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR, ensure_directories

ensure_directories()

INPUT_FILE = Path(PROCESSED_DIR) / "merged_daily_model.csv"
OUTPUT_FILE = Path(PROCESSED_DIR) / "merged_daily_features.csv"


def make_lag_features(df, col, lags=[1, 3, 7]):
    """Generate lag features like pm25_lag1, pm25_lag3, pm25_lag7"""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def make_rolling_features(df, col, windows=[3, 7]):
    """Rolling means like pm25_roll3, pm25_roll7"""
    for w in windows:
        df[f"{col}_roll{w}"] = df[col].rolling(w).mean()
    return df


def feature_engineering(df):

    # Sort for correct rolling/lag generation
    df = df.sort_values("date")

    POLLUTANTS = ["pm25", "pm10", "no2", "o3", "co"]

    # Create lag & rolling features for each pollutant
    for col in POLLUTANTS:
        df = make_lag_features(df, col)
        df = make_rolling_features(df, col)

    # Weather interactions
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_x_humidity"] = df["temperature"] * df["humidity"]

    # Traffic feature
    if "traffic_volume" in df.columns:
        df["traffic_category"] = pd.qcut(
            df["traffic_volume"],
            q=3,
            labels=["LowTraffic", "MedTraffic", "HighTraffic"]
        )

    # Fill NaNs created by lags
    df = df.fillna(0)

    return df


def load_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing merged file: {INPUT_FILE}. Run `python -m src.merge --sample` first."
        )
    return pd.read_csv(INPUT_FILE, parse_dates=["date"])


def save_output(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved feature dataset: {OUTPUT_FILE}")


def main(sample=True):
    df = load_input()
    df = feature_engineering(df)
    save_output(df)
    print("features: done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(sample=True)
