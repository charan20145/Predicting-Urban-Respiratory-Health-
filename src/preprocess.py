"""
src/preprocess.py

Preprocessing pipeline for sample data.

Usage:
    python -m src.preprocess --sample
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.config import RAW_DIR, PROCESSED_DIR, ensure_directories

ensure_directories()


def _read_raw_csv(name: str) -> pd.DataFrame:
    """
    Read a raw csv saved by data_fetch, for example: raw_air_sample.csv
    """
    p = Path(RAW_DIR) / f"raw_{name}_sample.csv"
    if not p.exists():
        raise FileNotFoundError(f"Raw sample file not found: {p}")
    df = pd.read_csv(p)
    return df


def normalize_timestamp(df: pd.DataFrame, col_candidates=("timestamp", "date", "datetime")) -> pd.DataFrame:
    """
    Ensure there is a datetime column named 'date' (date only) and 'timestamp' (datetime),
    and set the 'date' column to a pandas datetime (UTC-naive).
    """
    # find a column that looks like timestamp
    for c in col_candidates:
        if c in df.columns:
            dt_col = c
            break
    else:
        # try to detect a datetime-like column
        possible = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible:
            dt_col = possible[0]
        else:
            raise ValueError("No datetime-like column found (expected 'date' or 'timestamp').")

    # convert
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    # create a normalized date (date only) and timestamp columns
    if "timestamp" not in df.columns:
        df["timestamp"] = df[dt_col]
    df["date"] = df["timestamp"].dt.normalize()
    return df


def normalize_pollutant_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for unit conversions. For sample data we assume values are already µg/m3 for PM
    and ppb for gases. This function will fill missing pollutant columns with NaN if absent.
    """
    df = df.copy()
    expected = ["pm25", "pm10", "no2", "o3", "co"]
    # some datasets use pm25_mean etc. Try to detect and map.
    for ex in expected:
        if ex not in df.columns:
            # check for ex + '_mean'
            mean_name = f"{ex}_mean"
            if mean_name in df.columns:
                df[ex] = df[mean_name]
            else:
                df[ex] = np.nan
    return df


def handle_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple missingness handling:
    - For time series: forward-fill then interpolate numeric columns.
    - Clip obviously invalid pollutant values (negative).
    """
    df = df.copy()
    # sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # clip negatives for pollutant related columns (if present)
    pollutant_cols = [c for c in num_cols if c.lower().startswith(("pm", "no", "o3", "co"))]
    for c in pollutant_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan

    # fill small gaps: forward-fill then interpolate numeric columns
    df[num_cols] = df[num_cols].ffill().bfill().interpolate(method="linear", limit_direction="both")

    # remove duplicates
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"])
    else:
        df = df.drop_duplicates()
    return df


def build_cleaned_merged(save_path: Path = None) -> Path:
    """
    Build cleaned merged dataset for modeling. Reads the raw merged sample produced by data_fetch.
    Saves cleaned CSV to data/processed/cleaned_merged_daily.csv
    Returns saved path.
    """
    merged = _read_raw_csv("merged")
    # normalize timestamps/columns
    merged = normalize_timestamp(merged, col_candidates=("date", "timestamp"))
    merged = normalize_pollutant_units(merged)
    merged = handle_missing_and_outliers(merged)
    # round numeric pollutant stats
    for col in ["pm25", "pm10", "no2", "o3", "co"]:
        if col in merged.columns:
            merged[col] = merged[col].round(4)
    # ensure output dir
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "cleaned_merged_daily.csv" if save_path is None else save_path
    merged.to_csv(out, index=False)
    return out


def main(sample=True):
    if sample:
        path = build_cleaned_merged()
        print(f"Preprocess: cleaned merged file saved to {path}")
    else:
        raise NotImplementedError("Full preprocessing not implemented in this template.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run preprocessing on sample files")
    args = parser.parse_args()
    if args.sample:
        main(sample=True)
    else:
        main(sample=True)
