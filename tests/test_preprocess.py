# tests/test_preprocess.py

import os
from pathlib import Path
import pandas as pd
import src.preprocess as pp

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def test_build_cleaned_merged_exists():
    # ensure raw merged exists
    raw_merged = RAW_DIR / "raw_merged_sample.csv"
    if not raw_merged.exists():
        # try to run data_fetch to produce it
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "src.data_fetch", "--sample"], check=True)
    assert raw_merged.exists(), "raw_merged_sample.csv must exist for preprocessing test"

    out = pp.build_cleaned_merged()
    assert out.exists(), "cleaned merged file was not written"
    df = pd.read_csv(out)
    # basic checks
    assert "date" in df.columns or "timestamp" in df.columns
    # numeric pollutant columns exist or at least are present
    assert any(c in df.columns for c in ["pm25", "pm10", "no2", "o3", "co"])
