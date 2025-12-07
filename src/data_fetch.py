"""
src/data_fetch.py

Usage:
    python -m src.data_fetch --sample
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

from src.config import SAMPLES_DIR, RAW_DIR, DOCS_DIR, ensure_directories

ensure_directories()

def _safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

def load_sample_files(samples_dir: Path):
    files = {
        "air": samples_dir / "air_sample.csv",
        "weather": samples_dir / "weather_sample.csv",
        "traffic": samples_dir / "traffic_sample.csv",
        "health": samples_dir / "health_sample.csv",
        "merged": samples_dir / "merged_daily.csv",
        "population": samples_dir / "population_by_district.csv",
        "districts": samples_dir / "districts_geo.json",
    }
    loaded = {}
    for k, p in files.items():
        if p.exists():
            if p.suffix.lower() == ".csv":
                df = _safe_read_csv(p)
                if df is not None:
                    loaded[k] = df
                    print(f"Loaded sample '{k}' ({len(df)} rows) from {p.name}")
            else:
                try:
                    loaded[k] = p.read_text(encoding="utf-8")
                    print(f"Loaded sample '{k}' (text) from {p.name}")
                except Exception as e:
                    print(f"Warning: could not read {p}: {e}")
        else:
            print(f"Sample file not found: {p}")
    return loaded

def save_raw_samples(loaded: dict, raw_dir: Path, maxrows: int = 200):
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for k, obj in loaded.items():
        if isinstance(obj, pd.DataFrame):
            out = raw_dir / f"raw_{k}_sample.csv"
            obj.head(maxrows).to_csv(out, index=False)
            meta.append({
                "source": k,
                "filename": str(out),
                "rows_saved": min(len(obj), maxrows),
                "date_fetched": datetime.utcnow().isoformat(),
                "notes": "sample saved"
            })
            print(f"Saved: {out}")
        else:
            out = raw_dir / f"raw_{k}_sample.json"
            out.write_text(str(obj))
            meta.append({
                "source": k,
                "filename": str(out),
                "rows_saved": None,
                "date_fetched": datetime.utcnow().isoformat(),
                "notes": "text saved"
            })
            print(f"Saved: {out}")
    return meta

def write_metadata(meta_rows, docs_dir: Path):
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "data_metadata.csv"
    import pandas as pd
    existing = []
    if out.exists():
        try:
            existing = pd.read_csv(out).to_dict(orient="records")
        except Exception:
            existing = []
    all_rows = existing + meta_rows
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"Wrote metadata to {out}")

def main(sample=True):
    samples_dir = Path(SAMPLES_DIR)
    raw_dir = Path(RAW_DIR)
    docs_dir = Path(DOCS_DIR)

    if sample:
        loaded = load_sample_files(samples_dir)
        if not loaded:
            print(f"No sample files found in {samples_dir}.")
            sys.exit(2)
        meta_rows = save_raw_samples(loaded, raw_dir)
        write_metadata(meta_rows, docs_dir)
        print("data_fetch: sample mode complete.")
    else:
        print("Full API fetch not implemented.")
        sys.exit(3)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full:
        main(sample=False)
    else:
        main(sample=True)
