import pandas as pd
from pathlib import Path

def load_health_data(sample=True):
    if sample:
        p_health = Path("data/samples/health_sample.csv")
    else:
        p_health = Path("data/raw/health.csv")

    if not p_health.exists():
        print(f"Health file not found: {p_health}")
        return None

    try:
        return pd.read_csv(p_health)
    except Exception as e:
        print("Error reading health file:", e)
        return None


def create_target(df_merged: pd.DataFrame, df_health: pd.DataFrame) -> pd.DataFrame:
    """
    Merge admissions or health counts into merged dataset by city & date.
    Looks for respiratory_admissions -> hospital_admissions -> other names.
    """
    df = df_merged.copy()

    # normalize merged date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if df_health is not None:
        # ensure date exists and normalize
        if "date" not in df_health.columns:
            df_health["date"] = pd.to_datetime(df_health.iloc[:, 0], errors="coerce")
        else:
            df_health["date"] = pd.to_datetime(df_health["date"], errors="coerce")
        df_health["date"] = df_health["date"].dt.normalize()

        # choose priority columns
        priority_cols = [
            "respiratory_admissions",
            "hospital_admissions",
            "admissions",
            "cases",
            "count",
            "value",
            "hospital_adm",
            "admission_count",
        ]

        count_col = next((c for c in priority_cols if c in df_health.columns), None)

        if count_col is None:
            numeric_cols = (
                df_health.select_dtypes(include="number")
                .columns.difference(["date", "city"])
                .tolist()
            )
            count_col = numeric_cols[0] if numeric_cols else None

        # detect city column
        city_col = next(
            (c for c in ["city", "City", "location", "district", "area"]
             if c in df_health.columns),
            "city",
        )
        if city_col not in df_health.columns:
            df_health["city"] = "Unknown"
            city_col = "city"

        if count_col is not None:
            df_health_agg = (
                df_health[[city_col, "date", count_col]]
                .groupby([city_col, "date"], as_index=False)
                .agg({count_col: "sum"})
                .rename(columns={city_col: "city", count_col: "admissions"})
            )
            print(f"Using health column '{count_col}' (mapped to 'admissions').")

            df = df.merge(df_health_agg, on=["city", "date"], how="left")
            df["admissions"] = df["admissions"].fillna(0).astype(int)
            return df

    # fallback if no health file or no numeric admissions columns
    print("No valid admissions column found. Creating heuristic from pm25 mean.")
    if "pm25_mean" in df.columns:
        df["admissions"] = (df["pm25_mean"] > df["pm25_mean"].mean()).astype(int)
    elif "pm25" in df.columns:
        df["admissions"] = (df["pm25"] > df["pm25"].mean()).astype(int)
    else:
        df["admissions"] = 0
    return df


def main(sample=True):
    p_clean = Path("data/processed/cleaned_merged_daily.csv")
    if not p_clean.exists():
        print("Missing cleaned_merged_daily.csv. Run preprocess first.")
        return

    df = pd.read_csv(p_clean)
    df_health = load_health_data(sample=sample)

    df = create_target(df, df_health)

    # save outputs
    out_full = Path("data/processed/merged_daily_full.csv")
    out_model = Path("data/processed/merged_daily_model.csv")

    out_full.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_full, index=False)
    df.to_csv(out_model, index=False)

    print(f"Saved:\n - {out_full}\n - {out_model}")


if __name__ == "__main__":
    main(sample=True)
