# app.py
"""
Streamlit dashboard for:
Predicting Urban Respiratory Health Risks — Multi-source environmental & health data integration.

Place this file in the project root and run:
  & ".\.venv\Scripts\Activate.ps1"   # (PowerShell)
  streamlit run app.py

Requirements (minimal):
  pip install streamlit pandas numpy plotly joblib scikit-learn

Optional (explainability):
  pip install shap matplotlib
"""

from pathlib import Path
from typing import Optional, Dict, List
import json
import textwrap

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# Optional SHAP
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -----------------------
# Configuration / paths
# -----------------------
ROOT = Path(".")
DATA_FEATURES = ROOT / "data" / "processed" / "merged_daily_features.csv"
METRICS_PATH = ROOT / "results" / "metrics.csv"
MODEL_DIR = ROOT / "models"
FEATURE_IMPORTANCE_DIR = ROOT / "results"

MODEL_FILES = {
    "xgboost": MODEL_DIR / "xgboost.pkl",
    "random_forest": MODEL_DIR / "random_forest.pkl",
    "linear_regression": MODEL_DIR / "linear_regression.pkl",
}

# optional: a json mapping with feature columns saved by training code (if present)
FEATURE_COLS_JSON = MODEL_DIR / "feature_columns.json"


# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def load_features(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        # Normalise timezone-less datetimes to dates for safer comparisons later
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    except Exception as e:
        st.error(f"Failed to load features file: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_feature_columns_json(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_df_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def short_float(x, ndigits=4):
    try:
        return float(f"{x:.{ndigits}f}")
    except Exception:
        return x


# -----------------------
# UI helpers
# -----------------------
def top_kpi_section(df: pd.DataFrame):
    total_rows = len(df)
    cities = int(df["city"].nunique()) if "city" in df.columns else "N/A"
    date_min = df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else "N/A"
    date_max = df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else "N/A"

    kpi1 = df["pm25"].mean() if "pm25" in df.columns else np.nan
    kpi2 = df["pm10"].mean() if "pm10" in df.columns else np.nan
    kpi3 = df["respiratory_admissions"].mean() if "respiratory_admissions" in df.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("Cities", cities)
    c3.metric("Date range", f"{date_min} → {date_max}")
    c4.metric("Avg PM2.5", short_float(kpi1) if not np.isnan(kpi1) else "N/A")

    st.markdown(
        f"""
**Summary stats (subset shown):**  
- Average PM10: {short_float(kpi2) if not np.isnan(kpi2) else 'N/A'}  
- Avg respiratory admissions (daily): {short_float(kpi3) if not np.isnan(kpi3) else 'N/A'}  
"""
    )


def align_input_to_model_features(model, input_df: pd.DataFrame, available_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that input_df contains the same columns (and order) the model expects.
    Strategy:
      1) If model.feature_names_in_ exists -> use it.
      2) Else if models/feature_columns.json exists -> use that.
      3) Else fall back to numeric columns from available_df (sorted).
    Missing columns are filled with 0. Extra columns are dropped.
    """
    # Determine expected feature names
    expected = None
    if hasattr(model, "feature_names_in_"):
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            expected = None

    if expected is None and FEATURE_COLS_JSON.exists():
        expected = load_feature_columns_json(FEATURE_COLS_JSON)

    if expected is None:
        # fallback: use numeric columns from the dataset excluding target-like columns
        drop_like = {"respiratory_admissions", "admissions", "next_day_respiratory_admissions", "date", "city"}
        numeric_cols = [c for c in available_df.select_dtypes(include=[np.number]).columns if c not in drop_like]
        expected = sorted(numeric_cols)

    # Build aligned DataFrame
    aligned = pd.DataFrame(columns=expected)
    for col in expected:
        if col in input_df.columns:
            aligned[col] = input_df[col]
        elif col in available_df.columns:
            # if the available_df has the column but not in input (rare), copy latest value
            aligned[col] = float(available_df[col].iloc[-1]) if pd.notna(available_df[col].iloc[-1]) else 0.0
        else:
            aligned[col] = 0.0

    # Ensure types numeric where possible
    aligned = aligned.astype(float, errors="ignore")
    return aligned


# -----------------------
# Layout & main logic
# -----------------------
st.set_page_config(layout="wide", page_title="Urban Respiratory Health Dashboard")
st.title("Predicting Urban Respiratory Health Risks — Dashboard")
st.caption("Multi-source environmental & health data — dataset preview, model performance and explainability")

# Load data and metrics
df = load_features(DATA_FEATURES)
metrics = load_metrics(METRICS_PATH)

# Sidebar controls
st.sidebar.header("Controls")
st.sidebar.write("Select city / dates and model to inspect. Use pipeline commands in sidebar if files are missing.")

if df is None:
    st.sidebar.error(f"Feature file not found at: {DATA_FEATURES}")
else:
    cities = ["All"] + sorted(df["city"].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("City", options=cities)
    # safe date defaults
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Model for prediction / explain", options=list(MODEL_FILES.keys()))
show_shap = st.sidebar.checkbox("Show SHAP explanations (for tree models)", value=False)
st.sidebar.markdown("---")
if st.sidebar.button("Reload data"):
    st.experimental_rerun()
st.sidebar.markdown("Run pipeline if files are missing:\n`python -m src.features --sample` then `python -m src.models --sample`")

# Main area
if df is None:
    st.warning("Processed feature dataset not found. Run the preprocessing / feature pipeline and then refresh this page.")
    st.stop()

# Convert date inputs to timestamps for comparison
start_date, end_date = date_range
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

# Filter
df_filtered = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
if selected_city != "All":
    df_filtered = df_filtered[df_filtered["city"] == selected_city]

# KPIs and preview
st.header("Dataset overview")
top_kpi_section(df_filtered)

col_left, col_right = st.columns([2.5, 1])

with col_left:
    st.subheader("Data preview")
    st.dataframe(df_filtered.head(200), use_container_width=True)

    st.markdown("### Time series")
    if "date" in df_filtered.columns and "respiratory_admissions" in df_filtered.columns:
        fig = px.line(df_filtered.sort_values("date"), x="date", y="respiratory_admissions",
                      color="city" if selected_city == "All" else None, title="Daily respiratory admissions")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'respiratory_admissions' or 'date' column available for plotting.")

    st.markdown("### Pollution trends (select pollutant)")
    pollutants = [c for c in ["pm25", "pm10", "no2", "o3", "co"] if c in df_filtered.columns]
    if pollutants:
        pollutant = st.selectbox("Pollutant", options=pollutants)
        fig2 = px.line(df_filtered.sort_values("date"), x="date", y=pollutant,
                       color="city" if selected_city == "All" else None, title=f"{pollutant} over time")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No pollutant columns available in data.")

    # Download filtered CSV
    st.download_button("Download filtered sample (CSV)", data=save_df_to_bytes(df_filtered.head(1000)),
                       file_name="filtered_sample.csv")

with col_right:
    st.subheader("Model metrics")
    if metrics is not None:
        st.dataframe(metrics, use_container_width=True)
    else:
        st.info("No metrics file found. Run model training to create results/metrics.csv")

    st.markdown("### Model artifacts")
    for k, path in MODEL_FILES.items():
        exists = path.exists()
        st.write(f"- {k}: {'available' if exists else 'missing'}")
        if exists:
            with open(path, "rb") as f:
                b = f.read()
            st.download_button(f"Download {k} model", data=b, file_name=path.name, key=f"dl_{k}")

# Feature importances display
st.markdown("---")
st.header("Feature importances")
fi_files = {
    "LinearRegression": FEATURE_IMPORTANCE_DIR / "feature_importances_linear_regression.csv",
    "RandomForest": FEATURE_IMPORTANCE_DIR / "feature_importances_random_forest.csv",
    "XGBoost": FEATURE_IMPORTANCE_DIR / "feature_importances_xgboost.csv",
}
cols = st.columns(3)
for col, (title, p) in zip(cols, fi_files.items()):
    with col:
        st.subheader(title)
        if p.exists():
            try:
                df_imp = pd.read_csv(p)
                df_imp_sorted = df_imp.sort_values("importance", ascending=False).head(30)
                st.dataframe(df_imp_sorted.reset_index(drop=True))
                fig = px.bar(df_imp_sorted.head(15).sort_values("importance"), x="importance", y="feature", orientation="h")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Could not read importance file. Check CSV format.")
        else:
            st.info("No importance file.")

# Prediction UI
st.markdown("---")
st.header("Predict next-day respiratory admissions (single-row prediction)")

if selected_city == "All":
    st.info("Select a single city from the sidebar to use the prediction UI.")
else:
    df_city = df[df["city"] == selected_city].sort_values("date")
    if df_city.empty:
        st.warning("No data rows for the selected city.")
    else:
        latest_row = df_city.iloc[-1]
        st.markdown(f"**Using latest row for {selected_city}:** date = {latest_row['date'].strftime('%Y-%m-%d')}")

        # numeric features available
        drop_cols = {"respiratory_admissions", "admissions", "next_day_respiratory_admissions", "date"}
        numeric_cols = [c for c in df_city.select_dtypes(include=[np.number]).columns if c not in drop_cols]

        st.markdown("Adjust numeric inputs (optional) — only numeric features are shown.")
        input_values = {}
        for col in numeric_cols:
            default = float(latest_row[col]) if pd.notna(latest_row[col]) else 0.0
            input_values[col] = st.number_input(col, value=default, format="%.6f")

        X_input = pd.DataFrame([input_values])

        model_path = MODEL_FILES.get(model_choice)
        if model_path is None or not model_path.exists():
            st.warning(f"Model not found: {model_choice}. Train models with `python -m src.models --sample` and retry.")
        else:
            model = load_model(model_path)
            if model is None:
                st.error("Failed to load model file (it may be corrupt or incompatible).")
            else:
                # Align columns to model expectations (safe)
                try:
                    X_aligned = align_input_to_model_features(model, X_input, df_city)
                    preds = model.predict(X_aligned)
                    st.metric("Predicted next-day respiratory admissions (model output)", f"{float(preds[0]):.4f}")
                except Exception as e:
                    st.error(f"Model prediction error: {e}")

                # SHAP local explainability for tree models
                if show_shap and SHAP_AVAILABLE and hasattr(model, "predict"):
                    if model_choice in ("xgboost", "random_forest"):
                        st.subheader("Local SHAP explanation (tree model)")
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_vals = explainer.shap_values(X_aligned)
                            arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                            shap_contrib = pd.Series(np.abs(arr[0]), index=X_aligned.columns).sort_values(ascending=False)
                            st.write("Top contributing features (absolute SHAP):")
                            st.dataframe(shap_contrib.head(20).to_frame("abs_shap"))
                            fig, ax = plt.subplots(figsize=(6, 4))
                            shap_contrib.head(15).plot.barh(ax=ax)
                            ax.invert_yaxis()
                            ax.set_xlabel("abs(SHAP)")
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"SHAP explanation failed: {e}")
                    else:
                        st.info("SHAP local plots are most informative for tree models (random_forest, xgboost).")
                elif show_shap and not SHAP_AVAILABLE:
                    st.info("Install `shap` (pip install shap) to enable SHAP explainability.")

# Optional map
if "latitude" in df.columns and "longitude" in df.columns:
    st.markdown("---")
    st.header("Map")
    map_df = df_filtered.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].dropna()
    st.map(map_df)

# Footer / instructions
st.markdown("---")
st.markdown(
    textwrap.dedent(
        """
    )
)
