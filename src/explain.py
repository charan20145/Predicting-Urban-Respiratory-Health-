"""
src/explain.py

Generate SHAP explanation for a saved model.
Usage:
    python -m src.explain --model models/xgboost.pkl
"""
from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, ensure_directories

ensure_directories()

def load_data_and_model(model_path: Path):
    df = pd.read_csv(Path(PROCESSED_DIR) / "merged_daily_features.csv", parse_dates=["date"])
    X = df.select_dtypes(include=["number"]).fillna(0)
    model = joblib.load(model_path)
    return X, model

def explain_model(model_path: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    X, model = load_data_and_model(model_path)
    # use a sample subset for speed
    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    out_dir = Path(RESULTS_DIR) / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=200)
    # save shap values to CSV (reduced)
    sv_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
    sv_df.to_csv(out_dir / "shap_values_sample.csv", index=False)
    print(f"Saved SHAP summary image and values to {out_dir}")

def main(model_path=None):
    if model_path is None:
        model_path = Path(MODELS_DIR) / "xgboost.pkl"
    explain_model(str(model_path))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    main(model_path=args.model)
