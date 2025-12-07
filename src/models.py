"""
src/models.py

Train baseline and advanced models on the feature dataset, save trained models,
append metrics to results/metrics.csv, and write some diagnostics.

Usage:
    # from project root with virtualenv activated and PYTHONPATH=.
    python -m src.models --sample
    # or for verbose output
    python -u -m src.models --sample
"""

from __future__ import annotations
import os
from pathlib import Path
import joblib
from datetime import datetime
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: xgboost if available
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42


def save_model_and_metrics(model, model_name, mae, rmse, r2,
                           results_dir="results", models_dir="models"):
    """
    Save trained model and append evaluation metrics to results/metrics.csv.
    Creates the folders and file if they do not already exist.
    Returns tuple (model_path, metrics_path).
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)

    # Prepare metrics row
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    metrics_path = os.path.join(results_dir, "metrics.csv")
    df_row = pd.DataFrame([row])

    if not os.path.exists(metrics_path):
        df_row.to_csv(metrics_path, index=False)
    else:
        df_row.to_csv(metrics_path, index=False, mode="a", header=False)

    return model_path, metrics_path


def load_feature_data(path: str = "data/processed/merged_daily_features.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature file not found: {p}. Run `python -m src.features --sample` first.")
    df = pd.read_csv(p, parse_dates=["date"], low_memory=False)
    return df


def prepare_X_y(df: pd.DataFrame, target_col: str = "admissions"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in features.")
    # Basic drop columns that are obviously non-feature
    drop_cols = ["date", "city", "district", "timestamp"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors="ignore")
    y = df[target_col].values

    # Convert categorical dtypes to numeric (one-hot for low-cardinality)
    # Simple rule: object columns -> one-hot
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # final fill/convert
    X = X.fillna(0)
    # ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test,
                       results_dir="results", models_dir="models"):
    # fit
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(mse ** 0.5)
    r2 = r2_score(y_test, preds)

    # save model and metrics
    model_path, metrics_path = save_model_and_metrics(model, model_name, mae, rmse, r2,
                                                     results_dir=results_dir, models_dir=models_dir)

    # try to save feature importances (if available)
    fi_path = None
    try:
        if hasattr(model, "coef_"):
            # linear model coefficients
            coefs = model.coef_
            feat_names = X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(len(coefs))]
            df_coefs = pd.DataFrame({"feature": feat_names, "importance": coefs})
            fi_path = os.path.join(results_dir, f"feature_importances_{model_name}.csv")
            df_coefs.to_csv(fi_path, index=False)
        elif hasattr(model, "feature_importances_"):
            feat_names = X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(len(model.feature_importances_))]
            df_fi = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
            fi_path = os.path.join(results_dir, f"feature_importances_{model_name}.csv")
            df_fi.sort_values("importance", ascending=False).to_csv(fi_path, index=False)
    except Exception:
        fi_path = None

    # return dict of results
    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "feature_importances_path": fi_path,
    }


def run_grid_search_rf(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def run_grid_search_xgb(X_train, y_train):
    if not _HAS_XGB:
        raise RuntimeError("XGBoost is not installed in the environment.")
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1]
    }
    xgb = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    gs = GridSearchCV(xgb, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def main(sample=True):
    # paths
    results_dir = "results"
    models_dir = "models"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # load features
    df = load_feature_data("data/processed/merged_daily_features.csv")
    print(f"Loaded features: data/processed/merged_daily_features.csv")

    # prepare X, y
    X, y = prepare_X_y(df, target_col="admissions")
    print(f"Training with {len(X)} rows and {X.shape[1]} features.\n")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # models to train
    models_to_run = []

    # linear regression
    models_to_run.append(("linear_regression", LinearRegression()))

    # random forest (use grid-search to pick hyperparams)
    models_to_run.append(("random_forest", "grid_search_rf"))

    # xgboost if available
    if _HAS_XGB:
        models_to_run.append(("xgboost", "grid_search_xgb"))

    summary_rows = []

    for name, mdl in models_to_run:
        print(f"Training model: {name}")
        if name == "random_forest" and mdl == "grid_search_rf":
            best_rf, best_params = run_grid_search_rf(X_train, y_train)
            model_obj = best_rf
            print(f"Best params (RF): {best_params}")
        elif name == "xgboost" and mdl == "grid_search_xgb":
            best_xgb, best_params = run_grid_search_xgb(X_train, y_train)
            model_obj = best_xgb
            print(f"Best params (XGB): {best_params}")
        else:
            model_obj = mdl

        res = train_and_evaluate(model_obj, name, X_train, X_test, y_train, y_test,
                                 results_dir=results_dir, models_dir=models_dir)

        print(f"{name} results -> MAE: {res['mae']:.4f}  RMSE: {res['rmse']:.4f}  R²: {res['r2']:.4f}")
        if res.get("feature_importances_path"):
            print(f"Saved feature importances to {res['feature_importances_path']}")
        print(f"Saved metrics to {res['metrics_path']} and model to {res['model_path']}\n")

        summary_rows.append(res)

    # write simple summary file
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Model training summary\n")
        fh.write(f"Generated at: {datetime.utcnow().isoformat()}\n\n")
        for r in summary_rows:
            fh.write(f"{r['model_name']}: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}, R2={r['r2']:.4f}\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="use sample pipeline (keeps behavior identical)")
    args = parser.parse_args()
    main(sample=True)
