from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
DOCS_DIR = DATA_DIR / "docs"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SHAP_DIR = RESULTS_DIR / "shap"

def ensure_directories():
    dirs = [
        DATA_DIR, RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, DOCS_DIR,
        MODELS_DIR, RESULTS_DIR, PLOTS_DIR, SHAP_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Directories ensured.")
