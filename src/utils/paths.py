# In src/utils/paths.py (or src/paths.py)
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Common Directory Paths ---
DATA_DIR = PROJECT_ROOT / "Data"
SRC_DIR = PROJECT_ROOT / "src"
LOGS_DIR = PROJECT_ROOT / "logs"
CSV_DIR = DATA_DIR / "files_csv"
JSON_DIR = DATA_DIR / "files_json"
IMG_DIR = DATA_DIR / "images_file"
DATA_PIPE = SRC_DIR / "data_pipeline"
MODELS = SRC_DIR / "models"
MODELSS = MODELS / "models"

# --- Common File Paths (for output/input) ---
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DB_PATH =  DATA_PIPE / "data_store.db" # Or DATA_DIR / "data_store.db"

# Feature Engineering Output / Data Cleaning Input
ALL_FEATURE_CSV = CSV_DIR / "all_feature.csv"

# Data Cleaning Output
ALL_CLEANED_CSV = CSV_DIR / "all_features_cleaned.csv"

# Cluster Stocks Results
ALL_Stocluster_CSV = CSV_DIR / "stock_clusters.csv"

# Prophet Output
PROPHET_METRICS_JSON = JSON_DIR / "prophet_metrics.json"

# Recommend json Output
RECOMMEND_JSON= JSON_DIR / "recommendation_summary.json"

# Recommend json Output
TICKER_MAPPING= JSON_DIR / "ticker_mapping.json"

# Signal Confluence Output
CONFLUENCE_CSV = CSV_DIR / "signal_confluence_results.csv"

# Daily Recomendation
RECOMMEND_CSV = CSV_DIR / "daily_recommendations.csv"

# Stock predictability leaderboard
STOCK_PREDICTABILY = CSV_DIR / "stock_predictability_leaderboard.csv"

# Recommending Portfolio
RECOMMEND_PORTFOLIO = CSV_DIR / "recommended_portfolio.csv"

#session Csv
SESSION_DATA = CSV_DIR / "session_data.csv"

#Feature session csv
FEATURE_SESSION_DATA = CSV_DIR / "session_features.csv"


MODEL_PATH = MODELSS / "session_rf_model.pkl"

SCALER_PATH = MODELSS / "session_scaler.pkl"

FEATURE_PATH = MODELSS / "feature_names.pkl"

def check_and_create_dirs():
    """Ensure all necessary output directories exist."""
    print(os.makedirs(CSV_DIR, exist_ok=True))
    print(os.makedirs(JSON_DIR, exist_ok=True))
    print(os.makedirs(LOGS_DIR, exist_ok=True))
    print(os.makedirs(DATA_PIPE, exist_ok=True))
    print(os.makedirs(MODELS, exist_ok=True))
    print(os.makedirs(MODELSS, exist_ok=True))