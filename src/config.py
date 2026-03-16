from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

RAW_FILE = RAW_DIR / "Data_Train.xlsx"
CLEAN_FILE = PROCESSED_DIR / "train_clean.csv"
FEATURE_FILE = PROCESSED_DIR / "train_features.csv"
TARGET_COL = "Price"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "journey_day",
    "journey_month",
    "dep_hour",
    "dep_minute",
    "arrival_hour",
    "arrival_minute",
    "duration_minutes",
    "total_stops_num",
]

CATEGORICAL_FEATURES = [
    "Airline",
    "Source",
    "Destination",
    "Route",
    "Additional_Info",
]
