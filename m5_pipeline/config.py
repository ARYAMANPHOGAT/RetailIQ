"""Paths and fixed experiment settings for the isolated M5 pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "m5"
SALES_PATH = DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"
PRICES_PATH = DATA_DIR / "sell_prices.csv"

PIPELINE_DIR = ROOT_DIR / "m5_pipeline"
MODELS_DIR = ROOT_DIR / "models" / "m5"
LOGS_DIR = ROOT_DIR / "logs" / "m5"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "m5"
VALIDATION_REPORT_PATH = OUTPUTS_DIR / "validation_report.json"
VALIDATION_LOG_PATH = LOGS_DIR / "validation.log"
FEATURES_DIR = OUTPUTS_DIR / "features"
FEATURE_METADATA_PATH = FEATURES_DIR / "feature_metadata.json"
SERIES_METADATA_PATH = FEATURES_DIR / "series_metadata.csv"
FEATURE_LOG_PATH = LOGS_DIR / "features.log"
TRAINING_LOG_PATH = LOGS_DIR / "training.log"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
ANALYSIS_LOG_PATH = LOGS_DIR / "analysis.log"

SERIES_ID_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
FIRST_SALES_DAY = 1
LAST_SALES_DAY = 1941
VALIDATION_DAYS = 28
TEST_DAY_START = 1914
TEST_DAY_END = 1941

# Memory and reproducibility controls to be used by later pipeline stages.
CSV_CHUNK_ROWS = 100_000
RANDOM_SEED = 42
MAX_TRAIN_ROWS_PER_MODEL = 2_000_000
SERIES_BATCH_SIZE = 256
LIGHTGBM_MAX_BOOST_ROUNDS = 1000
LIGHTGBM_EARLY_STOPPING_ROUNDS = 80
LIGHTGBM_LOG_PERIOD = 25
LIGHTGBM_NUM_THREADS = 8
ANALYSIS_SAMPLE_ROWS = 50_000
ANALYSIS_PERMUTATION_REPEATS = 3
