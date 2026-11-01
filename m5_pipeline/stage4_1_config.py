"""Isolated settings for a small Stage 4 same-weekday feature experiment."""
from . import config as Base
from . import stage4_config as Stage4

ROOT_DIR = Base.OUTPUTS_DIR / "stage4_1"
RESULTS_DIR = ROOT_DIR / "results"
MODEL_DIR = Base.MODELS_DIR / "stage4_1"
LOG_DIR = Base.LOGS_DIR / "stage4_1"
MODEL_PATH = MODEL_DIR / "m5_lightgbm_stage4_1.txt"
MODEL_METADATA_PATH = MODEL_DIR / "m5_training_metadata.json"
METRICS_PATH = RESULTS_DIR / "stage4_1_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "stage4_1_validation_predictions.csv.gz"
IMPORTANCE_PATH = RESULTS_DIR / "stage4_1_feature_importance.csv"
TRAINING_LOG_PATH = LOG_DIR / "training.log"

FEATURES_DIR = Stage4.FEATURES_DIR  # read-only source; no shards are regenerated
SAMPLE_ROWS = Stage4.SAMPLE_ROWS
RANDOM_SEED = Stage4.RANDOM_SEED
