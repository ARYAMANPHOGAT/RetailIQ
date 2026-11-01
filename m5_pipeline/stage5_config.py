"""Isolated settings and paths for the Stage 5 hybrid experiment."""
from . import config as Base
from . import stage4_config as Stage4

ROOT_DIR = Base.OUTPUTS_DIR / "stage5"
MODEL_DIR = Base.MODELS_DIR / "stage5"
RESULTS_DIR = ROOT_DIR / "results"
LOG_DIR = Base.LOGS_DIR / "stage5"
FEATURES_DIR = Stage4.FEATURES_DIR  # read-only reuse

MODEL_PATH = MODEL_DIR / "m5_hybrid_stage5.txt"
MODEL_METADATA_PATH = MODEL_DIR / "m5_hybrid_metadata.json"
VALIDATION_METRICS_PATH = RESULTS_DIR / "stage5_validation_metrics.json"
VALIDATION_PREDICTIONS_PATH = RESULTS_DIR / "stage5_validation_predictions.csv.gz"
SELECTION_PATH = RESULTS_DIR / "validation_selection.json"
TEST_METRICS_PATH = RESULTS_DIR / "stage5_test_metrics.json"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "stage5_test_predictions.csv.gz"
TRAINING_LOG_PATH = LOG_DIR / "training.log"
TEST_LOG_PATH = LOG_DIR / "test_evaluation.log"

# Frozen primary settings approved for this experiment; no search is performed.
INTERMITTENT_RATE_THRESHOLD = 0.20
MIN_HISTORY_DAYS = 84
TSB_ALPHA = 0.10
TSB_BETA = 0.10
SAMPLE_ROWS = Stage4.SAMPLE_ROWS
RANDOM_SEED = Stage4.RANDOM_SEED
SERIES_BATCH_SIZE = Stage4.SERIES_BATCH_SIZE
