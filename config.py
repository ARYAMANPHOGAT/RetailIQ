"""
RetailIQ - central configuration.

Every path, threshold and modelling constant lives here so it can be found
and changed in ONE obvious place (including the inventory-risk thresholds).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project folder, so the project is portable)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "retail_store_inventory.csv"
DEMO_DATA_PATH = DATA_DIR / "demo" / "demo_retail_sample.csv"
TEMPLATE_PATH = DATA_DIR / "templates" / "retailiq_upload_template.csv"

MODELS_DIR = ROOT_DIR / "models"
DEMAND_MODEL_PATH = MODELS_DIR / "demand_model.joblib"
SEGMENTATION_PATH = MODELS_DIR / "segmentation_bundle.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

DB_PATH = ROOT_DIR / "database" / "retailiq.db"
LOG_PATH = ROOT_DIR / "logs" / "retailiq.log"
CSS_PATH = ROOT_DIR / "assets" / "style.css"

TRAIN_COMMAND = "python train.py"

# --------------------------------------------------------------------------
# Dataset schema (verified against the real retail_store_inventory.csv)
# --------------------------------------------------------------------------
DATE_COL = "Date"
STORE_COL = "Store ID"
PRODUCT_COL = "Product ID"
SALES_COL = "Units Sold"            # forecasting target (next day)
INVENTORY_COL = "Inventory Level"
PRICE_COL = "Price"
DISCOUNT_COL = "Discount"

# Columns an upload MUST contain (used by forecasting, risk and segmentation)
REQUIRED_COLUMNS = {
    DATE_COL: "date",        # YYYY-MM-DD, one row per store-product-day
    STORE_COL: "text",       # e.g. S001
    PRODUCT_COL: "text",     # e.g. P0001
    INVENTORY_COL: "number", # stock level recorded for that day (>= 0)
    SALES_COL: "number",     # units sold that day (>= 0)
    PRICE_COL: "number",     # selling price (>= 0)
    DISCOUNT_COL: "number",  # discount in percent (>= 0)
}

# Columns that exist in the original file. They are accepted in uploads but
# are NOT used by the models (see README: leakage / future-value rules).
OPTIONAL_COLUMNS = [
    "Category", "Region", "Units Ordered", "Demand Forecast",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing", "Seasonality",
]
LEAKY_COLUMNS = ["Demand Forecast"]  # ~0.997 correlated with Units Sold -> never a feature

# --------------------------------------------------------------------------
# Forecasting constants
# --------------------------------------------------------------------------
LAGS = (1, 7)              # lag_k = units sold k days before the target day
ROLL_WINDOW = 7            # rolling mean/std over the 7 days up to the cutoff
MIN_HISTORY_DAYS = 7       # consecutive days needed to forecast one series
TEST_FRACTION = 0.20       # final 20% of unique target dates = holdout test
CV_SPLITS = 3              # date-grouped expanding-window folds for tuning
RANDOM_STATE = 42

RF_PARAM_GRID = {          # deliberately modest so training stays practical
    "model__n_estimators": [100],
    "model__max_depth": [8, 16],
    "model__min_samples_leaf": [20, 100],
    "model__max_features": [0.33],
}

# --------------------------------------------------------------------------
# Inventory risk rules (rule-based, NOT a trained stockout model)
#   coverage_days = current inventory / recent average daily demand
# --------------------------------------------------------------------------
RISK_RECENT_DAYS = 7          # "recent average daily demand" window (calendar days)
RISK_HIGH_MAX_DAYS = 1.0      # coverage <  1.0 day            -> High
RISK_MEDIUM_MAX_DAYS = 3.0    # 1.0 <= coverage < 3.0 days     -> Medium, else Low
MIN_DEMAND_PER_DAY = 0.5      # below this, demand is treated as ~zero (no division)
TARGET_COVER_DAYS = 3.0       # replenishment ESTIMATE brings cover up to this level

# --------------------------------------------------------------------------
# Segmentation (K-Means on 28-day store-product behaviour profiles)
# --------------------------------------------------------------------------
SEGMENT_WINDOW_DAYS = 28
SEGMENT_FEATURES = [
    "avg_units_sold", "demand_std", "avg_inventory",
    "sell_through_rate", "avg_price", "avg_discount",
]
K_RANGE_EVALUATED = range(2, 9)   # k values shown in elbow / silhouette charts
K_RANGE_ALLOWED = range(3, 7)     # k chosen = best silhouette within 3..6 segments

# --------------------------------------------------------------------------
# Upload limits
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
MAX_UPLOAD_MB = 50
MAX_ROWS = 500_000
