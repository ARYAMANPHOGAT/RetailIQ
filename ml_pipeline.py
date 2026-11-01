"""
RetailIQ - machine-learning and data logic (no Streamlit code in here).

Sections
  1. Logging and friendly errors
  2. Loading + validating uploaded/raw data
  3. Next-day feature engineering (leak-free lag / rolling features)
  4. Model building, chronological splitting and metrics
  5. Inventory risk rules (rule-based, not a trained model)
  6. Product segmentation helpers (K-Means + PCA)
  7. Artifact loading and the full analysis run used by the app
"""
from __future__ import annotations

import io
import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

# ==========================================================================
# 1. Logging and friendly errors
# ==========================================================================

def get_logger() -> logging.Logger:
    """Technical details go to logs/retailiq.log, never to the user's screen."""
    logger = logging.getLogger("retailiq")
    if not logger.handlers:
        C.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(C.LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class UserFacingError(Exception):
    """An error whose message is safe and helpful to show in the UI."""


# ==========================================================================
# 2. Loading + validating data
# ==========================================================================

def read_table(source, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel file (path or uploaded bytes) into a DataFrame."""
    ext = Path(filename).suffix.lower()
    if ext not in C.ALLOWED_EXTENSIONS:
        raise UserFacingError(
            f"Unsupported file type '{ext or 'none'}'. Please upload a .csv or .xlsx file."
        )
    if isinstance(source, (bytes, bytearray)):
        size_mb = len(source) / 1_048_576
        if size_mb > C.MAX_UPLOAD_MB:
            raise UserFacingError(f"File is {size_mb:.1f} MB; the limit is {C.MAX_UPLOAD_MB} MB.")
        if len(source) == 0:
            raise UserFacingError("The file is empty.")
        source = io.BytesIO(source)
    try:
        if ext == ".csv":
            df = pd.read_csv(source)
        else:
            df = pd.read_excel(source, engine="openpyxl")
    except pd.errors.EmptyDataError:
        raise UserFacingError("The file is empty or has no header row.")
    except Exception as exc:  # corrupted / wrong format
        get_logger().exception("Could not read %s", filename)
        raise UserFacingError(
            f"Could not read '{filename}'. The file may be corrupted or not a real "
            f"{ext} file ({type(exc).__name__})."
        )
    if df.empty or df.shape[1] == 0:
        raise UserFacingError("The file contains no data rows.")
    if len(df) > C.MAX_ROWS:
        raise UserFacingError(f"The file has {len(df):,} rows; the limit is {C.MAX_ROWS:,}.")
    return df


@dataclass
class ValidationResult:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    clean: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)
    preprocessing: list = field(default_factory=list)
    series_table: pd.DataFrame | None = None


def _rows_hint(index, limit: int = 8) -> str:
    """Spreadsheet-style row numbers (header = row 1)."""
    rows = [int(i) + 2 for i in list(index)[:limit]]
    more = f" (+{len(index) - limit} more)" if len(index) > limit else ""
    return ", ".join(map(str, rows)) + more


def validate_dataset(raw: pd.DataFrame, known_stores=None, known_products=None) -> ValidationResult:
    """Check an uploaded table against the documented schema.

    Nothing is silently fixed: invalid values block the analysis with an
    exact message. Only harmless normalisation (trimming header spaces,
    matching header case, parsing types) is applied and reported.
    """
    res = ValidationResult(ok=False)
    df = raw.copy()

    # --- column names: trim + case-insensitive match to the documented names
    canonical = {c.lower(): c for c in list(C.REQUIRED_COLUMNS) + C.OPTIONAL_COLUMNS}
    rename, unknown = {}, []
    for col in df.columns:
        key = str(col).strip().lower()
        if key in canonical:
            if col != canonical[key]:
                rename[col] = canonical[key]
        else:
            unknown.append(str(col))
    if rename:
        df = df.rename(columns=rename)
        res.preprocessing.append(f"Matched column names: {', '.join(f'{a!r}->{b!r}' for a, b in rename.items())}")
    if df.columns.duplicated().any():
        res.errors.append("Two or more columns have the same name after trimming spaces/case.")
        return res

    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        res.errors.append(
            "Missing required column(s): " + ", ".join(missing)
            + ". Download the template to see the exact headers."
        )
        res.summary = {"rows": len(df), "columns": list(map(str, raw.columns))}
        return res
    if unknown:
        res.warnings.append(f"Unrecognised column(s) ignored: {', '.join(unknown)}")
    present_optional = [c for c in C.OPTIONAL_COLUMNS if c in df.columns]
    if any(c in df.columns for c in C.LEAKY_COLUMNS):
        res.warnings.append(
            "'Demand Forecast' is present but is never used as a model input "
            "(it is ~0.997 correlated with Units Sold and would leak the answer)."
        )

    # --- missing values in required columns (never filled silently)
    for col in C.REQUIRED_COLUMNS:
        na = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if na.any():
            res.errors.append(f"'{col}' has {int(na.sum())} empty value(s) at row(s) {_rows_hint(df.index[na])}.")

    # --- types
    dates = pd.to_datetime(df[C.DATE_COL], errors="coerce")
    bad_dates = dates.isna() & df[C.DATE_COL].notna()
    if bad_dates.any():
        res.errors.append(
            f"'{C.DATE_COL}' has {int(bad_dates.sum())} value(s) that are not valid dates at row(s) "
            f"{_rows_hint(df.index[bad_dates])}. Use YYYY-MM-DD."
        )
    df[C.DATE_COL] = dates.dt.normalize()
    if (dates.dropna() != dates.dropna().dt.normalize()).any():
        res.preprocessing.append("Time-of-day removed from 'Date' (daily data expected).")

    for col in (C.STORE_COL, C.PRODUCT_COL):
        df[col] = df[col].astype(str).str.strip()

    for col, kind in C.REQUIRED_COLUMNS.items():
        if kind != "number":
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        bad = num.isna() & df[col].notna() & (df[col].astype(str).str.strip() != "")
        if bad.any():
            res.errors.append(f"'{col}' has {int(bad.sum())} non-numeric value(s) at row(s) {_rows_hint(df.index[bad])}.")
        neg = num < 0
        if neg.any():
            res.errors.append(f"'{col}' has {int(neg.sum())} negative value(s) at row(s) {_rows_hint(df.index[neg])}.")
        df[col] = num
    over = df[C.DISCOUNT_COL] > 100
    if over.any():
        res.errors.append(f"'{C.DISCOUNT_COL}' must be a percentage 0-100; {int(over.sum())} value(s) exceed 100 at row(s) {_rows_hint(df.index[over])}.")

    # --- duplicate keys
    key = [C.DATE_COL, C.STORE_COL, C.PRODUCT_COL]
    dup = df.duplicated(subset=key, keep=False) & df[C.DATE_COL].notna()
    exact_dups = int(raw.duplicated().sum())
    if dup.any():
        res.errors.append(
            f"{int(dup.sum())} row(s) share the same Date + Store ID + Product ID "
            f"(each store-product needs one row per day) at row(s) {_rows_hint(df.index[dup])}."
        )

    # optional columns: report missing values but they are not used by models
    opt_missing = {c: int(df[c].isna().sum()) for c in present_optional if df[c].isna().any()}
    if opt_missing:
        res.warnings.append(f"Optional columns with empty values (not used by the models): {opt_missing}")

    res.summary = {
        "rows": int(len(df)),
        "columns": list(map(str, raw.columns)),
        "dtypes": {str(c): str(t) for c, t in raw.dtypes.items()},
        "missing_by_column": {str(c): int(v) for c, v in raw.isna().sum().items()},
        "exact_duplicate_rows": exact_dups,
        "duplicate_key_rows": int(dup.sum()),
        "optional_columns_present": present_optional,
    }
    if res.errors:
        return res

    # --- continuity / sufficiency per store-product series
    df = df.sort_values([C.STORE_COL, C.PRODUCT_COL, C.DATE_COL]).reset_index(drop=True)
    rows = []
    for (s, p), g in df.groupby([C.STORE_COL, C.PRODUCT_COL], sort=True):
        d = g[C.DATE_COL]
        last = d.max()
        span = (last - d.min()).days + 1
        dset = set(d)
        last7 = all((last - pd.Timedelta(days=i)) in dset for i in range(C.MIN_HISTORY_DAYS))
        last28 = all((last - pd.Timedelta(days=i)) in dset for i in range(C.SEGMENT_WINDOW_DAYS))
        rows.append({
            "Store ID": s, "Product ID": p, "First date": d.min().date(), "Last date": last.date(),
            "Days": len(d), "Missing days": span - len(d),
            "Forecastable": last7, "Segmentable": last28,
        })
    series = pd.DataFrame(rows)
    res.series_table = series
    n_fc = int(series["Forecastable"].sum())

    gaps = series[series["Missing days"] > 0]
    if len(gaps):
        res.warnings.append(
            f"{len(gaps)} store-product series have missing days. Missing days are NOT filled; "
            "features that would need them are skipped."
        )
    not_fc = series[~series["Forecastable"]]
    if len(not_fc):
        res.warnings.append(
            f"{len(not_fc)} series lack the {C.MIN_HISTORY_DAYS} consecutive most-recent days needed "
            "for lag-7 / rolling-7 features and will not be forecast."
        )
    not_seg = series[~series["Segmentable"]]
    if len(not_seg):
        res.warnings.append(
            f"{len(not_seg)} series have fewer than {C.SEGMENT_WINDOW_DAYS} consecutive recent days, "
            "so they cannot be assigned a segment."
        )
    if known_stores is not None and known_products is not None:
        new_s = sorted(set(df[C.STORE_COL]) - set(known_stores))
        new_p = sorted(set(df[C.PRODUCT_COL]) - set(known_products))
        if new_s or new_p:
            res.warnings.append(
                "IDs not seen in training (store: " + (", ".join(new_s[:5]) or "none")
                + "; product: " + (", ".join(new_p[:5]) or "none")
                + "). They are encoded as 'unknown', so forecasts rely on sales-history and calendar features only."
            )
    if n_fc == 0:
        res.errors.append(
            f"No store-product series has {C.MIN_HISTORY_DAYS} consecutive days of history ending on its "
            "latest date, so a next-day forecast cannot be made reliably. Add more daily history."
        )

    res.summary.update({
        "date_start": str(df[C.DATE_COL].min().date()),
        "date_end": str(df[C.DATE_COL].max().date()),
        "unique_dates": int(df[C.DATE_COL].nunique()),
        "stores": int(df[C.STORE_COL].nunique()),
        "products": int(df[C.PRODUCT_COL].nunique()),
        "series": int(len(series)),
        "forecastable_series": n_fc,
        "segmentable_series": int(series["Segmentable"].sum()),
        "series_with_gaps": int(len(gaps)),
    })
    res.preprocessing.append(
        f"Parsed dates, converted {', '.join(c for c, k in C.REQUIRED_COLUMNS.items() if k == 'number')} to numbers, "
        "trimmed ID text, sorted by store, product and date."
    )
    res.preprocessing.append("No values were imputed or invented.")
    keep = list(C.REQUIRED_COLUMNS) + present_optional
    res.clean = df[keep]
    res.ok = not res.errors
    return res


# ==========================================================================
# 3. Feature engineering for NEXT-DAY forecasting
#
# For each store-product and each cutoff day t we predict Units Sold on t+1.
#   lag_1        = units sold on t       (1 day before the target)
#   lag_7        = units sold on t-6     (7 days before the target)
#   roll_mean_7  = mean of units sold t-6 .. t   (only days known at cutoff)
#   roll_std_7   = std  of units sold t-6 .. t
#   target_dow / target_month = calendar of the target day t+1 (always known)
# Missing calendar days stay NaN (never filled), so any window touching a
# gap is dropped instead of using fabricated history.
# ==========================================================================
CATEGORICAL_FEATURES = [C.STORE_COL, C.PRODUCT_COL, "target_dow", "target_month"]
NUMERIC_FEATURES = ["lag_1", "lag_7", "roll_mean_7", "roll_std_7"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (series, cutoff day) with leak-free features.

    Column `target` holds the actual next-day sales when observed; rows with
    `is_future=True` are the next-day forecasts beyond the last date in the data.
    """
    data = df[[C.DATE_COL, C.STORE_COL, C.PRODUCT_COL, C.SALES_COL]].copy()
    data[C.DATE_COL] = pd.to_datetime(data[C.DATE_COL]).dt.normalize()
    parts = []
    for (store, product), g in data.groupby([C.STORE_COL, C.PRODUCT_COL], sort=True):
        g = g.set_index(C.DATE_COL).sort_index()
        days = pd.date_range(g.index.min(), g.index.max(), freq="D")
        sales = g[C.SALES_COL].astype(float).reindex(days)  # gaps -> NaN (not filled)
        f = pd.DataFrame({
            "cutoff_date": days,
            "lag_1": sales.values,
            "lag_7": sales.shift(C.LAGS[1] - 1).values,
            "roll_mean_7": sales.rolling(C.ROLL_WINDOW, min_periods=C.ROLL_WINDOW).mean().values,
            "roll_std_7": sales.rolling(C.ROLL_WINDOW, min_periods=C.ROLL_WINDOW).std().values,
            "target": sales.shift(-1).values,  # the label: sales on t+1
        })
        f[C.STORE_COL] = store
        f[C.PRODUCT_COL] = product
        f["is_future"] = f["cutoff_date"] == days[-1]
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["cutoff_date", "target_date", "target", "is_future"])
    feats = pd.concat(parts, ignore_index=True)
    feats = feats.dropna(subset=NUMERIC_FEATURES)
    # keep rows with an observed target (training/backtest) or the final future row
    feats = feats[feats["target"].notna() | feats["is_future"]].copy()
    feats["target_date"] = feats["cutoff_date"] + pd.Timedelta(days=1)
    feats["target_dow"] = feats["target_date"].dt.dayofweek.astype(int)
    feats["target_month"] = feats["target_date"].dt.month.astype(int)
    cols = [C.STORE_COL, C.PRODUCT_COL, "cutoff_date", "target_date", "target_dow", "target_month",
            *NUMERIC_FEATURES, "target", "is_future"]
    return feats[cols].reset_index(drop=True)


# ==========================================================================
# 4. Models, chronological split and metrics
# ==========================================================================

def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("numbers", StandardScaler(), NUMERIC_FEATURES),
    ])


def make_pipeline(kind: str) -> Pipeline:
    if kind == "Linear Regression":
        model = LinearRegression()
    elif kind == "Random Forest":
        model = RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


def chronological_split(feats: pd.DataFrame, test_fraction: float = C.TEST_FRACTION):
    """Split by calendar date: the final `test_fraction` of unique target dates is the test set.
    All rows sharing a date land in the same split."""
    dates = np.sort(feats["target_date"].unique())
    n_test = max(1, int(math.ceil(len(dates) * test_fraction)))
    split_date = pd.Timestamp(dates[-n_test])
    train = feats[feats["target_date"] < split_date].reset_index(drop=True)
    test = feats[feats["target_date"] >= split_date].reset_index(drop=True)
    return train, test, split_date


def date_grouped_folds(target_dates: pd.Series, n_splits: int = C.CV_SPLITS):
    """Expanding-window folds over UNIQUE dates (TimeSeriesSplit on dates, then mapped to rows).
    Validation dates are always later than training dates and never split across folds."""
    unique = np.sort(target_dates.unique())
    folds = []
    for tr_d, va_d in TimeSeriesSplit(n_splits=n_splits).split(unique):
        tr_idx = np.flatnonzero(target_dates.isin(unique[tr_d]).to_numpy())
        va_idx = np.flatnonzero(target_dates.isin(unique[va_d]).to_numpy())
        folds.append((tr_idx, va_idx))
    return folds


def predict_units(model, X: pd.DataFrame) -> np.ndarray:
    """Predictions are clipped at 0 because negative demand is impossible."""
    return np.clip(model.predict(X[FEATURE_COLUMNS]), 0, None)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


# ==========================================================================
# 5. Inventory risk (transparent rules - NOT a learned stockout probability)
# ==========================================================================

def classify_risk(coverage_days) -> str:
    if coverage_days is None or (isinstance(coverage_days, float) and math.isnan(coverage_days)):
        return "Low"  # ~zero recent demand: nothing is being consumed
    if coverage_days < C.RISK_HIGH_MAX_DAYS:
        return "High"
    if coverage_days < C.RISK_MEDIUM_MAX_DAYS:
        return "Medium"
    return "Low"


def inventory_risk(df: pd.DataFrame) -> pd.DataFrame:
    """One row per store-product using its latest recorded day.

    recent_avg_demand = mean Units Sold over the last RISK_RECENT_DAYS calendar days
    coverage_days     = current inventory / recent_avg_demand
    """
    rows = []
    for (s, p), g in df.groupby([C.STORE_COL, C.PRODUCT_COL], sort=True):
        g = g.sort_values(C.DATE_COL)
        last_date = g[C.DATE_COL].max()
        inv = float(g.loc[g[C.DATE_COL] == last_date, C.INVENTORY_COL].iloc[0])
        recent = g[g[C.DATE_COL] > last_date - pd.Timedelta(days=C.RISK_RECENT_DAYS)]
        avg = float(recent[C.SALES_COL].mean())
        if avg < C.MIN_DEMAND_PER_DAY:
            coverage = float("nan")
            note = "No meaningful recent demand"
        else:
            coverage = inv / avg
            note = ""
        risk = classify_risk(coverage)
        need = 0 if math.isnan(coverage) else max(0, math.ceil(C.TARGET_COVER_DAYS * avg - inv))
        if risk == "High":
            attention = "Replenish urgently (estimate)"
        elif risk == "Medium":
            attention = "Plan replenishment (estimate)"
        else:
            attention = note or "No action needed"
        rows.append({
            C.STORE_COL: s, C.PRODUCT_COL: p, "latest_date": last_date.date(),
            "current_inventory": inv, "recent_avg_demand": round(avg, 2),
            "days_used": int(len(recent)),
            "coverage_days": None if math.isnan(coverage) else round(coverage, 2),
            "risk_level": risk, "est_replenishment_units": int(need), "attention": attention,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Segmentation helpers
# ==========================================================================

def _profile(g: pd.DataFrame) -> dict:
    sold = g[C.SALES_COL].astype(float)
    inv = g[C.INVENTORY_COL].astype(float)
    return {
        "avg_units_sold": sold.mean(),
        "demand_std": sold.std(ddof=0),
        "avg_inventory": inv.mean(),
        "sell_through_rate": sold.sum() / inv.sum() if inv.sum() > 0 else 0.0,
        "avg_price": g[C.PRICE_COL].astype(float).mean(),
        "avg_discount": g[C.DISCOUNT_COL].astype(float).mean(),
    }


def window_profiles(df: pd.DataFrame, window: int = C.SEGMENT_WINDOW_DAYS) -> pd.DataFrame:
    """Training profiles: non-overlapping complete `window`-day blocks per series."""
    rows = []
    for (s, p), g in df.groupby([C.STORE_COL, C.PRODUCT_COL], sort=True):
        g = g.sort_values(C.DATE_COL).set_index(C.DATE_COL)
        start, end = g.index.min(), g.index.max()
        n_blocks = ((end - start).days + 1) // window
        for b in range(n_blocks):
            lo = start + pd.Timedelta(days=b * window)
            hi = lo + pd.Timedelta(days=window - 1)
            block = g.loc[lo:hi]
            if len(block) == window:  # only complete, gap-free blocks
                rows.append({C.STORE_COL: s, C.PRODUCT_COL: p, "window_start": lo.date(), **_profile(block)})
    return pd.DataFrame(rows)


def latest_profiles(df: pd.DataFrame, window: int = C.SEGMENT_WINDOW_DAYS) -> pd.DataFrame:
    """Analysis profiles: the most recent complete `window` days of each series."""
    rows = []
    for (s, p), g in df.groupby([C.STORE_COL, C.PRODUCT_COL], sort=True):
        last = g[C.DATE_COL].max()
        block = g[g[C.DATE_COL] > last - pd.Timedelta(days=window)]
        if len(block) == window:
            rows.append({C.STORE_COL: s, C.PRODUCT_COL: p, **_profile(block)})
    return pd.DataFrame(rows, columns=[C.STORE_COL, C.PRODUCT_COL, *C.SEGMENT_FEATURES])


FEATURE_WORDS = {
    "avg_units_sold": "sales", "demand_std": "demand variability", "avg_inventory": "stock",
    "sell_through_rate": "sell-through", "avg_price": "price", "avg_discount": "discount",
}


def describe_clusters(centers_scaled: np.ndarray) -> list[str]:
    """Readable labels from each centroid's two most distinctive features (z-scores).
    These describe measured averages; they are not ground-truth categories."""
    labels = []
    for row in centers_scaled:
        top = np.argsort(-np.abs(row))[:2]
        parts = [f"{'Higher' if row[i] > 0 else 'Lower'} {FEATURE_WORDS[C.SEGMENT_FEATURES[i]]}" for i in top]
        labels.append(", ".join(parts))
    return labels


def assign_segments(bundle: dict, profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return profiles.assign(cluster=pd.Series(dtype=int), pc1=pd.Series(dtype=float), pc2=pd.Series(dtype=float))
    Z = bundle["scaler"].transform(profiles[C.SEGMENT_FEATURES])
    out = profiles.copy()
    out["cluster"] = bundle["kmeans"].predict(Z).astype(int) + 1  # 1-based for display
    pcs = bundle["pca"].transform(Z)
    out["pc1"], out["pc2"] = pcs[:, 0], pcs[:, 1]
    out["segment_label"] = out["cluster"].map(lambda c: f"Segment {c}: {bundle['labels'][c - 1]}")
    return out


# ==========================================================================
# 7. Artifacts and the full analysis run
# ==========================================================================

def load_artifacts() -> dict:
    """Load saved models. Never trains. Returns a status dict the UI can display."""
    st = {"state": "ready", "problems": [], "warnings": [], "model": None, "segmentation": None, "metadata": None}
    missing = [p.name for p in (C.DEMAND_MODEL_PATH, C.SEGMENTATION_PATH, C.METADATA_PATH) if not p.exists()]
    if missing:
        st["state"] = "missing"
        st["problems"].append(f"Missing model file(s): {', '.join(missing)}. Run `{C.TRAIN_COMMAND}`.")
        return st
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            meta = json.loads(C.METADATA_PATH.read_text(encoding="utf-8"))
            model = joblib.load(C.DEMAND_MODEL_PATH)
            seg = joblib.load(C.SEGMENTATION_PATH)
        for w in caught:
            if "version" in str(w.message).lower():
                st["warnings"].append("Model files were saved with a different scikit-learn version. "
                                      f"If results look wrong, re-run `{C.TRAIN_COMMAND}`.")
                break
    except Exception as exc:
        get_logger().exception("Artifact load failed")
        st["state"] = "corrupt"
        st["problems"].append(
            f"Model files could not be loaded ({type(exc).__name__}). They may be corrupted or from an "
            f"incompatible library version. Run `{C.TRAIN_COMMAND}` to rebuild them."
        )
        return st
    if meta.get("feature_columns") != FEATURE_COLUMNS or not hasattr(model, "predict") \
            or not {"scaler", "kmeans", "pca", "labels"} <= set(seg):
        st["state"] = "incompatible"
        st["problems"].append(f"Model files do not match this version of RetailIQ. Run `{C.TRAIN_COMMAND}`.")
        return st
    if meta.get("sklearn_version") != sklearn.__version__ and not st["warnings"]:
        st["warnings"].append(
            f"Models trained with scikit-learn {meta.get('sklearn_version')}, running {sklearn.__version__}."
        )
    st.update(model=model, segmentation=seg, metadata=meta)
    return st


def run_analysis(clean: pd.DataFrame, artifacts: dict, progress=None) -> dict:
    """Forecast + risk + segmentation for a validated dataset. Never retrains."""
    def step(pct, msg):
        if progress:
            progress(pct, msg)

    model, seg, meta = artifacts["model"], artifacts["segmentation"], artifacts["metadata"]
    df = clean.copy()
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])

    step(10, "Building lag and rolling features")
    feats = build_features(df)
    future = feats[feats["is_future"]].copy()
    hist = feats[~feats["is_future"]].copy()
    if future.empty:
        raise UserFacingError("No series has enough consecutive history to forecast the next day.")

    step(35, "Forecasting next-day demand")
    future["predicted_units"] = predict_units(model, future)
    forecasts = future.rename(columns={"cutoff_date": "latest_date", "target_date": "forecast_date"})[
        [C.STORE_COL, C.PRODUCT_COL, "latest_date", "forecast_date", "lag_1", "roll_mean_7", "predicted_units"]
    ].reset_index(drop=True)

    backtest = pd.DataFrame(columns=[C.STORE_COL, C.PRODUCT_COL, "target_date", "actual_units", "predicted_units"])
    bt_metrics = None
    if not hist.empty:
        hist["predicted_units"] = predict_units(model, hist)
        backtest = hist.rename(columns={"target": "actual_units"})[
            [C.STORE_COL, C.PRODUCT_COL, "target_date", "actual_units", "predicted_units"]
        ].reset_index(drop=True)
        bt_metrics = regression_metrics(backtest["actual_units"], backtest["predicted_units"])

    step(60, "Calculating inventory risk")
    risk = inventory_risk(df)
    risk = risk.merge(forecasts[[C.STORE_COL, C.PRODUCT_COL, "predicted_units"]], how="left",
                      on=[C.STORE_COL, C.PRODUCT_COL]).rename(columns={"predicted_units": "next_day_forecast"})

    step(80, "Assigning product segments")
    seg_df = assign_segments(seg, latest_profiles(df))
    insights = risk.merge(
        seg_df[[C.STORE_COL, C.PRODUCT_COL, "cluster", "segment_label", "pc1", "pc2", *C.SEGMENT_FEATURES]]
        if not seg_df.empty else pd.DataFrame(columns=[C.STORE_COL, C.PRODUCT_COL, "cluster", "segment_label", "pc1", "pc2"]),
        how="left", on=[C.STORE_COL, C.PRODUCT_COL],
    )

    step(92, "Summarising")
    daily = df.groupby(C.DATE_COL).agg(units_sold=(C.SALES_COL, "sum"), inventory=(C.INVENTORY_COL, "sum"),
                                       avg_price=(C.PRICE_COL, "mean")).reset_index()
    daily[C.DATE_COL] = daily[C.DATE_COL].dt.strftime("%Y-%m-%d")
    last_date = df[C.DATE_COL].max()
    recent = df[df[C.DATE_COL] > last_date - pd.Timedelta(days=7)][C.SALES_COL].sum()
    prior = df[(df[C.DATE_COL] <= last_date - pd.Timedelta(days=7)) &
               (df[C.DATE_COL] > last_date - pd.Timedelta(days=14))][C.SALES_COL].sum()
    train_end = pd.Timestamp(meta["training"]["train_target_end"])
    overlap = bool((backtest["target_date"] <= train_end).any()) if len(backtest) else False
    unseen = sorted(set(df[C.STORE_COL]) - set(meta["known_stores"])) + \
        sorted(set(df[C.PRODUCT_COL]) - set(meta["known_products"]))
    summary = {
        "records": int(len(df)),
        "stores": int(df[C.STORE_COL].nunique()),
        "products": int(df[C.PRODUCT_COL].nunique()),
        "series": int(df.groupby([C.STORE_COL, C.PRODUCT_COL]).ngroups),
        "date_start": str(df[C.DATE_COL].min().date()),
        "date_end": str(last_date.date()),
        "total_units_sold": float(df[C.SALES_COL].sum()),
        "avg_units_per_series_day": float(df[C.SALES_COL].mean()),
        "current_inventory_total": float(risk["current_inventory"].sum()),
        "units_last7": float(recent),
        "units_prev7": float(prior) if prior > 0 else None,
        "forecast_date": str(forecasts["forecast_date"].max().date()),
        "forecast_series": int(len(forecasts)),
        "segmented_series": int(seg_df["cluster"].notna().sum()) if not seg_df.empty else 0,
        "risk_counts": {k: int((risk["risk_level"] == k).sum()) for k in ("Low", "Medium", "High")},
        "backtest_metrics": bt_metrics,
        "backtest_overlaps_training": overlap,
        "unseen_ids": unseen,
    }
    step(100, "Done")
    return {"summary": summary, "forecasts": forecasts, "backtest": backtest,
            "insights": insights, "daily": daily}
