"""
RetailIQ offline training script.

    python train.py

1. Inspects the raw dataset and prints a verification report.
2. Builds leak-free next-day features and splits chronologically by date.
3. Tunes Random Forest with date-grouped time-series folds, compares it to
   Linear Regression, and selects the model with the lower validation MAE.
4. Evaluates both on the untouched final 20% of dates (holdout test).
5. Fits K-Means + PCA on 28-day store-product behaviour profiles.
6. Writes demo data + upload template, then saves all artifacts ATOMICALLY:
   files are written to a staging folder first and only moved into models/
   after everything succeeded, so a failed run never breaks the last good models.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler

import config as C
import ml_pipeline as ml


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
def inspect_raw(raw: pd.DataFrame) -> dict:
    """Verify the real file instead of trusting descriptions of it."""
    df = raw.copy()
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    key = [C.STORE_COL, C.PRODUCT_COL]
    per_key = df.groupby(key + [C.DATE_COL]).size()
    gaps = df.sort_values(key + [C.DATE_COL]).groupby(key)[C.DATE_COL].diff().dt.days.dropna()
    stable = {c: int(df.groupby(C.PRODUCT_COL)[c].nunique().max()) for c in ("Category", "Region") if c in df}
    report = {
        "shape": list(df.shape),
        "columns": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "date_start": str(df[C.DATE_COL].min().date()),
        "date_end": str(df[C.DATE_COL].max().date()),
        "unique_dates": int(df[C.DATE_COL].nunique()),
        "stores": sorted(df[C.STORE_COL].unique().tolist()),
        "products": sorted(df[C.PRODUCT_COL].unique().tolist()),
        "missing_values_total": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "max_rows_per_store_product_date": int(per_key.max()),
        "series_count": int(df.groupby(key).ngroups),
        "all_series_daily_continuous": bool((gaps == 1).all()),
        "corr_demand_forecast_vs_units_sold": float(df["Demand Forecast"].corr(df[C.SALES_COL]))
        if "Demand Forecast" in df else None,
        "explicit_stockout_label_present": any("stockout" in c.lower() for c in raw.columns),
        "max_distinct_values_per_product": stable,
        "units_sold_never_exceeds_inventory": bool((df[C.SALES_COL] <= df[C.INVENTORY_COL]).all()),
        "category_values": {c: sorted(map(str, df[c].unique())) for c in
                            ("Category", "Region", "Weather Condition", "Seasonality") if c in df},
        "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())]
                           for c in raw.select_dtypes("number").columns},
    }
    return report


def naive_baselines(test: pd.DataFrame) -> dict:
    """Simple reference forecasts so model scores can be judged honestly."""
    y = test["target"]
    return {
        "Naive: yesterday's sales (lag 1)": ml.regression_metrics(y, test["lag_1"]),
        "Naive: 7-day rolling mean": ml.regression_metrics(y, test["roll_mean_7"]),
    }


def grouped_importance(rf_pipeline) -> dict:
    """Random Forest impurity importance, summed back to the original input columns."""
    pre = rf_pipeline.named_steps["preprocess"]
    names = pre.get_feature_names_out()
    imp = rf_pipeline.named_steps["model"].feature_importances_
    out = {}
    for name, value in zip(names, imp):
        block, col = name.split("__", 1)
        base = next((c for c in ml.CATEGORICAL_FEATURES if col.startswith(c + "_")), col) \
            if block == "categories" else col
        out[base] = out.get(base, 0.0) + float(value)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------
def train_demand(feats: pd.DataFrame):
    hist = feats[~feats["is_future"]].reset_index(drop=True)
    train, test, split_date = ml.chronological_split(hist)
    assert train["target_date"].max() < test["target_date"].min(), "train/test dates overlap!"
    log(f"Train rows {len(train):,} (target dates {train.target_date.min().date()} .. "
        f"{train.target_date.max().date()}); test rows {len(test):,} "
        f"({test.target_date.min().date()} .. {test.target_date.max().date()})")

    folds = ml.date_grouped_folds(train["target_date"])
    fold_info = [{"train_dates": f"{train.target_date.iloc[tr].min().date()} .. {train.target_date.iloc[tr].max().date()}",
                  "validation_dates": f"{train.target_date.iloc[va].min().date()} .. {train.target_date.iloc[va].max().date()}"}
                 for tr, va in folds]
    X_tr, y_tr = train[ml.FEATURE_COLUMNS], train["target"]
    X_te, y_te = test[ml.FEATURE_COLUMNS], test["target"]
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

    log("Cross-validating Linear Regression (date-grouped folds)")
    lr = ml.make_pipeline("Linear Regression")
    lr_cv = cross_validate(lr, X_tr, y_tr, cv=folds, scoring=scoring)
    lr_cv_summary = {"MAE": float(-lr_cv["test_MAE"].mean()), "RMSE": float(-lr_cv["test_RMSE"].mean()),
                     "R2": float(lr_cv["test_R2"].mean())}
    lr.fit(X_tr, y_tr)

    log(f"Grid-searching Random Forest ({len(folds)} date-grouped folds)")
    t0 = time.time()
    grid = GridSearchCV(ml.make_pipeline("Random Forest"), C.RF_PARAM_GRID, cv=folds,
                        scoring=scoring, refit="MAE", n_jobs=1)
    grid.fit(X_tr, y_tr)
    best = grid.best_index_
    rf_cv_summary = {"MAE": float(-grid.cv_results_["mean_test_MAE"][best]),
                     "RMSE": float(-grid.cv_results_["mean_test_RMSE"][best]),
                     "R2": float(grid.cv_results_["mean_test_R2"][best])}
    rf = grid.best_estimator_
    log(f"  done in {time.time() - t0:.0f}s, best params {grid.best_params_}")
    grid_table = [{"params": {k.replace("model__", ""): v for k, v in p.items()},
                   "cv_MAE": float(-m)} for p, m in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_MAE"])]

    # Selection uses VALIDATION scores only (the test set is never used for choosing)
    selected = "Random Forest" if rf_cv_summary["MAE"] < lr_cv_summary["MAE"] else "Linear Regression"
    models = {"Linear Regression": lr, "Random Forest": rf}
    test_metrics = {name: ml.regression_metrics(y_te, ml.predict_units(m, X_te)) for name, m in models.items()}
    log(f"Selected by validation MAE: {selected}")
    for name, m in test_metrics.items():
        log(f"  holdout {name}: MAE {m['MAE']:.2f}  RMSE {m['RMSE']:.2f}  R2 {m['R2']:.4f}")

    chosen = models[selected]
    sample = test.sample(min(6000, len(test)), random_state=C.RANDOM_STATE)
    perm = permutation_importance(chosen, sample[ml.FEATURE_COLUMNS], sample["target"],
                                  scoring="neg_mean_absolute_error", n_repeats=5,
                                  random_state=C.RANDOM_STATE, n_jobs=1)
    perm_imp = {c: {"mean": float(m), "std": float(s)} for c, m, s in
                zip(ml.FEATURE_COLUMNS, perm.importances_mean, perm.importances_std)}
    perm_imp = dict(sorted(perm_imp.items(), key=lambda kv: -kv[1]["mean"]))

    test = test.assign(pred=ml.predict_units(chosen, X_te))
    daily = test.groupby("target_date").agg(actual=("target", "sum"), predicted=("pred", "sum")).reset_index()
    daily["target_date"] = daily["target_date"].dt.strftime("%Y-%m-%d")

    info = {
        "selected_model": selected,
        "selection_rule": "lowest mean validation MAE across date-grouped expanding-window folds (test set not used)",
        "cv_folds": fold_info,
        "cv_scores": {"Linear Regression": lr_cv_summary, "Random Forest": rf_cv_summary},
        "rf_best_params": {k.replace("model__", ""): v for k, v in grid.best_params_.items()},
        "rf_grid": grid_table,
        "test_metrics": test_metrics,
        "naive_baselines": naive_baselines(test),
        "permutation_importance_selected": perm_imp,
        "rf_impurity_importance": grouped_importance(rf),
        "holdout_daily_totals": daily.to_dict(orient="list"),
        "training": {
            "train_target_start": str(train.target_date.min().date()),
            "train_target_end": str(train.target_date.max().date()),
            "test_target_start": str(test.target_date.min().date()),
            "test_target_end": str(test.target_date.max().date()),
            "split_date": str(split_date.date()),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "test_fraction_of_dates": C.TEST_FRACTION,
            "deployed_model_trained_on": "training period only (so holdout scores describe the deployed model)",
        },
    }
    return chosen, info


# --------------------------------------------------------------------------
def train_segmentation(raw: pd.DataFrame):
    prof = ml.window_profiles(raw)
    log(f"Segmentation profiles: {len(prof):,} complete {C.SEGMENT_WINDOW_DAYS}-day store-product windows")
    scaler = StandardScaler().fit(prof[C.SEGMENT_FEATURES])
    Z = scaler.transform(prof[C.SEGMENT_FEATURES])
    inertia, silhouette = {}, {}
    for k in C.K_RANGE_EVALUATED:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
        inertia[k] = float(km.inertia_)
        silhouette[k] = float(silhouette_score(Z, km.labels_))
    best_k = max(C.K_RANGE_ALLOWED, key=lambda k: silhouette[k])
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
    pca = PCA(n_components=2, random_state=C.RANDOM_STATE).fit(Z)
    labels = ml.describe_clusters(kmeans.cluster_centers_)
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=C.SEGMENT_FEATURES)
    centers.insert(0, "cluster", range(1, best_k + 1))
    centers["size_in_training"] = np.bincount(kmeans.labels_, minlength=best_k)
    pcs = pca.transform(Z)
    log(f"  k chosen = {best_k} (silhouette {silhouette[best_k]:.3f}); PCA variance "
        f"{pca.explained_variance_ratio_.sum():.1%}")
    bundle = {
        "scaler": scaler, "kmeans": kmeans, "pca": pca, "labels": labels,
        "features": C.SEGMENT_FEATURES, "window_days": C.SEGMENT_WINDOW_DAYS,
        "k": best_k, "inertia": inertia, "silhouette": silhouette,
        "centers": centers.round(3).to_dict(orient="records"),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_loadings": pd.DataFrame(pca.components_.T, index=C.SEGMENT_FEATURES, columns=["PC1", "PC2"]).round(3).to_dict(),
        "training_points": {"pc1": pcs[:, 0].round(3).tolist(), "pc2": pcs[:, 1].round(3).tolist(),
                            "cluster": (kmeans.labels_ + 1).tolist()},
    }
    summary = {k: v for k, v in bundle.items() if k in ("k", "inertia", "silhouette", "labels", "centers",
                                                        "pca_explained_variance", "window_days", "features")}
    summary["profiles"] = int(len(prof))
    return bundle, summary


# --------------------------------------------------------------------------
def make_demo_and_template(raw: pd.DataFrame, test_start: str) -> dict:
    """Demo = last 28 days of 10 store-product series (5 stores x 2 products).
    28 days are needed: 7 for lag/rolling features and 28 for a segmentation profile.
    These dates fall inside the holdout test period, so demo comparisons are out-of-sample."""
    df = raw.copy()
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    stores = sorted(df[C.STORE_COL].unique())[:5]
    products = sorted(df[C.PRODUCT_COL].unique())
    pairs = [(s, products[(2 * i + j) % len(products)]) for i, s in enumerate(stores) for j in range(2)]
    last = df[C.DATE_COL].max()
    window = df[df[C.DATE_COL] > last - pd.Timedelta(days=C.SEGMENT_WINDOW_DAYS)]
    demo = pd.concat([window[(window[C.STORE_COL] == s) & (window[C.PRODUCT_COL] == p)] for s, p in pairs])
    demo = demo.sort_values([C.DATE_COL, C.STORE_COL, C.PRODUCT_COL])
    demo[C.DATE_COL] = demo[C.DATE_COL].dt.strftime("%Y-%m-%d")
    C.DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(C.DEMO_DATA_PATH, index=False)

    # Template: required columns only, 8 consecutive days for one series (valid for forecasting)
    s, p = pairs[0]
    tpl = raw[(raw[C.STORE_COL] == s) & (raw[C.PRODUCT_COL] == p)].head(8)[list(C.REQUIRED_COLUMNS)]
    C.TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tpl.to_csv(C.TEMPLATE_PATH, index=False)
    return {"demo_rows": int(len(demo)), "demo_series": len(pairs),
            "demo_date_start": demo[C.DATE_COL].min(), "demo_date_end": demo[C.DATE_COL].max(),
            "demo_out_of_sample": demo[C.DATE_COL].min() >= test_start}


# --------------------------------------------------------------------------
def main() -> int:
    if not C.RAW_DATA_PATH.exists():
        print(f"ERROR: dataset not found at {C.RAW_DATA_PATH}\n"
              "Place retail_store_inventory.csv in data/raw/ (see README).")
        return 1
    started = time.time()
    log(f"Loading {C.RAW_DATA_PATH.name}")
    raw = pd.read_csv(C.RAW_DATA_PATH)
    report = inspect_raw(raw)
    log(f"Shape {report['shape']}, dates {report['date_start']} .. {report['date_end']}, "
        f"{len(report['stores'])} stores, {len(report['products'])} products, "
        f"missing {report['missing_values_total']}, duplicates {report['duplicate_rows']}")
    log(f"Daily continuous series: {report['all_series_daily_continuous']}; "
        f"corr(Demand Forecast, Units Sold) = {report['corr_demand_forecast_vs_units_sold']:.4f} -> excluded")
    log(f"Distinct Category/Region values per product: {report['max_distinct_values_per_product']} "
        "-> not stable product attributes, excluded from features")

    check = ml.validate_dataset(raw)
    if not check.ok:
        print("ERROR: the raw dataset failed validation:\n - " + "\n - ".join(check.errors))
        return 1
    clean = check.clean.copy()
    clean[C.DATE_COL] = pd.to_datetime(clean[C.DATE_COL])

    feats = ml.build_features(clean)
    log(f"Feature rows: {len(feats):,} (features: {', '.join(ml.FEATURE_COLUMNS)})")
    model, demand_info = train_demand(feats)
    seg_bundle, seg_info = train_segmentation(clean)
    demo_info = make_demo_and_template(raw, demand_info["training"]["test_target_start"])

    version = f"RIQ-{datetime.now():%Y%m%d-%H%M}"
    metadata = {
        "model_version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "dataset_file": C.RAW_DATA_PATH.name,
        "data_inspection": report,
        "target": "Units Sold on the next calendar day (t+1) for each Store ID + Product ID",
        "feature_columns": ml.FEATURE_COLUMNS,
        "categorical_features": ml.CATEGORICAL_FEATURES,
        "numeric_features": ml.NUMERIC_FEATURES,
        "excluded_columns": {
            "Demand Forecast": "~0.997 correlated with Units Sold: target leakage",
            "Price, Discount, Holiday/Promotion, Weather Condition, Competitor Pricing, Units Ordered, Inventory Level":
                "time-varying; their target-day values are not known at the forecast cutoff",
            "Category, Region, Seasonality": "change from day to day for the same product in this synthetic file, so they are not stable identifiers",
        },
        "prediction_clipping": "negative predictions are clipped to 0",
        "known_stores": report["stores"], "known_products": report["products"],
        **demand_info,
        "segmentation": seg_info,
        "demo": demo_info,
        "risk_rules": {"recent_days": C.RISK_RECENT_DAYS, "high_below_days": C.RISK_HIGH_MAX_DAYS,
                       "medium_below_days": C.RISK_MEDIUM_MAX_DAYS, "min_demand_per_day": C.MIN_DEMAND_PER_DAY,
                       "target_cover_days": C.TARGET_COVER_DAYS},
        "training_seconds": None,
    }

    # ---- atomic save: staging folder first, then replace the live files
    staging = C.MODELS_DIR / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    joblib.dump(model, staging / C.DEMAND_MODEL_PATH.name)
    joblib.dump(seg_bundle, staging / C.SEGMENTATION_PATH.name)
    joblib.load(staging / C.DEMAND_MODEL_PATH.name)  # verify the files load before publishing
    joblib.load(staging / C.SEGMENTATION_PATH.name)
    metadata["training_seconds"] = round(time.time() - started, 1)
    (staging / C.METADATA_PATH.name).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    for name in (C.DEMAND_MODEL_PATH.name, C.SEGMENTATION_PATH.name, C.METADATA_PATH.name):
        os.replace(staging / name, C.MODELS_DIR / name)
    shutil.rmtree(staging, ignore_errors=True)
    log(f"Saved artifacts to {C.MODELS_DIR} (version {version}) in {metadata['training_seconds']}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # keep the last good models; report clearly
        shutil.rmtree(C.MODELS_DIR / "_staging", ignore_errors=True)
        print(f"\nTRAINING FAILED: {type(exc).__name__}: {exc}\n"
              "Previously saved models (if any) were left unchanged.")
        raise
