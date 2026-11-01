"""Unit tests: loading/validation, features, leakage, splitting, models, risk, segmentation."""
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import config as C
import ml_pipeline as ml
from conftest import make_series


# ---------------------------------------------------------------- loading & schema
def test_raw_dataset_schema_matches_documentation(raw_df):
    assert raw_df.shape == (73100, 15)
    for col in list(C.REQUIRED_COLUMNS) + C.OPTIONAL_COLUMNS:
        assert col in raw_df.columns
    v = ml.validate_dataset(raw_df)
    assert v.ok, v.errors
    assert v.summary["stores"] == 5 and v.summary["products"] == 20
    assert v.summary["date_start"] == "2022-01-01" and v.summary["date_end"] == "2024-01-01"
    assert v.summary["series_with_gaps"] == 0


def test_read_table_rejects_bad_files():
    with pytest.raises(ml.UserFacingError, match="Unsupported file type"):
        ml.read_table(b"a,b\n1,2", "data.txt")
    with pytest.raises(ml.UserFacingError, match="empty"):
        ml.read_table(b"", "data.csv")
    with pytest.raises(ml.UserFacingError, match="Could not read"):
        ml.read_table(b"\x00\x01not-a-zip" * 50, "data.xlsx")


def test_read_table_reads_excel():
    buf = io.BytesIO()
    make_series(10).to_excel(buf, index=False)
    df = ml.read_table(buf.getvalue(), "x.xlsx")
    assert len(df) == 10 and ml.validate_dataset(df).ok


def test_validation_missing_column_and_header_normalisation():
    df = make_series(10).drop(columns=["Units Sold"])
    v = ml.validate_dataset(df)
    assert not v.ok and "Units Sold" in v.errors[0]
    df2 = make_series(10).rename(columns={"Units Sold": "  units sold "})
    v2 = ml.validate_dataset(df2)
    assert v2.ok and "Units Sold" in v2.clean.columns
    assert any("Matched column names" in p for p in v2.preprocessing)


def test_validation_reports_exact_bad_rows():
    df = make_series(10).astype({"Units Sold": object, "Date": object})
    df.loc[2, "Units Sold"] = "abc"
    df.loc[4, "Price"] = -1
    df.loc[6, "Date"] = "31/31/2024"
    df.loc[7, "Inventory Level"] = None
    v = ml.validate_dataset(df)
    text = " | ".join(v.errors)
    assert not v.ok
    assert "non-numeric" in text and "row(s) 4" in text          # index 2 -> spreadsheet row 4
    assert "negative" in text and "row(s) 6" in text
    assert "not valid dates" in text and "row(s) 8" in text
    assert "'Inventory Level' has 1 empty" in text


def test_validation_duplicates_and_insufficient_history():
    df = make_series(10)
    v = ml.validate_dataset(pd.concat([df, df.iloc[[3]]]))
    assert not v.ok and any("share the same Date" in e for e in v.errors)
    short = ml.validate_dataset(make_series(5))
    assert not short.ok and any("cannot be made reliably" in e for e in short.errors)


def test_validation_gaps_and_unseen_ids():
    df = make_series(30)
    df = df.drop(index=[10])  # one missing day in the middle
    new = make_series(10, store="S999", product="P9999")
    v = ml.validate_dataset(pd.concat([df, new]), known_stores=["S001"], known_products=["P0001"])
    assert v.ok
    assert any("missing days" in w for w in v.warnings)
    assert any("S999" in w for w in v.warnings)
    st = v.series_table.set_index("Store ID")
    assert bool(st.loc["S001", "Forecastable"]) and not bool(st.loc["S999", "Segmentable"])


# ---------------------------------------------------------------- features & leakage
def test_features_lags_and_rolling_are_correct():
    df = make_series(12)  # sales = 1..12 on consecutive days
    f = ml.build_features(df)
    hist = f[~f["is_future"]]
    row = hist[hist["cutoff_date"] == pd.Timestamp("2024-01-08")].iloc[0]  # sales that day = 8
    assert row["lag_1"] == 8                       # sales on the cutoff day (target - 1)
    assert row["lag_7"] == 2                       # sales on target - 7 = cutoff - 6
    assert row["roll_mean_7"] == pytest.approx(np.mean([2, 3, 4, 5, 6, 7, 8]))
    assert row["target"] == 9                      # next-day sales
    assert row["target_date"] == pd.Timestamp("2024-01-09")
    assert row["target_dow"] == pd.Timestamp("2024-01-09").dayofweek
    first = hist["cutoff_date"].min()
    assert first == pd.Timestamp("2024-01-07")     # needs 7 observed days


def test_features_independent_of_input_order():
    df = pd.concat([make_series(15), make_series(15, store="S002", sales=list(range(50, 65)))])
    a = ml.build_features(df)
    b = ml.build_features(df.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(a, b)


def test_no_future_information_in_features():
    df = make_series(20)
    base = ml.build_features(df)
    cutoff = pd.Timestamp("2024-01-10")
    changed = df.copy()
    changed.loc[pd.to_datetime(changed["Date"]) > cutoff, "Units Sold"] = 9999  # target day and later
    after = ml.build_features(changed)
    cols = ml.NUMERIC_FEATURES + ["target_dow", "target_month"]
    r0 = base[base["cutoff_date"] == cutoff][cols].reset_index(drop=True)
    r1 = after[after["cutoff_date"] == cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(r0, r1)
    assert after[after["cutoff_date"] == cutoff]["target"].iloc[0] == 9999


def test_gaps_are_not_filled():
    df = make_series(20)
    df = df[df["Date"] != "2024-01-10"]
    f = ml.build_features(df)
    gap = pd.Timestamp("2024-01-10")
    hist = f[~f["is_future"]]
    # no row may use the missing day as a feature (window t-6..t) or as its target
    for _, r in hist.iterrows():
        assert not (r["cutoff_date"] - pd.Timedelta(days=6) <= gap <= r["cutoff_date"])
        assert r["target_date"] != gap


def test_future_row_is_next_calendar_day():
    f = ml.build_features(make_series(10))
    fut = f[f["is_future"]]
    assert len(fut) == 1 and fut["target_date"].iloc[0] == pd.Timestamp("2024-01-11")
    assert pd.isna(fut["target"].iloc[0])


def test_demand_forecast_never_a_feature():
    assert "Demand Forecast" not in ml.FEATURE_COLUMNS
    meta = json.loads(C.METADATA_PATH.read_text())
    assert "Demand Forecast" not in meta["feature_columns"]
    assert meta["data_inspection"]["corr_demand_forecast_vs_units_sold"] > 0.99


# ---------------------------------------------------------------- splitting
def test_chronological_split_by_date(raw_df):
    feats = ml.build_features(raw_df.head(5000))
    hist = feats[~feats["is_future"]]
    train, test, split = ml.chronological_split(hist)
    assert train["target_date"].max() < test["target_date"].min() == split
    assert set(train["target_date"]).isdisjoint(set(test["target_date"]))
    n_dates = hist["target_date"].nunique()
    assert test["target_date"].nunique() == int(np.ceil(n_dates * C.TEST_FRACTION))


def test_date_grouped_folds_keep_time_order():
    dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=40), 3))
    for tr, va in ml.date_grouped_folds(dates, n_splits=3):
        assert dates.iloc[tr].max() < dates.iloc[va].min()
        assert set(dates.iloc[tr]).isdisjoint(set(dates.iloc[va]))


def test_saved_metadata_split_has_no_overlap():
    meta = json.loads(C.METADATA_PATH.read_text())
    tr = meta["training"]
    assert tr["train_target_end"] < tr["test_target_start"]
    assert meta["selected_model"] in ("Linear Regression", "Random Forest")
    cv = meta["cv_scores"]
    best_cv = min(cv, key=lambda k: cv[k]["MAE"])
    assert meta["selected_model"] == best_cv   # chosen by validation, not test


# ---------------------------------------------------------------- artifacts & prediction
def test_artifacts_load_and_predict_non_negative(demo_df):
    arts = ml.load_artifacts()
    assert arts["state"] == "ready", arts["problems"]
    feats = ml.build_features(demo_df)
    pred = ml.predict_units(arts["model"], feats)
    assert len(pred) == len(feats) and (pred >= 0).all()


def test_unseen_ids_do_not_crash():
    arts = ml.load_artifacts()
    f = ml.build_features(make_series(10, store="S_NEW", product="P_NEW"))
    assert np.isfinite(ml.predict_units(arts["model"], f)).all()


def test_missing_and_corrupt_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DEMAND_MODEL_PATH", tmp_path / "none.joblib")
    assert ml.load_artifacts()["state"] == "missing"
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"garbage")
    monkeypatch.setattr(C, "DEMAND_MODEL_PATH", bad)
    st = ml.load_artifacts()
    assert st["state"] == "corrupt" and "train.py" in st["problems"][0]


def test_artifact_roundtrip(tmp_path):
    model = ml.make_pipeline("Linear Regression")
    f = ml.build_features(pd.concat([make_series(30), make_series(30, store="S002")]))
    hist = f[~f["is_future"]]
    model.fit(hist[ml.FEATURE_COLUMNS], hist["target"])
    joblib.dump(model, tmp_path / "m.joblib")
    again = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(model.predict(hist[ml.FEATURE_COLUMNS]), again.predict(hist[ml.FEATURE_COLUMNS]))


def test_run_analysis_output_schema(demo_df):
    arts = ml.load_artifacts()
    v = ml.validate_dataset(demo_df)
    res = ml.run_analysis(v.clean, arts)
    f = res["forecasts"]
    assert list(f.columns) == ["Store ID", "Product ID", "latest_date", "forecast_date", "lag_1", "roll_mean_7", "predicted_units"]
    assert len(f) == 10 and (f["forecast_date"] == f["latest_date"] + pd.Timedelta(days=1)).all()
    assert res["summary"]["forecast_date"] == "2024-01-02"
    assert res["backtest"]["actual_units"].notna().all()
    assert res["summary"]["backtest_overlaps_training"] is False   # demo is out-of-sample
    assert set(res["insights"]["risk_level"]) <= {"Low", "Medium", "High"}


def test_metrics():
    m = ml.regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1


# ---------------------------------------------------------------- inventory risk
def test_classify_risk_thresholds():
    assert ml.classify_risk(0.0) == "High"
    assert ml.classify_risk(C.RISK_HIGH_MAX_DAYS - 0.01) == "High"
    assert ml.classify_risk(C.RISK_HIGH_MAX_DAYS) == "Medium"
    assert ml.classify_risk(C.RISK_MEDIUM_MAX_DAYS) == "Low"
    assert ml.classify_risk(float("nan")) == "Low"


def test_inventory_risk_rules_and_zero_demand():
    normal = make_series(10, sales=[10] * 10, inventory=15)          # cover 1.5 days -> Medium
    zero = make_series(10, product="P0002", sales=[0] * 10, inventory=50)
    empty = make_series(10, product="P0003", sales=[20] * 10, inventory=0)
    df = pd.concat([normal, zero, empty])
    df["Date"] = pd.to_datetime(df["Date"])
    r = ml.inventory_risk(df).set_index("Product ID")
    assert r.loc["P0001", "coverage_days"] == 1.5 and r.loc["P0001", "risk_level"] == "Medium"
    assert r.loc["P0001", "est_replenishment_units"] == int(np.ceil(C.TARGET_COVER_DAYS * 10 - 15))
    assert pd.isna(r.loc["P0002", "coverage_days"]) and r.loc["P0002", "risk_level"] == "Low"
    assert r.loc["P0002", "est_replenishment_units"] == 0
    assert r.loc["P0003", "coverage_days"] == 0 and r.loc["P0003", "risk_level"] == "High"
    assert (r["days_used"] == C.RISK_RECENT_DAYS).all()


# ---------------------------------------------------------------- segmentation / PCA
def test_window_profiles_only_complete_windows():
    df = make_series(60)
    df["Date"] = pd.to_datetime(df["Date"])
    assert len(ml.window_profiles(df)) == 2  # 60 days -> 2 complete 28-day blocks
    gappy = df[df["Date"] != pd.Timestamp("2024-01-05")]
    assert len(ml.window_profiles(gappy)) == 1


def test_segment_assignment_and_pca(demo_df):
    arts = ml.load_artifacts()
    bundle = arts["segmentation"]
    df = demo_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    prof = ml.latest_profiles(df)
    seg = ml.assign_segments(bundle, prof)
    assert len(seg) == 10
    assert seg["cluster"].between(1, bundle["k"]).all()
    assert seg[["pc1", "pc2"]].notna().all().all()
    assert bundle["k"] in C.K_RANGE_ALLOWED
    assert bundle["k"] == max(C.K_RANGE_ALLOWED, key=lambda k: bundle["silhouette"][k])
    assert len(bundle["labels"]) == bundle["k"]
