"""Integration tests: SQLite history, CSV/PDF exports, atomic training save, and the Streamlit app (AppTest)."""
import sqlite3

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config as C
import ml_pipeline as ml
import storage as S


@pytest.fixture(scope="module")
def demo_analysis():
    arts = ml.load_artifacts()
    v = ml.validate_dataset(pd.read_csv(C.DEMO_DATA_PATH))
    return ml.run_analysis(v.clean, arts), arts["metadata"]


# ---------------------------------------------------------------- SQLite
def test_save_read_delete(demo_analysis):
    res, meta = demo_analysis
    aid = S.save_analysis("Demo", "demo.csv", res, meta["model_version"])
    runs = S.list_runs()
    assert list(runs["analysis_id"]) == [aid] and runs["status"].iloc[0] == "Completed"
    run_meta, loaded = S.load_run(aid)
    assert run_meta["mode"] == "Demo" and run_meta["record_count"] == 280
    assert len(loaded["forecasts"]) == len(res["forecasts"])
    assert loaded["forecasts"]["predicted_units"].round(6).tolist() == res["forecasts"]["predicted_units"].round(6).tolist()
    assert len(loaded["backtest"]) == len(res["backtest"])
    assert loaded["insights"]["risk_level"].tolist() == res["insights"]["risk_level"].tolist()
    S.delete_run(aid)
    assert S.list_runs().empty
    with sqlite3.connect(C.DB_PATH) as con:  # cascade removed child rows
        assert con.execute("SELECT COUNT(*) FROM prediction_results").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM product_insights").fetchone()[0] == 0


def test_actual_is_null_for_future_forecasts(demo_analysis):
    res, meta = demo_analysis
    aid = S.save_analysis("Demo", "demo.csv", res, meta["model_version"])
    with sqlite3.connect(C.DB_PATH) as con:
        n_null = con.execute("SELECT COUNT(*) FROM prediction_results WHERE kind='next_day' AND actual_units IS NULL "
                             "AND analysis_id=?", (aid,)).fetchone()[0]
    assert n_null == len(res["forecasts"])


def test_failed_save_leaves_nothing(demo_analysis):
    res, meta = demo_analysis
    broken = dict(res, insights=res["insights"].drop(columns=["risk_level"]))
    with pytest.raises(Exception):
        S.save_analysis("Live", "broken.csv", broken, meta["model_version"])
    assert S.list_runs().empty


def test_history_preserved_and_clear(demo_analysis):
    res, meta = demo_analysis
    a = S.save_analysis("Demo", "a.csv", res, meta["model_version"])
    b = S.save_analysis("Live", "b.csv", res, meta["model_version"])
    assert set(S.list_runs()["analysis_id"]) == {a, b}
    assert set(S.list_runs()["mode"]) == {"Demo", "Live"}
    S.clear_history()
    assert S.list_runs().empty and S.load_run(a) is None


# ---------------------------------------------------------------- exports
def test_csv_and_pdf_exports(demo_analysis):
    res, meta = demo_analysis
    csv = S.to_csv_bytes(res["forecasts"]).decode()
    assert csv.splitlines()[0].startswith("Store ID,Product ID")
    aid = S.save_analysis("Demo", "demo.csv", res, meta["model_version"])
    run_meta, loaded = S.load_run(aid)
    pdf = S.build_pdf_report(run_meta, loaded, meta)
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000
    assert S.build_pdf_report(run_meta, loaded, None)[:4] == b"%PDF"  # works without model metadata


# ---------------------------------------------------------------- training safety
def test_failed_training_keeps_last_good_models(tmp_path, monkeypatch):
    import train
    models = tmp_path / "models"
    models.mkdir()
    for name in (C.DEMAND_MODEL_PATH.name, C.SEGMENTATION_PATH.name, C.METADATA_PATH.name):
        (models / name).write_text("last-good")
    monkeypatch.setattr(C, "MODELS_DIR", models)
    monkeypatch.setattr(train, "train_demand", lambda feats: ({"model": "new"}, {"training": {"test_target_start": "2023"}}))
    # an unpicklable bundle makes the 2nd joblib.dump fail AFTER the 1st file was staged
    monkeypatch.setattr(train, "train_segmentation", lambda df: ({"bad": lambda: None}, {}))
    monkeypatch.setattr(train, "make_demo_and_template", lambda raw, start: {})
    with pytest.raises(Exception):
        train.main()
    for name in (C.DEMAND_MODEL_PATH.name, C.SEGMENTATION_PATH.name, C.METADATA_PATH.name):
        assert (models / name).read_text() == "last-good"


# ---------------------------------------------------------------- Streamlit app (headless)
def _app():
    return AppTest.from_file(str(C.ROOT_DIR / "app.py"), default_timeout=120)


def test_app_starts_and_runs_demo_through_all_pages():
    at = _app().run()
    assert not at.exception
    assert any("Welcome to RetailIQ" in m.value for m in at.markdown)
    at.button(key="w_demo").click().run()
    assert not at.exception and at.session_state["run_meta"]["mode"] == "Demo"
    for page in ["Dataset Management", "Demand Forecasting", "Inventory Risk Analysis", "Product Segmentation",
                 "Model Performance", "History & Reports", "M5 Forecasting", "M5 Model Performance", "Overview"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
        assert not [e.value for e in at.error], (page, [e.value for e in at.error])
    assert len(S.list_runs()) == 1


def test_app_without_models_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DEMAND_MODEL_PATH", tmp_path / "missing.joblib")
    at = _app().run()
    assert not at.exception
    assert any("train.py" in c.value for c in at.code)
    at.session_state["page"] = "Model Performance"
    at.run()
    assert not at.exception and any("Models are not ready" in m.value for m in at.markdown)


def test_mode_switch_requires_confirmation_and_reset_keeps_history():
    at = _app().run()
    at.button(key="w_demo").click().run()
    at.button(key="mode_live").click().run()
    assert at.session_state["pending_mode"] == "Live" and at.session_state["mode"] == "Demo"
    at.button(key="dlg_cancel").click().run()
    assert at.session_state["mode"] == "Demo" and at.session_state["analysis"] is not None
    at.button(key="mode_live").click().run()
    at.button(key="dlg_switch").click().run()
    assert at.session_state["mode"] == "Live" and at.session_state["analysis"] is None
    assert len(S.list_runs()) == 1  # switching never deletes history
