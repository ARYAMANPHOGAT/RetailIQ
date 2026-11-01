"""
RetailIQ - local SQLite history and CSV / PDF exports.

Tables (minimal by design)
  analysis_runs       one row per successful analysis (metadata + JSON summary)
  prediction_results  next-day forecasts and backtest predictions (actual only when observed)
  product_insights    inventory risk + segment assignment per store-product
Raw uploaded files are NOT stored - only analysis results.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    mode             TEXT NOT NULL CHECK (mode IN ('Demo', 'Live')),
    source_name      TEXT NOT NULL,
    record_count     INTEGER NOT NULL,
    forecast_date    TEXT,
    latest_data_date TEXT,
    model_version    TEXT,
    status           TEXT NOT NULL,
    summary_json     TEXT
);
CREATE TABLE IF NOT EXISTS prediction_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL REFERENCES analysis_runs(analysis_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('next_day', 'backtest')),
    store_id        TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    forecast_date   TEXT NOT NULL,
    predicted_units REAL NOT NULL,
    actual_units    REAL,          -- NULL for future dates (no actual exists yet)
    latest_date     TEXT,
    lag_1           REAL,
    roll_mean_7     REAL
);
CREATE INDEX IF NOT EXISTS idx_pred_analysis ON prediction_results(analysis_id);
CREATE TABLE IF NOT EXISTS product_insights (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id             INTEGER NOT NULL REFERENCES analysis_runs(analysis_id) ON DELETE CASCADE,
    store_id                TEXT NOT NULL,
    product_id              TEXT NOT NULL,
    latest_date             TEXT,
    current_inventory       REAL,
    recent_avg_demand       REAL,
    days_used               INTEGER,
    coverage_days           REAL,  -- NULL when recent demand is ~zero
    risk_level              TEXT,
    est_replenishment_units INTEGER,
    attention               TEXT,
    next_day_forecast       REAL,
    cluster                 INTEGER,
    segment_label           TEXT,
    pc1                     REAL,
    pc2                     REAL,
    profile_json            TEXT
);
CREATE INDEX IF NOT EXISTS idx_insight_analysis ON product_insights(analysis_id);
"""


@contextmanager
def connect():
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(C.DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        con.commit()


def db_status() -> tuple[bool, str]:
    try:
        init_db()
        with connect() as con:
            n = con.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
        return True, f"{n} saved analys{'is' if n == 1 else 'es'}"
    except Exception as exc:
        return False, f"Database unavailable ({type(exc).__name__})"


def _none(v):
    """Convert NaN/NA to None so SQLite stores NULL."""
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


def save_analysis(mode: str, source_name: str, analysis: dict, model_version: str, extra: dict | None = None) -> int:
    """Save one successful analysis in a single transaction (all-or-nothing)."""
    init_db()
    s = analysis["summary"]
    payload = {"summary": s, "daily": analysis["daily"].to_dict(orient="list"), **(extra or {})}
    with connect() as con:
        try:
            cur = con.execute(
                "INSERT INTO analysis_runs (created_at, mode, source_name, record_count, forecast_date, "
                "latest_data_date, model_version, status, summary_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), mode, source_name, int(s["records"]),
                 s["forecast_date"], s["date_end"], model_version, "Completed", json.dumps(payload, default=str)),
            )
            aid = cur.lastrowid
            f = analysis["forecasts"]
            con.executemany(
                "INSERT INTO prediction_results (analysis_id, kind, store_id, product_id, forecast_date, "
                "predicted_units, actual_units, latest_date, lag_1, roll_mean_7) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(aid, "next_day", r[C.STORE_COL], r[C.PRODUCT_COL], str(pd.Timestamp(r["forecast_date"]).date()),
                  float(r["predicted_units"]), None, str(pd.Timestamp(r["latest_date"]).date()),
                  float(r["lag_1"]), float(r["roll_mean_7"])) for _, r in f.iterrows()],
            )
            b = analysis["backtest"]
            con.executemany(
                "INSERT INTO prediction_results (analysis_id, kind, store_id, product_id, forecast_date, "
                "predicted_units, actual_units) VALUES (?,?,?,?,?,?,?)",
                [(aid, "backtest", r[C.STORE_COL], r[C.PRODUCT_COL], str(pd.Timestamp(r["target_date"]).date()),
                  float(r["predicted_units"]), float(r["actual_units"])) for _, r in b.iterrows()],
            )
            ins = analysis["insights"]
            con.executemany(
                "INSERT INTO product_insights (analysis_id, store_id, product_id, latest_date, current_inventory, "
                "recent_avg_demand, days_used, coverage_days, risk_level, est_replenishment_units, attention, "
                "next_day_forecast, cluster, segment_label, pc1, pc2, profile_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(aid, r[C.STORE_COL], r[C.PRODUCT_COL], str(r["latest_date"]), float(r["current_inventory"]),
                  float(r["recent_avg_demand"]), int(r["days_used"]), _none(r["coverage_days"]), r["risk_level"],
                  int(r["est_replenishment_units"]), r["attention"], _none(r["next_day_forecast"]),
                  None if _none(r.get("cluster")) is None else int(r["cluster"]), _none(r.get("segment_label")),
                  _none(r.get("pc1")), _none(r.get("pc2")),
                  json.dumps({k: _none(r.get(k)) for k in C.SEGMENT_FEATURES}))
                 for _, r in ins.iterrows()],
            )
            con.commit()
            return int(aid)
        except Exception:
            con.rollback()  # nothing from a failed save is kept
            raise


def list_runs() -> pd.DataFrame:
    init_db()
    with connect() as con:
        return pd.read_sql_query(
            "SELECT analysis_id, created_at, mode, source_name, record_count, forecast_date, "
            "latest_data_date, model_version, status FROM analysis_runs ORDER BY analysis_id DESC", con)


def load_run(analysis_id: int) -> tuple[dict, dict] | None:
    """Rebuild a saved analysis (same structure as ml_pipeline.run_analysis) without re-running it."""
    init_db()
    with connect() as con:
        run = pd.read_sql_query("SELECT * FROM analysis_runs WHERE analysis_id = ?", con, params=(analysis_id,))
        if run.empty:
            return None
        preds = pd.read_sql_query("SELECT * FROM prediction_results WHERE analysis_id = ?", con, params=(analysis_id,))
        ins = pd.read_sql_query("SELECT * FROM product_insights WHERE analysis_id = ?", con, params=(analysis_id,))
    meta = run.iloc[0].to_dict()
    payload = json.loads(meta.pop("summary_json") or "{}")
    rename = {"store_id": C.STORE_COL, "product_id": C.PRODUCT_COL}
    nd = preds[preds["kind"] == "next_day"].rename(columns=rename)
    forecasts = pd.DataFrame({
        C.STORE_COL: nd[C.STORE_COL], C.PRODUCT_COL: nd[C.PRODUCT_COL],
        "latest_date": pd.to_datetime(nd["latest_date"]), "forecast_date": pd.to_datetime(nd["forecast_date"]),
        "lag_1": nd["lag_1"], "roll_mean_7": nd["roll_mean_7"], "predicted_units": nd["predicted_units"],
    }).reset_index(drop=True)
    bt = preds[preds["kind"] == "backtest"].rename(columns=rename)
    backtest = pd.DataFrame({
        C.STORE_COL: bt[C.STORE_COL], C.PRODUCT_COL: bt[C.PRODUCT_COL],
        "target_date": pd.to_datetime(bt["forecast_date"]), "actual_units": bt["actual_units"],
        "predicted_units": bt["predicted_units"],
    }).reset_index(drop=True)
    ins = ins.rename(columns=rename)
    profiles = pd.DataFrame([json.loads(p or "{}") for p in ins["profile_json"]], index=ins.index,
                            columns=C.SEGMENT_FEATURES)
    insights = pd.concat([ins.drop(columns=["id", "analysis_id", "profile_json"]), profiles], axis=1)
    insights["cluster"] = insights["cluster"].astype("Int64")
    analysis = {"summary": payload.get("summary", {}), "forecasts": forecasts, "backtest": backtest,
                "insights": insights, "daily": pd.DataFrame(payload.get("daily", {})),
                "validation": payload.get("validation")}
    return meta, analysis


def delete_run(analysis_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM analysis_runs WHERE analysis_id = ?", (analysis_id,))
        con.commit()


def clear_history() -> None:
    with connect() as con:
        con.execute("DELETE FROM analysis_runs")
        con.commit()


# ==========================================================================
# Exports
# ==========================================================================

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _latin(text) -> str:
    """fpdf core fonts are Latin-1 only; replace common Unicode punctuation."""
    s = str(text)
    for a, b in {"–": "-", "—": "-", "·": "|", "²": "2", "≥": ">=", "≤": "<=", "→": "->", "’": "'", "“": '"', "”": '"', "…": "..."}.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def header(self):
        self.set_fill_color(16, 16, 16)
        self.rect(0, 0, self.w, 16, "F")
        self.set_fill_color(255, 170, 22)
        self.rect(0, 16, self.w, 0.8, "F")
        self.set_xy(10, 4)
        self.set_text_color(255, 170, 22)
        self.set_font("Helvetica", "B", 14)
        self.cell(40, 8, "RETAILIQ")
        self.set_text_color(200, 200, 200)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 8, "Analysis Report", align="R")
        self.ln(16)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _latin(f"Generated {datetime.now():%Y-%m-%d %H:%M} | Synthetic-data project | Page {self.page_no()}"), align="C")

    def h2(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(217, 145, 11)
        self.cell(0, 7, _latin(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_text_color(30, 30, 30)

    def para(self, text, size=9.5):
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, _latin(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def kv(self, pairs):
        self.set_font("Helvetica", "", 9.5)
        for k, v in pairs:
            self.set_text_color(90, 90, 90)
            self.cell(55, 5.5, _latin(k))
            self.set_text_color(20, 20, 20)
            self.cell(0, 5.5, _latin(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, df: pd.DataFrame, widths=None):
        if df.empty:
            self.para("(no rows)")
            return
        widths = widths or [(self.w - 20) / len(df.columns)] * len(df.columns)
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(40, 40, 40)
        self.set_text_color(255, 255, 255)
        for col, w in zip(df.columns, widths):
            self.cell(w, 6, _latin(col), border=0, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        for i, (_, row) in enumerate(df.iterrows()):
            self.set_fill_color(245, 245, 245) if i % 2 else self.set_fill_color(255, 255, 255)
            for v, w in zip(row, widths):
                self.cell(w, 5.5, _latin("-" if pd.isna(v) else v)[:40], border=0, fill=True)
            self.ln()


def _fmt(v, nd=2):
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:,.{nd}f}"


def build_pdf_report(run_meta: dict, analysis: dict, model_meta: dict | None) -> bytes:
    s = analysis["summary"]
    pdf = _Report(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()

    pdf.h2("1. Analysis metadata")
    pdf.kv([
        ("Analysis ID", str(run_meta.get("analysis_id", "-"))),
        ("Created", str(run_meta.get("created_at", "-"))),
        ("Mode", str(run_meta.get("mode", "-"))),
        ("Source", str(run_meta.get("source_name", "-"))),
        ("Model version", str(run_meta.get("model_version", "-"))),
    ])

    pdf.h2("2. Dataset summary")
    pdf.kv([
        ("Records", f"{s['records']:,}"), ("Stores / Products", f"{s['stores']} / {s['products']}"),
        ("Store-product series", str(s["series"])), ("Date range", f"{s['date_start']} to {s['date_end']}"),
        ("Total units sold", _fmt(s["total_units_sold"], 0)),
        ("Avg units per series-day", _fmt(s["avg_units_per_series_day"])),
    ])

    pdf.h2("3. Model and forecast details")
    if model_meta:
        sel = model_meta["selected_model"]
        tr = model_meta["training"]
        pdf.para(f"Selected model: {sel} (chosen by lowest validation MAE on date-grouped folds). "
                 f"Trained on target dates {tr['train_target_start']} to {tr['train_target_end']}; "
                 f"holdout test {tr['test_target_start']} to {tr['test_target_end']}.")
        rows = [{"Model": n, "MAE": _fmt(m["MAE"]), "RMSE": _fmt(m["RMSE"]), "R2": _fmt(m["R2"], 4)}
                for n, m in {**model_meta["test_metrics"], **model_meta["naive_baselines"]}.items()]
        pdf.ln(1)
        pdf.table(pd.DataFrame(rows), widths=[90, 30, 30, 30])
        pdf.ln(2)
    pdf.para(f"Next-day forecast date: {s['forecast_date']} (latest data date {s['date_end']}). "
             f"{s['forecast_series']} series forecast. Values are model predictions, not observations.")
    bm = s.get("backtest_metrics")
    if bm:
        note = " Note: part of this file overlaps the training period, so these are partly in-sample." \
            if s.get("backtest_overlaps_training") else ""
        pdf.para(f"Backtest on this file ({bm['n']} predictions with known actuals): MAE {_fmt(bm['MAE'])}, "
                 f"RMSE {_fmt(bm['RMSE'])}, R2 {_fmt(bm['R2'], 4)}.{note}")
    f = analysis["forecasts"].copy()
    f = f.sort_values("predicted_units", ascending=False).head(25)
    pdf.ln(1)
    pdf.table(pd.DataFrame({
        "Store": f[C.STORE_COL], "Product": f[C.PRODUCT_COL],
        "Forecast date": pd.to_datetime(f["forecast_date"]).dt.strftime("%Y-%m-%d"),
        "Last obs. sales": f["lag_1"].round(0), "7-day avg": f["roll_mean_7"].round(1),
        "Predicted units": f["predicted_units"].round(1),
    }), widths=[22, 24, 32, 32, 30, 40])

    pdf.h2("4. Inventory risk summary (rule-based)")
    rc = s["risk_counts"]
    pdf.para(f"High: {rc['High']} | Medium: {rc['Medium']} | Low: {rc['Low']}. Coverage days = current inventory / "
             f"average daily units sold over the last {C.RISK_RECENT_DAYS} days. High < {C.RISK_HIGH_MAX_DAYS:g} day, "
             f"Medium < {C.RISK_MEDIUM_MAX_DAYS:g} days, otherwise Low. Replenishment quantities are estimates to reach "
             f"{C.TARGET_COVER_DAYS:g} days of cover, not purchase orders.")
    ins = analysis["insights"].copy()
    order = {"High": 0, "Medium": 1, "Low": 2}
    ins = ins.assign(_o=ins["risk_level"].map(order)).sort_values(["_o", "coverage_days"]).head(20)
    pdf.ln(1)
    pdf.table(pd.DataFrame({
        "Store": ins[C.STORE_COL], "Product": ins[C.PRODUCT_COL], "Inventory": ins["current_inventory"].round(0),
        "Avg demand": ins["recent_avg_demand"].round(1), "Cover (days)": ins["coverage_days"],
        "Risk": ins["risk_level"], "Est. replenish": ins["est_replenishment_units"],
    }), widths=[20, 22, 26, 28, 28, 22, 34])

    pdf.h2("5. Segmentation summary (K-Means)")
    seg = analysis["insights"].dropna(subset=["cluster"])
    if model_meta:
        sg = model_meta["segmentation"]
        sil = sg["silhouette"].get(str(sg["k"]), sg["silhouette"].get(sg["k"], float("nan")))
        pdf.para(f"K-Means with k = {sg['k']} on {sg['window_days']}-day behaviour profiles "
                 f"(silhouette {sil:.3f}). "
                 "Segment names are generated from measured centroid values; they are descriptions, not ground truth.")
    if seg.empty:
        pdf.para("No series in this analysis had enough consecutive history for a segment.")
    else:
        counts = seg.groupby("segment_label").size().reset_index(name="Series")
        pdf.table(counts.rename(columns={"segment_label": "Segment"}), widths=[150, 30])

    pdf.h2("6. Limitations")
    pdf.para(
        "- The training data is a synthetic Kaggle dataset, not real retailer data.\n"
        "- In this dataset next-day sales are essentially unrelated to past sales, so holdout R2 is close to 0; "
        "forecasts are close to each series' typical level and should be read as baseline estimates.\n"
        "- Inventory risk uses transparent rules; it is not a learned stockout probability (the data has no stockout label).\n"
        "- Price, promotion, weather and other future-day inputs are not used because they are unknown at forecast time.\n"
        "- Feature importance shows association, not causation."
    )
    return bytes(pdf.output())
