"""
RetailIQ - Intelligent Retail Demand Forecasting & Inventory Risk Prediction System
Streamlit dashboard.  Launch with:  streamlit run app.py

The app only LOADS saved models (models/). It never trains; run `python train.py` for that.
"""
from __future__ import annotations

import hashlib
import html as htmllib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
import ml_pipeline as ml
import storage as S

st.set_page_config(page_title="RetailIQ", page_icon=str(C.ROOT_DIR / "assets" / "favicon.png"),
                   layout="wide", initial_sidebar_state="expanded")
LOG = ml.get_logger()

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PAGES = [
    ("Overview", ":material/home:", "Get a complete view of your retail demand, inventory and product insights."),
    ("Dataset Management", ":material/database:", "Load the demo data or upload your own file, then validate it before analysis."),
    ("Demand Forecasting", ":material/monitoring:", "Next-day Units Sold predictions for every store-product series."),
    ("Inventory Risk Analysis", ":material/shield:", "Rule-based inventory coverage: how many days current stock lasts at recent demand."),
    ("Product Segmentation", ":material/hub:", "K-Means segments of 28-day store-product behaviour, visualised with PCA."),
    ("Model Performance", ":material/bar_chart:", "Genuine holdout results from the offline training run."),
    ("History & Reports", ":material/description:", "Saved analyses, CSV exports and PDF reports."),
    ("M5 Forecasting", ":material/insights:", "Separate Walmart M5 next-day forecasts; never mixed with RetailIQ uploads."),
    ("M5 Model Performance", ":material/analytics:", "Read-only validation and final holdout evidence for M5 model versions."),
]
AMBER, AMBER2, TEXT, MUTED, LINE = "#FFAA16", "#D9910B", "#F2F2F2", "#A5A5A5", "#393939"
RISK_COLORS = {"High": "#E5534B", "Medium": AMBER, "Low": "#3FB950"}
SEG_COLORS = ["#FFAA16", "#F2F2F2", "#8C8C8C", "#B87708", "#FFD580", "#5E5E5E"]
METRIC_HELP = {
    "MAE": "Mean Absolute Error: the average size of a forecast miss, in units. Lower is better.",
    "RMSE": "Root Mean Squared Error: like MAE but punishes large misses more. Lower is better.",
    "R2": "R-squared: share of the variation in actual sales explained by the model. 1 = perfect, "
          "0 = no better than always predicting the average, below 0 = worse than the average.",
}
STRIPES = ('<svg width="30" height="15" viewBox="0 0 30 15"><polygon points="6,0 11,0 5,15 0,15" fill="#FFAA16"/>'
           '<polygon points="15,0 20,0 14,15 9,15" fill="#FFAA16"/><polygon points="24,0 29,0 23,15 18,15" fill="#D9910B"/></svg>')
_SVG = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFAA16" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{}</svg>'
ICONS = {
    "cart": _SVG.format('<circle cx="9" cy="20" r="1.4"/><circle cx="18" cy="20" r="1.4"/><path d="M2 3h3l2.6 12.2a1.5 1.5 0 0 0 1.5 1.2h8.6a1.5 1.5 0 0 0 1.5-1.1L21 8H6"/>'),
    "box": _SVG.format('<path d="M21 8l-9-5-9 5 9 5 9-5z"/><path d="M3 8v8l9 5 9-5V8"/><path d="M12 13v8"/>'),
    "store": _SVG.format('<path d="M3 9l1.5-5h15L21 9"/><path d="M3 9h18v1.5a3 3 0 0 1-6 0 3 3 0 0 1-6 0 3 3 0 0 1-6 0z"/><path d="M5 13v8h14v-8"/><path d="M10 21v-5h4v5"/>'),
    "calendar": _SVG.format('<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>'),
    "layers": _SVG.format('<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>'),
    "rows": _SVG.format('<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M3 9h18M3 14h18M9 4v16"/>'),
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def md(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def esc(x) -> str:
    return htmllib.escape(str(x))


def section(title: str, hint: str = "") -> None:
    h = f'<span class="hint">{esc(hint)}</span>' if hint else ""
    md(f'<div class="riq-section">{STRIPES}<span class="txt">{esc(title)}</span>{h}</div>')


def note(text: str) -> None:
    md(f'<div class="riq-note">{text}</div>')


def chip(text: str, kind: str = "grey") -> str:
    return f'<span class="riq-chip {kind}">{esc(text)}</span>'


def fmt(v, nd=2, dash="-"):
    try:
        return dash if v is None or pd.isna(v) else f"{float(v):,.{nd}f}"
    except (TypeError, ValueError):
        return dash


def style_fig(fig: go.Figure, height: int = 300, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12, family="Segoe UI, Inter, Roboto, Arial, sans-serif"),
        hoverlabel=dict(bgcolor="#1D1D1D", bordercolor=LINE, font_color=TEXT), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color="#CFCFCF")),
    )
    fig.update_xaxes(gridcolor="#262626", linecolor=LINE, zeroline=False, tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor="#262626", linecolor=LINE, zeroline=False, tickfont=dict(color=MUTED), automargin=True)
    return fig


def show_fig(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=key)


def spark(values, w: int = 92, h: int = 38) -> str:
    v = [float(x) for x in values if pd.notna(x)][-14:]
    if len(v) < 2:
        return ""
    lo, hi = min(v) * 0.85, max(v)
    rng = (hi - lo) or 1
    bw = w / len(v)
    bars = []
    for i, x in enumerate(v):
        bh = max(3, h * (x - lo) / rng)
        col = AMBER if i == len(v) - 1 else AMBER2
        bars.append(f'<rect x="{i * bw + 1:.1f}" y="{h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" '
                    f'fill="{col}" opacity="{0.45 + 0.55 * (i + 1) / len(v):.2f}"/>')
    return f'<svg width="{w}" height="{h}">{"".join(bars)}</svg>'


def kpi(icon: str, title: str, value: str, sub: str = "", delta=None, sparkvals=None, small=False) -> str:
    d = ""
    if delta is not None:
        cls, arrow = ("up", "&#9650;") if delta >= 0 else ("down", "&#9660;")
        d = f'<span class="d {cls}">{arrow} {abs(delta):.1f}%</span> '
    sp = f'<div class="spark">{spark(sparkvals, 70, 30)}</div>' if sparkvals is not None else ""
    return (f'<div class="riq-kpi"><div class="top"><div class="ico">{ICONS[icon]}</div><div class="t">{title}</div></div>'
            f'<div class="v{" small" if small else ""}">{value}</div>'
            f'<div class="bottom"><div class="s">{d}{sub}</div>{sp}</div></div>')


# --------------------------------------------------------------------------
# State, artifacts, callbacks
# --------------------------------------------------------------------------
def init_state() -> None:
    for k, v in dict(page="Overview", mode="Demo", analysis=None, run_meta=None, pending_mode=None,
                     upload=None, uploader_nonce=0, confirm=None, show_about=False, flash=None).items():
        st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def _load_artifacts(stamp):
    return ml.load_artifacts()


def artifacts() -> dict:
    """Cached by file modification time, so a new `python train.py` is picked up on the next rerun."""
    stamp = tuple(p.stat().st_mtime if p.exists() else 0 for p in
                  (C.DEMAND_MODEL_PATH, C.SEGMENTATION_PATH, C.METADATA_PATH))
    return _load_artifacts(stamp)


def model_meta() -> dict | None:
    a = artifacts()
    return a["metadata"] if a["state"] == "ready" else None


def set_page(p: str) -> None:
    st.session_state.page = p


def clear_session_analysis() -> None:
    st.session_state.analysis = None
    st.session_state.run_meta = None


def switch_mode(m: str) -> None:
    st.session_state.mode = m
    clear_session_analysis()
    st.session_state.upload = None
    st.session_state.uploader_nonce += 1  # resets the file uploader widget
    st.session_state.pending_mode = None


def request_mode(m: str) -> None:
    if m == st.session_state.mode:
        return
    if st.session_state.analysis is not None:
        st.session_state.pending_mode = m  # ask first (see confirm_mode_dialog)
    else:
        switch_mode(m)


def go_live_upload() -> None:
    st.session_state.page = "Dataset Management"
    request_mode("Live")


def cancel_pending() -> None:
    st.session_state.pending_mode = None


def cancel_confirm() -> None:
    st.session_state.confirm = None


def reset_analysis() -> None:
    clear_session_analysis()
    st.session_state.flash = ("ok", "Current analysis cleared from this session. Saved history is unchanged.")


def open_run(aid: int) -> None:
    loaded = S.load_run(int(aid))
    if loaded is None:
        st.session_state.flash = ("err", f"Analysis #{aid} no longer exists.")
        return
    meta, analysis = loaded
    st.session_state.mode = meta["mode"]
    st.session_state.analysis = normalise(analysis)
    st.session_state.run_meta = meta
    st.session_state.page = "Overview"
    st.session_state.flash = ("ok", f"Opened saved analysis #{aid} ({meta['mode']}).")


def normalise(a: dict) -> dict:
    """Make fresh and reopened analyses look identical to the page code."""
    a = dict(a)
    f = a["forecasts"].copy()
    for c in ("latest_date", "forecast_date"):
        f[c] = pd.to_datetime(f[c])
    b = a["backtest"].copy()
    b["target_date"] = pd.to_datetime(b["target_date"])
    ins = a["insights"].copy()
    ins["latest_date"] = ins["latest_date"].astype(str)
    ins["cluster"] = pd.to_numeric(ins.get("cluster"), errors="coerce").astype("Int64")
    for c in ("coverage_days", "next_day_forecast", "pc1", "pc2"):
        if c in ins:
            ins[c] = pd.to_numeric(ins[c], errors="coerce")
    d = pd.DataFrame(a["daily"]).copy()
    if not d.empty:
        d[C.DATE_COL] = pd.to_datetime(d[C.DATE_COL])
    a.update(forecasts=f, backtest=b, insights=ins, daily=d)
    return a


# --------------------------------------------------------------------------
# Dialogs
# --------------------------------------------------------------------------
@st.dialog("Switch mode?", on_dismiss=cancel_pending)
def confirm_mode_dialog(target: str) -> None:
    st.write(f"Switching to **{target} Mode** closes the analysis currently on screen. "
             "It is already saved and can be reopened from **History & Reports**.")
    c1, c2 = st.columns(2)
    if c1.button(f"Switch to {target}", type="primary", width="stretch", key="dlg_switch"):
        switch_mode(target)
        st.rerun()  # documented way to close a dialog
    if c2.button("Cancel", width="stretch", key="dlg_cancel"):
        cancel_pending()
        st.rerun()


def _do_confirmed(kind: str, arg) -> None:
    if kind == "delete":
        S.delete_run(int(arg))
        if st.session_state.run_meta and st.session_state.run_meta.get("analysis_id") == int(arg):
            clear_session_analysis()
        st.session_state.flash = ("ok", f"Analysis #{arg} deleted.")
    elif kind == "clear":
        S.clear_history()
        st.session_state.flash = ("ok", "All saved history cleared.")
    st.session_state.confirm = None


@st.dialog("Please confirm", on_dismiss=cancel_confirm)
def confirm_action_dialog(kind: str, arg) -> None:
    if kind == "delete":
        st.write(f"Delete saved analysis **#{arg}** and all of its stored results? This cannot be undone.")
    else:
        st.write("Delete **all** saved analyses from the local database? This cannot be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Delete", type="primary", width="stretch", key="dlg_delete"):
        _do_confirmed(kind, arg)
        st.rerun()
    if c2.button("Cancel", width="stretch", key="dlg_cancel2"):
        cancel_confirm()
        st.rerun()


@st.dialog("About & Methodology", width="large", on_dismiss=lambda: st.session_state.update(show_about=False))
def about_dialog() -> None:
    meta = model_meta()
    st.markdown(f"""
**RetailIQ** forecasts next-day demand, flags inventory risk and segments store-product behaviour using
classical machine learning (scikit-learn).

**Data honesty.** The training data is the Kaggle *Retail Store Inventory Forecasting* file
(`retail_store_inventory.csv`, 73,100 rows, 5 stores x 20 products, 2022-01-01 to 2024-01-01).
It is **synthetic** - not real retailer operations data.

**Forecasting.** Target = Units Sold on the next day (t+1) per Store ID + Product ID. Inputs use only
information known at the end of day t: sales lag 1 and lag 7, 7-day rolling mean/std, the target day's
weekday and month, and the store/product IDs. Linear Regression (baseline) and Random Forest are compared;
the one with the lower **validation** MAE on date-grouped time-series folds is selected. The final 20% of dates
are held out for testing and never used for choosing.

**Excluded on purpose.** *Demand Forecast* (~0.997 correlated with Units Sold - leakage); price, discount,
promotion, weather, competitor price, units ordered and inventory for the target day (not known in advance);
Category/Region/Seasonality (they change daily for the same product in this file).

**Inventory risk** is rule-based coverage = inventory / recent average daily sales (last {C.RISK_RECENT_DAYS} days).
High < {C.RISK_HIGH_MAX_DAYS:g} day, Medium < {C.RISK_MEDIUM_MAX_DAYS:g} days, else Low. There is no stockout label in
the data, so no stockout classifier is trained. Replenishment numbers are estimates, not purchase orders.

**Segmentation.** K-Means on six standardised features of {C.SEGMENT_WINDOW_DAYS}-day behaviour profiles; k guided by
the elbow method and silhouette score; PCA projects the profiles to 2-D for display only.

**Privacy.** Everything runs locally. Uploaded files are processed in memory and are **not** stored; only analysis
results are saved in `database/retailiq.db`. Technical errors are logged to `logs/retailiq.log`.
""")
    if meta:
        st.caption(f"Model version {meta['model_version']} - trained {meta['trained_at']} - "
                   f"scikit-learn {meta['sklearn_version']}")


# --------------------------------------------------------------------------
# Layout: sidebar + header
# --------------------------------------------------------------------------
BRAND = ('<div class="riq-brand"><svg width="40" height="26" viewBox="0 0 40 26">'
         '<polygon points="10,0 18,0 8,26 0,26" fill="#FFAA16"/><polygon points="24,0 32,0 22,26 14,26" fill="#FFAA16"/>'
         '<polygon points="38,0 40,0 40,4 31,26 28,26" fill="#D9910B"/></svg>'
         '<div><div class="word">RETAIL<b>IQ</b></div><div class="tag">Intelligent retail analytics</div></div></div>')


def sidebar() -> None:
    mode = st.session_state.mode
    with st.sidebar:
        md(BRAND)
        md('<div class="riq-side-label">Mode</div>')
        c1, c2 = st.columns(2, gap="small")
        c1.button("Demo Mode", key="mode_demo", type="primary" if mode == "Demo" else "secondary",
                  width="stretch", on_click=request_mode, args=("Demo",))
        c2.button("Live Analysis", key="mode_live", type="primary" if mode == "Live" else "secondary",
                  width="stretch", on_click=request_mode, args=("Live",))
        md('<div class="riq-side-label">Navigate</div>')
        with st.container(key="navgroup"):
            for name, icon, _ in PAGES:
                st.button(name, key="nav_" + name.lower().replace(" ", "_"), icon=icon, width="stretch",
                          type="primary" if st.session_state.page == name else "tertiary",
                          on_click=set_page, args=(name,))
        st.divider()
        meta = st.session_state.run_meta
        if meta:
            aid = meta.get("analysis_id")
            md(f'<div class="riq-muted">Current analysis</div>'
               f'<div style="margin:4px 0 8px">{chip("#" + str(aid) if aid else "unsaved", "amber")} '
               f'{chip(meta["mode"], "grey")}<br><span class="riq-muted">{esc(meta["source_name"])}</span></div>')
            st.button("Reset analysis", icon=":material/restart_alt:", width="stretch", on_click=reset_analysis,
                      help="Clears the analysis from this session only. Saved history is kept.")
        if st.button("About & Methodology", icon=":material/info:", type="tertiary", width="stretch"):
            st.session_state.show_about = True
        st.caption("Runs locally. Uploaded files are not stored; only analysis results are saved.")


def header() -> None:
    page = st.session_state.page
    sub = next(s for n, _, s in PAGES if n == page)
    arts = artifacts()
    state = arts["state"]
    m_dot = "ok" if state == "ready" and not arts["warnings"] else ("warn" if state == "ready" else "bad")
    m_txt = {"ready": "Loaded", "missing": "Not trained", "corrupt": "Load error", "incompatible": "Incompatible"}[state]
    db_ok, db_msg = S.db_status()
    mode = st.session_state.mode
    md(f'''<div class="riq-header"><div><h1>{esc(page if page != "Overview" else "Overview")}</h1><div class="sub">{esc(sub)}</div></div>
<div class="riq-status">
<div class="item"><div><div class="lbl">Mode</div><div class="val">{chip(mode + " Mode", "amber")}</div></div></div>
<div class="item"><div><div class="lbl">{"RetailIQ Model Status" if page == "M5 Forecasting" else "Model Status"}</div><div class="val"><span class="riq-dot {m_dot}"></span>{m_txt}</div></div></div>
<div class="item"><div><div class="lbl">Database Status</div><div class="val"><span class="riq-dot {"ok" if db_ok else "bad"}"></span>{"Connected" if db_ok else "Error"}</div><div class="lbl">{esc(db_msg)}</div></div></div>
</div></div>'''.replace("\n", ""))
    for w in arts["warnings"]:
        st.warning(w, icon=":material/warning:")
    meta = st.session_state.run_meta
    if meta and page not in ("Overview", "Model Performance", "History & Reports", "Dataset Management"):
        aid = meta.get("analysis_id")
        md(f'<div class="riq-muted" style="margin:-6px 0 12px">Showing analysis {chip("#" + str(aid) if aid else "unsaved", "amber")} '
           f'{chip(meta["mode"], "grey")} &nbsp;{esc(meta["source_name"])} &middot; {esc(meta.get("created_at", ""))}</div>')


def empty_state(title: str, text: str, actions: bool = True) -> None:
    with st.container(key="card_empty"):
        md(f'<div class="riq-hero"><h2>{esc(title)}</h2><p>{text}</p></div>')
        if actions:
            c1, c2, _ = st.columns([1, 1, 2])
            if c1.button("Launch demo analysis", icon=":material/play_circle:", type="primary", width="stretch",
                         key="empty_demo", disabled=st.session_state.mode != "Demo",
                         help=None if st.session_state.mode == "Demo" else "Switch to Demo Mode first"):
                if run_demo(st):
                    st.rerun()
            c2.button("Go to Dataset Management", icon=":material/database:", width="stretch", key="empty_ds",
                      on_click=set_page, args=("Dataset Management",))


def require_analysis(what: str):
    if st.session_state.analysis is None:
        empty_state("No analysis loaded", f"Run the demo analysis or upload a dataset in Live mode to see {what}. "
                    "You can also reopen a saved analysis from History & Reports.")
        return None
    return st.session_state.analysis


def models_missing_box() -> None:
    arts = artifacts()
    with st.container(key="card_models_missing"):
        md('<div class="riq-hero"><h2>Models are not ready</h2></div>')
        for p in arts["problems"]:
            st.error(p, icon=":material/error:")
        st.markdown("Open a terminal in the project folder and run:")
        st.code(C.TRAIN_COMMAND, language="bash")
        st.caption("Training uses data/raw/retail_store_inventory.csv and takes about a minute. Then refresh this page.")


# --------------------------------------------------------------------------
# Running analyses
# --------------------------------------------------------------------------
def run_and_store(mode: str, source_name: str, clean: pd.DataFrame, validation_summary: dict, area) -> bool:
    arts = artifacts()
    if arts["state"] != "ready":
        area.error(" ".join(arts["problems"]) or f"Models are not ready. Run `{C.TRAIN_COMMAND}`.")
        return False
    bar = area.progress(0, text="Starting analysis")
    try:
        res = ml.run_analysis(clean, arts, progress=lambda p, m: bar.progress(min(int(p), 100), text=m))
    except ml.UserFacingError as exc:
        bar.empty()
        area.error(str(exc))
        return False
    except Exception:
        LOG.exception("Analysis failed")
        bar.empty()
        area.error("The analysis failed unexpectedly. Technical details were written to logs/retailiq.log.")
        return False
    version = arts["metadata"]["model_version"]
    try:
        aid = S.save_analysis(mode, source_name, res, version, {"validation": validation_summary})
        meta, analysis = S.load_run(aid)
        st.session_state.flash = ("ok", f"Analysis #{aid} completed and saved to history.")
    except Exception:
        LOG.exception("Saving analysis failed")
        meta = {"analysis_id": None, "created_at": datetime.now().isoformat(timespec="seconds"), "mode": mode,
                "source_name": source_name, "model_version": version, "record_count": res["summary"]["records"],
                "status": "Not saved"}
        analysis = {**res, "validation": validation_summary}
        st.session_state.flash = ("warn", "Results are shown but could not be saved to history (database error).")
    st.session_state.analysis = normalise(analysis)
    st.session_state.run_meta = meta
    return True


def load_demo_validation():
    if not C.DEMO_DATA_PATH.exists():
        return None, None
    raw = pd.read_csv(C.DEMO_DATA_PATH)
    meta = model_meta()
    v = ml.validate_dataset(raw, meta["known_stores"] if meta else None, meta["known_products"] if meta else None)
    return raw, v


def run_demo(area) -> bool:
    raw, v = load_demo_validation()
    if v is None:
        area.error(f"Demo dataset not found at {C.DEMO_DATA_PATH}. Run `{C.TRAIN_COMMAND}` to create it.")
        return False
    if not v.ok:
        area.error("Demo dataset failed validation: " + "; ".join(v.errors))
        return False
    return run_and_store("Demo", C.DEMO_DATA_PATH.name, v.clean, v.summary, area)


# ==========================================================================
# PAGE 1 - Overview
# ==========================================================================
def page_overview() -> None:
    a = st.session_state.analysis
    if a is None:
        return welcome()
    s, daily, ins = a["summary"], a["daily"], a["insights"]
    prev = s.get("units_prev7")
    delta = (s["units_last7"] - prev) / prev * 100 if prev else None
    high = s["risk_counts"]["High"]
    cards = [
        kpi("cart", "Total Units Sold", f"{s['total_units_sold']:,.0f}",
            "last 7 days vs prior 7" if delta is not None else "full date range", delta,
            daily["units_sold"].tolist() if len(daily) else None),
        kpi("rows", "Records", f"{s['records']:,}", f"{s['series']} store-product series"),
        kpi("store", "Stores &middot; Products", f"{s['stores']} &middot; {s['products']}",
            f"avg {s['avg_units_per_series_day']:.1f} units per<br>store-product-day"),
        kpi("layers", "Current Inventory", f"{s['current_inventory_total']:,.0f}",
            f"units on latest day<br>{high} high-risk", None,
            daily["inventory"].tolist() if len(daily) else None),
        kpi("calendar", "Date Range", f"{s['date_start']}<br>to {s['date_end']}",
            f"Next-day forecast: {s['forecast_date']}", small=True),
    ]
    for col, card in zip(st.columns(5, gap="small"), cards):
        with col:
            md(card)
    if high:
        md(f'<div class="riq-alert" style="margin-top:12px"><b>Inventory alert:</b> {high} store-product series have less than '
           f'{C.RISK_HIGH_MAX_DAYS:g} day of inventory cover at recent demand.</div>')
    else:
        st.write("")

    left, right = st.columns([1.55, 1], gap="small")
    with left, st.container(key="card_trend", height=410):
        h1, h2 = st.columns([2, 1])
        with h1:
            section("Sales Trend", f"daily totals, {s['date_start']} to {s['date_end']}")
        metric = h2.selectbox("Metric", ["Units Sold", "Inventory Level", "Average Price"], label_visibility="collapsed")
        col = {"Units Sold": "units_sold", "Inventory Level": "inventory", "Average Price": "avg_price"}[metric]
        fig = go.Figure(go.Scatter(x=daily[C.DATE_COL], y=daily[col], mode="lines", line=dict(color=AMBER, width=1.8),
                                   fill="tozeroy", fillcolor="rgba(255,170,22,0.10)", name=metric))
        show_fig(style_fig(fig, 320, legend=False).update_yaxes(title=metric), key="ov_trend")
    with right, st.container(key="card_fc", height=410):
        h1, h2 = st.columns([2.2, 1])
        with h1:
            section("Next-Day Forecast", "top 6")
        h2.button("View all", icon=":material/arrow_forward:", icon_position="right", key="ov_fc_all",
                  on_click=set_page, args=("Demand Forecasting",), width="stretch")
        f = a["forecasts"].sort_values("predicted_units", ascending=False).head(6)
        rows = "".join(
            f'<tr><td>{esc(r[C.STORE_COL])}</td><td>{esc(r[C.PRODUCT_COL])}</td>'
            f'<td class="num">{r["predicted_units"]:.0f}</td><td class="num">{r["roll_mean_7"]:.0f}</td>'
            f'<td class="num">{trend_icon(r["predicted_units"], r["roll_mean_7"])}</td></tr>' for _, r in f.iterrows())
        md(f'<table class="riq-table"><tr><th>Store</th><th>Product</th><th class="num">Predicted</th>'
           f'<th class="num">7-day avg</th><th class="num">vs avg</th></tr>{rows}</table>'
           f'<div class="riq-muted" style="margin-top:8px">Forecast for {s["forecast_date"]}. Predicted values are model estimates.</div>')

    c1, c2, c3 = st.columns([1, 1, 1.05], gap="small")
    with c1, st.container(key="card_risk", height=350):
        section("Inventory Risk Distribution")
        risk_donut(s["risk_counts"], key="ov_donut", height=250)
    with c2, st.container(key="card_pca", height=350):
        section("Product Segmentation (PCA)")
        seg = ins.dropna(subset=["cluster"])
        if seg.empty:
            st.caption(f"No series had {C.SEGMENT_WINDOW_DAYS} consecutive days for a segment profile.")
        else:
            show_fig(pca_figure(seg, 260, background=False), key="ov_pca")
    with c3, st.container(key="card_perf", height=350):
        h1, h2 = st.columns([2, 1.1])
        with h1:
            section("Model Performance")
        h2.button("Details", icon=":material/arrow_forward:", icon_position="right", key="ov_perf",
                  on_click=set_page, args=("Model Performance",), width="stretch")
        perf_table(compact=True)

    c1, c2 = st.columns([1.55, 1], gap="small")
    with c1, st.container(key="card_hist", height=275):
        section("Recent Analysis History")
        recent_history(limit=4)
    with c2, st.container(key="card_qa", height=275):
        section("Quick Actions")
        quick_actions()


def trend_icon(pred, avg) -> str:
    if not avg:
        return "-"
    r = (pred - avg) / avg
    if r > 0.05:
        return f'<span style="color:{AMBER}">&#9650; {r:+.0%}</span>'
    if r < -0.05:
        return f'<span style="color:{MUTED}">&#9660; {r:+.0%}</span>'
    return f'<span style="color:{MUTED}">= {r:+.0%}</span>'


def risk_donut(counts: dict, key: str, height: int = 240) -> None:
    total = sum(counts.values())
    labels = ["Low", "Medium", "High"]
    d1, d2 = st.columns([1.1, 1])
    with d1:
        fig = go.Figure(go.Pie(labels=[f"{l} Risk" for l in labels], values=[counts[l] for l in labels], hole=0.64,
                               marker=dict(colors=[RISK_COLORS[l] for l in labels], line=dict(color="#1D1D1D", width=2)),
                               sort=False, textinfo="none", hovertemplate="%{label}: %{value} series<extra></extra>"))
        fig.add_annotation(text=f"<b>{total}</b><br>series", showarrow=False, font=dict(size=17, color=TEXT))
        show_fig(style_fig(fig, height, legend=False), key=key)
    with d2:
        items = "".join(
            f'<div class="riq-legend-row"><span class="riq-dot" '
            f'style="background:{RISK_COLORS[l]};width:11px;height:11px"></span><span style="flex:1">{l} Risk</span>'
            f'<span class="riq-muted">({counts[l]})&nbsp; {counts[l] / total:.0%}</span></div>' for l in labels) if total else ""
        md(f'<div style="padding-top:36px">{items}</div>')


def pca_figure(seg: pd.DataFrame, height: int, background: bool = True) -> go.Figure:
    fig = go.Figure()
    bundle = artifacts().get("segmentation")
    if background and bundle:
        tp = bundle["training_points"]
        fig.add_trace(go.Scatter(x=tp["pc1"], y=tp["pc2"], mode="markers", name="Training profiles",
                                 marker=dict(size=4, color="#3A3A3A"), hoverinfo="skip"))
    for c in sorted(seg["cluster"].dropna().unique()):
        part = seg[seg["cluster"] == c]
        fig.add_trace(go.Scatter(
            x=part["pc1"], y=part["pc2"], mode="markers", name=f"Segment {c}",
            marker=dict(size=10, color=SEG_COLORS[(int(c) - 1) % len(SEG_COLORS)], line=dict(color="#101010", width=1)),
            text=part[C.STORE_COL] + " · " + part[C.PRODUCT_COL] + "<br>" + part["segment_label"].astype(str),
            hovertemplate="%{text}<extra></extra>"))
    fig = style_fig(fig, height)
    fig.update_xaxes(title="PC 1")
    fig.update_yaxes(title="PC 2")
    fig.update_layout(legend=dict(orientation="v", x=1.01, y=1, yanchor="top"))
    return fig


def perf_table(compact: bool = False) -> None:
    meta = model_meta()
    if not meta:
        st.caption(f"Models not trained. Run `{C.TRAIN_COMMAND}`.")
        return
    sel = meta["selected_model"]
    rows = ""
    for name, m in meta["test_metrics"].items():
        cls = ' class="sel"' if name == sel else ""
        tag = " &#9679;" if name == sel else ""
        rows += (f'<tr{cls}><td>{name}{tag}</td><td class="num">{m["MAE"]:.2f}</td><td class="num">{m["RMSE"]:.2f}</td>'
                 f'<td class="num">{m["R2"]:.3f}</td></tr>')
    if not compact:
        for name, m in meta["naive_baselines"].items():
            rows += (f'<tr><td style="color:{MUTED}">{name}</td><td class="num">{m["MAE"]:.2f}</td>'
                     f'<td class="num">{m["RMSE"]:.2f}</td><td class="num">{m["R2"]:.3f}</td></tr>')
    sg = meta["segmentation"]
    sil = sg["silhouette"].get(str(sg["k"]), 0)
    rows += f'<tr><td>K-Means (silhouette, k={sg["k"]})</td><td class="num">-</td><td class="num">-</td><td class="num">{sil:.3f}</td></tr>'
    md(f'<table class="riq-table"><tr><th>Model</th><th class="num">MAE</th><th class="num">RMSE</th>'
       f'<th class="num">R&sup2;</th></tr>{rows}</table>'
       f'<div class="riq-muted" style="margin-top:8px">&#9679; selected by validation MAE. Holdout test: '
       f'{meta["training"]["test_target_start"]} to {meta["training"]["test_target_end"]}.</div>')


def recent_history(limit: int = 4) -> None:
    try:
        runs = S.list_runs().head(limit)
    except Exception:
        LOG.exception("history")
        st.caption("History unavailable (database error).")
        return
    if runs.empty:
        st.caption("No saved analyses yet.")
        return
    widths = [1.5, 2.2, 0.9, 0.8, 1.1, 0.6]
    hdr = st.columns(widths)
    for c, t in zip(hdr, ["Date & Time", "Source", "Records", "Mode", "Status", ""]):
        c.markdown(f'<span class="riq-muted">{t}</span>', unsafe_allow_html=True)
    for _, r in runs.iterrows():
        cols = st.columns(widths, vertical_alignment="center")
        cols[0].markdown(f"<small>{esc(str(r['created_at']).replace('T', ' ')[:16])}</small>", unsafe_allow_html=True)
        cols[1].markdown(f"<small>{esc(r['source_name'])}</small>", unsafe_allow_html=True)
        cols[2].markdown(f"<small>{int(r['record_count']):,}</small>", unsafe_allow_html=True)
        cols[3].markdown(f"<small>{esc(r['mode'])}</small>", unsafe_allow_html=True)
        cols[4].markdown(chip(r["status"], "green"), unsafe_allow_html=True)
        cols[5].button("", icon=":material/visibility:", key=f"open_{r['analysis_id']}", type="tertiary",
                       help=f"Open analysis #{r['analysis_id']}", on_click=open_run, args=(int(r["analysis_id"]),))


def quick_actions() -> None:
    a, meta = st.session_state.analysis, st.session_state.run_meta
    r1, r2 = st.columns(2, gap="small"), st.columns(2, gap="small")
    q1, q2, q3, q4 = r1[0], r1[1], r2[0], r2[1]
    if st.session_state.mode == "Demo":
        if q1.button("Run Analysis", icon=":material/play_circle:", key="qa_run", width="stretch",
                     help="Re-run the demo analysis with the saved models"):
            if run_demo(st):
                st.rerun()
    else:
        q1.button("Run Analysis", icon=":material/play_circle:", key="qa_run", width="stretch",
                  on_click=set_page, args=("Dataset Management",), help="Upload and validate a file, then run")
    q2.button("Upload Dataset", icon=":material/upload:", key="qa_upload", width="stretch", on_click=go_live_upload,
              help="Switch to Live Analysis and upload a CSV/Excel file")
    tpl = C.TEMPLATE_PATH.read_bytes() if C.TEMPLATE_PATH.exists() else b""
    q3.download_button("Download Template", tpl, file_name="retailiq_upload_template.csv", mime="text/csv",
                       icon=":material/description:", key="qa_tpl", width="stretch", disabled=not tpl,
                       on_click="ignore")
    pdf = b""
    if a is not None:
        try:
            pdf = S.build_pdf_report(meta, a, model_meta())
        except Exception:
            LOG.exception("PDF build failed")
    q4.download_button("Generate Report", pdf, file_name=f"retailiq_analysis_{meta.get('analysis_id') if meta else 'x'}.pdf",
                       mime="application/pdf", icon=":material/picture_as_pdf:", key="qa_pdf", width="stretch",
                       disabled=not pdf, on_click="ignore", help="PDF report of the current analysis")


def welcome() -> None:
    arts = artifacts()
    meta = arts["metadata"] if arts["state"] == "ready" else None
    md('<div class="riq-hero"><h2>Welcome to RetailIQ</h2><p>Next-day demand forecasting, rule-based inventory risk and '
       'K-Means product segmentation, built with classical machine learning. Choose how you want to start.</p></div>')
    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(key="card_choice_demo"):
        md('<div class="riq-choice"><div class="kicker">No upload needed</div><h3>Demo Mode</h3>'
           f'<p>Explore real outputs from the saved models on a prepared sample: 10 store-product series x '
           f'{C.SEGMENT_WINDOW_DAYS} days from the end of the dataset (inside the holdout test period).</p></div>')
        if st.session_state.mode != "Demo":
            st.button("Switch to Demo Mode", key="w_demo_switch", on_click=request_mode, args=("Demo",), width="stretch")
        elif st.button("Launch Demo Analysis", icon=":material/play_circle:", type="primary", key="w_demo",
                       width="stretch", disabled=arts["state"] != "ready"):
            if run_demo(st):
                st.rerun()
    with c2, st.container(key="card_choice_live"):
        md('<div class="riq-choice"><div class="kicker">Your data</div><h3>Live Analysis Mode</h3>'
           '<p>Upload a CSV or Excel file with the documented columns. It is validated first; forecasts use the saved '
           'models (nothing is retrained) and results are saved to local history.</p></div>')
        st.button("Start Live Analysis", icon=":material/upload:", key="w_live", width="stretch",
                  on_click=go_live_upload)

    st.write("")
    s1, s2, s3 = st.columns(3, gap="small")
    with s1, st.container(key="card_st_model", height=200):
        section("Model readiness")
        if meta:
            sel = meta["selected_model"]
            md(f'{chip("Loaded", "green")} <span class="riq-muted">&nbsp;{esc(meta["model_version"])}</span>'
               f'<div style="margin-top:10px">Selected: <b>{sel}</b><br>Holdout MAE {meta["test_metrics"][sel]["MAE"]:.2f} '
               f'&middot; R&sup2; {meta["test_metrics"][sel]["R2"]:.3f}<br><span class="riq-muted">Trained {esc(meta["trained_at"])}</span></div>')
        else:
            md(chip({"missing": "Not trained", "corrupt": "Load error", "incompatible": "Incompatible"}[arts["state"]], "red"))
            st.caption(" ".join(arts["problems"]))
            st.code(C.TRAIN_COMMAND, language="bash")
    with s2, st.container(key="card_st_db", height=200):
        section("Database")
        ok, msg = S.db_status()
        md(f'{chip("Connected" if ok else "Error", "green" if ok else "red")}<div style="margin-top:10px">{esc(msg)}<br>'
           f'<span class="riq-muted">Local SQLite file: database/retailiq.db</span></div>')
    with s3, st.container(key="card_st_demo", height=200):
        section("Demo dataset")
        if meta and C.DEMO_DATA_PATH.exists():
            d = meta["demo"]
            md(f'{chip("Available", "green")}<div style="margin-top:10px">{d["demo_rows"]} rows &middot; {d["demo_series"]} series<br>'
               f'{d["demo_date_start"]} to {d["demo_date_end"]}<br><span class="riq-muted">'
               f'{"Out-of-sample: after the training cutoff" if d["demo_out_of_sample"] else "Overlaps training period"}</span></div>')
        else:
            md(chip("Missing", "red"))
            st.caption(f"Created by `{C.TRAIN_COMMAND}`.")
    note("<b>Data note:</b> the bundled training data is a <b>synthetic</b> Kaggle dataset, not real retailer data. "
         "See <i>About &amp; Methodology</i> in the sidebar.")
    with st.container(key="card_hist_welcome"):
        section("Recent Analysis History")
        recent_history(limit=5)


# ==========================================================================
# PAGE 2 - Dataset Management
# ==========================================================================
def schema_expander() -> None:
    with st.expander("Upload schema and template", icon=":material/table:"):
        req = pd.DataFrame([{"Column": c, "Type": t, "Required": "Yes"} for c, t in C.REQUIRED_COLUMNS.items()])
        opt = pd.DataFrame([{"Column": c, "Type": "any", "Required": "Optional (accepted, not used by models)"}
                            for c in C.OPTIONAL_COLUMNS])
        st.dataframe(pd.concat([req, opt]), hide_index=True, width="stretch")
        st.markdown(
            f"- One row per **Date + Store ID + Product ID** (no duplicates), daily data.\n"
            f"- Forecasting a series needs its **{C.MIN_HISTORY_DAYS} most recent consecutive days**; a segment needs "
            f"**{C.SEGMENT_WINDOW_DAYS}**.\n- Numbers must be non-negative; Discount is a percent (0-100).\n"
            f"- Files: .csv or .xlsx, up to {C.MAX_UPLOAD_MB} MB. Missing values are not filled - fix them in the file.")
        if C.TEMPLATE_PATH.exists():
            st.download_button("Download CSV template", C.TEMPLATE_PATH.read_bytes(), "retailiq_upload_template.csv",
                               "text/csv", icon=":material/download:", on_click="ignore", key="tpl_ds")


def render_validation(v: ml.ValidationResult, raw: pd.DataFrame | None, name: str, size_bytes: int | None) -> None:
    status = chip("Valid - ready for analysis", "green") if v.ok else chip("Not valid - fix the issues below", "red")
    s = v.summary
    size = f"{size_bytes / 1024:,.1f} KB" if size_bytes else "-"
    md(f'<div style="margin:4px 0 10px"><b>{esc(name)}</b> &nbsp;<span class="riq-muted">{size} &middot; '
       f'{s.get("rows", 0):,} rows &middot; {len(s.get("columns", []))} columns</span> &nbsp; {status}</div>')
    for e in v.errors:
        st.error(e, icon=":material/error:")
    for w in v.warnings:
        st.warning(w, icon=":material/warning:")
    if "date_start" in s:
        cols = st.columns(5)
        for c, (lbl, val) in zip(cols, [("Date range", f"{s['date_start']} to {s['date_end']}"), ("Stores", s["stores"]),
                                        ("Products", s["products"]), ("Forecastable series", f"{s['forecastable_series']} / {s['series']}"),
                                        ("Segmentable series", f"{s['segmentable_series']} / {s['series']}")]):
            c.markdown(f'<div class="riq-muted">{lbl}</div><div style="font-weight:600">{val}</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Preview", "Columns & types", "Missing & duplicates", "Series continuity", "Preprocessing"])
    with tabs[0]:
        if raw is not None:
            st.dataframe(raw.head(200), hide_index=True, width="stretch", height=320)
            st.caption(f"First {min(200, len(raw))} of {len(raw):,} rows.")
    with tabs[1]:
        def role(c):
            if c in C.REQUIRED_COLUMNS:
                return "Required"
            if c in C.LEAKY_COLUMNS:
                return "Optional - never used (leakage)"
            if c in C.OPTIONAL_COLUMNS:
                return "Optional - not used by models"
            return "Unrecognised - ignored"
        dt = s.get("dtypes", {})
        st.dataframe(pd.DataFrame([{"Column": c, "Detected type": dt.get(c, "-"), "Role": role(c.strip())}
                                   for c in s.get("columns", [])]), hide_index=True, width="stretch")
    with tabs[2]:
        miss = s.get("missing_by_column", {})
        st.dataframe(pd.DataFrame({"Column": list(miss), "Missing values": list(miss.values())}),
                     hide_index=True, width="stretch")
        st.markdown(f"Exact duplicate rows: **{s.get('exact_duplicate_rows', '-')}** &nbsp;|&nbsp; "
                    f"Rows sharing Date + Store ID + Product ID: **{s.get('duplicate_key_rows', '-')}**")
    with tabs[3]:
        if v.series_table is not None:
            st.dataframe(v.series_table, hide_index=True, width="stretch", height=320)
        else:
            st.caption("Available once the basic checks pass.")
    with tabs[4]:
        for p in v.preprocessing or ["No preprocessing applied yet."]:
            st.markdown(f"- {p}")


def page_dataset() -> None:
    schema_expander()
    if st.session_state.mode == "Demo":
        with st.container(key="card_demo_ds"):
            section("Demo dataset", "prepared sample - clearly labelled demo data")
            raw, v = load_demo_validation()
            if v is None:
                st.error(f"Demo dataset missing. Run `{C.TRAIN_COMMAND}` to create {C.DEMO_DATA_PATH.name}.")
                return
            note(f"<b>Why 280 rows?</b> 10 store-product series x {C.SEGMENT_WINDOW_DAYS} consecutive days. Forecasting needs "
                 f"{C.MIN_HISTORY_DAYS} days of history per series and a segment profile needs {C.SEGMENT_WINDOW_DAYS}. "
                 "The days come from the end of the original file, inside the holdout period, so the demo's "
                 "actual-vs-predicted comparisons are out-of-sample.")
            render_validation(v, raw, C.DEMO_DATA_PATH.name, C.DEMO_DATA_PATH.stat().st_size)
            if st.button("Run Demo Analysis", icon=":material/play_circle:", type="primary", key="ds_run_demo",
                         disabled=not v.ok or artifacts()["state"] != "ready"):
                if run_demo(st):
                    st.rerun()
            st.caption("Switch to **Live Analysis** in the sidebar to upload your own file.")
        return

    with st.container(key="card_upload"):
        section("Upload a dataset", "CSV or Excel (.xlsx)")
        up = st.file_uploader("Choose a file", type=["csv", "xlsx"], key=f"uploader_{st.session_state.uploader_nonce}",
                              help=f"Max {C.MAX_UPLOAD_MB} MB. The file is processed in memory and not stored.")
        if up is None:
            st.session_state.upload = None
            st.caption("Waiting for a file. Download the template above to see the exact format.")
            return
        data = up.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        cached = st.session_state.upload
        if not cached or cached["hash"] != digest or cached["name"] != up.name:
            meta = model_meta()
            entry = {"hash": digest, "name": up.name, "size": len(data), "raw": None, "v": None, "error": None}
            try:
                with st.spinner("Reading and validating"):
                    raw = ml.read_table(data, up.name)
                    entry["raw"] = raw
                    entry["v"] = ml.validate_dataset(raw, meta["known_stores"] if meta else None,
                                                     meta["known_products"] if meta else None)
            except ml.UserFacingError as exc:
                entry["error"] = str(exc)
            except Exception:
                LOG.exception("Validation crashed for %s", up.name)
                entry["error"] = "The file could not be validated. Technical details were written to logs/retailiq.log."
            st.session_state.upload = entry
        entry = st.session_state.upload
        if entry["error"]:
            md(f'<b>{esc(entry["name"])}</b> &nbsp;{chip("Not valid", "red")}')
            st.error(entry["error"], icon=":material/error:")
            return
        render_validation(entry["v"], entry["raw"], entry["name"], entry["size"])
        ready = entry["v"].ok and artifacts()["state"] == "ready"
        if artifacts()["state"] != "ready":
            st.error(f"Models are not ready - run `{C.TRAIN_COMMAND}` first.")
        if st.button("Run Analysis", icon=":material/play_circle:", type="primary", key="ds_run_live", disabled=not ready,
                     help=None if ready else "Enabled once the file passes validation"):
            if run_and_store("Live", entry["name"], entry["v"].clean, entry["v"].summary, st):
                st.session_state.page = "Overview"
                st.rerun()


# ==========================================================================
# PAGE 3 - Demand Forecasting
# ==========================================================================
def page_forecast() -> None:
    a = require_analysis("next-day forecasts")
    if a is None:
        return
    s, f, bt = a["summary"], a["forecasts"], a["backtest"]
    meta = model_meta()
    cols = st.columns(4, gap="small")
    for c, (t, v, sub) in zip(cols, [
        ("Forecast date", s["forecast_date"], "the day being predicted"),
        ("Latest data date", s["date_end"], "last observed day in the data"),
        ("Series forecast", f"{s['forecast_series']} / {s['series']}", "need 7 consecutive recent days"),
        ("Model", meta["selected_model"] if meta else "-", (st.session_state.run_meta or {}).get("model_version", "")),
    ]):
        c.markdown(kpi("calendar" if "date" in t.lower() else "box", t, esc(v), esc(sub), small=True), unsafe_allow_html=True)
    st.write("")
    with st.container(key="card_fc_table"):
        section("Next-day forecasts", "filter, sort and export")
        c1, c2 = st.columns(2)
        stores = c1.multiselect("Stores", sorted(f[C.STORE_COL].unique()), placeholder="All stores")
        prods = c2.multiselect("Products", sorted(f[C.PRODUCT_COL].unique()), placeholder="All products")
        mask = pd.Series(True, index=f.index)
        if stores:
            mask &= f[C.STORE_COL].isin(stores)
        if prods:
            mask &= f[C.PRODUCT_COL].isin(prods)
        view = f[mask]
        table = pd.DataFrame({
            "Store ID": view[C.STORE_COL], "Product ID": view[C.PRODUCT_COL],
            "Latest data date": view["latest_date"].dt.date, "Forecast date": view["forecast_date"].dt.date,
            "Last observed sales": view["lag_1"], "7-day avg sales": view["roll_mean_7"],
            "PREDICTED units": view["predicted_units"].round(1),
        })
        st.dataframe(table, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(table)),
                     column_config={"7-day avg sales": st.column_config.NumberColumn(format="%.1f"),
                                    "Last observed sales": st.column_config.NumberColumn(format="%d"),
                                    "PREDICTED units": st.column_config.NumberColumn(format="%.1f", help="Model prediction, not an observation")})
        st.download_button("Export forecasts (CSV)", S.to_csv_bytes(table), "retailiq_next_day_forecasts.csv", "text/csv",
                           icon=":material/download:", on_click="ignore", key="dl_fc")

    left, right = st.columns([1.4, 1], gap="small")
    with left, st.container(key="card_fc_series", height=470):
        section("Series history and forecast")
        options = [f"{r[C.STORE_COL]} · {r[C.PRODUCT_COL]}" for _, r in view.iterrows()] or ["-"]
        pick = st.selectbox("Series", options, label_visibility="collapsed")
        if pick != "-":
            sid, pid = pick.split(" · ")
            hist = bt[(bt[C.STORE_COL] == sid) & (bt[C.PRODUCT_COL] == pid)].sort_values("target_date")
            nxt = f[(f[C.STORE_COL] == sid) & (f[C.PRODUCT_COL] == pid)]
            hist = hist.tail(120)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["target_date"], y=hist["actual_units"], name="Actual (observed)",
                                     mode="lines+markers", line=dict(color=TEXT, width=1.4), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=hist["target_date"], y=hist["predicted_units"], name="Predicted (backtest)",
                                     mode="lines", line=dict(color=AMBER, width=1.8, dash="dot")))
            fig.add_trace(go.Scatter(x=nxt["forecast_date"], y=nxt["predicted_units"], name="Next-day forecast",
                                     mode="markers", marker=dict(color=AMBER, size=13, symbol="diamond",
                                                                 line=dict(color="#101010", width=1))))
            show_fig(style_fig(fig, 360).update_yaxes(title="Units sold"), key="fc_series")
            st.caption("Actual-vs-predicted is shown only for days with real observations. The diamond is the "
                       "forecast for a day with no actual yet.")
    with right, st.container(key="card_fc_bt", height=470):
        section("Backtest on this data", "predictions vs known actuals")
        bm = s.get("backtest_metrics")
        if bm:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{bm['MAE']:.2f}", help=METRIC_HELP["MAE"])
            m2.metric("RMSE", f"{bm['RMSE']:.2f}", help=METRIC_HELP["RMSE"])
            m3.metric("R²", f"{bm['R2']:.3f}", help=METRIC_HELP["R2"])
            agg = bt.groupby("target_date")[["actual_units", "predicted_units"]].sum().reset_index()
            fig = go.Figure([
                go.Scatter(x=agg["target_date"], y=agg["actual_units"], name="Actual total", line=dict(color=TEXT, width=1.3)),
                go.Scatter(x=agg["target_date"], y=agg["predicted_units"], name="Predicted total",
                           line=dict(color=AMBER, width=1.8, dash="dot")),
            ])
            show_fig(style_fig(fig, 210), key="fc_agg")
            st.caption(f"{bm['n']:,} predictions with known actuals (all series, daily totals shown).")
            if s.get("backtest_overlaps_training"):
                st.warning("Part of this file falls inside the model's training period, so some of these "
                           "comparisons are in-sample and look better than true forecasts.", icon=":material/warning:")
        else:
            st.caption("No day in this file has both complete features and a known next-day actual.")
    note("<b>Reading these forecasts:</b> in this synthetic dataset, next-day sales are essentially unrelated to past sales "
         "(holdout R&sup2; &asymp; 0). The model therefore predicts values close to each series' typical level; it still beats "
         "naive rules such as &lsquo;same as yesterday&rsquo;. See Model Performance.")


# ========================================================================== 
# M5 Forecasting (separate Walmart dataset path)
# ========================================================================== 
def page_m5_forecasting() -> None:
    st.markdown("Forecast Walmart M5 product-store demand with the selected M5 model. "
                "This page reads only `data/m5/`; it does not use uploaded or synthetic RetailIQ data.")
    try:
        from m5_pipeline.retailiq_integration import build_next_day_forecasts, load_deployment_info
        deployment, model_meta, validation = load_deployment_info()
    except Exception as exc:
        st.warning(f"The separate M5 model is unavailable: {exc}")
        return

    st.caption(f"Model: **{deployment['model_version']}** · trained on {model_meta['training_rows_used']:,} sampled rows · "
               "selected using validation performance; final test excluded from selection. Stage 5 remains unpromoted.")
    validation_key = {"stage4": "stage4_new_features", "stage4_1": "stage4_1_same_weekday"}.get(deployment["model_variant"])
    if validation_key and validation_key in validation:
        metric = validation[validation_key]["overall"]
        cols = st.columns(3)
        cols[0].metric("Validation MAE", f"{metric['MAE']:.4f}")
        cols[1].metric("Validation RMSE", f"{metric['RMSE']:.4f}")
        cols[2].metric("Validation WAPE", f"{metric['WAPE_percent']:.2f}%")
    if deployment.get("test_metrics_path") and Path(deployment["test_metrics_path"]).is_file():
        try:
            test_metrics = json.loads(Path(deployment["test_metrics_path"]).read_text(encoding="utf-8"))
            test_key = "stage4_selected" if deployment["model_variant"] == "stage4" else "stage4_1_selected"
            if test_key in test_metrics:
                metric = test_metrics[test_key]["overall"]
                cols = st.columns(3)
                cols[0].metric("Final test MAE", f"{metric['MAE']:.4f}")
                cols[1].metric("Final test RMSE", f"{metric['RMSE']:.4f}")
                cols[2].metric("Final test WAPE", f"{metric['WAPE_percent']:.2f}%")
        except (OSError, ValueError, KeyError):
            st.info("Saved final-test metrics could not be loaded; forecast generation remains available.")
    st.info("M5 sales data has no inventory-level field. Inventory risk remains in the existing RetailIQ dataset workflow.")

    if st.button("Generate next-day M5 forecast", type="primary", key="m5_generate_forecast"):
        try:
            with st.spinner("Building cutoff-safe features for M5 product-store series…"):
                st.session_state.m5_forecasts = build_next_day_forecasts()
                st.session_state.m5_model_version = deployment["model_version"]
        except Exception as exc:
            LOG.exception("M5 forecast generation failed")
            st.error(f"M5 forecast generation failed: {exc}")
            return

    forecasts = st.session_state.get("m5_forecasts")
    if forecasts is not None:
        if st.session_state.get("m5_model_version") != deployment["model_version"]:
            st.session_state.m5_forecasts = None
            st.info("The selected M5 model changed. Generate a fresh forecast.")
            return
        section("Next-day Walmart M5 forecasts", f"target day d_{forecasts.attrs.get('forecast_day', '-')}")
        st.caption(f"{len(forecasts):,} product-store series · forecast date {forecasts.forecast_date.iloc[0].date()} · "
                   f"model {forecasts.attrs.get('model_version', deployment['model_version'])}")
        st.dataframe(forecasts.nlargest(100, "predicted_units"), hide_index=True, width="stretch")
        st.download_button("Download all M5 forecasts (CSV)",
                           forecasts.to_csv(index=False).encode("utf-8"),
                           "retailiq_m5_next_day_forecasts.csv", "text/csv",
                           key="m5_forecast_download", on_click="ignore")


@st.cache_data(show_spinner=False)
def _m5_performance_predictions(path: str):
    """Cache the saved holdout file and a deterministic sample used for point charts."""
    frame = pd.read_csv(path, usecols=["series_idx", "target_day", "actual", "stage4_lightgbm"])
    frame.rename(columns={"stage4_lightgbm": "predicted"}, inplace=True)
    daily = frame.groupby("target_day", as_index=False)[["actual", "predicted"]].sum()
    sample = frame.sample(n=min(20_000, len(frame)), random_state=42).sort_values(["series_idx", "target_day"])
    sample["residual"] = sample["predicted"] - sample["actual"]
    return daily, sample


def page_m5_performance() -> None:
    """Read-only model evidence page. It only reads persisted M5 artifacts."""
    from m5_pipeline.performance_report import comparison_table, series_extremes, tier_comparison

    root = Path(C.ROOT_DIR)
    output = root / "outputs" / "m5"
    model_dir = root / "models" / "m5" / "stage4"
    required = [root / "models/m5/retailiq_selected_model.json",
                model_dir / "m5_lightgbm_stage4.txt", model_dir / "m5_training_metadata.json",
                output / "m5_metrics.json", output / "stage4/results/stage4_metrics.json",
                output / "stage4/results/stage4_test_metrics.json",
                output / "stage4/results/stage4_validation_predictions.csv.gz",
                output / "stage4/results/stage4_test_predictions.csv.gz",
                output / "stage4/results/stage4_feature_importance.csv",
                output / "stage4_analysis/series_error_metrics.csv"]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        st.error("M5 performance evidence is incomplete. Missing: " + ", ".join(missing))
        return

    try:
        deployment = json.loads((root / "models/m5/retailiq_selected_model.json").read_text(encoding="utf-8"))
        metadata = json.loads((model_dir / "m5_training_metadata.json").read_text(encoding="utf-8"))
        comparison = comparison_table(root)
        val_tiers = tier_comparison(root, "Validation")
        test_tiers = tier_comparison(root, "Final test")
        feature_importance = pd.read_csv(output / "stage4/results/stage4_feature_importance.csv")
        val_daily, val_sample = _m5_performance_predictions(str(output / "stage4/results/stage4_validation_predictions.csv.gz"))
        test_daily, test_sample = _m5_performance_predictions(str(output / "stage4/results/stage4_test_predictions.csv.gz"))
        best, worst = series_extremes(output / "stage4_analysis/series_error_metrics.csv", "test")
    except Exception as exc:
        LOG.exception("Could not load saved M5 performance artifacts")
        st.error(f"Could not read the saved M5 evidence: {exc}")
        return

    st.title("M5 Model Performance")
    st.caption("Read-only report from saved artifacts. No training, tuning, or model selection is performed here.")
    if deployment.get("model_variant") != "stage4":
        st.warning("The selected-model manifest does not currently identify Stage 4.")
    else:
        st.success(f"Selected model: **{deployment.get('model_version', 'Stage 4 LightGBM')}** · Walmart M5 · Stage 5 is not promoted.")

    train_days = metadata.get("training_period_target_days", [None, None])
    val_days = metadata.get("validation_period_target_days", [None, None])
    test_days = [1914, 1941]
    summary_cols = st.columns(4)
    summary = [
        ("Training sample", f"{metadata.get('training_rows_used', 0):,} rows", f"{metadata.get('training_series_used', 0):,} of 30,490 series · {metadata.get('sample_policy', 'saved sampling policy') }"),
        ("Train period", f"d_{train_days[0]}–d_{train_days[1]}", "training target days"),
        ("Validation", f"d_{val_days[0]}–d_{val_days[1]}", f"{metadata.get('validation_rows', 0):,} rows"),
        ("Final test", f"d_{test_days[0]}–d_{test_days[1]}", "28-day holdout · evaluated after selection"),
    ]
    for col, (label, value, help_text) in zip(summary_cols, summary):
        col.metric(label, value, help=help_text)
    st.caption(f"Training metrics: not saved in the available artifacts. Best iteration: {metadata.get('best_iteration', 'not recorded')}; seed: {metadata.get('random_seed', 'not recorded')}; LightGBM model file: `{deployment.get('model_path', '')}`.")

    st.subheader("Model comparison")
    st.dataframe(comparison, hide_index=True, width="stretch", column_config={
        "MAE": st.column_config.NumberColumn(format="%.4f"), "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "WAPE (%)": st.column_config.NumberColumn(format="%.2f%%")})
    st.download_button("Download metric comparison (CSV)", comparison.to_csv(index=False).encode("utf-8"),
                       "m5_model_comparison.csv", "text/csv", key="m5_perf_metrics_csv", on_click="ignore")
    st.caption("Stage 4.1 is shown for validation only; it has no final-test result. The Seasonal Naive baseline has overall split metrics, but no saved demand-tier breakdown.")

    st.subheader("Actual and predicted demand")
    split_choice = st.radio("Saved split", ["Validation", "Final test"], horizontal=True, key="m5_perf_split")
    daily, sample = (val_daily, val_sample) if split_choice == "Validation" else (test_daily, test_sample)
    actual_pred = go.Figure([go.Scatter(x=daily.target_day, y=daily.actual, mode="lines+markers", name="Actual", line=dict(color=TEXT)),
                             go.Scatter(x=daily.target_day, y=daily.predicted, mode="lines+markers", name="Stage 4 predicted", line=dict(color=AMBER))])
    actual_pred.update_layout(title="All-series daily demand total", xaxis_title="M5 target day", yaxis_title="Units")
    show_fig(style_fig(actual_pred, 340), key="m5_actual_pred_daily")
    scatter = go.Figure(go.Scattergl(x=sample.actual, y=sample.predicted, mode="markers", name="Series-day rows",
                                    marker=dict(color=AMBER, size=4, opacity=.32),
                                    hovertemplate="Actual %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>"))
    maximum = float(max(sample.actual.max(), sample.predicted.max()))
    scatter.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line=dict(color=TEXT, dash="dash"))
    scatter.update_layout(title="Actual versus predicted · deterministic 20,000-row sample", xaxis_title="Actual units", yaxis_title="Predicted units")
    show_fig(style_fig(scatter, 340, legend=False), key="m5_actual_pred_scatter")
    left, right = st.columns(2)
    with left:
        hist = go.Figure(go.Histogram(x=sample.residual, nbinsx=60, marker_color=AMBER))
        hist.update_layout(title="Prediction error distribution", xaxis_title="Residual (predicted − actual)", yaxis_title="Sampled rows")
        show_fig(style_fig(hist, 310, legend=False), key="m5_residual_hist")
    with right:
        residual = go.Figure(go.Scattergl(x=sample.predicted, y=sample.residual, mode="markers",
                                          marker=dict(color=AMBER, size=4, opacity=.3)))
        residual.add_hline(y=0, line=dict(color=TEXT, dash="dash"))
        residual.update_layout(title="Residuals versus predicted demand", xaxis_title="Predicted units", yaxis_title="Residual")
        show_fig(style_fig(residual, 310, legend=False), key="m5_residual_vs_pred")
    st.caption("Point-level residual charts use a reproducible sample of at most 20,000 saved prediction rows. Daily totals use all rows in the saved file.")

    st.subheader("Stage 4 feature importance")
    if {"feature", "gain"}.issubset(feature_importance.columns):
        importance = feature_importance.nlargest(15, "gain").sort_values("gain")
        fig = go.Figure(go.Bar(x=importance.gain, y=importance.feature, orientation="h", marker_color=AMBER))
        fig.update_layout(title="Top 15 features by LightGBM gain", xaxis_title="Gain", yaxis_title="Feature")
        show_fig(style_fig(fig, max(360, 24 * len(importance) + 90), legend=False), key="m5_feature_importance")
    else:
        st.info("Saved feature importance file does not contain expected feature and gain columns.")

    st.subheader("Demand-tier performance")
    split_label = "Validation" if split_choice == "Validation" else "Final test"
    tier_data = val_tiers if split_choice == "Validation" else test_tiers
    st.dataframe(tier_data, hide_index=True, width="stretch", column_config={
        "MAE": st.column_config.NumberColumn(format="%.4f"), "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "WAPE (%)": st.column_config.NumberColumn(format="%.2f%%")})
    st.caption(f"Demand tiers use the persisted Stage 3 cutpoints for the {split_label.lower()} comparison. Stage 4.1 appears only in validation because no test evaluation was saved.")
    if not tier_data.empty:
        fig = go.Figure()
        for model in tier_data.Model.unique():
            part = tier_data[tier_data.Model.eq(model)]
            fig.add_trace(go.Bar(x=part["Demand tier"], y=part["WAPE (%)"], name=model))
        fig.update_layout(title=f"WAPE by demand tier · {split_label}", barmode="group", xaxis_title="Demand tier", yaxis_title="WAPE (%)")
        show_fig(style_fig(fig, 340), key="m5_tier_wape")

    st.subheader("Best- and worst-performing product-store series")
    left, right = st.columns(2)
    with left:
        st.markdown("**Lowest Stage 4 test MAE**")
        st.dataframe(best, hide_index=True, width="stretch")
    with right:
        st.markdown("**Highest Stage 4 test MAE**")
        st.dataframe(worst, hide_index=True, width="stretch")
    st.caption("Series ranking uses saved Stage 4 final-test per-series error summaries; it is descriptive reporting and was not used for model selection.")

    with st.expander("How to read the metrics and evidence"):
        st.markdown("""
        - **MAE** is the average absolute forecast error in units; lower is better.
        - **RMSE** is in units too, but squares errors first, so large misses weigh more heavily.
        - **WAPE** is total absolute error divided by total actual demand, shown as a percentage. It is **not an accuracy percentage** and does not mean `100 − WAPE` is accuracy. It can exceed 100%, especially when sparse products have little actual demand.
        - Negative mean residual indicates overall underprediction; positive indicates overprediction. Stage 4's saved test summary shows negative mean bias.
        - Results cover the saved M5 evaluation populations; the model used a 2,000,000-row global sample without replacement for training, spanning all 30,490 series. The Stage 3/4/5 comparison metrics are persisted; Stage 4.1 is validation-only.
        """)
    st.info("Strengths: Stage 4 reduces validation MAE and WAPE versus Stage 3 and improves low/medium-tier MAE/WAPE. Limits: low- and medium-demand WAPE remains around 100%, and Stage 4 validation RMSE is slightly higher than Stage 3. Recommended future work: test occurrence-plus-size or Croston-style intermittent models, validate per-store/category calibration, and model event/price changes with spike-aware features. Stage 4 remains selected; this page does not promote models.")


# ========================================================================== 
# PAGE 4 - Inventory Risk Analysis
# ==========================================================================
def page_risk() -> None:
    a = require_analysis("inventory risk")
    if a is None:
        return
    ins, counts = a["insights"], a["summary"]["risk_counts"]
    c = st.columns(3, gap="small")
    desc = {"High": f"coverage &lt; {C.RISK_HIGH_MAX_DAYS:g} day", "Medium": f"{C.RISK_HIGH_MAX_DAYS:g} to &lt; {C.RISK_MEDIUM_MAX_DAYS:g} days",
            "Low": f"&ge; {C.RISK_MEDIUM_MAX_DAYS:g} days or ~zero demand"}
    for col, lvl in zip(c, ["High", "Medium", "Low"]):
        col.markdown(f'<div class="riq-risk {lvl}"><div class="riq-muted">{lvl} risk</div><div class="n">{counts[lvl]}</div>'
                     f'<div class="riq-muted">{desc[lvl]}</div></div>', unsafe_allow_html=True)
    note(f"<b>How risk is calculated (rules, not a trained model):</b> coverage days = current inventory / average daily "
         f"units sold over the last {C.RISK_RECENT_DAYS} days of each series. If recent demand is below "
         f"{C.MIN_DEMAND_PER_DAY:g} unit/day, coverage is not computed (no division by ~zero) and the series is Low risk. "
         f"<b>Estimated replenishment</b> = units needed to reach {C.TARGET_COVER_DAYS:g} days of cover - an estimate, not a purchase order. "
         "The dataset has no stockout label, so no stockout probability is claimed. Thresholds live in <code>config.py</code>.")
    with st.container(key="card_risk_table"):
        section("Store-product inventory risk", "click a column header to sort")
        f1, f2 = st.columns([2, 1])
        levels = f1.pills("Risk level", ["High", "Medium", "Low"], selection_mode="multi", default=["High", "Medium", "Low"])
        view = ins[ins["risk_level"].isin(levels or [])].copy()
        view["_o"] = view["risk_level"].map({"High": 0, "Medium": 1, "Low": 2})
        view = view.sort_values(["_o", "coverage_days"], na_position="last")
        table = pd.DataFrame({
            "Store ID": view[C.STORE_COL], "Product ID": view[C.PRODUCT_COL], "Latest date": view["latest_date"],
            "Current inventory": view["current_inventory"], "Recent avg daily demand": view["recent_avg_demand"],
            "Coverage (days)": view["coverage_days"], "Risk level": view["risk_level"],
            "Est. replenishment (units)": view["est_replenishment_units"], "Attention": view["attention"],
            "Next-day forecast": view["next_day_forecast"].round(1),
        })
        styled = table.style.map(lambda v: f"color: {RISK_COLORS.get(v, TEXT)}; font-weight: 600", subset=["Risk level"]) \
            .format({"Current inventory": "{:,.0f}", "Recent avg daily demand": "{:.1f}", "Coverage (days)": "{:.2f}",
                     "Next-day forecast": "{:.1f}"}, na_rep="-")
        st.dataframe(styled, hide_index=True, width="stretch", height=min(460, 38 + 35 * max(len(table), 1)))
        f2.download_button("Export risk table (CSV)", S.to_csv_bytes(table), "retailiq_inventory_risk.csv", "text/csv",
                           icon=":material/download:", on_click="ignore", key="dl_risk", width="stretch")
    with st.container(key="card_risk_chart"):
        section("Coverage by series", "lowest 30 shown")
        cov = ins.dropna(subset=["coverage_days"]).sort_values("coverage_days").head(30)
        if cov.empty:
            st.caption("No series has measurable recent demand.")
        else:
            labels = cov[C.STORE_COL] + " · " + cov[C.PRODUCT_COL]
            fig = go.Figure(go.Bar(x=cov["coverage_days"], y=labels, orientation="h",
                                   marker_color=[RISK_COLORS[r] for r in cov["risk_level"]],
                                   hovertemplate="%{y}: %{x:.2f} days<extra></extra>"))
            for x, t in ((C.RISK_HIGH_MAX_DAYS, "High"), (C.RISK_MEDIUM_MAX_DAYS, "Medium")):
                fig.add_vline(x=x, line=dict(color=MUTED, dash="dash", width=1),
                              annotation_text=f"{t} < {x:g}d", annotation_font_color=MUTED)
            fig = style_fig(fig, max(260, 22 * len(cov) + 60), legend=False)
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Days of cover at recent demand")
            show_fig(fig, key="risk_bar")


# ==========================================================================
# PAGE 5 - Product Segmentation
# ==========================================================================
def page_segments() -> None:
    a = require_analysis("product segments")
    if a is None:
        return
    arts = artifacts()
    if arts["state"] != "ready":
        return models_missing_box()
    bundle, sg = arts["segmentation"], arts["metadata"]["segmentation"]
    seg = a["insights"].dropna(subset=["cluster"])
    k = bundle["k"]
    cols = st.columns(4, gap="small")
    for c, (t, v, sub) in zip(cols, [
        ("Segments (k)", str(k), "best silhouette within 3-6"),
        ("Silhouette", f"{bundle['silhouette'][k]:.3f}", "-1 to 1; higher = better separated"),
        ("Training profiles", f"{sg['profiles']:,}", f"{C.SEGMENT_WINDOW_DAYS}-day store-product windows"),
        ("Segmented here", f"{len(seg)} / {len(a['insights'])}", f"need {C.SEGMENT_WINDOW_DAYS} consecutive days"),
    ]):
        c.markdown(kpi("layers", t, v, sub, small=True), unsafe_allow_html=True)
    st.write("")
    if seg.empty:
        st.info(f"No series in this analysis has {C.SEGMENT_WINDOW_DAYS} consecutive recent days, so none could be "
                "assigned to a segment. Training results are shown below.", icon=":material/info:")
    left, right = st.columns([1.5, 1], gap="small")
    with left, st.container(key="card_seg_pca", height=500):
        section("PCA projection", f"PC1+PC2 explain {sum(bundle['pca_explained_variance']):.0%} of scaled variance")
        if seg.empty:
            seg_plot = pd.DataFrame(columns=["cluster", "pc1", "pc2", C.STORE_COL, C.PRODUCT_COL, "segment_label"])
        else:
            seg_plot = seg
        show_fig(pca_figure(seg_plot, 380, background=True), key="seg_pca")
        st.caption("Grey points: training profiles. Coloured points: series in this analysis. "
                   "PCA is for visualisation only, not model evaluation.")
    with right, st.container(key="card_seg_sizes", height=500):
        section("Segment sizes in this analysis")
        if not seg.empty:
            sizes = seg.groupby("cluster").size().reindex(range(1, k + 1), fill_value=0)
            fig = go.Figure(go.Bar(x=[f"Segment {i}" for i in sizes.index], y=sizes.values,
                                   marker_color=[SEG_COLORS[(i - 1) % len(SEG_COLORS)] for i in sizes.index],
                                   hovertemplate="%{x}: %{y} series<extra></extra>"))
            show_fig(style_fig(fig, 220, legend=False).update_yaxes(title="Series"), key="seg_sizes")
        for i, lbl in enumerate(bundle["labels"], start=1):
            md(f'<div style="margin:4px 0"><span class="riq-dot" style="background:{SEG_COLORS[(i - 1) % len(SEG_COLORS)]}"></span>'
               f'&nbsp; <b>Segment {i}</b> <span class="riq-muted">{esc(lbl)}</span></div>')

    with st.container(key="card_seg_profile"):
        section("Segment characteristics", "centroid averages from training profiles (original units)")
        centers = pd.DataFrame(bundle["centers"]).round(2)
        here = seg.groupby("cluster").size() if not seg.empty else pd.Series(dtype=int)
        centers["series_here"] = centers["cluster"].map(here).fillna(0).astype(int)
        centers.insert(1, "description", bundle["labels"])
        st.dataframe(centers.rename(columns={
            "cluster": "Segment", "description": "Auto description", "avg_units_sold": "Avg units sold/day",
            "demand_std": "Demand std", "avg_inventory": "Avg inventory", "sell_through_rate": "Sell-through",
            "avg_price": "Avg price", "avg_discount": "Avg discount %", "size_in_training": "Profiles (training)",
            "series_here": "Series (this analysis)"}), hide_index=True, width="stretch")
        note("<b>Interpretation:</b> descriptions are generated from each centroid's two most distinctive standardised "
             "features. They summarise measured averages and are not ground-truth categories. In this synthetic data "
             f"profiles differ only slightly (silhouette {bundle['silhouette'][k]:.2f}), so segments are weakly separated.")

    e1, e2 = st.columns(2, gap="small")
    ks = list(bundle["inertia"].keys())
    with e1, st.container(key="card_elbow"):
        section("Elbow method", "inertia (within-cluster sum of squares)")
        fig = go.Figure(go.Scatter(x=ks, y=[bundle["inertia"][x] for x in ks], mode="lines+markers",
                                   line=dict(color=AMBER), marker=dict(size=[13 if x == k else 7 for x in ks])))
        show_fig(style_fig(fig, 250, legend=False).update_xaxes(title="k", dtick=1).update_yaxes(title="Inertia"), key="elbow")
    with e2, st.container(key="card_sil"):
        section("Silhouette score", f"chosen k = {k}")
        fig = go.Figure(go.Bar(x=ks, y=[bundle["silhouette"][x] for x in ks],
                               marker_color=[AMBER if x == k else ("#5E5E5E" if x in C.K_RANGE_ALLOWED else "#333") for x in ks]))
        show_fig(style_fig(fig, 250, legend=False).update_xaxes(title="k", dtick=1).update_yaxes(title="Silhouette"), key="sil")
    st.caption("k is chosen as the highest silhouette within 3-6 segments (k=2 is shown but excluded as too coarse to be useful).")

    with st.container(key="card_seg_series"):
        section("Series and their segments")
        if not seg.empty:
            st.dataframe(pd.DataFrame({
                "Store ID": seg[C.STORE_COL], "Product ID": seg[C.PRODUCT_COL], "Segment": seg["cluster"],
                "Description": seg["segment_label"], "Avg units sold": seg["avg_units_sold"].round(1),
                "Demand std": seg["demand_std"].round(1), "Avg inventory": seg["avg_inventory"].round(0),
                "Sell-through": seg["sell_through_rate"].round(3), "Avg price": seg["avg_price"].round(2),
                "Avg discount %": seg["avg_discount"].round(1), "PC1": seg["pc1"].round(2), "PC2": seg["pc2"].round(2),
            }), hide_index=True, width="stretch")
        with st.expander("PCA loadings (how each feature contributes to PC1 / PC2)"):
            st.dataframe(pd.DataFrame(bundle["pca_loadings"]), width="stretch")


# ==========================================================================
# PAGE 6 - Model Performance
# ==========================================================================
def page_performance() -> None:
    meta = model_meta()
    if not meta:
        return models_missing_box()
    sel, tr = meta["selected_model"], meta["training"]
    cols = st.columns(4, gap="small")
    for c, (t, v, sub) in zip(cols, [
        ("Selected model", sel, "lowest validation MAE"),
        ("Model version", meta["model_version"], f"trained {meta['trained_at'].replace('T', ' ')}"),
        ("Training period", f"{tr['train_target_start']}<br>to {tr['train_target_end']}", f"{tr['train_rows']:,} rows"),
        ("Holdout test period", f"{tr['test_target_start']}<br>to {tr['test_target_end']}", f"{tr['test_rows']:,} rows · final 20% of dates"),
    ]):
        c.markdown(kpi("box" if c is cols[0] else "calendar", t, v, esc(sub), small=True), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.1, 1], gap="small")
    with left, st.container(key="card_perf_main", height=430):
        section("Holdout test results", "never used for selection or tuning")
        m = meta["test_metrics"][sel]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"MAE ({sel})", f"{m['MAE']:.2f}", help=METRIC_HELP["MAE"])
        m2.metric("RMSE", f"{m['RMSE']:.2f}", help=METRIC_HELP["RMSE"])
        m3.metric("R²", f"{m['R2']:.4f}", help=METRIC_HELP["R2"])
        perf_table(compact=False)
    with right, st.container(key="card_perf_chart", height=430):
        section("Actual vs predicted on the holdout", "daily totals, all 100 series")
        d = pd.DataFrame(meta["holdout_daily_totals"])
        fig = go.Figure([
            go.Scatter(x=d["target_date"], y=d["actual"], name="Actual", line=dict(color=TEXT, width=1.2)),
            go.Scatter(x=d["target_date"], y=d["predicted"], name=f"Predicted ({sel})", line=dict(color=AMBER, width=2)),
        ])
        show_fig(style_fig(fig, 340).update_yaxes(title="Units sold (sum)"), key="perf_holdout")
    note("<b>Why is R&sup2; about zero?</b> The data inspection found that next-day Units Sold is uncorrelated with past sales "
         "(lag 1, lag 7 and rolling means all have correlation &asymp; 0.00), as expected from synthetic, independently generated "
         "rows. No honest next-day model can explain much variance here, so both models learn each series' typical level. "
         "They still beat naive rules (yesterday's sales, 7-day mean) on MAE. The leaky <i>Demand Forecast</i> column would "
         "give R&sup2; &asymp; 0.99, which is why it is excluded.")

    c1, c2 = st.columns(2, gap="small")
    with c1, st.container(key="card_perm", height=460):
        section("Feature importance (selected model)", "permutation, holdout")
        pi = meta["permutation_importance_selected"]
        names = list(pi)[::-1]
        fig = go.Figure(go.Bar(x=[pi[n]["mean"] for n in names], y=names, orientation="h",
                               error_x=dict(type="data", array=[pi[n]["std"] for n in names], color=MUTED),
                               marker_color=[AMBER if pi[n]["mean"] > 0 else "#5E5E5E" for n in names]))
        show_fig(style_fig(fig, 290, legend=False).update_xaxes(title="Increase in MAE when the feature is shuffled"), key="perm")
        st.caption("Values near 0 (or negative) mean shuffling the feature does not hurt accuracy - consistent with "
                   "the absence of signal. Importance shows association, not causation.")
    with c2, st.container(key="card_imp", height=460):
        section("RF impurity importance", "one-hot columns summed per input")
        ri = meta["rf_impurity_importance"]
        names = list(ri)[::-1]
        fig = go.Figure(go.Bar(x=[ri[n] for n in names], y=names, orientation="h", marker_color=AMBER2))
        show_fig(style_fig(fig, 290, legend=False).update_xaxes(title="Share of impurity reduction"), key="imp")
        st.caption("Impurity importance is measured on training data and favours continuous features even when they "
                   "carry no real signal, so read it together with the permutation chart.")

    with st.container(key="card_cv"):
        section("Validation and tuning", "date-grouped expanding-window folds inside the training period")
        t1, t2, t3 = st.tabs(["Cross-validation", "Random Forest grid", "Features & exclusions"])
        with t1:
            folds = pd.DataFrame(meta["cv_folds"])
            folds.index = [f"Fold {i + 1}" for i in range(len(folds))]
            st.dataframe(folds, width="stretch")
            cv = pd.DataFrame(meta["cv_scores"]).T.round(4)
            cv.index.name = "Model (mean of folds)"
            st.dataframe(cv, width="stretch")
            st.caption(f"Selection rule: {meta['selection_rule']}. Predictions are clipped at 0.")
        with t2:
            g = pd.DataFrame([{**r["params"], "CV MAE": round(r["cv_MAE"], 3)} for r in meta["rf_grid"]])
            st.dataframe(g, hide_index=True, width="stretch")
            st.caption(f"Best parameters: {meta['rf_best_params']} (GridSearchCV scored on validation folds only).")
        with t3:
            st.markdown("**Inputs used:** " + ", ".join(f"`{c}`" for c in meta["feature_columns"]))
            st.markdown("- `lag_1` = sales on the cutoff day (target - 1); `lag_7` = sales on target - 7\n"
                        "- `roll_mean_7` / `roll_std_7` = mean / std of the 7 days ending on the cutoff day\n"
                        "- `target_dow` / `target_month` = calendar of the forecast day\n"
                        "- Store ID, Product ID, weekday and month are one-hot encoded (unknown IDs ignored safely); "
                        "numeric features are standardised. The encoder is saved inside the model pipeline.")
            st.dataframe(pd.DataFrame([{"Excluded": k, "Reason": v} for k, v in meta["excluded_columns"].items()]),
                         hide_index=True, width="stretch")
            ins = meta["data_inspection"]
            st.markdown(
                f"**Verified data checks:** {ins['shape'][0]:,} rows x {ins['shape'][1]} columns; {ins['date_start']} to "
                f"{ins['date_end']}; {len(ins['stores'])} stores x {len(ins['products'])} products; missing values "
                f"{ins['missing_values_total']}; duplicate rows {ins['duplicate_rows']}; every series daily and continuous: "
                f"{ins['all_series_daily_continuous']}; corr(Demand Forecast, Units Sold) = "
                f"{ins['corr_demand_forecast_vs_units_sold']:.4f}; explicit stockout label present: "
                f"{ins['explicit_stockout_label_present']}.")


# ==========================================================================
# PAGE 7 - History & Reports
# ==========================================================================
def page_history() -> None:
    try:
        runs = S.list_runs()
    except Exception:
        LOG.exception("list_runs failed")
        st.error("The history database could not be read. Technical details were written to logs/retailiq.log.")
        return
    note("<b>Storage &amp; privacy:</b> results are stored locally in <code>database/retailiq.db</code> (SQLite). "
         "Uploaded files themselves are never stored. Deleting here removes the saved results permanently.")
    if runs.empty:
        empty_state("No saved analyses yet", "Every successful Demo or Live analysis is saved here automatically.")
        return
    with st.container(key="card_hist_list"):
        h1, h2 = st.columns([3, 1])
        with h1:
            section("Saved analyses", f"{len(runs)} total")
        flt = h2.segmented_control("Mode filter", ["All", "Demo", "Live"], default="All", label_visibility="collapsed")
        view = runs if flt in (None, "All") else runs[runs["mode"] == flt]
        st.dataframe(view.rename(columns={
            "analysis_id": "ID", "created_at": "Created", "mode": "Mode", "source_name": "Source", "record_count": "Records",
            "forecast_date": "Forecast date", "latest_data_date": "Latest data date", "model_version": "Model version",
            "status": "Status"}), hide_index=True, width="stretch", height=min(380, 38 + 35 * max(len(view), 1)))
        if view.empty:
            st.caption("No analyses for this filter.")
            return
        labels = {int(r.analysis_id): f"#{r.analysis_id} · {str(r.created_at).replace('T', ' ')} · {r.mode} · {r.source_name}"
                  for r in view.itertuples()}
        aid = st.selectbox("Select an analysis", list(labels), format_func=labels.get)
        loaded = S.load_run(aid)
        if loaded is None:
            st.error("This analysis no longer exists.")
            return
        meta, analysis = loaded
        analysis = normalise(analysis)
        b = st.columns(5, gap="small")
        b[0].button("Open analysis", icon=":material/visibility:", type="primary", width="stretch",
                    on_click=open_run, args=(aid,), key="h_open")
        fc = analysis["forecasts"].assign(forecast_date=lambda d: d["forecast_date"].dt.date,
                                          latest_date=lambda d: d["latest_date"].dt.date)
        b[1].download_button("Forecasts CSV", S.to_csv_bytes(fc), f"retailiq_{aid}_forecasts.csv", "text/csv",
                             icon=":material/download:", width="stretch", on_click="ignore", key="h_fc")
        b[2].download_button("Insights CSV", S.to_csv_bytes(analysis["insights"]), f"retailiq_{aid}_insights.csv",
                             "text/csv", icon=":material/download:", width="stretch", on_click="ignore", key="h_ins")
        try:
            pdf = S.build_pdf_report(meta, analysis, model_meta())
        except Exception:
            LOG.exception("PDF failed")
            pdf = b""
        b[3].download_button("PDF report", pdf, f"retailiq_{aid}_report.pdf", "application/pdf",
                             icon=":material/picture_as_pdf:", width="stretch", on_click="ignore", key="h_pdf", disabled=not pdf)
        b[4].button("Delete", icon=":material/delete:", width="stretch", key="h_del",
                    on_click=lambda: st.session_state.update(confirm=("delete", aid)))
        s = analysis["summary"]
        st.caption(f"{s['records']:,} records · {s['series']} series · {s['date_start']} to {s['date_end']} · forecast date "
                   f"{s['forecast_date']} · risk H/M/L {s['risk_counts']['High']}/{s['risk_counts']['Medium']}/{s['risk_counts']['Low']}"
                   f" · model {meta['model_version']}")
    st.button("Clear all history", icon=":material/delete_sweep:", key="h_clear",
              on_click=lambda: st.session_state.update(confirm=("clear", None)))


# ==========================================================================
# Main
# ==========================================================================
ROUTES = {
    "Overview": page_overview, "Dataset Management": page_dataset, "Demand Forecasting": page_forecast,
    "Inventory Risk Analysis": page_risk, "Product Segmentation": page_segments,
    "Model Performance": page_performance, "History & Reports": page_history,
    "M5 Forecasting": page_m5_forecasting,
    "M5 Model Performance": page_m5_performance,
}


def main() -> None:
    init_state()
    if C.CSS_PATH.exists():
        md(f"<style>{C.CSS_PATH.read_text(encoding='utf-8')}</style>")
    sidebar()
    header()
    flash = st.session_state.flash
    if flash:
        kind, msg = flash
        st.toast(msg, icon={"ok": ":material/check_circle:", "warn": ":material/warning:", "err": ":material/error:"}[kind])
        st.session_state.flash = None
    try:
        ROUTES[st.session_state.page]()
    except Exception:
        LOG.exception("Page %s failed", st.session_state.page)
        st.error("Something went wrong while showing this page. Technical details were written to logs/retailiq.log. "
                 "Try Reset analysis in the sidebar.")
    # at most one dialog per run
    if st.session_state.pending_mode:
        confirm_mode_dialog(st.session_state.pending_mode)
    elif st.session_state.confirm:
        confirm_action_dialog(*st.session_state.confirm)
    elif st.session_state.show_about:
        st.session_state.show_about = False
        about_dialog()


main()
