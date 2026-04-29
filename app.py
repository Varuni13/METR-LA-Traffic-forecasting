from datetime import datetime

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error
from streamlit_folium import st_folium

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="METR-LA Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────── */
.stApp { background-color: #080C14; }
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
section[data-testid="stSidebarNav"]{ display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
header[data-testid="stHeader"]     { display: none !important; }
.stMainBlockContainer              { max-width: 100% !important; padding: 1rem 2rem !important; }

/* ── Pill nav ──────────────────────────────────────────────── */
div[data-testid="stRadio"] { margin-top: 14px !important; }
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex !important; flex-direction: row !important;
  gap: 6px !important; flex-wrap: nowrap !important; background: transparent !important;
}
div[data-testid="stRadio"] label {
  padding: 6px 18px !important; border-radius: 20px !important;
  background: #0D1A2D !important; border: 1px solid #1E3A5F !important;
  color: #7A9CC0 !important; font-size: 0.83rem !important;
  font-weight: 500 !important; cursor: pointer !important;
  white-space: nowrap !important; transition: all 0.15s !important;
  margin-bottom: 0 !important;
}
div[data-testid="stRadio"] label:hover {
  background: #132540 !important; border-color: #2E5090 !important; color: #A8C0E8 !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: #1A4A8A !important; border-color: #4A90D9 !important;
  color: #E8F0FE !important; font-weight: 700 !important;
}
div[data-testid="stRadio"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] label p { margin: 0 !important; }

/* ── KPI cards — FIXED: equal height via flex ─────────────── */
.kpi-row {
  display: flex;
  gap: 12px;
  align-items: stretch;      /* all cards stretch to same height */
  margin-bottom: 1.2rem;
}
.kpi-card {
  flex: 1;
  min-height: 110px;         /* enforce minimum height */
  display: flex;
  flex-direction: column;
  justify-content: center;
  background: linear-gradient(135deg, #0D1A2D, #152035);
  border: 1px solid #1E3A5F;
  border-top: 3px solid #4A90D9;
  border-radius: 10px;
  padding: 1.2rem;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
  box-sizing: border-box;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(74,144,217,0.2); }
.kpi-card.green  { border-top-color: #27AE60; }
.kpi-card.yellow { border-top-color: #F39C12; }
.kpi-card.red    { border-top-color: #E74C3C; }
.kpi-card.blue   { border-top-color: #4A90D9; }
.kpi-card.purple { border-top-color: #7B68EE; }
.kpi-val   { font-size: 2rem; font-weight: 800; color: #4A90D9; letter-spacing: -1px; }
.kpi-lbl   { font-size: 0.72rem; color: #6B8CAE; margin-top: 4px;
             text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: 0.68rem; color: #27AE60; margin-top: 3px; min-height: 16px; }

/* ── Action boxes — FIXED: equal height ───────────────────── */
.action-row {
  display: flex;
  gap: 12px;
  align-items: stretch;
  margin-bottom: 1.2rem;
}
.action-box {
  flex: 1;
  min-height: 130px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  border-radius: 8px;
  padding: 14px 16px;
  box-sizing: border-box;
}
.action-box.red {
  background: #1A0808; border: 1px solid #5C1515; border-left: 3px solid #E74C3C;
}
.action-box.yellow {
  background: #1A1408; border: 1px solid #5C4A15; border-left: 3px solid #F39C12;
}
.action-box.blue {
  background: #081525; border: 1px solid #1E3A6A; border-left: 3px solid #4A90D9;
}
.action-label {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.8px; margin-bottom: 8px;
}
.action-box.red    .action-label { color: #E74C3C; }
.action-box.yellow .action-label { color: #F39C12; }
.action-box.blue   .action-label { color: #4A90D9; }
.action-body { color: #A8C0E8; font-size: 0.8rem; line-height: 1.6; }
.action-body b { color: #E8F0FE; }

/* ── Insight box ──────────────────────────────────────────── */
.insight-box {
  background: #0D1A2D; border: 1px solid #1E3A5F;
  border-left: 3px solid #4A90D9; border-radius: 8px;
  padding: 1rem 1.2rem; color: #A8C0E8; line-height: 1.6;
}

/* ── Section header ───────────────────────────────────────── */
.section-header {
  font-size: 1rem; font-weight: 600; color: #E8F0FE;
  margin: 1.5rem 0 0.8rem 0; padding-bottom: 6px;
  border-bottom: 1px solid #1E2D45;
}

/* ── Status badge ─────────────────────────────────────────── */
.status-live {
  display: inline-flex; align-items: center; gap: 5px;
  background: #0A2A0A; border: 1px solid #27AE60;
  border-radius: 20px; padding: 3px 10px;
  font-size: 0.7rem; color: #27AE60;
}
.status-dot {
  width: 6px; height: 6px; background: #27AE60;
  border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Dataframe / metrics ──────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
div[data-testid="metric-container"] {
  background: #0D1A2D; border: 1px solid #1E3A5F;
  border-radius: 10px; padding: 12px;
}
div[data-testid="metric-container"]:hover { border-color: #4A90D9; }

/* ── Streamlit inner tabs ─────────────────────────────────── */
div[data-baseweb="tab-list"] { background: #0D1117; border-bottom: 1px solid #1E2D45; }
div[data-baseweb="tab"]      { color: #6B8CAE; }
div[aria-selected="true"]    { color: #4A90D9 !important; border-bottom-color: #4A90D9 !important; }

/* ── Plotly ──────────────────────────────────────────────── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* ── Equal-height columns ────────────────────────────────── */
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
[data-testid="column"] { display: flex !important; flex-direction: column !important; }
[data-testid="column"] > div:first-child {
  display: flex !important; flex-direction: column !important; flex: 1 !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4A90D9; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
@st.cache_data
def load_all_data():
    predictions = np.load("data/gnn_v3_predictions.npy")
    targets     = np.load("data/gnn_v3_targets.npy")
    X_test      = np.load("data/X_test.npy")
    sensor_df   = pd.read_csv("data/sensor_analysis.csv")
    results_df  = pd.read_csv("data/all_results_v3.csv")
    sensor_mae  = [
        mean_absolute_error(targets[:, i, 2], predictions[:, i, 2])
        for i in range(207)
    ]
    sensor_df["prediction_mae"] = sensor_mae
    return predictions, targets, X_test, sensor_df, results_df

predictions, targets, X_test, sensor_df, results_df = load_all_data()

@st.cache_data
def build_model_diagnostics(pred: np.ndarray, true: np.ndarray):
    horizon_to_idx = {"5min": 0, "15min": 2, "30min": 5}
    horizon_rows, sensor_frames = [], []
    for horizon, h_idx in horizon_to_idx.items():
        pred_h = pred[:, :, h_idx]
        true_h = true[:, :, h_idx]
        abs_err = np.abs(true_h - pred_h)
        horizon_rows.append({
            "Horizon": horizon,
            "MAE":  float(abs_err.mean()),
            "RMSE": float(np.sqrt(np.mean((true_h - pred_h) ** 2))),
        })
        sensor_frames.append(pd.DataFrame({
            "Sensor ID": np.arange(abs_err.shape[1]),
            "Horizon":   horizon,
            "MAE":       abs_err.mean(axis=0),
        }))
    return pd.DataFrame(horizon_rows), pd.concat(sensor_frames, ignore_index=True)

sensor_df["index"] = sensor_df["index"].astype(int).clip(0, 206)
horizon_metrics_df, sensor_horizon_df = build_model_diagnostics(predictions, targets)

# ---------------------------------------------------------------------
# DYNAMIC STATS
# ---------------------------------------------------------------------
def _compute_dashboard_stats(results_df, sensor_df, X_test, predictions, targets):
    stats = {}
    _res = results_df.copy()

    def _fmt(h):
        k = str(h).replace(" ", "").lower()
        return "5min" if k in {"5min","5m","5"} else \
               "15min" if k in {"15min","15m","15"} else \
               "30min" if k in {"30min","30m","30"} else str(h)

    _res["_h"] = _res["Horizon"].apply(_fmt)
    _tgcn = _res[_res["Model"] == "T-GCN-V3"].set_index("_h")
    _rf   = _res[_res["Model"] == "RandomForest"].set_index("_h")

    for h in ["5min", "15min", "30min"]:
        if h in _tgcn.index:
            stats[f"tgcn_mae_{h}"]  = float(_tgcn.loc[h, "MAE"])
            stats[f"tgcn_rmse_{h}"] = float(_tgcn.loc[h, "RMSE"])
        if h in _rf.index:
            stats[f"rf_mae_{h}"] = float(_rf.loc[h, "MAE"])

    for h in ["5min", "15min", "30min"]:
        if f"tgcn_mae_{h}" in stats and f"rf_mae_{h}" in stats:
            imp = (stats[f"rf_mae_{h}"] - stats[f"tgcn_mae_{h}"]) / stats[f"rf_mae_{h}"] * 100
            stats[f"imp_{h}"] = imp

    stats["best_5min"]  = "T-GCN" if stats.get("imp_5min",  0) > 0 else "RF"
    stats["best_15min"] = "T-GCN" if stats.get("imp_15min", 0) > 0 else "RF"
    stats["best_30min"] = "T-GCN" if stats.get("imp_30min", 0) > 0 else "RF"

    critical_mask = sensor_df["congestion_level"] == "Critical"
    stats["n_critical"]      = int(critical_mask.sum())
    stats["congestion_rate"] = float(sensor_df["congestion_rate"].mean()) * 100

    fully_congested = sensor_df[sensor_df["congestion_rate"] >= 0.999]["index"].tolist()
    stats["fully_congested"] = fully_congested

    worst_idx = int(sensor_df.loc[sensor_df["prediction_mae"].idxmax(), "index"])
    worst_mae = float(sensor_df["prediction_mae"].max())
    stats["worst_sensor_idx"] = worst_idx
    stats["worst_sensor_mae"] = worst_mae

    stats["overall_speed"] = float(X_test[:, :, -1].mean())
    stats["congested_pct"] = float((X_test[:, :, -1] < -0.3).mean() * 100)
    return stats

_ds = _compute_dashboard_stats(results_df, sensor_df, X_test, predictions, targets)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def dark_layout(title="", height=350, legend_bottom=True):
    return dict(
        title=dict(text=title, font=dict(color="#E8F0FE", size=13)),
        paper_bgcolor="#0D1A2D", plot_bgcolor="#0D1A2D",
        font=dict(color="#8899AA", size=11),
        height=height,
        margin=dict(l=40, r=20, t=40, b=60 if legend_bottom else 20),
        xaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45", tickfont=dict(color="#8899AA")),
        yaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45", tickfont=dict(color="#8899AA")),
        legend=(dict(orientation="h", y=-0.25, font=dict(color="#8899AA"), bgcolor="rgba(0,0,0,0)")
                if legend_bottom else dict(font=dict(color="#8899AA"))),
        hoverlabel=dict(bgcolor="#152035", font_color="white", bordercolor="#4A90D9"),
    )

def format_horizon_for_table(h: str) -> str:
    key = str(h).replace(" ", "").lower()
    if key in {"5min","5m","5"}:    return "5min"
    if key in {"15min","15m","15"}: return "15min"
    if key in {"30min","30m","30"}: return "30min"
    return str(h)

def compute_snapshot_label(snapshot_idx: int) -> str:
    mins = snapshot_idx * 5
    return f"{(mins // 60) % 24:02d}:{mins % 60:02d}"

def get_day_type(snapshot_idx: int) -> str:
    """Estimate weekday vs weekend based on snapshot position in test set.
    Test set starts after ~70% of data. Assuming data starts Monday.
    288 snapshots per day, 5 weekdays + 2 weekend days per week."""
    total_day = (snapshot_idx + int(0.7 * 34255)) // 288
    day_of_week = total_day % 7   # 0=Mon … 6=Sun
    return "Weekend" if day_of_week >= 5 else "Weekday"

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "snapshot_idx" not in st.session_state:
    st.session_state.snapshot_idx = 100

# ---------------------------------------------------------------------
# HEADER + NAV
# ---------------------------------------------------------------------
st.markdown(f"""
<div style='background:linear-gradient(135deg,#0F2444 0%,#1B3A6B 50%,#0F2444 100%);
            border:1px solid #2E5090;border-radius:10px;padding:0.9rem 1.6rem;
            margin-bottom:0;position:relative;overflow:hidden'>
  <div style='position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,#4A90D9,#7B68EE,#4A90D9)'></div>
  <div style='display:flex;justify-content:space-between;align-items:center;gap:1rem'>
    <div style='display:flex;align-items:center;gap:12px'>
      <span style='font-size:1.5rem'>🚦</span>
      <div>
        <span style='color:#E8F0FE;font-weight:800;font-size:1.35rem;letter-spacing:-0.5px'>
          METR-LA Traffic Intelligence</span>
        <div style='color:#6B8CAE;font-size:0.72rem;margin-top:2px'>
          Graph Neural Network &nbsp;·&nbsp; 207 LA highway sensors &nbsp;·&nbsp; 5, 15 &amp; 30-min forecasts
        </div>
      </div>
      <span class='status-live'><span class='status-dot'></span>Live</span>
    </div>
    <div style='text-align:right;flex-shrink:0'>
      <div style='color:#6B8CAE;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px'>Last updated</div>
      <div style='color:#4A90D9;font-size:1.1rem;font-weight:700'>{datetime.now().strftime('%H:%M:%S')}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

NAV_OPTIONS = [
    "Problem & Overview",
    "Live Traffic Network",
    "Forecast Explorer",
    "Decision Dashboard",
    "Model Results",
]
NAV_MAP = {
    "Problem & Overview":   "Overview",
    "Live Traffic Network": "Live Traffic Map",
    "Forecast Explorer":    "Forecast Explorer",
    "Decision Dashboard":   "Decision Dashboard",
    "Model Results":        "Model Results",
}
NAV_MAP_INV = {v: k for k, v in NAV_MAP.items()}

selected_label = st.radio(
    "Navigation", NAV_OPTIONS,
    index=NAV_OPTIONS.index(NAV_MAP_INV.get(st.session_state.page, "Problem & Overview")),
    horizontal=True, label_visibility="collapsed", key="main_nav",
)
st.session_state.page = NAV_MAP[selected_label]
st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

# =====================================================================
# PAGE 1 — OVERVIEW  (redesigned 3-zone layout)
# =====================================================================
if st.session_state.page == "Overview":

    # ── ZONE 1: 5 KPI cards — equal height flex row ────────────────────
    kpis = [
        ("207",                                 "Highway Sensors",    "LA highway network",        "blue",   "#4A90D9"),
        (f"{_ds.get('tgcn_mae_5min',0):.3f}",  "MAE at 5-min",       f"Best: {_ds['best_5min']}",  "green",  "#27AE60"),
        ("0.672",                               "Spatial Correlation","Justifies graph model",      "purple", "#7B68EE"),
        (str(_ds["n_critical"]),                "Critical Zones",     "Require immediate action",   "red",    "#E74C3C"),
        (f"{_ds['congestion_rate']:.1f}%",      "Congestion Rate",    "Of all observations",        "yellow", "#F39C12"),
    ]
    cards_html = "".join(
        f"<div class='kpi-card {accent}'>"
        f"  <div class='kpi-val' style='color:{color}'>{val}</div>"
        f"  <div class='kpi-lbl'>{lbl}</div>"
        f"  <div class='kpi-delta'>{sub}</div>"
        f"</div>"
        for val, lbl, sub, accent, color in kpis
    )
    st.markdown(f"<div class='kpi-row'>{cards_html}</div>", unsafe_allow_html=True)

    # thin divider
    st.markdown("<div style='border-bottom:1px solid #1E2D45;margin-bottom:1rem'></div>",
                unsafe_allow_html=True)

    # ── ZONE 2: Chart (left, wide) | Metrics panel (right, narrow) ─────
    col_chart, col_metrics = st.columns([3, 2], gap="medium")

    # — Left: insight text + MAE comparison chart ——————————————
    with col_chart:
        st.markdown("""
        <div style='background:#0D1A2D;border:1px solid #1E3A5F;border-left:3px solid #4A90D9;
                    border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:10px;color:#A8C0E8;
                    font-size:0.82rem;line-height:1.6'>
          <span style='color:#4A90D9;font-weight:700'>Why Graph ML?</span> &nbsp;
          Standard models treat each sensor independently. But LA highway sensors are
          <b style='color:#E8F0FE'>spatially connected</b> — a slowdown at one sensor
          spreads to neighbours within minutes. Our T-GCN model captures this using the
          road network graph (<b style='color:#E8F0FE'>1,722 connections</b> between
          <b style='color:#E8F0FE'>207 sensors</b>).
        </div>
        """, unsafe_allow_html=True)

        results_local = results_df[
            results_df["Model"].isin(["Persistence", "RandomForest", "T-GCN-V3"])
        ].copy()
        results_local["Horizon"] = results_local["Horizon"].apply(format_horizon_for_table)
        horizon_order = ["5min", "15min", "30min"]
        color_map = {"Persistence": "#3498DB", "RandomForest": "#27AE60", "T-GCN-V3": "#E74C3C"}
        width_map = {"Persistence": 1.5,       "RandomForest": 1.5,       "T-GCN-V3": 3}

        rf_d   = results_local[results_local["Model"] == "RandomForest"].set_index("Horizon").reindex(horizon_order).reset_index()
        tgcn_d = results_local[results_local["Model"] == "T-GCN-V3"].set_index("Horizon").reindex(horizon_order).reset_index()

        fig_mae = go.Figure()
        # shaded gap between RF and T-GCN
        fig_mae.add_trace(go.Scatter(
            x=rf_d["Horizon"].tolist() + rf_d["Horizon"].tolist()[::-1],
            y=rf_d["MAE"].tolist() + tgcn_d["MAE"].tolist()[::-1],
            fill="toself", fillcolor="rgba(39,174,96,0.12)",
            line=dict(width=0), showlegend=True,
            name="T-GCN advantage (shaded)", hoverinfo="skip",
        ))
        for model in ["Persistence", "RandomForest", "T-GCN-V3"]:
            d = results_local[results_local["Model"] == model].set_index("Horizon").reindex(horizon_order).reset_index()
            fig_mae.add_trace(go.Scatter(
                x=d["Horizon"], y=d["MAE"],
                name="T-GCN (ours)" if model == "T-GCN-V3" else model,
                mode="lines+markers",
                line=dict(color=color_map[model], width=width_map[model]),
                marker=dict(size=10 if model == "T-GCN-V3" else 7,
                            symbol="diamond" if model == "T-GCN-V3" else "circle"),
            ))
        fig_mae.update_layout(**dark_layout(
            "Model Comparison: MAE by Horizon — lower is better", height=340))
        fig_mae.update_yaxes(title_text="MAE (normalized units)")
        fig_mae.update_xaxes(title_text="Forecast Horizon")
        st.plotly_chart(fig_mae, use_container_width=True)

    # — Right: performance table + 4 stat mini-cards ——————————
    with col_metrics:
        _mae5  = _ds.get("tgcn_mae_5min",  0)
        _mae15 = _ds.get("tgcn_mae_15min", 0)
        _mae30 = _ds.get("tgcn_mae_30min", 0)
        _imp5  = _ds.get("imp_5min",  0)
        _imp15 = _ds.get("imp_15min", 0)
        _imp30 = _ds.get("imp_30min", 0)

        def _badge(imp):
            c = "#27AE60" if imp > 0 else "#F39C12"
            return (f"<span style='background:{c}22;border:1px solid {c}55;border-radius:4px;"
                    f"padding:2px 8px;color:{c};font-size:0.7rem;font-weight:700'>"
                    f"{'+'if imp>0 else ''}{imp:.1f}%</span>")

        perf_rows = "".join(
            f"<tr style='border-bottom:1px solid #1A2D45'>"
            f"<td style='padding:10px 10px 10px 0;color:#7A9CC0;font-size:0.78rem'>{h}</td>"
            f"<td style='padding:10px 8px;color:#4A90D9;font-size:1.05rem;font-weight:800'>{mae:.3f}</td>"
            f"<td style='padding:10px 0;text-align:right'>{_badge(imp)}</td>"
            f"</tr>"
            for h, mae, imp in [
                ("5-min",  _mae5,  _imp5),
                ("15-min", _mae15, _imp15),
                ("30-min", _mae30, _imp30),
            ]
        )

        _fc_ids   = _ds["fully_congested"]
        _fc_label = ", ".join(f"S{i}" for i in _fc_ids) if _fc_ids else "None"
        _fc_color = "#E74C3C" if _fc_ids else "#27AE60"
        _fc_bg    = "#1A0A0A" if _fc_ids else "#0A1E0F"
        _fc_border= "#5C1515" if _fc_ids else "#1A5C2A"

        st.markdown(
            f"<div style='display:flex;flex-direction:column;gap:10px'>"
            f"<div style='background:#0D1A2D;border:1px solid #1E3A5F;border-radius:10px;padding:12px 16px'>"
            f"<div style='color:#6B8CAE;font-size:0.65rem;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;margin-bottom:4px'>T-GCN vs Random Forest</div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr style='border-bottom:1px solid #1E3A5F'>"
            f"<th style='padding:6px 10px 6px 0;color:#4A6A8A;font-size:0.65rem;font-weight:600;text-align:left;text-transform:uppercase'>Horizon</th>"
            f"<th style='padding:6px 8px;color:#4A6A8A;font-size:0.65rem;font-weight:600;text-align:left;text-transform:uppercase'>MAE</th>"
            f"<th style='padding:6px 0;color:#4A6A8A;font-size:0.65rem;font-weight:600;text-align:right;text-transform:uppercase'>vs RF</th>"
            f"</tr>{perf_rows}</table></div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>"
            f"<div style='background:#0A1525;border:1px solid #1E3A6A;border-top:2px solid #4A90D9;border-radius:8px;padding:10px 12px'>"
            f"<div style='color:#6B8CAE;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.8px'>Spatial r</div>"
            f"<div style='color:#4A90D9;font-size:1.3rem;font-weight:800;margin:4px 0'>0.672</div>"
            f"<div style='color:#6B8CAE;font-size:0.65rem'>GNN justified</div></div>"
            f"<div style='background:{_fc_bg};border:1px solid {_fc_border};border-top:2px solid {_fc_color};border-radius:8px;padding:10px 12px'>"
            f"<div style='color:#6B8CAE;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.8px'>Always congested</div>"
            f"<div style='color:{_fc_color};font-size:1.1rem;font-weight:800;margin:4px 0'>{_fc_label}</div>"
            f"<div style='color:#6B8CAE;font-size:0.65rem'>100% congestion rate</div></div>"
            f"<div style='background:#0A1E0F;border:1px solid #1A5C2A;border-top:2px solid #27AE60;border-radius:8px;padding:10px 12px'>"
            f"<div style='color:#6B8CAE;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.8px'>Parameters</div>"
            f"<div style='color:#27AE60;font-size:1.3rem;font-weight:800;margin:4px 0'>27,750</div>"
            f"<div style='color:#6B8CAE;font-size:0.65rem'>Lightweight model</div></div>"
            f"<div style='background:#1A1408;border:1px solid #5C4A15;border-top:2px solid #F39C12;border-radius:8px;padding:10px 12px'>"
            f"<div style='color:#6B8CAE;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.8px'>Training time</div>"
            f"<div style='color:#F39C12;font-size:1.1rem;font-weight:800;margin:4px 0'>16 min</div>"
            f"<div style='color:#6B8CAE;font-size:0.65rem'>RTX 3060 GPU</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── ZONE 3: How It Works — 3 horizontal cards + Training Summary ───
    st.markdown("<div style='border-bottom:1px solid #1E2D45;margin:1.2rem 0'></div>",
                unsafe_allow_html=True)

    # Section label
    st.markdown(
        "<div style='color:#E8F0FE;font-size:0.9rem;font-weight:700;margin-bottom:10px'>"
        "How The System Works</div>",
        unsafe_allow_html=True,
    )

    # 3 steps as horizontal equal columns + 1 narrow training summary column
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")

    step_data = [
        (c1, "1. Collect", "🔵",
         "207 highway sensors report traffic speeds every 5 minutes. Each reading captures speed + time of day.",
         "5-min intervals", "#4A90D9"),
        (c2, "2. Predict", "🟣",
         "T-GCN processes the last 60 minutes across all 207 connected sensors simultaneously — 12 timesteps, 1,722 road edges.",
         "12 time steps · K=2 hops", "#7B68EE"),
        (c3, "3. Act", "🟢",
         "Operations team sees predicted congestion zones 15 minutes ahead. Drivers repositioned BEFORE demand spikes.",
         "15-min lookahead window", "#27AE60"),
    ]

    for col, title, icon, desc, stat, color in step_data:
        with col:
            st.markdown(f"""
            <div style='background:#0D1A2D;border:1px solid #1E3A5F;border-top:3px solid {color};
                        border-radius:10px;padding:14px 14px;height:100%;box-sizing:border-box'>
              <div style='color:{color};font-size:0.75rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.8px;margin-bottom:8px'>{title}</div>
              <div style='color:#A8C0E8;font-size:0.78rem;line-height:1.55;margin-bottom:10px'>{desc}</div>
              <div style='background:{color}18;border:1px solid {color}44;border-radius:4px;
                          padding:4px 8px;color:{color};font-size:0.68rem;font-weight:600;
                          display:inline-block'>{stat}</div>
            </div>
            """, unsafe_allow_html=True)

    with c4:
        training_rows = [
            ("Architecture", "T-GCN (GConvGRU)"),
            ("Parameters",   "27,750"),
            ("Best epoch",   "10 / 30"),
            ("Loss fn",      "Huber Loss"),
            ("Optimizer",    "Adam + Cosine LR"),
        ]
        rows_html = "".join(
            f"<tr style='border-bottom:1px solid #1A2D45'>"
            f"<td style='color:#6B8CAE;font-size:0.72rem;padding:7px 10px 7px 0;white-space:nowrap'>{k}</td>"
            f"<td style='color:#E8F0FE;font-size:0.72rem;font-weight:600;padding:7px 0;text-align:right'>{v}</td>"
            f"</tr>"
            for k, v in training_rows
        )
        st.markdown(f"""
        <div style='background:#0D1A2D;border:1px solid #1E3A5F;border-top:3px solid #6B8CAE;
                    border-radius:10px;padding:14px;height:100%;box-sizing:border-box'>
          <div style='color:#6B8CAE;font-size:0.75rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.8px;margin-bottom:8px'>Training Summary</div>
          <table style='width:100%;border-collapse:collapse'>{rows_html}</table>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE 2 — LIVE TRAFFIC MAP
# =====================================================================
elif st.session_state.page == "Live Traffic Map":
    st.markdown("### Live Traffic Network — Los Angeles")

    n_snapshots  = int(X_test.shape[0])
    max_snapshot = max(0, n_snapshots - 1)
    st.session_state.snapshot_idx = min(st.session_state.snapshot_idx, max_snapshot)

    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Map Controls**")
        map_mode    = st.selectbox("Color sensors by:", ["Congestion Rate", "Prediction Error", "Mean Speed"])
        show_labels = st.checkbox("Show sensor labels", value=False)

        st.markdown("---")
        st.markdown("**Filter by congestion level:**")
        show_critical = st.checkbox("Critical (>60%)",  value=True)
        show_high     = st.checkbox("High (25–60%)",    value=True)
        show_medium   = st.checkbox("Medium (10–25%)",  value=True)
        show_low      = st.checkbox("Low (<10%)",       value=True)

        st.markdown("---")
        st.markdown("**Time Animation**")
        st.session_state.snapshot_idx = st.slider(
            "Test snapshot:", 0, max_snapshot, st.session_state.snapshot_idx,
            help="Drag to animate traffic over time",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("◀ Back"):
                st.session_state.snapshot_idx = max(0, st.session_state.snapshot_idx - 1)
                st.rerun()
        with c2:
            if st.button("Reset"):
                st.session_state.snapshot_idx = 0
                st.rerun()
        with c3:
            if st.button("Next ▶"):
                st.session_state.snapshot_idx = min(max_snapshot, st.session_state.snapshot_idx + 1)
                st.rerun()

        snapshot  = st.session_state.snapshot_idx
        time_str  = compute_snapshot_label(snapshot)
        day_type  = get_day_type(snapshot)

        st.markdown("---")
        st.write(f"Frame **{snapshot}** / {max_snapshot}")

        # FIX: Show time AND day type for business context
        st.markdown(
            f"<div style='background:#0D1A2D;border:1px solid #1E3A5F;border-radius:6px;"
            f"padding:8px 12px;margin:4px 0'>"
            f"<div style='color:#6B8CAE;font-size:0.68rem;text-transform:uppercase'>Approx Time</div>"
            f"<div style='color:#E8F0FE;font-size:1rem;font-weight:700'>{time_str}</div>"
            f"<div style='color:#4A90D9;font-size:0.72rem;margin-top:2px'>{day_type} pattern</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        snap_mean = float(X_test[snapshot, :, -1].mean())

        # FIX: Plain-language congestion status with normalized value as secondary info
        if snap_mean > 0.1:
            status_color, status_text, status_bg = "#27AE60", "Free Flow", "#0A1E0F"
        elif snap_mean > -0.2:
            status_color, status_text, status_bg = "#F39C12", "Moderate Traffic", "#1A1408"
        else:
            status_color, status_text, status_bg = "#E74C3C", "Heavy Congestion", "#1A0808"

        st.markdown(
            f"<div style='background:{status_bg};border:1px solid {status_color}55;"
            f"border-left:3px solid {status_color};border-radius:6px;padding:8px 12px;margin-top:4px'>"
            f"<div style='color:{status_color};font-size:0.75rem;font-weight:700'>{status_text}</div>"
            f"<div style='color:#6B8CAE;font-size:0.68rem;margin-top:2px'>"
            f"Network avg speed: {snap_mean:+.3f} (normalized)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    def render_map(snapshot_idx: int):
        m = folium.Map(location=[34.05, -118.30], zoom_start=11, control_scale=True, tiles=None)
        folium.TileLayer(
            tiles="https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="© Google", name="Google Streets",
            overlay=False, control=True, show=True, max_zoom=20,
            subdomains=["mt0","mt1","mt2","mt3"],
        ).add_to(m)
        folium.TileLayer(
            tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="© Google", name="Google Satellite",
            overlay=False, control=True, show=False, max_zoom=20,
            subdomains=["mt0","mt1","mt2","mt3"],
        ).add_to(m)
        folium.LayerControl().add_to(m)

        current_snap_speeds = X_test[snapshot_idx, :, -1]
        current_predictions = predictions[snapshot_idx, :, 2]

        for _, row in sensor_df.iterrows():
            level = str(row["congestion_level"])
            if level == "Critical" and not show_critical: continue
            if level == "High"     and not show_high:     continue
            if level == "Medium"   and not show_medium:   continue
            if level == "Low"      and not show_low:      continue

            idx = int(row["index"])
            if not (0 <= idx < 207): continue

            if map_mode == "Congestion Rate":
                cong = float(row["congestion_rate"])
                # FIX: Colors match legend exactly — same thresholds as legend
                fc = "#8B0000" if cong > 0.6 else "#E74C3C" if cong > 0.35 else "#F39C12" if cong > 0.2 else "#F7DC6F"
                radius = max(5, cong * 22)
                popup_extra = f"Congestion rate: {cong:.1%}"
            elif map_mode == "Prediction Error":
                mae  = float(row["prediction_mae"])
                norm = max(0, min(1, (mae - 0.12) / (0.35 - 0.12)))
                r_val = int(norm * 200 + 55)
                g_val = int((1 - norm) * 180 + 20)
                fc = f"#{r_val:02x}{g_val:02x}30"
                radius = max(5, mae * 28)
                popup_extra = f"Prediction MAE (15min): {mae:.4f}"
            else:
                spd = float(current_snap_speeds[idx])
                fc = "#27AE60" if spd > 0.1 else "#F39C12" if spd > -0.2 else "#E74C3C"
                radius = 7
                popup_extra = f"Current normalized speed: {spd:+.3f}"

            pred_15     = float(current_predictions[idx])
            popup_html  = f"""
            <div style='font-family:Arial;font-size:12px;min-width:180px;color:#111'>
              <b style='color:#1B3A6B'>Sensor {idx}</b><br>
              <hr style='margin:4px 0'>
              {popup_extra}<br>
              <b>15-min prediction: {pred_15:+.3f}</b><br>
              Congestion level: <b>{level}</b><br>
              Location: {row['latitude']:.4f}°N, {abs(row['longitude']):.4f}°W
            </div>"""
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius, color="white", weight=0.5,
                fill=True, fill_color=fc, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"S{idx}: {level}" if show_labels else f"S{idx} | {level} | Click for details",
            ).add_to(m)
        return m

    with col_map:
        current_map = render_map(st.session_state.snapshot_idx)
        st_folium(current_map, height=560, use_container_width=True,
                  key=f"map-{st.session_state.snapshot_idx}-{map_mode}")

        # FIX: Legend colors now match exactly what is rendered on the map
        if map_mode == "Congestion Rate":
            legend_items = [
                ("#8B0000", "Critical > 60%"),
                ("#E74C3C", "High 35–60%"),
                ("#F39C12", "Moderate 20–35%"),
                ("#F7DC6F", "Low < 20%"),
            ]
        elif map_mode == "Prediction Error":
            legend_items = [
                ("#c83050", "High error  MAE > 0.30"),
                ("#e07030", "Medium  0.20–0.30"),
                ("#b0b030", "Low  MAE < 0.20"),
            ]
        else:
            legend_items = [
                ("#27AE60", "Free flow  speed > +0.1"),
                ("#F39C12", "Moderate  −0.2 to +0.1"),
                ("#E74C3C", "Congested  speed < −0.2"),
            ]

        dots = "".join(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<div style='width:11px;height:11px;border-radius:50%;background:{c};flex-shrink:0'></div>"
            f"<span style='color:#8899AA;font-size:0.76rem'>{l}</span></div>"
            for c, l in legend_items
        )
        st.markdown(
            f"<div style='background:#0D1A2D;border:1px solid #1E3A5F;border-radius:8px;"
            f"padding:10px 14px;margin-top:8px;display:inline-block'>"
            f"<div style='color:#E8F0FE;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:1px;margin-bottom:6px'>{map_mode}</div>{dots}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Click a sensor for details. Use the slider or Back / Next buttons to animate over time.")


# =====================================================================
# PAGE 3 — FORECAST EXPLORER
# =====================================================================
elif st.session_state.page == "Forecast Explorer":
    st.markdown("### Forecast Explorer")
    st.caption("Explore T-GCN predictions vs actual speeds for any sensor and time period")

    col1, col2 = st.columns([1, 3])

    with col1:
        sensor_id = st.selectbox("Select Sensor:", range(207), format_func=lambda x: f"Sensor {x:03d}")
        horizon   = st.radio("Forecast Horizon:", ["5 min", "15 min", "30 min"])
        h_idx     = {"5 min": 0, "15 min": 2, "30 min": 5}[horizon]

        max_points = int(targets.shape[0])
        n_points   = st.slider("Data points to show:", 50, min(500, max_points), min(220, max_points))

        info  = sensor_df[sensor_df["index"] == sensor_id].iloc[0]
        level = str(info["congestion_level"])
        st.markdown("---")
        st.markdown("**Sensor Info**")
        st.metric("Congestion Rate", f"{info['congestion_rate']:.1%}")
        st.metric("MAE (15min)",     f"{info['prediction_mae']:.4f}")
        if level == "Critical": st.error(f"Level: {level}")
        elif level == "High":   st.warning(f"Level: {level}")
        else:                   st.success(f"Level: {level}")

        st.markdown("**Location**")
        mini_m = folium.Map(location=[info["latitude"], info["longitude"]],
                            zoom_start=14, tiles="CartoDB dark_matter")
        folium.CircleMarker(
            location=[info["latitude"], info["longitude"]],
            radius=12, color="red", fill=True, fill_color="red", fill_opacity=0.9,
        ).add_to(mini_m)
        st_folium(mini_m, height=200, use_container_width=True)

    with col2:
        pred_s  = predictions[-n_points:, sensor_id, h_idx]
        true_s  = targets[-n_points:, sensor_id, h_idx]
        time_x  = list(range(n_points))
        mae_val = mean_absolute_error(true_s, pred_s)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_x, y=true_s.tolist(), name="Actual Speed",
            line=dict(color="#27AE60", width=2), fill="tozeroy",
            fillcolor="rgba(39,174,96,0.10)"))
        fig.add_trace(go.Scatter(x=time_x, y=pred_s.tolist(),
            name=f"T-GCN ({horizon})", line=dict(color="#E74C3C", width=2, dash="dot")))
        error = np.abs(pred_s - true_s)
        fig.add_trace(go.Scatter(x=time_x, y=(pred_s + error).tolist(),
            line=dict(width=0), showlegend=False,
            fillcolor="rgba(231,76,60,0.14)", fill="tonexty"))
        fig.update_layout(**dark_layout(
            f"Sensor {sensor_id} — {horizon} Forecast | MAE = {mae_val:.4f}", height=340))
        fig.update_xaxes(title_text="Time Snapshot (test set, most recent)")
        fig.update_yaxes(title_text="Normalized Speed")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=true_s.tolist(), y=pred_s.tolist(), mode="markers",
                marker=dict(color="#4A90D9", opacity=0.55, size=5), name="Predictions"))
            mn = float(min(true_s.min(), pred_s.min()))
            mx = float(max(true_s.max(), pred_s.max()))
            fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                line=dict(color="#E74C3C", dash="dash"), name="Perfect prediction"))
            fig2.update_layout(**dark_layout("Predicted vs Actual", height=290))
            fig2.update_xaxes(title_text="Actual")
            fig2.update_yaxes(title_text="Predicted")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=(pred_s - true_s).tolist(), nbinsx=30,
                marker_color="#4A90D9", name="Error"))
            fig3.update_layout(**dark_layout("Error Distribution", height=290))
            fig3.update_xaxes(title_text="Error (predicted − actual)")
            fig3.update_yaxes(title_text="Count")
            st.plotly_chart(fig3, use_container_width=True)


# =====================================================================
# PAGE 4 — DECISION DASHBOARD
# =====================================================================
elif st.session_state.page == "Decision Dashboard":
    st.markdown("### Decision Dashboard")
    st.caption("Operational recommendations powered by T-GCN traffic intelligence")

    # FIX: Equal-height action boxes via a single HTML flex row
    st.markdown("""
    <div class='action-row'>
      <div class='action-box red'>
        <div class='action-label'>🔴 Immediate Action</div>
        <div class='action-body'>
          Sensors <b>16 &amp; 196</b> are congested <b>100%</b> of observed time.<br><br>
          Deploy <b>5 permanent extra drivers</b> within 1 km of this zone.<br>
          <span style='color:#E74C3C;font-size:0.72rem'>Location: 34.07°N 118.23°W</span>
        </div>
      </div>
      <div class='action-box yellow'>
        <div class='action-label'>📅 Daily Schedule</div>
        <div class='action-body'>
          Pre-position drivers at:<br>
          • <b>6:45am</b> — before morning rush (7am)<br>
          • <b>4:45pm</b> — before evening rush (5pm)<br><br>
          Evening congestion is <b>33% worse</b> than morning.
        </div>
      </div>
      <div class='action-box blue'>
        <div class='action-label'>📱 App Integration</div>
        <div class='action-body'>
          Show ETA uncertainty flag for <b>42 sensors</b> where prediction MAE &gt; 0.245.<br><br>
          Customer message:<br>
          <i>"ETA may vary due to complex traffic conditions."</i>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Priority Sensor Zones")
        action_df = pd.DataFrame({
            "Priority": ["CRITICAL", "CRITICAL", "HIGH", "HIGH", "STANDARD"],
            "Zone": ["Sensors 16, 196", "Sensor 56", "58 sensors (25–60% congested)",
                     "42 sensors (low reliability)", "All 207 sensors"],
            "Condition": ["100% congested", "91% congested", "Peak hours",
                          "MAE > 0.245", "Evening rush 5–7pm"],
            "Recommended Action": [
                "Permanent driver buffer (5+ drivers)",
                "Pre-position 6:45am & 4:45pm daily",
                "Surge pricing: speed < −0.3",
                "ETA uncertainty flag in app",
                "Extra incentives from 4:30pm Mon–Fri",
            ],
        })
        st.dataframe(action_df, hide_index=True, use_container_width=True, height=230)

        # FIX: Bar chart with correct severity order and matching colors
        st.markdown("#### Congestion Level Distribution")
        level_order  = ["Critical", "High", "Medium", "Low"]
        level_colors = {
            "Critical": "#8B0000",   # dark red — matches map Critical
            "High":     "#E74C3C",   # red      — matches map High
            "Medium":   "#F39C12",   # orange   — matches map Moderate
            "Low":      "#F7DC6F",   # yellow   — matches map Low
        }
        level_counts = (
            sensor_df["congestion_level"]
            .value_counts()
            .reindex(level_order, fill_value=0)
            .reset_index()
        )
        level_counts.columns = ["Level", "Count"]
        fig = px.bar(
            level_counts, x="Level", y="Count", color="Level",
            color_discrete_map=level_colors,
            category_orders={"Level": level_order},   # enforce severity order
        )
        fig.update_layout(**dark_layout("Sensors by Congestion Level", height=260, legend_bottom=False))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Number of Sensors")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### System Metrics")
        _n_high   = int((sensor_df["congestion_level"] == "High").sum())
        _avg_cong = float(sensor_df["congestion_rate"].mean()) * 100
        _fc_label = ", ".join(f"S{i}" for i in _ds["fully_congested"]) or "None"
        st.metric("Critical zones",      str(_ds["n_critical"]),         "Require permanent staffing")
        st.metric("High priority zones", str(_n_high),                   "Rush hour coverage")
        st.metric("Avg congestion rate", f"{_avg_cong:.1f}%",            "Across all sensors")
        st.metric("Most congested",      _fc_label,                      "100% congestion rate")
        st.metric("Hardest to predict",  f"S{_ds['worst_sensor_idx']}",  f"MAE = {_ds['worst_sensor_mae']:.3f}")

        st.markdown("#### Limitations")
        st.markdown("""
- Data from 2012 — pre-pandemic patterns
- Highway sensors only, no surface streets
- 30-min predictions less reliable (MAE ≈ 0.275)
- No weather or incident features included
""")


# =====================================================================
# PAGE 5 — MODEL RESULTS
# =====================================================================
elif st.session_state.page == "Model Results":
    st.markdown("### Model Results")

    tab1, tab2, tab3 = st.tabs(["Performance Table", "Error Analysis", "Architecture"])

    with tab1:
        working_results = results_df.copy()
        working_results["Horizon"] = working_results["Horizon"].apply(format_horizon_for_table)
        horizon_order = ["5min", "15min", "30min"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MAE Results (lower = better)**")
            mae_pivot = (working_results.pivot(index="Horizon", columns="Model", values="MAE")
                         .reindex(horizon_order))
            st.dataframe(mae_pivot.style.highlight_min(axis=1, color="#1E5C1E"),
                         use_container_width=True)
        with col2:
            st.markdown("**RMSE Results (lower = better)**")
            rmse_pivot = (working_results.pivot(index="Horizon", columns="Model", values="RMSE")
                          .reindex(horizon_order))
            st.dataframe(rmse_pivot.style.highlight_min(axis=1, color="#1E5C1E"),
                         use_container_width=True)

        # FIX: Add credibility note beneath the tables
        st.markdown(
            "<div style='color:#6B8CAE;font-size:0.72rem;margin-top:4px;margin-bottom:12px'>"
            "ℹ️ RF trained separately per horizon on correct targets. "
            "All models evaluated on identical held-out test set (last 20% of data, chronological split)."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Improvement of T-GCN vs Random Forest**")
        rf_df   = working_results[working_results["Model"] == "RandomForest"].set_index("Horizon").reindex(horizon_order)
        tgcn_df = working_results[working_results["Model"] == "T-GCN-V3"].set_index("Horizon").reindex(horizon_order)

        imp_rows = []
        for h in horizon_order:
            rf_mae   = float(rf_df.loc[h, "MAE"])
            tgcn_mae = float(tgcn_df.loc[h, "MAE"])
            imp_rows.append({
                "Horizon":    h,
                "RF MAE":     round(rf_mae,   4),
                "T-GCN MAE":  round(tgcn_mae, 4),
                "Improvement": f"{(rf_mae - tgcn_mae)/rf_mae*100:+.1f}%",
            })
        st.dataframe(pd.DataFrame(imp_rows), hide_index=True, use_container_width=True)

        # FIX: Removed redundant summary cards — improvement table above already
        # shows all three horizons clearly. Kept only the key insight box.
        st.markdown("---")
        st.markdown(
            "<div class='insight-box'>"
            "<b style='color:#E8F0FE'>Key finding:</b> T-GCN beats Random Forest at <b>all three horizons</b> "
            "(+4.9% at 5min, +11.3% at 15min, +11.8% at 30min). The spatial advantage is largest at "
            "15 and 30 minutes — exactly when Uber needs reliable advance warning to pre-position drivers."
            "</div>",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("**Model-driven error diagnostics (computed live from test predictions)**")
        c1, c2 = st.columns(2)
        with c1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(x=horizon_metrics_df["Horizon"], y=horizon_metrics_df["MAE"],
                                   name="MAE", marker_color="#4A90D9"))
            fig_h.update_layout(**dark_layout("T-GCN MAE by Horizon", height=300, legend_bottom=False))
            fig_h.update_yaxes(title_text="MAE")
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=horizon_metrics_df["Horizon"], y=horizon_metrics_df["RMSE"],
                                   name="RMSE", marker_color="#E67E22"))
            fig_r.update_layout(**dark_layout("T-GCN RMSE by Horizon", height=300, legend_bottom=False))
            fig_r.update_yaxes(title_text="RMSE")
            st.plotly_chart(fig_r, use_container_width=True)

        sensor_rank = sensor_df[["index","latitude","longitude","congestion_rate",
                                  "prediction_mae","congestion_level"]].copy()
        sensor_rank = sensor_rank.sort_values("prediction_mae", ascending=False)
        sensor_rank.columns = ["Sensor ID","Latitude","Longitude","Congestion Rate","MAE (15min)","Level"]
        sensor_rank["Congestion %"] = sensor_rank["Congestion Rate"] * 100

        fig_rank = px.scatter(
            sensor_rank, x="Congestion %", y="MAE (15min)", color="Level",
            hover_data=["Sensor ID","Latitude","Longitude"],
            color_discrete_map={"Critical":"#8B0000","High":"#E74C3C","Medium":"#F39C12","Low":"#F7DC6F"},
        )
        fig_rank.update_layout(**dark_layout("Sensor Reliability vs Congestion", height=340))
        fig_rank.update_xaxes(title_text="Congestion Rate (%)")
        st.plotly_chart(fig_rank, use_container_width=True)

        st.dataframe(sensor_rank[["Sensor ID","Level","Congestion Rate","MAE (15min)"]].head(30),
                     hide_index=True, use_container_width=True, height=320)

        st.markdown("**Top high-error sensors by horizon**")
        selected_h = st.selectbox("Horizon", ["5min","15min","30min"], key="results_horizon")
        top_error  = (sensor_horizon_df[sensor_horizon_df["Horizon"] == selected_h]
                      .sort_values("MAE", ascending=False).head(20))
        fig_top = px.bar(top_error, x="Sensor ID", y="MAE",
                         color="MAE", color_continuous_scale="OrRd")
        fig_top.update_layout(**dark_layout(f"Top 20 Sensors by MAE ({selected_h})",
                                            height=320, legend_bottom=False))
        st.plotly_chart(fig_top, use_container_width=True)

    with tab3:
        st.markdown("**Model Architecture — T-GCN**")
        st.markdown("""
- **Architecture:** T-GCN (Temporal Graph Convolutional Network using GConvGRU)
- **Input:** 207 sensors × last 12 timesteps × 1 feature (normalized speed)
- **Graph:** 1,722 road edges, K=2 hop neighborhood
- **Output:** 6 future timesteps per sensor (5, 10, 15, 20, 25, 30 min ahead)
- **Evaluated at:** 5-min, 15-min, 30-min horizons
- **Key design:** Sequential timestep processing — each of the 12 steps is fed one at a time
  through GConvGRU, so the hidden state accumulates both spatial and temporal context
- **Parameters:** 27,750 total
- **Loss function:** Huber Loss (robust to outliers)
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4) with CosineAnnealingLR scheduler
""")
        selected_sensor  = st.selectbox("Inspect sensor:", range(207),
                                         format_func=lambda x: f"Sensor {x:03d}", key="diag_sensor")
        selected_horizon = st.selectbox("Forecast horizon:",
                                         [("5min",0),("15min",2),("30min",5)],
                                         format_func=lambda x: x[0], key="diag_horizon")
        h_name, h_idx = selected_horizon
        pred_line = predictions[:, selected_sensor, h_idx]
        true_line = targets[:,    selected_sensor, h_idx]
        err_line  = np.abs(pred_line - true_line)

        fig_beh = go.Figure()
        fig_beh.add_trace(go.Scatter(x=np.arange(len(pred_line)), y=true_line,
            name="Actual", line=dict(color="#2ECC71", width=2)))
        fig_beh.add_trace(go.Scatter(x=np.arange(len(pred_line)), y=pred_line,
            name="Predicted", line=dict(color="#E74C3C", width=2, dash="dot")))
        fig_beh.update_layout(**dark_layout(f"Sensor {selected_sensor} — Full Test Period ({h_name})", height=330))
        fig_beh.update_xaxes(title_text="Snapshot")
        fig_beh.update_yaxes(title_text="Normalized Speed")
        st.plotly_chart(fig_beh, use_container_width=True)

        fig_err = px.histogram(pd.DataFrame({"Absolute Error": err_line}), x="Absolute Error",
                               nbins=40, color_discrete_sequence=["#4A90D9"])
        fig_err.update_layout(**dark_layout(
            f"Error distribution — Sensor {selected_sensor} ({h_name})",
            height=280, legend_bottom=False))
        st.plotly_chart(fig_err, use_container_width=True)


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid #1E2D45;padding:1rem 0;margin-top:2rem;
            display:flex;justify-content:space-between;align-items:center;
            gap:1rem;flex-wrap:wrap'>
  <div style='color:#3A4A5A;font-size:0.73rem'>METR-LA Traffic Intelligence</div>
  <div style='color:#3A4A5A;font-size:0.73rem'>
    EM627 Spatial Data Science · T-GCN Graph Neural Network · 207 sensors · RTX 3060 GPU
  </div>
  <div style='color:#3A4A5A;font-size:0.73rem'>April 2026</div>
</div>
""", unsafe_allow_html=True)