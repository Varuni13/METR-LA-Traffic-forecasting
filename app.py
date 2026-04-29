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
/* Hide sidebar and its toggle button entirely */
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
section[data-testid="stSidebarNav"]{ display: none !important; }
.stMainBlockContainer              { max-width: 100% !important; padding: 3.5rem 2rem 1rem 2rem !important; }

/* ── Top header bar ────────────────────────────────────────── */
.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #0D1117;
  border-bottom: 1px solid #1E2D45;
  padding: 10px 20px;
  margin-bottom: 0;
  border-radius: 8px 8px 0 0;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-title { color: #E8F0FE; font-weight: 700; font-size: 0.95rem; line-height: 1.2; }
.brand-sub   { color: #6B8CAE; font-size: 0.7rem; }
.top-bar-right { display: flex; align-items: center; gap: 16px; }
.ts-label { color: #6B8CAE; font-size: 0.7rem; }
.ts-value { color: #4A90D9; font-size: 0.85rem; font-weight: 600; }

/* ── Pill nav ──────────────────────────────────────────────── */
.nav-bar {
  background: #0D1117;
  border-bottom: 1px solid #1E2D45;
  padding: 6px 20px 0 20px;
  display: flex;
  gap: 4px;
  margin-bottom: 20px;
  border-radius: 0 0 8px 8px;
}
/* Radio styled as pill tabs */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex !important;
  flex-direction: row !important;
  gap: 4px !important;
  flex-wrap: nowrap !important;
  background: transparent !important;
}
div[data-testid="stRadio"] label {
  padding: 7px 16px !important;
  border-radius: 6px 6px 0 0 !important;
  background: transparent !important;
  color: #6B8CAE !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  white-space: nowrap !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.15s !important;
  margin-bottom: 0 !important;
}
div[data-testid="stRadio"] label:hover {
  color: #A8C0E8 !important;
  background: #0F1E30 !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
  color: #4A90D9 !important;
  font-weight: 700 !important;
  border-bottom: 2px solid #4A90D9 !important;
  background: #0F1E30 !important;
}
/* Hide radio circle dot */
div[data-testid="stRadio"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] label p { margin: 0 !important; }

/* ── KPI cards ────────────────────────────────────────────── */
.kpi-card {
  background: linear-gradient(135deg, #0D1A2D, #152035);
  border: 1px solid #1E3A5F;
  border-top: 3px solid #4A90D9;
  border-radius: 10px; padding: 1.2rem; text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(74,144,217,0.2); }
.kpi-card.green  { border-top-color: #27AE60; }
.kpi-card.yellow { border-top-color: #F39C12; }
.kpi-card.red    { border-top-color: #E74C3C; }
.kpi-card.blue   { border-top-color: #4A90D9; }
.kpi-card.purple { border-top-color: #7B68EE; }
.kpi-val   { font-size: 2rem;  font-weight: 800; color: #4A90D9; letter-spacing: -1px; }
.kpi-lbl   { font-size: 0.72rem; color: #6B8CAE; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: 0.68rem; color: #27AE60; margin-top: 3px; }

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

/* ── Main header block ────────────────────────────────────── */
.main-header {
  background: linear-gradient(135deg, #0F2444 0%, #1B3A6B 50%, #0F2444 100%);
  padding: 1.6rem 2rem; border-radius: 10px;
  border: 1px solid #2E5090;
  box-shadow: 0 4px 24px rgba(74,144,217,0.15);
  margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.main-header::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, #4A90D9, #7B68EE, #4A90D9);
}

/* ── Sidebar elements ─────────────────────────────────────── */
.sidebar-metric {
  background: #0D1A2D; border: 1px solid #1E3A5F;
  border-radius: 8px; padding: 8px 12px; margin: 4px 0;
  display: flex; justify-content: space-between; align-items: center;
}
.sidebar-metric-val { color: #4A90D9; font-weight: 600; font-size: 0.85rem; }
.sidebar-alert {
  background: #1A0A0A; border: 1px solid #5C1515;
  border-left: 3px solid #E74C3C; border-radius: 6px;
  padding: 7px 10px; margin: 4px 0; font-size: 0.75rem; color: #E88080;
}
.sidebar-section-label {
  color: #4A90D9; font-weight: 700; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 1.5px;
  margin-bottom: 8px; padding-left: 8px;
  border-left: 3px solid #4A90D9;
}

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
# DYNAMIC STATS  — computed once from loaded data, used throughout UI
# ---------------------------------------------------------------------
def _compute_dashboard_stats(results_df, sensor_df, X_test, predictions, targets):
    stats = {}

    # Model MAE/RMSE per horizon from results_df
    _res = results_df.copy()
    def _fmt(h):
        k = str(h).replace(" ", "").lower()
        return "5min" if k in {"5min","5m","5"} else "15min" if k in {"15min","15m","15"} else "30min" if k in {"30min","30m","30"} else str(h)
    _res["_h"] = _res["Horizon"].apply(_fmt)
    _tgcn = _res[_res["Model"] == "T-GCN-V3"].set_index("_h")
    _rf   = _res[_res["Model"] == "RandomForest"].set_index("_h")

    for h in ["5min", "15min", "30min"]:
        if h in _tgcn.index:
            stats[f"tgcn_mae_{h}"]  = float(_tgcn.loc[h, "MAE"])
            stats[f"tgcn_rmse_{h}"] = float(_tgcn.loc[h, "RMSE"])
        if h in _rf.index:
            stats[f"rf_mae_{h}"]    = float(_rf.loc[h, "MAE"])

    # Improvement % T-GCN vs RF per horizon
    for h in ["5min", "15min", "30min"]:
        if f"tgcn_mae_{h}" in stats and f"rf_mae_{h}" in stats:
            imp = (stats[f"rf_mae_{h}"] - stats[f"tgcn_mae_{h}"]) / stats[f"rf_mae_{h}"] * 100
            stats[f"imp_{h}"] = imp

    # Best model per horizon
    stats["best_5min"]  = "T-GCN" if stats.get("imp_5min",  0) > 0 else "RF"
    stats["best_15min"] = "T-GCN" if stats.get("imp_15min", 0) > 0 else "RF"
    stats["best_30min"] = "T-GCN" if stats.get("imp_30min", 0) > 0 else "RF"

    # Congestion stats from sensor_df
    critical_mask = sensor_df["congestion_level"] == "Critical"
    stats["n_critical"]      = int(critical_mask.sum())
    stats["congestion_rate"] = float(sensor_df["congestion_rate"].mean()) * 100

    # Sensors with 100% congestion
    fully_congested = sensor_df[sensor_df["congestion_rate"] >= 0.999]["index"].tolist()
    stats["fully_congested"] = fully_congested

    # Hardest-to-predict sensor
    worst_idx = int(sensor_df.loc[sensor_df["prediction_mae"].idxmax(), "index"])
    worst_mae = float(sensor_df["prediction_mae"].max())
    stats["worst_sensor_idx"] = worst_idx
    stats["worst_sensor_mae"] = worst_mae

    # Overall speed and congested %
    stats["overall_speed"]  = float(X_test[:, :, -1].mean())
    stats["congested_pct"]  = float((X_test[:, :, -1] < -0.3).mean() * 100)

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
    if key in {"5min", "5m", "5"}:   return "5min"
    if key in {"15min", "15m", "15"}: return "15min"
    if key in {"30min", "30m", "30"}: return "30min"
    return str(h)

def compute_snapshot_label(snapshot_idx: int) -> str:
    mins = snapshot_idx * 5
    return f"{(mins // 60) % 24:02d}:{mins % 60:02d}"

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "snapshot_idx" not in st.session_state:
    st.session_state.snapshot_idx = 100

# ---------------------------------------------------------------------
# UNIFIED HEADER + NAV  (rendered once, above all pages)
# ---------------------------------------------------------------------

# Build stats and alert strings for the header
_fc_badges_hdr = "  ".join(
    f"<span style='background:#2D0F0F;border:1px solid #8B2020;border-left:2px solid #E74C3C;"
    f"border-radius:3px;padding:2px 8px;font-size:0.7rem;color:#E88080'>S{i} 100% congested</span>"
    for i in _ds["fully_congested"]
)
_high_sensors = sensor_df[
    (sensor_df["congestion_level"] == "High") & (sensor_df["congestion_rate"] >= 0.85)
].sort_values("congestion_rate", ascending=False).head(1)
_high_badge_hdr = ""
if not _high_sensors.empty:
    _hs = _high_sensors.iloc[0]
    _high_badge_hdr = (
        f"<span style='background:#2D1E08;border:1px solid #8B5E20;border-left:2px solid #F39C12;"
        f"border-radius:3px;padding:2px 8px;font-size:0.7rem;color:#F0C070'>"
        f"S{int(_hs['index'])} {_hs['congestion_rate']:.0%} congested</span>"
    )
_all_alerts = (_fc_badges_hdr + ("  " if _fc_badges_hdr and _high_badge_hdr else "") + _high_badge_hdr)

_stat_pills = "  ·  ".join([
    f"<span style='color:#E8F0FE;font-weight:600'>207</span> <span style='color:#4A6A8A'>sensors</span>",
    f"<span style='color:#E8F0FE;font-weight:600'>{_ds['congested_pct']:.1f}%</span> <span style='color:#4A6A8A'>congested</span>",
    f"<span style='color:#E8F0FE;font-weight:600'>{_ds['n_critical']}</span> <span style='color:#4A6A8A'>critical zones</span>",
    f"<span style='color:#E8F0FE;font-weight:600'>{_ds.get('tgcn_mae_5min',0):.3f}</span> <span style='color:#4A6A8A'>MAE 5-min</span>",
])

st.markdown(f"""
<div style='background:linear-gradient(135deg,#0F2444 0%,#1B3A6B 50%,#0F2444 100%);
            border:1px solid #2E5090;border-radius:10px;padding:1.2rem 1.6rem;
            margin-bottom:0;position:relative;overflow:hidden'>
  <div style='position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,#4A90D9,#7B68EE,#4A90D9)'></div>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem'>
    <div>
      <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
        <span style='font-size:1.6rem'>🚦</span>
        <span style='color:#E8F0FE;font-weight:800;font-size:1.4rem;letter-spacing:-0.5px'>
          METR-LA Traffic Intelligence</span>
        <span class='status-live'><span class='status-dot'></span>Live</span>
      </div>
      <div style='color:#8AAACE;font-size:0.8rem;margin-bottom:8px'>
        Graph Neural Network · Real-time forecasting across 207 Los Angeles highway sensors
        · Predicting 5, 15 and 30 minutes ahead
      </div>
      <div style='font-size:0.78rem'>{_stat_pills}</div>
      <div style='margin-top:8px;display:flex;gap:6px;flex-wrap:wrap'>{_all_alerts}</div>
    </div>
    <div style='text-align:right;flex-shrink:0'>
      <div style='color:#6B8CAE;font-size:0.68rem'>Last updated</div>
      <div style='color:#4A90D9;font-size:1rem;font-weight:700'>{datetime.now().strftime('%H:%M:%S')}</div>
      <div style='color:#3A5A7A;font-size:0.68rem;margin-top:6px'>
        Test set: 6,851 snapshots<br>Training: 23,978 snapshots
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Pill nav
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
    "",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(NAV_MAP_INV.get(st.session_state.page, "Problem & Overview")),
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)
st.session_state.page = NAV_MAP[selected_label]

st.markdown("<div style='border-bottom:1px solid #1E2D45;margin-bottom:20px'></div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PAGE 1 — OVERVIEW
# ---------------------------------------------------------------------
if st.session_state.page == "Overview":
    kpis = [
        ("207",
         "Highway Sensors", "LA highway network", "blue"),
        (f"{_ds.get('tgcn_mae_5min', 0):.3f}",
         "MAE at 5-min", f"Best: {_ds['best_5min']}", "green"),
        ("0.672",
         "Spatial Correlation", "Justifies graph model", "purple"),
        (str(_ds["n_critical"]),
         "Critical Zones", "Require immediate action", "red"),
        (f"{_ds['congestion_rate']:.1f}%",
         "Congestion Rate", "Of all observations", "yellow"),
    ]
    cols = st.columns(5)
    for col, (val, lbl, sub, accent) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card {accent}'>
              <div class='kpi-val'>{val}</div>
              <div class='kpi-lbl'>{lbl}</div>
              <div class='kpi-delta'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Why Graph ML?")
        st.markdown("""
        <div class='insight-box'>
        Standard models treat each sensor independently. But LA highway sensors are
        <b>spatially connected</b> — a slowdown at one sensor spreads to neighbours
        within minutes. Our T-GCN model captures this using the road network graph
        (1,722 connections between 207 sensors).
        </div>
        """, unsafe_allow_html=True)

        results_df_local = results_df[
            results_df["Model"].isin(["Persistence", "RandomForest", "T-GCN-V3"])
        ].copy()
        results_df_local["Horizon"] = results_df_local["Horizon"].apply(format_horizon_for_table)

        horizon_order = ["5min", "15min", "30min"]
        color_map = {"Persistence": "#3498DB", "RandomForest": "#27AE60", "T-GCN-V3": "#E74C3C"}
        width_map = {"Persistence": 1.5, "RandomForest": 1.5, "T-GCN-V3": 3}

        rf_d = (results_df_local[results_df_local["Model"] == "RandomForest"]
                .set_index("Horizon").reindex(horizon_order).reset_index())
        tgcn_d = (results_df_local[results_df_local["Model"] == "T-GCN-V3"]
                  .set_index("Horizon").reindex(horizon_order).reset_index())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rf_d["Horizon"].tolist() + rf_d["Horizon"].tolist()[::-1],
            y=rf_d["MAE"].tolist() + tgcn_d["MAE"].tolist()[::-1],
            fill="toself", fillcolor="rgba(39,174,96,0.12)",
            line=dict(width=0), showlegend=True,
            name="T-GCN improvement", hoverinfo="skip",
        ))
        for model in ["Persistence", "RandomForest", "T-GCN-V3"]:
            d = (results_df_local[results_df_local["Model"] == model]
                 .set_index("Horizon").reindex(horizon_order).reset_index())
            fig.add_trace(go.Scatter(
                x=d["Horizon"], y=d["MAE"],
                name="T-GCN (ours)" if model == "T-GCN-V3" else model,
                mode="lines+markers",
                line=dict(color=color_map[model], width=width_map[model]),
                marker=dict(size=9 if model == "T-GCN-V3" else 6),
            ))
        fig.update_layout(**dark_layout("T-GCN vs Baselines — MAE by Horizon", height=320))
        fig.update_yaxes(title_text="MAE")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Key Results")
        _mae5  = _ds.get("tgcn_mae_5min",  0)
        _mae15 = _ds.get("tgcn_mae_15min", 0)
        _mae30 = _ds.get("tgcn_mae_30min", 0)
        _imp5  = _ds.get("imp_5min",  0)
        _imp15 = _ds.get("imp_15min", 0)
        _imp30 = _ds.get("imp_30min", 0)

        _5min_msg  = f"5-min:  {_ds['best_5min']} best — MAE={_mae5:.3f}  ({_imp5:+.1f}% vs RF)"
        _15min_msg = f"15-min: {_ds['best_15min']} best — MAE={_mae15:.3f} ({_imp15:+.1f}% vs RF)"
        _30min_msg = f"30-min: {_ds['best_30min']} best — MAE={_mae30:.3f} ({_imp30:+.1f}% vs RF)"

        (st.success if _imp5  > 0 else st.warning)(_5min_msg)
        (st.success if _imp15 > 0 else st.warning)(_15min_msg)
        (st.success if _imp30 > 0 else st.warning)(_30min_msg)

        st.info("r=0.672 spatial autocorrelation confirms graph value")

        _fc_ids = _ds["fully_congested"]
        if _fc_ids:
            _fc_str = ", ".join(f"S{i}" for i in _fc_ids)
            st.error(f"Sensors {_fc_str}: congested 100% of time")
        else:
            st.success("No sensors at 100% congestion")

        st.markdown("### Training Summary")
        st.markdown("""
| | |
|---|---|
| Architecture | T-GCN (GConvGRU) |
| Parameters | 27,750 |
| Training time | 16 minutes |
| GPU | RTX 3060 |
| Best epoch | 10/30 |
| Loss | Huber Loss |
""")

    st.markdown("---")
    st.markdown("<div class='section-header'>How The System Works</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    steps = [
        ("1. Collect",
         "207 highway sensors report traffic speeds every 5 minutes. Each reading captures speed + time of day for that road location.",
         "5-min intervals", "#4A90D9"),
        ("2. Predict",
         "T-GCN processes the last 60 minutes of data across all 207 connected sensors simultaneously, predicting the next 5, 15, and 30 minutes.",
         "12 time steps input", "#7B68EE"),
        ("3. Act",
         "Operations team sees predicted congestion zones 15 minutes ahead. Drivers are repositioned BEFORE demand spikes — not after.",
         "15-min lookahead", "#27AE60"),
    ]
    for col, (title, desc, stat, color) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#0D1A2D;border:1px solid #1E3A5F;border-top:3px solid {color};
                        border-radius:10px;padding:1.2rem 1.4rem;height:100%'>
              <div style='color:{color};font-weight:700;font-size:0.9rem;margin-bottom:8px;
                          text-transform:uppercase;letter-spacing:0.5px'>{title}</div>
              <div style='color:#8899AA;font-size:0.83rem;line-height:1.6;margin-bottom:12px'>{desc}</div>
              <div style='background:#0A1520;border:1px solid #1E3A5F;border-radius:6px;
                          padding:4px 10px;display:inline-block;color:{color};
                          font-size:0.72rem;font-weight:600;letter-spacing:0.5px'>{stat}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PAGE 2 — LIVE TRAFFIC MAP
# ---------------------------------------------------------------------
elif st.session_state.page == "Live Traffic Map":
    st.markdown("### Live Traffic Network — Los Angeles")

    n_snapshots  = int(X_test.shape[0])
    max_snapshot = max(0, n_snapshots - 1)
    st.session_state.snapshot_idx = min(st.session_state.snapshot_idx, max_snapshot)

    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Map Controls**")
        map_mode = st.radio("Color sensors by:", ["Congestion Rate", "Prediction Error", "Mean Speed"])
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
            help="Drag to animate over time",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("◀ Back"):
                st.session_state.snapshot_idx = max(0, st.session_state.snapshot_idx - 1)
                st.rerun()
        with c2:
            if st.button("Next ▶"):
                st.session_state.snapshot_idx = min(max_snapshot, st.session_state.snapshot_idx + 1)
                st.rerun()

        snapshot = st.session_state.snapshot_idx
        st.markdown("---")
        st.write(f"Frame **{snapshot}** / {max_snapshot}")
        st.write(f"Approx time: **{compute_snapshot_label(snapshot)}**")

        snap_mean = float(X_test[snapshot, :, -1].mean())
        if snap_mean > 0.1:
            st.success(f"Free flow — {snap_mean:.3f}")
        elif snap_mean > -0.2:
            st.warning(f"Moderate — {snap_mean:.3f}")
        else:
            st.error(f"Congested — {snap_mean:.3f}")

    def render_map(snapshot_idx: int):
        m = folium.Map(location=[34.05, -118.30], zoom_start=11, control_scale=True, tiles=None)
        folium.TileLayer(
            tiles="https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="© Google", name="Google Streets",
            overlay=False, control=True, show=True, max_zoom=20,
            subdomains=["mt0", "mt1", "mt2", "mt3"],
        ).add_to(m)
        folium.TileLayer(
            tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="© Google", name="Google Satellite",
            overlay=False, control=True, show=False, max_zoom=20,
            subdomains=["mt0", "mt1", "mt2", "mt3"],
        ).add_to(m)
        folium.LayerControl().add_to(m)

        current_snap_speeds  = X_test[snapshot_idx, :, -1]
        current_predictions  = predictions[snapshot_idx, :, 2]

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
                fc = "#C0392B" if cong > 0.6 else "#E74C3C" if cong > 0.35 else "#F39C12" if cong > 0.2 else "#F7DC6F"
                radius = max(5, cong * 22)
                popup_extra = f"Congestion: {cong:.1%}"
            elif map_mode == "Prediction Error":
                mae  = float(row["prediction_mae"])
                norm = max(0, min(1, (mae - 0.12) / (0.35 - 0.12)))
                r_val = int(norm * 200 + 55)
                g_val = int((1 - norm) * 180 + 20)
                fc = f"#{r_val:02x}{g_val:02x}30"
                radius = max(5, mae * 28)
                popup_extra = f"Pred MAE: {mae:.4f}"
            else:
                spd = float(current_snap_speeds[idx])
                fc = "#27AE60" if spd > 0.1 else "#F39C12" if spd > -0.2 else "#E74C3C"
                radius = 7
                popup_extra = f"Current speed: {spd:.3f}"

            pred_15 = float(current_predictions[idx])
            popup_html = f"""
            <div style='font-family:Arial;font-size:12px;min-width:170px;color:#111'>
              <b style='color:#1B3A6B'>Sensor {idx}</b><br>
              <hr style='margin:4px 0'>
              {popup_extra}<br>
              <b>15-min prediction: {pred_15:.3f}</b><br>
              Level: <b>{level}</b><br>
              Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}
            </div>"""
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius, color="white", weight=0.5,
                fill=True, fill_color=fc, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"S{idx}" if show_labels else f"S{idx} | {level} | Click for details",
            ).add_to(m)
        return m

    with col_map:
        current_map = render_map(st.session_state.snapshot_idx)
        st_folium(current_map, height=560, use_container_width=True,
                  key=f"map-{st.session_state.snapshot_idx}-{map_mode}")

        if map_mode == "Congestion Rate":
            legend_items = [("#C0392B","Critical > 60%"),("#E74C3C","High 35–60%"),
                            ("#F39C12","Moderate 20–35%"),("#F7DC6F","Low < 20%")]
        elif map_mode == "Prediction Error":
            legend_items = [("#c83050","High error MAE > 0.30"),
                            ("#e07030","Medium 0.20–0.30"),("#b0b030","Low < 0.20")]
        else:
            legend_items = [("#27AE60","Free flow > 0.1"),
                            ("#F39C12","Moderate −0.2 to 0.1"),("#E74C3C","Congested < −0.2")]

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
        st.caption("Click a sensor for details. Use the slider or step buttons to animate over time.")

# ---------------------------------------------------------------------
# PAGE 3 — FORECAST EXPLORER
# ---------------------------------------------------------------------
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
        pred_s = predictions[-n_points:, sensor_id, h_idx]
        true_s = targets[-n_points:, sensor_id, h_idx]
        time_x = list(range(n_points))
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
        fig.update_xaxes(title_text="Time Snapshot")
        fig.update_yaxes(title_text="Normalized Speed")
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=true_s.tolist(), y=pred_s.tolist(), mode="markers",
                marker=dict(color="#4A90D9", opacity=0.55, size=5), name="Predictions"))
            mn, mx = float(min(true_s.min(), pred_s.min())), float(max(true_s.max(), pred_s.max()))
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
            fig3.update_xaxes(title_text="Error (pred − actual)")
            fig3.update_yaxes(title_text="Count")
            st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 4 — DECISION DASHBOARD
# ---------------------------------------------------------------------
elif st.session_state.page == "Decision Dashboard":
    st.markdown("### Decision Dashboard")
    st.caption("Operational recommendations powered by T-GCN traffic intelligence")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("**IMMEDIATE ACTION**\n\nSensors 16 & 196 congested **100%** of time\n\n"
                 "Deploy 5 permanent extra drivers\nLocation: 34.07°N 118.23°W")
    with c2:
        st.warning("**DAILY SCHEDULE**\n\nPre-position drivers at:\n"
                   "• **6:45am** (before morning rush)\n• **4:45pm** (before evening rush)\n\n"
                   "Evening 33% worse than morning")
    with c3:
        st.info("**APP INTEGRATION**\n\nShow ETA uncertainty flag for\n42 sensors with MAE > 0.245\n\n"
                "Message: 'ETA may vary due\nto complex traffic patterns'")

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

        st.markdown("#### Congestion Level Distribution")
        level_counts = sensor_df["congestion_level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]
        fig = px.bar(level_counts, x="Level", y="Count", color="Level",
                     color_discrete_map={"Critical":"#E74C3C","High":"#F39C12",
                                         "Medium":"#3498DB","Low":"#27AE60"})
        fig.update_layout(**dark_layout("Sensors by Congestion Level", height=260, legend_bottom=False))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Number of Sensors")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### System Metrics")
        _n_high = int((sensor_df["congestion_level"] == "High").sum())
        _avg_cong = float(sensor_df["congestion_rate"].mean()) * 100
        _fc_label = ", ".join(f"S{i}" for i in _ds["fully_congested"]) or "None"
        st.metric("Critical zones",      str(_ds["n_critical"]),           "Require permanent staffing")
        st.metric("High priority zones", str(_n_high),                     "Rush hour coverage")
        st.metric("Avg congestion rate", f"{_avg_cong:.1f}%",              "Across all sensors")
        st.metric("Most congested",      _fc_label,                        "100% congestion rate")
        st.metric("Hardest to predict",  f"S{_ds['worst_sensor_idx']}",    f"MAE = {_ds['worst_sensor_mae']:.3f}")

        st.markdown("#### Limitations")
        st.markdown("""
- Data from 2012 — pre-pandemic
- Highways only, no surface streets
- 30-min less reliable (MAE=0.275)
- No weather/incident features
""")

# ---------------------------------------------------------------------
# PAGE 5 — MODEL RESULTS
# ---------------------------------------------------------------------
elif st.session_state.page == "Model Results":
    st.markdown("### Model Results")

    tab1, tab2, tab3 = st.tabs(["Performance Table", "Error Analysis", "Architecture"])

    with tab1:
        col1, col2 = st.columns(2)
        working_results = results_df.copy()
        working_results["Horizon"] = working_results["Horizon"].apply(format_horizon_for_table)
        horizon_order = ["5min", "15min", "30min"]

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

        st.markdown("**Improvement of T-GCN vs Random Forest**")
        rf_df   = working_results[working_results["Model"] == "RandomForest"].set_index("Horizon").reindex(horizon_order)
        tgcn_df = working_results[working_results["Model"] == "T-GCN-V3"].set_index("Horizon").reindex(horizon_order)
        imp_rows = []
        for h in horizon_order:
            rf_mae   = float(rf_df.loc[h, "MAE"])
            tgcn_mae = float(tgcn_df.loc[h, "MAE"])
            imp_rows.append({"Horizon": h, "RF MAE": round(rf_mae, 4),
                             "T-GCN MAE": round(tgcn_mae, 4),
                             "Improvement": f"{(rf_mae - tgcn_mae)/rf_mae*100:+.1f}%"})
        st.dataframe(pd.DataFrame(imp_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-header'>T-GCN Performance Summary</div>", unsafe_allow_html=True)
        summary_cols = st.columns(3)
        summaries = []
        for h in horizon_order:
            rf_mae   = float(rf_df.loc[h, "MAE"])
            tgcn_mae = float(tgcn_df.loc[h, "MAE"])
            imp      = (rf_mae - tgcn_mae) / rf_mae * 100
            summaries.append((h, f"{tgcn_mae:.3f}", f"{imp:+.1f}%",
                               "#27AE60" if imp >= 0 else "#E74C3C"))
        for col, (h, mae, imp, color) in zip(summary_cols, summaries):
            with col:
                st.markdown(f"""
                <div style='background:#0D1A2D;border:1px solid #1E3A5F;border-radius:10px;
                            padding:1.4rem;text-align:center'>
                  <div style='color:#8899AA;font-size:0.78rem;text-transform:uppercase;
                              letter-spacing:1px'>{h}</div>
                  <div style='color:#4A90D9;font-size:2rem;font-weight:800;margin:8px 0'>{mae}</div>
                  <div style='color:{color};font-size:1rem;font-weight:600'>{imp}</div>
                  <div style='color:#6B8CAE;font-size:0.72rem;margin-top:4px'>vs Random Forest</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**Model-driven error diagnostics (computed live from predictions)**")
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
        sensor_rank.columns = ["Sensor ID","Latitude","Longitude",
                                "Congestion Rate","MAE (15min)","Level"]
        sensor_rank["Congestion %"] = sensor_rank["Congestion Rate"] * 100

        fig_rank = px.scatter(sensor_rank, x="Congestion %", y="MAE (15min)", color="Level",
            hover_data=["Sensor ID","Latitude","Longitude"],
            color_discrete_map={"Critical":"#E74C3C","High":"#F39C12",
                                 "Medium":"#3498DB","Low":"#27AE60"})
        fig_rank.update_layout(**dark_layout("Sensor Reliability vs Congestion", height=340))
        fig_rank.update_xaxes(title_text="Congestion Rate (%)")
        st.plotly_chart(fig_rank, use_container_width=True)

        st.dataframe(sensor_rank[["Sensor ID","Level","Congestion Rate","MAE (15min)"]].head(30),
                     hide_index=True, use_container_width=True, height=320)

        st.markdown("**Top high-error sensors by horizon**")
        selected_h = st.selectbox("Horizon", ["5min","15min","30min"], key="results_horizon")
        top_error = (sensor_horizon_df[sensor_horizon_df["Horizon"] == selected_h]
                     .sort_values("MAE", ascending=False).head(20))
        fig_top = px.bar(top_error, x="Sensor ID", y="MAE",
                         color="MAE", color_continuous_scale="OrRd")
        fig_top.update_layout(**dark_layout(f"Top 20 Sensors by MAE ({selected_h})",
                                            height=320, legend_bottom=False))
        st.plotly_chart(fig_top, use_container_width=True)

    with tab3:
        st.markdown("**Model Architecture**")
        st.markdown("""
- **Architecture:** T-GCN (graph convolution + recurrent temporal modelling)
- **Sensors:** 207 LA highway detectors
- **Horizons:** 5 min, 15 min, 30 min
- **Objective:** minimise short-horizon forecasting error under spatial dependence
""")
        selected_sensor  = st.selectbox("Inspect sensor", range(207),
                                         format_func=lambda x: f"Sensor {x:03d}", key="diag_sensor")
        selected_horizon = st.selectbox("Forecast horizon",
                                         [("5min",0),("15min",2),("30min",5)],
                                         format_func=lambda x: x[0], key="diag_horizon")
        h_name, h_idx = selected_horizon
        pred_line = predictions[:, selected_sensor, h_idx]
        true_line = targets[:,    selected_sensor, h_idx]
        err_line  = np.abs(pred_line - true_line)

        diag_df = pd.DataFrame({"t": np.arange(len(pred_line)),
                                 "Actual": true_line, "Predicted": pred_line,
                                 "Absolute Error": err_line})

        fig_beh = go.Figure()
        fig_beh.add_trace(go.Scatter(x=diag_df["t"], y=diag_df["Actual"],
            name="Actual", line=dict(color="#2ECC71", width=2)))
        fig_beh.add_trace(go.Scatter(x=diag_df["t"], y=diag_df["Predicted"],
            name="Predicted", line=dict(color="#E74C3C", width=2, dash="dot")))
        fig_beh.update_layout(**dark_layout(
            f"Sensor {selected_sensor} — {h_name}", height=330))
        fig_beh.update_xaxes(title_text="Snapshot")
        fig_beh.update_yaxes(title_text="Normalized Speed")
        st.plotly_chart(fig_beh, use_container_width=True)

        fig_err = px.histogram(diag_df, x="Absolute Error", nbins=40,
                               color_discrete_sequence=["#4A90D9"])
        fig_err.update_layout(**dark_layout(
            f"Error distribution — Sensor {selected_sensor} ({h_name})",
            height=280, legend_bottom=False))
        st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
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
