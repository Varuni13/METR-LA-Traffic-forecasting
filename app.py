from pathlib import Path

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
    page_icon="TI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# DARK THEME CSS
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
  .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .block-container {
        padding-top: 1.6rem;
    }
  .main-header {
    background: linear-gradient(135deg, #1B3A6B 0%, #0F2444 100%);
    padding: 2rem; border-radius: 12px;
    border-left: 5px solid #4A90D9;
    margin-bottom: 1.5rem;
  }
  .kpi-card {
    background: linear-gradient(135deg, #1a1f2e, #242938);
    border: 1px solid #2E4080;
    border-radius: 10px; padding: 1.2rem;
    text-align: center;
  }
  .kpi-val { font-size: 2rem; font-weight: 800; color: #4A90D9; }
  .kpi-lbl { font-size: 0.8rem; color: #8899AA; margin-top: 4px; }
  .insight-box {
    background: #1a2035; border-left: 4px solid #4A90D9;
    padding: 1rem 1.2rem; border-radius: 6px; margin: 8px 0;
  }
  div[data-testid="metric-container"] {
    background: #1a1f2e; border: 1px solid #2E4080;
    border-radius: 8px; padding: 12px;
  }
    .top-nav-label {
        color: #8ea4cc;
        font-size: 0.74rem;
        margin-bottom: 0.3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid #31456f;
        background: linear-gradient(180deg, #16233e 0%, #121c33 100%);
        color: #a9bfde;
        font-size: 0.79rem;
        font-weight: 600;
        min-height: 2rem;
        padding: 0.15rem 0.8rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #4A90D9;
        color: #d8e7ff;
        box-shadow: 0 0 0 1px rgba(74, 144, 217, 0.25);
    }
    .stButton > button[kind="primary"] {
        border-color: #4A90D9;
        background: linear-gradient(180deg, #2a4f8c 0%, #223f6f 100%);
        color: #f3f8ff;
        box-shadow: 0 4px 14px rgba(55, 118, 205, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #3160a9 0%, #2a4f8c 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
@st.cache_data
def load_all_data():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    predictions = np.load("data/gnn_v3_predictions.npy")
    targets = np.load("data/gnn_v3_targets.npy")
    X_test = np.load("data/X_test.npy")
    sensor_df = pd.read_csv("data/sensor_analysis.csv")
    results_df = pd.read_csv("data/all_results_v3.csv")
    sensor_mae = [
        mean_absolute_error(targets[:, i, 2], predictions[:, i, 2])
        for i in range(207)
    ]
    sensor_df["prediction_mae"] = sensor_mae
    return predictions, targets, X_test, sensor_df, results_df


predictions, targets, X_test, sensor_df, results_df = load_all_data()


@st.cache_data
def build_model_diagnostics(pred: np.ndarray, true: np.ndarray):
    horizon_to_idx = {"5min": 0, "15min": 2, "30min": 5}
    horizon_rows = []
    sensor_frames = []

    for horizon, h_idx in horizon_to_idx.items():
        pred_h = pred[:, :, h_idx]
        true_h = true[:, :, h_idx]
        abs_err = np.abs(true_h - pred_h)

        horizon_rows.append(
            {
                "Horizon": horizon,
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt(np.mean((true_h - pred_h) ** 2))),
            }
        )

        sensor_frames.append(
            pd.DataFrame(
                {
                    "Sensor ID": np.arange(abs_err.shape[1]),
                    "Horizon": horizon,
                    "MAE": abs_err.mean(axis=0),
                }
            )
        )

    horizon_metrics = pd.DataFrame(horizon_rows)
    sensor_metrics = pd.concat(sensor_frames, ignore_index=True)
    return horizon_metrics, sensor_metrics

# Guard against unexpected index dtype/value issues.
sensor_df["index"] = sensor_df["index"].astype(int).clip(0, 206)
horizon_metrics_df, sensor_horizon_df = build_model_diagnostics(predictions, targets)

PAGES = {
    "Problem": "Overview",
    "Network": "Live Traffic Map",
    "Forecast": "Forecast Explorer",
    "Decision": "Decision Dashboard",
    "Results": "Model Results",
}

if "page" not in st.session_state:
    st.session_state.page = "Overview"

# ---------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------
def story_line(active_stage: str):
    st.markdown('<div class="top-nav-label">Story Navigation</div>', unsafe_allow_html=True)
    stages = ["Problem", "Network", "Forecast", "Decision", "Results"]
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages):
        active = stage == active_stage
        label = stage
        btn_type = "primary" if active else "secondary"
        if col.button(label, key=f"top-nav-{stage}", type=btn_type, use_container_width=True):
            st.session_state.page = PAGES[stage]
            st.rerun()


def format_horizon_for_table(h: str) -> str:
    # Handle slight variations that can appear in result files.
    key = str(h).replace(" ", "").lower()
    if key in {"5min", "5m", "5"}:
        return "5min"
    if key in {"15min", "15m", "15"}:
        return "15min"
    if key in {"30min", "30m", "30"}:
        return "30min"
    return str(h)


def compute_snapshot_label(snapshot_idx: int) -> str:
    mins = snapshot_idx * 5
    hh = (mins // 60) % 24
    mm = mins % 60
    return f"{hh:02d}:{mm:02d}"


# ---------------------------------------------------------------------
# PAGE 1: OVERVIEW  (Problem -> Network)
# ---------------------------------------------------------------------
if st.session_state.page == "Overview":
    story_line("Problem")

    st.markdown(
        """
        <div class="main-header">
          <h1 style="color:white;margin:0">METR-LA Traffic Intelligence</h1>
          <p style="color:#A8C0E8;margin:8px 0 0 0">
          Graph Neural Network for real-time traffic forecasting across
          207 Los Angeles highway sensors. Predicting 5, 15, and 30 minutes ahead.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("207", "Highway Sensors"),
        ("0.140", "MAE at 5-min"),
        ("0.672", "Spatial Correlation"),
        ("25", "Critical Zones"),
        ("22.8%", "Congestion Rate"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Why Graph ML?")
        st.markdown(
            """
            <div class="insight-box">
            Standard models treat each sensor independently. But LA highway sensors
            are <b>spatially connected</b> — a slowdown at one sensor spreads to neighbors
            within minutes. Our T-GCN model captures this using the road network graph
            (1,722 connections between 207 sensors).
            </div>
            """,
            unsafe_allow_html=True,
        )

        results_df_local = results_df[
            results_df["Model"].isin(["Persistence", "RandomForest", "T-GCN-V3"])
        ].copy()
        results_df_local["Horizon"] = results_df_local["Horizon"].apply(format_horizon_for_table)

        horizon_order = ["5min", "15min", "30min"]
        color_map = {
            "Persistence": "#3498DB",
            "RandomForest": "#27AE60",
            "T-GCN-V3": "#E74C3C",
        }
        width_map = {"Persistence": 1.5, "RandomForest": 1.5, "T-GCN-V3": 3}

        fig = go.Figure()
        for model in ["Persistence", "RandomForest", "T-GCN-V3"]:
            d = (
                results_df_local[results_df_local["Model"] == model]
                .set_index("Horizon")
                .reindex(horizon_order)
                .reset_index()
            )
            fig.add_trace(
                go.Scatter(
                    x=d["Horizon"],
                    y=d["MAE"],
                    name="T-GCN (ours)" if model == "T-GCN-V3" else model,
                    mode="lines+markers",
                    line=dict(color=color_map[model], width=width_map[model]),
                    marker=dict(size=9 if model == "T-GCN-V3" else 6),
                )
            )

        fig.update_layout(
            title="T-GCN Beats Baselines at Short Horizons",
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=320,
            legend=dict(orientation="h", y=-0.25),
            xaxis=dict(gridcolor="#2E4080"),
            yaxis=dict(gridcolor="#2E4080", title="MAE"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Key Results")
        st.success("Best at 5-min: T-GCN MAE=0.140 (-5.2% vs RF)")
        st.success("Best at 15-min: T-GCN MAE=0.200 (-0.3% vs RF)")
        st.warning("30-min: RF slightly better (temporal patterns dominate)")
        st.info("r=0.672 spatial autocorrelation confirms graph value")
        st.error("Sensors 16 and 196: congested 100% of time")

        st.markdown("### Training Summary")
        st.markdown(
            """
            | | |
            |---|---|
            | Architecture | T-GCN (GConvGRU) |
            | Parameters | 27,750 |
            | Training time | 16 minutes |
            | GPU | RTX 3060 |
            | Best epoch | 10/30 |
            | Loss | Huber Loss |
            """
        )

# ---------------------------------------------------------------------
# PAGE 2: LIVE TRAFFIC MAP  (Network -> Forecast) with time animation slider
# ---------------------------------------------------------------------
elif st.session_state.page == "Live Traffic Map":
    story_line("Network")
    st.markdown("### Live Traffic Network — Los Angeles")

    n_snapshots = int(X_test.shape[0])
    max_snapshot = max(0, n_snapshots - 1)

    if "snapshot_idx" not in st.session_state:
        st.session_state.snapshot_idx = min(100, max_snapshot)

    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Map Controls**")
        map_mode = st.radio(
            "Color sensors by:",
            ["Congestion Rate", "Prediction Error", "Mean Speed"],
        )
        show_labels = st.checkbox("Show sensor labels", value=False)

        st.markdown("---")
        st.markdown("**Filter by congestion level:**")
        show_critical = st.checkbox("Critical (>60%)", value=True)
        show_high = st.checkbox("High (25-60%)", value=True)
        show_medium = st.checkbox("Medium (10-25%)", value=True)
        show_low = st.checkbox("Low (<10%)", value=True)

        st.markdown("---")
        st.markdown("**Time Animation Slider**")
        st.session_state.snapshot_idx = st.slider(
            "Test snapshot (5-min intervals):",
            0,
            max_snapshot,
            st.session_state.snapshot_idx,
            help="Drag to see traffic changing over time",
        )

        step_col1, step_col2 = st.columns(2)
        with step_col1:
            if st.button("◀ Step"):
                st.session_state.snapshot_idx = max(0, st.session_state.snapshot_idx - 1)
                st.rerun()
        with step_col2:
            if st.button("Step ▶"):
                st.session_state.snapshot_idx = min(max_snapshot, st.session_state.snapshot_idx + 1)
                st.rerun()

        snapshot = st.session_state.snapshot_idx

        st.markdown("---")
        st.markdown("**Snapshot context**")
        st.write(f"Frame: **{snapshot}** / {max_snapshot}")
        st.write(f"Approx local time: **{compute_snapshot_label(snapshot)}**")

        snap_mean = float(X_test[snapshot, :, -1].mean())
        st.markdown("**Current conditions:**")
        if snap_mean > 0.1:
            st.success(f"Free flow\nMean speed: {snap_mean:.3f}")
        elif snap_mean > -0.2:
            st.warning(f"Moderate\nMean speed: {snap_mean:.3f}")
        else:
            st.error(f"Congested\nMean speed: {snap_mean:.3f}")

    def render_map(snapshot_idx: int):
        m = folium.Map(
            location=[34.05, -118.30],
            zoom_start=11,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )

        current_snap_speeds = X_test[snapshot_idx, :, -1]
        current_predictions = predictions[snapshot_idx, :, 2]  # 15-min predictions

        for _, row in sensor_df.iterrows():
            level = str(row["congestion_level"])
            if level == "Critical" and not show_critical:
                continue
            if level == "High" and not show_high:
                continue
            if level == "Medium" and not show_medium:
                continue
            if level == "Low" and not show_low:
                continue

            idx = int(row["index"])
            if idx < 0 or idx >= 207:
                continue

            if map_mode == "Congestion Rate":
                cong = float(row["congestion_rate"])
                if cong > 0.6:
                    fc = "#C0392B"
                elif cong > 0.35:
                    fc = "#E74C3C"
                elif cong > 0.2:
                    fc = "#F39C12"
                else:
                    fc = "#F7DC6F"
                radius = max(5, cong * 22)
                popup_extra = f"Congestion: {cong:.1%}"

            elif map_mode == "Prediction Error":
                mae = float(row["prediction_mae"])
                norm = (mae - 0.12) / (0.35 - 0.12)
                norm = max(0, min(1, norm))
                r_val = int(norm * 200 + 55)
                g_val = int((1 - norm) * 180 + 20)
                fc = f"#{r_val:02x}{g_val:02x}30"
                radius = max(5, mae * 28)
                popup_extra = f"Pred MAE: {mae:.4f}"

            else:
                spd = float(current_snap_speeds[idx])
                if spd > 0.1:
                    fc = "#27AE60"
                elif spd > -0.2:
                    fc = "#F39C12"
                else:
                    fc = "#E74C3C"
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
            </div>
            """

            tooltip = f"S{idx}" if show_labels else f"S{idx} | {level} | Click for details"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color="white",
                weight=0.5,
                fill=True,
                fill_color=fc,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=tooltip,
            ).add_to(m)

        return m

    with col_map:
        current_map = render_map(st.session_state.snapshot_idx)
        st_folium(current_map, height=560, use_container_width=True, key=f"map-{st.session_state.snapshot_idx}-{map_mode}")
        st.caption("Click a sensor for details. Use the slider or step buttons to animate over time.")

# ---------------------------------------------------------------------
# PAGE 3: FORECAST EXPLORER  (Forecast)
# ---------------------------------------------------------------------
elif st.session_state.page == "Forecast Explorer":
    story_line("Forecast")

    st.markdown("### Interactive Forecast Explorer")
    st.caption("Explore T-GCN predictions vs actual speeds for any sensor and time period")

    col1, col2 = st.columns([1, 3])

    with col1:
        sensor_id = st.selectbox("Select Sensor:", range(207), format_func=lambda x: f"Sensor {x:03d}")
        horizon = st.radio("Forecast Horizon:", ["5 min", "15 min", "30 min"])
        h_idx = {"5 min": 0, "15 min": 2, "30 min": 5}[horizon]

        max_points = int(targets.shape[0])
        n_points = st.slider("Data points to show:", 50, min(500, max_points), min(220, max_points))

        info = sensor_df[sensor_df["index"] == sensor_id].iloc[0]
        st.markdown("---")
        st.markdown("**Sensor Info**")
        st.metric("Congestion Rate", f"{info['congestion_rate']:.1%}")
        st.metric("MAE (15min)", f"{info['prediction_mae']:.4f}")

        level = str(info["congestion_level"])
        if level == "Critical":
            st.error(f"Level: {level}")
        elif level == "High":
            st.warning(f"Level: {level}")
        else:
            st.success(f"Level: {level}")

        st.markdown("**Sensor Location**")
        mini_m = folium.Map(
            location=[info["latitude"], info["longitude"]],
            zoom_start=14,
            tiles="CartoDB dark_matter",
            width=220,
            height=180,
        )
        folium.CircleMarker(
            location=[info["latitude"], info["longitude"]],
            radius=12,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
        ).add_to(mini_m)
        st_folium(mini_m, height=220, use_container_width=True)

    with col2:
        pred_s = predictions[-n_points:, sensor_id, h_idx]
        true_s = targets[-n_points:, sensor_id, h_idx]
        time_x = list(range(n_points))

        mae_val = mean_absolute_error(true_s, pred_s)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=true_s.tolist(),
                name="Actual Speed",
                line=dict(color="#27AE60", width=2),
                fill="tozeroy",
                fillcolor="rgba(39,174,96,0.10)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=pred_s.tolist(),
                name=f"T-GCN Prediction ({horizon})",
                line=dict(color="#E74C3C", width=2, dash="dot"),
            )
        )

        error = np.abs(pred_s - true_s)
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=(pred_s + error).tolist(),
                line=dict(width=0),
                showlegend=False,
                fillcolor="rgba(231,76,60,0.14)",
                fill="tonexty",
            )
        )

        fig.update_layout(
            title=f"Sensor {sensor_id} — {horizon} Forecast | MAE = {mae_val:.4f}",
            xaxis_title="Time Snapshot (test set)",
            yaxis_title="Normalized Speed",
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=340,
            legend=dict(orientation="h", y=-0.22),
            xaxis=dict(gridcolor="#2E4080"),
            yaxis=dict(gridcolor="#2E4080"),
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=true_s.tolist(),
                    y=pred_s.tolist(),
                    mode="markers",
                    marker=dict(color="#4A90D9", opacity=0.55, size=5),
                    name="Predictions",
                )
            )
            min_v = float(min(true_s.min(), pred_s.min()))
            max_v = float(max(true_s.max(), pred_s.max()))
            fig2.add_trace(
                go.Scatter(
                    x=[min_v, max_v],
                    y=[min_v, max_v],
                    line=dict(color="#E74C3C", dash="dash"),
                    name="Perfect prediction",
                )
            )
            fig2.update_layout(
                title="Predicted vs Actual",
                paper_bgcolor="#1a1f2e",
                plot_bgcolor="#1a1f2e",
                font=dict(color="white"),
                height=290,
                xaxis=dict(title="Actual", gridcolor="#2E4080"),
                yaxis=dict(title="Predicted", gridcolor="#2E4080"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            fig3 = go.Figure()
            fig3.add_trace(
                go.Histogram(
                    x=(pred_s - true_s).tolist(),
                    nbinsx=30,
                    marker_color="#4A90D9",
                    name="Prediction Error",
                )
            )
            fig3.update_layout(
                title="Error Distribution",
                paper_bgcolor="#1a1f2e",
                plot_bgcolor="#1a1f2e",
                font=dict(color="white"),
                height=290,
                xaxis=dict(title="Error (pred - actual)", gridcolor="#2E4080"),
                yaxis=dict(title="Count", gridcolor="#2E4080"),
            )
            st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------
# PAGE 4: DECISION DASHBOARD  (Decision)
# ---------------------------------------------------------------------
elif st.session_state.page == "Decision Dashboard":
    story_line("Decision")

    st.markdown("### Ride-Hailing Decision Dashboard")
    st.caption("Operational recommendations powered by T-GCN traffic intelligence")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.error(
            "**IMMEDIATE ACTION**\n\n"
            "Sensors 16 & 196 congested **100%** of time\n\n"
            "Deploy 5 permanent extra drivers\n"
            "Location: 34.07°N 118.23°W"
        )
    with c2:
        st.warning(
            "**DAILY SCHEDULE**\n\n"
            "Pre-position drivers at:\n"
            "• **6:45am** (before morning rush)\n"
            "• **4:45pm** (before evening rush)\n\n"
            "Evening 33% worse than morning"
        )
    with c3:
        st.info(
            "**APP INTEGRATION**\n\n"
            "Show ETA uncertainty flag for\n"
            "42 sensors with MAE > 0.245\n\n"
            "Message: 'ETA may vary due\n"
            "to complex traffic patterns'"
        )

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Priority Sensor Zones")
        action_df = pd.DataFrame(
            {
                "Priority": ["🔴 CRITICAL", "🔴 CRITICAL", "🟠 HIGH", "🟠 HIGH", "🟢 STANDARD"],
                "Zone": [
                    "Sensors 16, 196",
                    "Sensor 56",
                    "58 sensors (25-60% congested)",
                    "42 sensors (low reliability)",
                    "All 207 sensors",
                ],
                "Condition": [
                    "100% congested",
                    "91% congested",
                    "Peak hours",
                    "MAE > 0.245",
                    "Evening rush 5-7pm",
                ],
                "Recommended Action": [
                    "Permanent driver buffer (5+ drivers)",
                    "Pre-position 6:45am & 4:45pm daily",
                    "Surge pricing: speed < -0.3",
                    "ETA uncertainty flag in app",
                    "Extra incentives from 4:30pm Mon-Fri",
                ],
            }
        )
        st.dataframe(action_df, hide_index=True, use_container_width=True, height=230)

        st.markdown("#### Congestion Level Distribution")
        level_counts = sensor_df["congestion_level"].value_counts().reset_index()
        level_counts.columns = ["Level", "Count"]

        fig = px.bar(
            level_counts,
            x="Level",
            y="Count",
            color="Level",
            color_discrete_map={
                "Critical": "#E74C3C",
                "High": "#F39C12",
                "Medium": "#3498DB",
                "Low": "#27AE60",
            },
            title="Sensors by Congestion Level",
        )
        fig.update_layout(
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=260,
            showlegend=False,
            xaxis=dict(gridcolor="#2E4080"),
            yaxis=dict(gridcolor="#2E4080", title="Number of Sensors"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### System Metrics")
        st.metric("Critical zones", "25", "Require permanent staffing")
        st.metric("High priority zones", "58", "Rush hour coverage")
        st.metric("Avg congestion rate", "25.9%", "Across all sensors")
        st.metric("Most congested", "S16, S196", "100% congestion rate")
        st.metric("Hardest to predict", "S62", "MAE = 0.347")

        st.markdown("#### Limitations")
        st.markdown(
            """
            - Data from 2012 — pre-pandemic
            - Highways only, no surface streets
            - 30-min less reliable (MAE=0.275)
            - No weather/incident features
            """
        )

# ---------------------------------------------------------------------
# PAGE 5: MODEL RESULTS  (Results)
# ---------------------------------------------------------------------
elif st.session_state.page == "Model Results":
    story_line("Results")
    st.markdown("### Full Model Results")

    tab1, tab2, tab3 = st.tabs(["Performance Table", "Error Analysis", "Architecture"])

    with tab1:
        col1, col2 = st.columns(2)

        working_results = results_df.copy()
        working_results["Horizon"] = working_results["Horizon"].apply(format_horizon_for_table)
        horizon_order = ["5min", "15min", "30min"]

        with col1:
            st.markdown("**MAE Results (lower = better)**")
            mae_pivot = (
                working_results.pivot(index="Horizon", columns="Model", values="MAE")
                .reindex(horizon_order)
            )
            st.dataframe(
                mae_pivot.style.highlight_min(axis=1, color="#1E5C1E"),
                use_container_width=True,
            )

        with col2:
            st.markdown("**RMSE Results (lower = better)**")
            rmse_pivot = (
                working_results.pivot(index="Horizon", columns="Model", values="RMSE")
                .reindex(horizon_order)
            )
            st.dataframe(
                rmse_pivot.style.highlight_min(axis=1, color="#1E5C1E"),
                use_container_width=True,
            )

        st.markdown("**Improvement of T-GCN vs Random Forest**")
        imp_data = pd.DataFrame(
            {
                "Horizon": ["5min", "15min", "30min"],
                "RF MAE": [0.1477, 0.2005, 0.2631],
                "T-GCN MAE": [0.1400, 0.1998, 0.2752],
                "Improvement": ["+5.2%", "+0.3%", "-4.6%"],
            }
        )
        st.dataframe(imp_data, hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("**Model-driven error diagnostics (computed live from predictions)**")

        c1, c2 = st.columns(2)
        with c1:
            fig_h = go.Figure()
            fig_h.add_trace(
                go.Bar(
                    x=horizon_metrics_df["Horizon"],
                    y=horizon_metrics_df["MAE"],
                    name="MAE",
                    marker_color="#4A90D9",
                )
            )
            fig_h.update_layout(
                title="Computed T-GCN MAE by Horizon",
                paper_bgcolor="#1a1f2e",
                plot_bgcolor="#1a1f2e",
                font=dict(color="white"),
                height=300,
                xaxis=dict(gridcolor="#2E4080"),
                yaxis=dict(gridcolor="#2E4080", title="MAE"),
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with c2:
            fig_r = go.Figure()
            fig_r.add_trace(
                go.Bar(
                    x=horizon_metrics_df["Horizon"],
                    y=horizon_metrics_df["RMSE"],
                    name="RMSE",
                    marker_color="#E67E22",
                )
            )
            fig_r.update_layout(
                title="Computed T-GCN RMSE by Horizon",
                paper_bgcolor="#1a1f2e",
                plot_bgcolor="#1a1f2e",
                font=dict(color="white"),
                height=300,
                xaxis=dict(gridcolor="#2E4080"),
                yaxis=dict(gridcolor="#2E4080", title="RMSE"),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("**Sensor Error Ranking (interactive)**")
        sensor_rank = sensor_df[
            ["index", "latitude", "longitude", "congestion_rate", "prediction_mae", "congestion_level"]
        ].copy()
        sensor_rank = sensor_rank.sort_values("prediction_mae", ascending=False)
        sensor_rank.columns = [
            "Sensor ID",
            "Latitude",
            "Longitude",
            "Congestion Rate",
            "MAE (15min)",
            "Level",
        ]
        sensor_rank["Congestion %"] = sensor_rank["Congestion Rate"] * 100

        fig_rank = px.scatter(
            sensor_rank,
            x="Congestion %",
            y="MAE (15min)",
            color="Level",
            hover_data=["Sensor ID", "Latitude", "Longitude"],
            title="Sensor Reliability vs Congestion",
            color_discrete_map={
                "Critical": "#E74C3C",
                "High": "#F39C12",
                "Medium": "#3498DB",
                "Low": "#27AE60",
            },
        )
        fig_rank.update_layout(
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=340,
            xaxis=dict(gridcolor="#2E4080", title="Congestion Rate (%)"),
            yaxis=dict(gridcolor="#2E4080"),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.dataframe(
            sensor_rank[["Sensor ID", "Level", "Congestion Rate", "MAE (15min)"]].head(30),
            hide_index=True,
            use_container_width=True,
            height=320,
        )

        st.markdown("**Top high-error sensors by horizon**")
        selected_h = st.selectbox("Horizon", ["5min", "15min", "30min"], key="results_horizon")
        top_error = (
            sensor_horizon_df[sensor_horizon_df["Horizon"] == selected_h]
            .sort_values("MAE", ascending=False)
            .head(20)
        )
        fig_top = px.bar(
            top_error,
            x="Sensor ID",
            y="MAE",
            title=f"Top 20 Sensors by MAE ({selected_h})",
            color="MAE",
            color_continuous_scale="OrRd",
        )
        fig_top.update_layout(
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=320,
            xaxis=dict(gridcolor="#2E4080"),
            yaxis=dict(gridcolor="#2E4080"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with tab3:
        st.markdown("**Model Architecture and Behavior**")
        st.markdown(
            """
            - Architecture: T-GCN (graph convolution + recurrent temporal modeling)
            - Sensors: 207 LA highway detectors
            - Horizons: 5min, 15min, 30min
            - Objective: minimize short-horizon forecasting error under spatial dependence
            """
        )

        selected_sensor = st.selectbox(
            "Inspect temporal behavior for sensor",
            options=list(range(207)),
            format_func=lambda x: f"Sensor {x:03d}",
            key="diag_sensor",
        )
        selected_horizon = st.selectbox(
            "Forecast horizon",
            options=[("5min", 0), ("15min", 2), ("30min", 5)],
            format_func=lambda x: x[0],
            key="diag_horizon",
        )

        h_name, h_idx = selected_horizon
        pred_line = predictions[:, selected_sensor, h_idx]
        true_line = targets[:, selected_sensor, h_idx]
        err_line = np.abs(pred_line - true_line)

        diag_df = pd.DataFrame(
            {
                "t": np.arange(len(pred_line)),
                "Actual": true_line,
                "Predicted": pred_line,
                "Absolute Error": err_line,
            }
        )

        fig_behavior = go.Figure()
        fig_behavior.add_trace(
            go.Scatter(
                x=diag_df["t"],
                y=diag_df["Actual"],
                name="Actual",
                line=dict(color="#2ECC71", width=2),
            )
        )
        fig_behavior.add_trace(
            go.Scatter(
                x=diag_df["t"],
                y=diag_df["Predicted"],
                name="Predicted",
                line=dict(color="#E74C3C", width=2, dash="dot"),
            )
        )
        fig_behavior.update_layout(
            title=f"Sensor {selected_sensor} behavior at {h_name}",
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=330,
            xaxis=dict(gridcolor="#2E4080", title="Snapshot"),
            yaxis=dict(gridcolor="#2E4080", title="Normalized Speed"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_behavior, use_container_width=True)

        fig_err = px.histogram(
            diag_df,
            x="Absolute Error",
            nbins=40,
            title=f"Error distribution for Sensor {selected_sensor} ({h_name})",
            color_discrete_sequence=["#4A90D9"],
        )
        fig_err.update_layout(
            paper_bgcolor="#1a1f2e",
            plot_bgcolor="#1a1f2e",
            font=dict(color="white"),
            height=280,
            xaxis=dict(gridcolor="#2E4080"),
            yaxis=dict(gridcolor="#2E4080"),
        )
        st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7e8a9a;font-size:0.8rem'>"
    "METR-LA Traffic Intelligence | EM627 Spatial Data Science | "
    "T-GCN Graph Neural Network | 207 sensors | RTX 3060 GPU"
    "</div>",
    unsafe_allow_html=True,
)