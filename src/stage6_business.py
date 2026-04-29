import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    import folium
except ImportError:
    folium = None

try:
    import contextily as ctx
except ImportError:
    ctx = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

matplotlib.style.use("seaborn-v0_8")
matplotlib.rcParams["figure.dpi"] = 150
np.random.seed(42)

# Add this block near the top of stage6_business.py
# RIGHT AFTER the imports and BEFORE any function definitions

import urllib.request

def ensure_sensor_locations():
    """Download sensor locations from DCRNN repo if not already present."""
    sensor_loc_path = config.DATA_DIR / "sensor_locations.csv"

    if not sensor_loc_path.exists():
        print("  Sensor locations file not found. Downloading from DCRNN repo...")
        url = (
            "https://raw.githubusercontent.com/liyaguang/DCRNN/"
            "master/data/sensor_graph/graph_sensor_locations.csv"
        )
        try:
            urllib.request.urlretrieve(url, sensor_loc_path)
            print(f"  Downloaded sensor_locations.csv -> {sensor_loc_path}")
        except Exception as e:
            print(f"  [ERROR] Could not download sensor locations: {e}")
            print("  Manual fix: Download from https://github.com/liyaguang/DCRNN")
            print("  Place file at: data/sensor_locations.csv")
            raise
    else:
        print(f"  Sensor locations found: {sensor_loc_path}")

    return sensor_loc_path

# ---------------------------------------------------------------------------
# STEP 1 — LOAD ALL DATA
# ---------------------------------------------------------------------------
def step1_load():
    print("\n=== STEP 1: Loading sensor data ===")
    
    # FIXED (downloads automatically if missing):
    sensor_loc_path = ensure_sensor_locations()
    df_loc = pd.read_csv(sensor_loc_path)
    print(f"Loaded {len(df_loc)} sensor locations")
    print(df_loc.head(3).to_string())

    lat_ok = df_loc["latitude"].between(34.0, 34.2).all()
    lon_ok = df_loc["longitude"].between(-118.5, -118.1).all()
    print(f"Latitude  valid (34.0–34.2) : {lat_ok}")
    print(f"Longitude valid (-118.5–-118.1): {lon_ok}")

    all_preds   = np.load(config.DATA_DIR / "gnn_v3_predictions.npy")
    all_targets = np.load(config.DATA_DIR / "gnn_v3_targets.npy")
    X_test      = np.load(config.DATA_DIR / "X_test.npy")

    # Per-sensor statistics
    sensor_mean_speed = X_test[:, :, -1].mean(axis=0)   # (207,)
    sensor_std_speed  = X_test[:, :, -1].std(axis=0)    # (207,)

    global_mean = sensor_mean_speed.mean()
    global_std  = sensor_mean_speed.std()
    congestion_threshold = global_mean - 0.5 * global_std
    congestion_rate = np.mean(X_test[:, :, -1] < congestion_threshold, axis=0)  # (207,)

    sensor_mae = np.array([
        mean_absolute_error(all_targets[:, i, 2], all_preds[:, i, 2])
        for i in range(207)
    ])

    df_loc["mean_speed"]     = sensor_mean_speed
    df_loc["congestion_rate"] = congestion_rate
    df_loc["prediction_mae"] = sensor_mae

    df_loc["congestion_level"] = pd.cut(
        df_loc["congestion_rate"],
        bins=[0, 0.1, 0.25, 0.4, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )

    print("\nCongestion rate summary:")
    print(df_loc["congestion_rate"].describe().round(4).to_string())
    print("\nTop 5 most congested sensors:")
    print(
        df_loc.nlargest(5, "congestion_rate")[
            ["index", "sensor_id", "latitude", "longitude", "congestion_rate"]
        ].to_string(index=False)
    )
    print("Sensor data prepared")

    return df_loc, all_preds, all_targets, X_test


# ---------------------------------------------------------------------------
# Helpers — build GeoDataFrame and plot basemap
# ---------------------------------------------------------------------------
def _make_gdf(df_loc):
    """Return (gdf_web, x_coords, y_coords) or None if geopandas unavailable."""
    if gpd is None or ctx is None:
        return None
    try:
        from shapely.geometry import Point
        geometry = [
            Point(lon, lat)
            for lat, lon in zip(df_loc["latitude"], df_loc["longitude"])
        ]
        gdf = gpd.GeoDataFrame(df_loc.copy(), geometry=geometry, crs="EPSG:4326")
        gdf_web = gdf.to_crs(epsg=3857)
        x = gdf_web.geometry.x.values
        y = gdf_web.geometry.y.values
        return gdf_web, x, y
    except Exception:
        return None


def _add_basemap(ax):
    if ctx is not None:
        try:
            ctx.add_basemap(
                ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.6
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# STEP 2 — SENSOR NETWORK BASEMAP FIGURE
# ---------------------------------------------------------------------------
def step2_sensor_network(df_loc):
    print("\n=== STEP 2: Creating sensor network basemap figure ===")

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_result = _make_gdf(df_loc)

    if gdf_result is not None:
        gdf_web, x_coords, y_coords = gdf_result
        sc = ax.scatter(
            x_coords, y_coords,
            c=df_loc["mean_speed"].values,
            cmap="RdYlGn", s=80, alpha=0.85,
            edgecolors="gray", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Mean Normalized Speed", shrink=0.6)
        _add_basemap(ax)
        ax.set_axis_off()
        caption = (
            "Figure 12: 207 METR-LA highway sensors plotted on Los Angeles basemap. "
            "Color indicates mean normalized speed (green=fast, red=congested). "
            "Source: OpenStreetMap contributors."
        )
    else:
        sc = ax.scatter(
            df_loc["longitude"], df_loc["latitude"],
            c=df_loc["mean_speed"],
            cmap="RdYlGn", s=80, alpha=0.85,
            edgecolors="gray", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Mean Normalized Speed", shrink=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        caption = (
            "Figure 12: 207 METR-LA highway sensors on Los Angeles area map. "
            "Color indicates mean normalized speed (green=fast, red=congested). "
            "Basemap: coordinate reference only (contextily unavailable)."
        )

    ax.set_title(
        "METR-LA Road Sensor Network — Los Angeles\n"
        "207 Loop Detectors on LA Highways",
        fontsize=14, fontweight="bold", pad=15,
    )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(
        config.FIGURES_DIR / "fig12_sensor_network_map.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("Saved fig12_sensor_network_map.png")


# ---------------------------------------------------------------------------
# STEP 3 — CONGESTION HOTSPOT MAP
# ---------------------------------------------------------------------------
def step3_congestion_hotspots(df_loc):
    print("\n=== STEP 3: Congestion hotspot map ===")

    sizes = df_loc["congestion_rate"].values * 300 + 40
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_result = _make_gdf(df_loc)

    if gdf_result is not None:
        gdf_web, x_coords, y_coords = gdf_result
        sc = ax.scatter(
            x_coords, y_coords,
            c=df_loc["congestion_rate"].values,
            cmap="Reds", s=sizes, alpha=0.85,
            edgecolors="darkred", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Congestion Rate (fraction of time)", shrink=0.6)
        _add_basemap(ax)
        ax.set_axis_off()

        top5 = df_loc.nlargest(5, "congestion_rate")
        top5_idx = top5["index"].values
        for idx in top5_idx:
            row = df_loc[df_loc["index"] == idx].iloc[0]
            pos_in_gdf = df_loc[df_loc["index"] == idx].index[0]
            ax.annotate(
                f"S{int(row['index'])}\n({row['congestion_rate']:.0%})",
                xy=(x_coords[pos_in_gdf], y_coords[pos_in_gdf]),
                fontsize=8, color="darkred", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
    else:
        sc = ax.scatter(
            df_loc["longitude"], df_loc["latitude"],
            c=df_loc["congestion_rate"],
            cmap="Reds", s=sizes, alpha=0.85,
            edgecolors="darkred", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Congestion Rate (fraction of time)", shrink=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

        top5 = df_loc.nlargest(5, "congestion_rate")
        for _, row in top5.iterrows():
            ax.annotate(
                f"S{int(row['index'])}\n({row['congestion_rate']:.0%})",
                xy=(row["longitude"], row["latitude"]),
                fontsize=8, color="darkred", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    ax.set_title(
        "Traffic Congestion Hotspots — Los Angeles Highway Network\n"
        "Size and color indicate congestion frequency",
        fontsize=14, fontweight="bold", pad=15,
    )
    fig.text(
        0.5, 0.02,
        "Figure 13: Congestion hotspot map showing fraction of time each sensor "
        "operates below threshold speed. Larger, darker circles indicate chronic "
        "congestion zones requiring priority driver repositioning. "
        "Source: METR-LA dataset, OpenStreetMap contributors.",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    plt.savefig(
        config.FIGURES_DIR / "fig13_congestion_hotspots.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("Saved fig13_congestion_hotspots.png")


# ---------------------------------------------------------------------------
# STEP 4 — PREDICTION ERROR MAP
# ---------------------------------------------------------------------------
def step4_prediction_error_map(df_loc):
    print("\n=== STEP 4: Prediction error map ===")

    sizes = df_loc["prediction_mae"].values * 600 + 30
    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_result = _make_gdf(df_loc)

    if gdf_result is not None:
        gdf_web, x_coords, y_coords = gdf_result
        sc = ax.scatter(
            x_coords, y_coords,
            c=df_loc["prediction_mae"].values,
            cmap="RdYlGn_r", s=sizes, alpha=0.85,
            edgecolors="gray", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Prediction MAE (15min horizon)", shrink=0.6)
        _add_basemap(ax)
        ax.set_axis_off()

        top5_hard = df_loc.nlargest(5, "prediction_mae")
        for _, row in top5_hard.iterrows():
            pos = df_loc[df_loc["index"] == row["index"]].index[0]
            ax.annotate(
                f"S{int(row['index'])}\nMAE={row['prediction_mae']:.3f}",
                xy=(x_coords[pos], y_coords[pos]),
                fontsize=8, color="darkred", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )
    else:
        sc = ax.scatter(
            df_loc["longitude"], df_loc["latitude"],
            c=df_loc["prediction_mae"],
            cmap="RdYlGn_r", s=sizes, alpha=0.85,
            edgecolors="gray", linewidth=0.5,
        )
        plt.colorbar(sc, ax=ax, label="Prediction MAE (15min horizon)", shrink=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

        top5_hard = df_loc.nlargest(5, "prediction_mae")
        for _, row in top5_hard.iterrows():
            ax.annotate(
                f"S{int(row['index'])}\nMAE={row['prediction_mae']:.3f}",
                xy=(row["longitude"], row["latitude"]),
                fontsize=8, color="darkred", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    ax.set_title(
        "T-GCN Prediction Reliability Map — METR-LA\n"
        "Red sensors = high prediction error (less reliable ETA)",
        fontsize=14, fontweight="bold", pad=15,
    )
    fig.text(
        0.5, 0.02,
        "Figure 14: Prediction error map showing MAE per sensor at 15-minute horizon. "
        "Red sensors indicate locations where T-GCN predictions are less reliable — "
        "likely at highway intersections, on-ramps, or incident-prone sections. "
        "Source: METR-LA dataset, OpenStreetMap contributors.",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    plt.savefig(
        config.FIGURES_DIR / "fig14_prediction_error_map.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print("Saved fig14_prediction_error_map.png")


# ---------------------------------------------------------------------------
# STEP 5 — FOLIUM INTERACTIVE MAP
# ---------------------------------------------------------------------------
def step5_folium_map(df_loc):
    print("\n=== STEP 5: Interactive folium map ===")

    if folium is None:
        print("folium not available — skipping interactive map")
        return

    def get_color(rate):
        if rate > 0.4:  return "darkred"
        if rate > 0.25: return "red"
        if rate > 0.1:  return "orange"
        return "green"

    m = folium.Map(location=[34.05, -118.25], zoom_start=11, tiles="CartoDB positron")

    for _, row in df_loc.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(5, row["congestion_rate"] * 20),
            color=get_color(row["congestion_rate"]),
            fill=True,
            fill_color=get_color(row["congestion_rate"]),
            fill_opacity=0.7,
            popup=folium.Popup(
                f"Sensor {int(row['index'])}<br>"
                f"Sensor ID: {int(row['sensor_id'])}<br>"
                f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}<br>"
                f"Mean Speed: {row['mean_speed']:.3f} (normalized)<br>"
                f"Congestion Rate: {row['congestion_rate']:.1%}<br>"
                f"Prediction MAE: {row['prediction_mae']:.4f}<br>"
                f"Congestion Level: {row['congestion_level']}",
                max_width=250,
            ),
            tooltip=f"Sensor {int(row['index'])} — {row['congestion_level']} congestion",
        ).add_to(m)

    out_path = config.FIGURES_DIR / "interactive_sensor_map.html"
    m.save(str(out_path))
    print("Saved interactive_sensor_map.html")
    print("TIP: Open this HTML file in browser for interactive exploration")


# ---------------------------------------------------------------------------
# STEP 6 — BUSINESS OPERATIONS ANALYSIS
# ---------------------------------------------------------------------------
def step6_business_analysis(df_loc):
    print("\n=== STEP 6: Business operations analysis ===")

    critical_sensors   = df_loc[df_loc["congestion_level"] == "Critical"]
    high_sensors       = df_loc[df_loc["congestion_level"] == "High"]
    unreliable_sensors = df_loc[
        df_loc["prediction_mae"] > df_loc["prediction_mae"].quantile(0.8)
    ]

    print("=== RIDE-HAILING OPERATIONAL ZONES ===")
    print(f"Critical congestion sensors     : {len(critical_sensors)}")
    print(f"High congestion sensors         : {len(high_sensors)}")
    print(f"Low prediction reliability sensors: {len(unreliable_sensors)}")
    print()
    print("TOP PRIORITY ZONES FOR DRIVER REPOSITIONING:")
    top_priority = df_loc.nlargest(10, "congestion_rate")[
        ["index", "latitude", "longitude", "congestion_rate", "prediction_mae"]
    ]
    print(top_priority.to_string(index=False))

    print()
    print("PEAK HOURS FOR SURGE PRICING (from EDA):")
    print("  Morning rush: 7-9am  (speed drops to -0.3 normalized)")
    print("  Evening rush: 5-7pm  (speed drops to -0.4 normalized, more severe)")
    print("  Weekend: Similar pattern but less intense")

    print()
    print("HIGH RISK ETA ZONES (unreliable predictions):")
    print(
        unreliable_sensors[["index", "latitude", "longitude", "prediction_mae"]]
        .to_string(index=False)
    )

    df_loc.to_csv(config.DATA_DIR / "sensor_analysis.csv", index=False)
    print("\nSaved sensor_analysis.csv")

    return critical_sensors, high_sensors, unreliable_sensors


# ---------------------------------------------------------------------------
# STEP 7 — SAVE BUSINESS RECOMMENDATIONS
# ---------------------------------------------------------------------------
def step7_save_recommendations(df_loc, critical_sensors, unreliable_sensors):
    top3_coords = df_loc.nlargest(3, "congestion_rate")[
        ["index", "latitude", "longitude", "congestion_rate"]
    ]
    top3_str = "\n".join(
        f"           Sensor {int(r['index'])}: "
        f"({r['latitude']:.5f}, {r['longitude']:.5f}) "
        f"congestion={r['congestion_rate']:.1%}"
        for _, r in top3_coords.iterrows()
    )

    unreliable_ids = ", ".join(
        str(int(i)) for i in unreliable_sensors["index"].values
    )

    content = f"""RIDE-HAILING OPERATIONS RECOMMENDATIONS
METR-LA Traffic Forecasting -- T-GCN Model
==========================================

MODEL PERFORMANCE SUMMARY
--------------------------
T-GCN achieves MAE of 0.140 at 5-minute horizon
T-GCN achieves MAE of 0.200 at 15-minute horizon
T-GCN achieves MAE of 0.275 at 30-minute horizon
Outperforms Random Forest at 5min (+5.2%) and 15min (+0.3%)
Spatial autocorrelation r=0.672 confirms graph structure value

OPERATIONAL RECOMMENDATIONS
----------------------------
1. DRIVER PRE-POSITIONING
   Action: Deploy additional drivers to high-congestion zones
           15-20 minutes before predicted speed drops
   When:   Weekdays 6:45am (before 7am rush) and 4:45pm (before 5pm rush)
   Where:
{top3_str}
   Expected benefit: Reduce average customer wait time by 2-3 minutes

2. DYNAMIC PRICING TRIGGERS
   Action: Activate surge pricing when T-GCN predicts
           speed below threshold at 15-minute horizon
   Threshold: normalized speed < -0.3 (heavy congestion)
   Zones: Critical and High congestion sensors
   Expected benefit: Balance supply-demand during peak periods

3. ETA RELIABILITY FLAGS
   Action: Add uncertainty warning to customer app
           for sensors with MAE > 0.245 (top 20% hardest)
   Affected sensors: {unreliable_ids}
   Message to customer: "ETA may vary due to complex traffic conditions"

4. AIRPORT AND CORRIDOR MONITORING
   Action: Monitor sensor clusters near LAX and downtown corridors
   Peak risk windows: 5pm-7pm weekdays
   Recommended buffer: Add 15% to ETA for 30-minute predictions

LIMITATIONS
-----------
- Model trained on 2012 data -- may not capture current traffic patterns
- 30-minute predictions are less reliable (MAE=0.275 vs RF=0.263)
- Congestion prediction harder than free-flow (MAE 0.322 vs 0.164)
- Sensor coverage limited to highways -- surface streets not included

NEXT STEPS
----------
- Retrain with recent data including post-pandemic patterns
- Add weather and event data as additional features
- Implement real-time retraining pipeline
- Expand to surface street sensors for complete coverage
"""

    out_path = config.DATA_DIR / "business_recommendations.txt"
    with open(out_path, "w") as f:
        f.write(content)
    print("Saved business_recommendations.txt")


# ---------------------------------------------------------------------------
# STEP 8 — FINAL SUMMARY
# ---------------------------------------------------------------------------
def step8_final_summary(df_loc):
    critical_count   = (df_loc["congestion_level"] == "Critical").sum()
    high_count       = (df_loc["congestion_level"] == "High").sum()
    unreliable_count = (df_loc["prediction_mae"] > df_loc["prediction_mae"].quantile(0.8)).sum()

    print("\n" + "=" * 65)
    print("=== STAGE 6 COMPLETE ===")
    print("=" * 65)
    print("Figures saved:")
    print("  figures/fig12_sensor_network_map.png")
    print("  figures/fig13_congestion_hotspots.png")
    print("  figures/fig14_prediction_error_map.png")
    if folium is not None:
        print("  figures/interactive_sensor_map.html")
    print("Data saved:")
    print("  data/sensor_analysis.csv")
    print("  data/business_recommendations.txt")
    print("─" * 65)
    print("Key operational numbers:")
    print(f"  Critical congestion sensors      : {critical_count}")
    print(f"  High congestion sensors          : {high_count}")
    print(f"  Low reliability sensors (top 20%): {unreliable_count}")
    print(f"  Most congested sensor            : "
          f"{df_loc.nlargest(1, 'congestion_rate')['index'].values[0]}")
    print(f"  Hardest-to-predict sensor        : "
          f"{df_loc.nlargest(1, 'prediction_mae')['index'].values[0]}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df_loc, all_preds, all_targets, X_test = step1_load()
    step2_sensor_network(df_loc)
    step3_congestion_hotspots(df_loc)
    step4_prediction_error_map(df_loc)
    step5_folium_map(df_loc)
    critical_sensors, high_sensors, unreliable_sensors = step6_business_analysis(df_loc)
    step7_save_recommendations(df_loc, critical_sensors, unreliable_sensors)
    step8_final_summary(df_loc)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Stage 6 failed with the following traceback:")
        traceback.print_exc()
        sys.exit(1)
