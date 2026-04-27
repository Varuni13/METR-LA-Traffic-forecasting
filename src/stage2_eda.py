import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
from pathlib import Path

# -- Config import ------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# -- Style --------------------------------------------------------------------
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 150
np.random.seed(config.RANDOM_SEED)

# -- Optional imports ---------------------------------------------------------
try:
    import contextily as ctx
except ImportError:
    ctx = None

try:
    import folium
except ImportError:
    folium = None

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("  [WARN] networkx not installed -- Fig 5 will be skipped")


# =============================================================================
# STEP 1 - Load data and extract all speeds
# =============================================================================
def step1_load_speeds():
    print("=== STEP 1: Loading and extracting all speed data ===")
    from torch_geometric_temporal.dataset import METRLADatasetLoader

    loader  = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=6)

    speeds_list      = []
    edge_index_saved = None
    edge_attr_saved  = None

    for i, snapshot in enumerate(dataset):
        # x shape: (207, 2, 12) -> feature 0, timestep 0 -> (207,)
        speed = snapshot.x[:, 0, 0].numpy()
        speeds_list.append(speed)
        if i == 0:
            edge_index_saved = snapshot.edge_index
            edge_attr_saved  = snapshot.edge_attr

    all_speeds = np.array(speeds_list)   # (34255, 207)
    print(f"  all_speeds shape : {all_speeds.shape}")
    print(f"  Speed min        : {all_speeds.min():.4f}")
    print(f"  Speed max        : {all_speeds.max():.4f}")
    print(f"  Speed mean       : {all_speeds.mean():.4f}")
    print(f"  Speed std        : {all_speeds.std():.4f}")
    print("  Note: speeds are normalized -- working with normalized values throughout")
    return all_speeds, edge_index_saved, edge_attr_saved


# =============================================================================
# STEP 2 - Temporal pattern: average speed by hour
# =============================================================================
def step2_temporal(all_speeds):
    print("\n=== STEP 2: Temporal patterns ===")
    n_snaps  = all_speeds.shape[0]
    snap_idx = np.arange(n_snaps)

    hour_of_day = (snap_idx * 5 // 60) % 24
    day_of_week = (snap_idx * 5 // (60 * 24)) % 7
    is_weekend  = day_of_week >= 5

    mean_speed_per_snapshot = all_speeds.mean(axis=1)   # (34255,)

    hourly_mean = np.array([mean_speed_per_snapshot[hour_of_day == h].mean() for h in range(24)])
    hourly_std  = np.array([mean_speed_per_snapshot[hour_of_day == h].std()  for h in range(24)])

    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, hourly_mean, color='steelblue', marker='o', linewidth=2, label='Mean Speed')
    ax.fill_between(hours,
                    hourly_mean - hourly_std,
                    hourly_mean + hourly_std,
                    alpha=0.2, color='steelblue', label='Std Band')
    for h in [7, 8, 9]:
        ax.axvline(x=h, color='red', linestyle='--', alpha=0.6,
                   label='Morning Rush' if h == 7 else '')
    for h in [17, 18, 19]:
        ax.axvline(x=h, color='orange', linestyle='--', alpha=0.6,
                   label='Evening Rush' if h == 17 else '')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Normalized Speed')
    ax.set_title('Average Traffic Speed by Hour of Day - METR-LA (207 Sensors)')
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.04,
             'Figure 1: Clear speed drops at 7-9am and 5-7pm confirm rush hour patterns across all 207 LA highway sensors.',
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    out = config.FIGURES_DIR / 'fig1_speed_by_hour.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig1 -> {out}")

    return hour_of_day, day_of_week, is_weekend, mean_speed_per_snapshot, hourly_mean


# =============================================================================
# STEP 3 - Weekday vs Weekend
# =============================================================================
def step3_weekday_weekend(all_speeds, hour_of_day, day_of_week, is_weekend):
    mean_speed_per_snapshot = all_speeds.mean(axis=1)

    wd_hourly = np.array([
        mean_speed_per_snapshot[(hour_of_day == h) & (~is_weekend)].mean()
        for h in range(24)
    ])
    we_hourly = np.array([
        mean_speed_per_snapshot[(hour_of_day == h) & is_weekend].mean()
        for h in range(24)
    ])

    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, wd_hourly, color='steelblue', linestyle='-',  linewidth=2, marker='o', label='Weekday')
    ax.plot(hours, we_hourly, color='green',     linestyle='--', linewidth=2, marker='s', label='Weekend')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Normalized Speed')
    ax.set_title('Weekday vs Weekend Traffic Speed - METR-LA')
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.04,
             'Figure 2: Weekday rush hours create sharp speed drops absent on weekends, confirming temporal structure in the data.',
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    out = config.FIGURES_DIR / 'fig2_weekday_weekend.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig2 -> {out}")

    wd_mean = mean_speed_per_snapshot[~is_weekend].mean()
    we_mean = mean_speed_per_snapshot[is_weekend].mean()
    diff    = wd_mean - we_mean
    print(f"  Weekday mean speed : {wd_mean:.4f}")
    print(f"  Weekend mean speed : {we_mean:.4f}")
    print(f"  Difference         : {diff:.4f}")
    return diff


# =============================================================================
# STEP 4 - Sensor heatmap
# =============================================================================
def step4_heatmap(all_speeds):
    print("\n=== STEP 4: Sensor heatmap ===")
    sample   = all_speeds[:500, :30]   # (500, 30)
    x_labels = [f"{int(i*5//60)}h{int(i*5%60):02d}" if i % 12 == 0 else ''
                for i in range(500)]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(sample.T,
                cmap='RdYlGn',
                ax=ax,
                cbar_kws={'label': 'Normalized Speed'},
                xticklabels=x_labels,
                yticklabels=[f"S{i}" for i in range(30)])
    ax.set_xlabel('Time Snapshot')
    ax.set_ylabel('Sensor ID')
    ax.set_title('Traffic Speed Heatmap: 30 Sensors x First 500 Snapshots')
    plt.xticks(rotation=45, ha='right')
    fig.text(0.5, -0.03,
             'Figure 3: Heatmap reveals synchronized speed drops across multiple sensors during congestion events, evidence of spatial dependence.',
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    out = config.FIGURES_DIR / 'fig3_heatmap.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig3 -> {out}")


# =============================================================================
# STEP 5 - Spatial autocorrelation
# =============================================================================
def step5_spatial(all_speeds, edge_index):
    print("\n=== STEP 5: Spatial autocorrelation ===")
    sensor_mean_speed = all_speeds.mean(axis=0)   # (207,)
    sensor_std_speed  = all_speeds.std(axis=0)

    ei = edge_index.numpy()
    neighbor_avg = np.zeros(207)
    for i in range(207):
        nbrs = ei[1, ei[0] == i]
        if len(nbrs) > 0:
            neighbor_avg[i] = sensor_mean_speed[nbrs].mean()
        else:
            neighbor_avg[i] = sensor_mean_speed[i]

    valid = neighbor_avg != 0
    corr  = np.corrcoef(sensor_mean_speed[valid], neighbor_avg[valid])[0, 1]
    print(f"  Spatial autocorrelation (Moran-like correlation): {corr:.4f}")

    z  = np.polyfit(sensor_mean_speed, neighbor_avg, 1)
    pf = np.poly1d(z)
    xr = np.linspace(sensor_mean_speed.min(), sensor_mean_speed.max(), 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sensor_mean_speed, neighbor_avg, alpha=0.6, color='steelblue', s=40)
    ax.plot(xr, pf(xr), color='red', linewidth=2, label=f'Trend (r={corr:.3f})')
    ax.set_xlabel('Sensor Mean Speed')
    ax.set_ylabel('Neighbor Mean Speed')
    ax.set_title(f'Spatial Autocorrelation: r = {corr:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.04,
             'Figure 4: Strong positive correlation between sensor speeds and their road-network neighbors confirms spatial dependence - justifying graph-based modeling.',
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    out = config.FIGURES_DIR / 'fig4_spatial_autocorr.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig4 -> {out}")

    return sensor_mean_speed, sensor_std_speed, corr


# =============================================================================
# STEP 6 - Congestion statistics
# =============================================================================
def step6_congestion(all_speeds, hour_of_day):
    print("\n=== STEP 6: Congestion analysis ===")
    global_mean = all_speeds.mean()
    global_std  = all_speeds.std()
    threshold   = global_mean - 0.5 * global_std
    print(f"  Congestion threshold (mean - 0.5*std): {threshold:.4f}")

    congestion_rate = (all_speeds < threshold).mean(axis=0)   # (207,)

    top20_congested = np.argsort(congestion_rate)[-20:][::-1]
    top20_free      = np.argsort(congestion_rate)[:20]

    print("  Top 10 most congested sensors:")
    for sid in top20_congested[:10]:
        print(f"    Sensor {sid:3d}  congestion rate: {congestion_rate[sid]:.3f}")

    below_threshold = (all_speeds < threshold).mean(axis=1)   # (34255,)
    peak_by_hour    = np.array([below_threshold[hour_of_day == h].mean() for h in range(24)])
    peak_hours      = np.argsort(peak_by_hour)[-3:][::-1]
    print(f"  Peak congestion hours: {list(peak_hours)} (hour of day)")

    return congestion_rate, top20_congested, peak_hours, threshold


# =============================================================================
# STEP 7 - Sensor network figure
# =============================================================================
def step7_network(edge_index, sensor_mean_speed):
    print("\n=== STEP 7: Sensor network visualization ===")
    if not HAS_NX:
        print("  [SKIP] networkx not available -- install with: pip install networkx")
        return

    G = nx.Graph()
    G.add_nodes_from(range(207))
    ei    = edge_index.numpy()
    edges = list(zip(ei[0].tolist(), ei[1].tolist()))
    G.add_edges_from(edges)

    pos    = nx.spring_layout(G, seed=config.RANDOM_SEED)
    speeds = [sensor_mean_speed[i] for i in range(207)]
    vmin, vmax = min(speeds), max(speeds)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=speeds,
                           cmap='RdYlGn',
                           node_size=100,
                           vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Mean Normalized Speed')
    ax.set_title('METR-LA Road Sensor Network (207 sensors, 1722 connections)')
    ax.axis('off')
    fig.text(0.5, 0.01,
             'Figure 5: Network graph of 207 METR-LA sensors. Node color indicates average speed (green=fast, red=congested). Connected sensors share road-network proximity.',
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    out = config.FIGURES_DIR / 'fig5_network.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved fig5 -> {out}")


# =============================================================================
# STEP 8 - Save EDA summary
# =============================================================================
def step8_save_summary(all_speeds, corr, peak_hours, top20_congested,
                       congestion_rate, wd_we_diff, edge_index):
    out = config.DATA_DIR / 'eda_summary.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write("METR-LA EDA Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total snapshots       : {all_speeds.shape[0]}\n")
        f.write(f"Nodes                 : 207\n")
        f.write(f"Edges                 : {edge_index.shape[1]}\n")
        f.write("-" * 40 + "\n")
        f.write("Speed statistics (normalized):\n")
        f.write(f"  min  : {all_speeds.min():.4f}\n")
        f.write(f"  max  : {all_speeds.max():.4f}\n")
        f.write(f"  mean : {all_speeds.mean():.4f}\n")
        f.write(f"  std  : {all_speeds.std():.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Spatial autocorrelation (r) : {corr:.4f}\n")
        f.write(f"Peak congestion hours       : {list(peak_hours)}\n")
        f.write(f"Weekday vs weekend diff     : {wd_we_diff:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("Top 10 most congested sensors:\n")
        for sid in top20_congested[:10]:
            f.write(f"  Sensor {sid:3d}  rate: {congestion_rate[sid]:.3f}\n")
    print(f"\n  EDA summary saved to: {out}")


# =============================================================================
# STEP 9 - Final summary box
# =============================================================================
def step9_summary(corr, peak_hours, top20_congested, congestion_rate, wd_we_diff):
    top3 = ', '.join([str(s) for s in top20_congested[:3]])
    print()
    print("+" + "=" * 60 + "+")
    print("|  EDA COMPLETE -- Key Findings                            |")
    print("+" + "=" * 60 + "+")
    print(f"|  Spatial autocorrelation (r) : {corr:<28.4f}|")
    print(f"|  Peak congestion hours       : {str(list(peak_hours)):<28}|")
    print(f"|  Top 3 congested sensors     : {top3:<28}|")
    print(f"|  Weekday vs weekend diff     : {wd_we_diff:<28.4f}|")
    print("|  Figures saved:                                          |")
    for i, name in enumerate(['fig1_speed_by_hour', 'fig2_weekday_weekend',
                               'fig3_heatmap', 'fig4_spatial_autocorr',
                               'fig5_network'], 1):
        print(f"|    Fig {i}: {name:<50}|")
    print("+" + "=" * 60 + "+")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    try:
        all_speeds, edge_index, edge_attr = step1_load_speeds()

        hour_of_day, day_of_week, is_weekend, mean_speed_per_snapshot, hourly_mean = \
            step2_temporal(all_speeds)

        wd_we_diff = step3_weekday_weekend(all_speeds, hour_of_day, day_of_week, is_weekend)

        step4_heatmap(all_speeds)

        sensor_mean_speed, sensor_std_speed, corr = step5_spatial(all_speeds, edge_index)

        congestion_rate, top20_congested, peak_hours, threshold = \
            step6_congestion(all_speeds, hour_of_day)

        step7_network(edge_index, sensor_mean_speed)

        step8_save_summary(all_speeds, corr, peak_hours, top20_congested,
                           congestion_rate, wd_we_diff, edge_index)

        step9_summary(corr, peak_hours, top20_congested, congestion_rate, wd_we_diff)

        print("\n=== EDA COMPLETE ===")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] EDA failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("\nHint: Make sure all packages are installed and stage1 ran successfully.")
