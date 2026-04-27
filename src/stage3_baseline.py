import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 150
np.random.seed(config.RANDOM_SEED)

HORIZON_NAMES = ["5min", "15min", "30min"]
HORIZON_IDX   = [0, 2, 5]


# =============================================================================
# STEP 1 - Load and split data
# =============================================================================
def step1_load_split():
    print("=== STEP 1: Loading and splitting data ===")
    from torch_geometric_temporal.dataset import METRLADatasetLoader

    loader   = METRLADatasetLoader()
    dataset  = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=6)
    snapshots = list(dataset)

    total    = len(snapshots)
    train_end = int(0.7 * total)
    val_end   = int(0.8 * total)

    train_snapshots = snapshots[:train_end]
    val_snapshots   = snapshots[train_end:val_end]
    test_snapshots  = snapshots[val_end:]

    print(f"  Total snapshots : {total}")
    print(f"  Train           : {len(train_snapshots)}  (snapshots 0 – {train_end-1})")
    print(f"  Val             : {len(val_snapshots)}  (snapshots {train_end} – {val_end-1})")
    print(f"  Test            : {len(test_snapshots)}  (snapshots {val_end} – {total-1})")
    print("  Split is chronological - no data leakage")

    return train_snapshots, val_snapshots, test_snapshots


# =============================================================================
# STEP 2 - Extract arrays
# =============================================================================
def _extract_arrays(snapshots):
    X_list, y_list = [], []
    for snap in snapshots:
        X_list.append(snap.x[:, 0, :].numpy())   # (207, 12)
        y_list.append(snap.y[:, :].numpy())        # (207, 6)
    return np.array(X_list), np.array(y_list)


def step2_extract(train_snapshots, val_snapshots, test_snapshots):
    print("\n=== STEP 2: Extracting speed arrays ===")

    X_train, y_train = _extract_arrays(train_snapshots)
    X_val,   y_val   = _extract_arrays(val_snapshots)
    X_test,  y_test  = _extract_arrays(test_snapshots)

    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}   y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")

    np.save(config.DATA_DIR / 'X_train.npy', X_train)
    np.save(config.DATA_DIR / 'y_train.npy', y_train)
    np.save(config.DATA_DIR / 'X_val.npy',   X_val)
    np.save(config.DATA_DIR / 'y_val.npy',   y_val)
    np.save(config.DATA_DIR / 'X_test.npy',  X_test)
    np.save(config.DATA_DIR / 'y_test.npy',  y_test)
    print("  Arrays saved to data/")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# STEP 3 - Persistence baseline
# =============================================================================
def step3_persistence(X_test, y_test):
    print("\n=== STEP 3: Persistence Baseline ===")
    print("  Persistence = predict future speed = current speed (last observed)")

    results = {}
    last_speed = X_test[:, :, -1]   # (n_test, 207)

    for name, h in zip(HORIZON_NAMES, HORIZON_IDX):
        true = y_test[:, :, h].flatten()
        pred = last_speed.flatten()
        mae  = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        results[name] = {"MAE": mae, "RMSE": rmse}

    print(f"  {'Horizon':<8} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*26}")
    for name, m in results.items():
        print(f"  {name:<8} {m['MAE']:>8.4f} {m['RMSE']:>8.4f}")
    print("  Persistence baseline: predict current speed stays constant")

    return results


# =============================================================================
# STEP 4 - Historical average baseline
# =============================================================================
def step4_hist_avg(X_train, X_test, y_test):
    print("\n=== STEP 4: Historical Average Baseline ===")
    print("  For each sensor, predict its historical mean speed from training data")

    sensor_mean = X_train[:, :, -1].mean(axis=0)   # (207,)
    pred_base   = np.tile(sensor_mean, (X_test.shape[0], 1))   # (n_test, 207)

    results = {}
    for name, h in zip(HORIZON_NAMES, HORIZON_IDX):
        true = y_test[:, :, h].flatten()
        pred = pred_base.flatten()
        mae  = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        results[name] = {"MAE": mae, "RMSE": rmse}

    print(f"  {'Horizon':<8} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*26}")
    for name, m in results.items():
        print(f"  {name:<8} {m['MAE']:>8.4f} {m['RMSE']:>8.4f}")
    print("  Historical average: always predict each sensor's training mean")

    return results, sensor_mean


# =============================================================================
# STEP 5 - Random Forest baseline
# =============================================================================
def step5_random_forest(X_train, y_train, X_test, y_test):
    print("\n=== STEP 5: Random Forest Baseline ===")
    print("  Training RF on flattened features - this may take 2-3 minutes...")

    X_train_flat = X_train.reshape(-1, 12)             # (n_train*207, 12)
    y_train_flat = y_train[:, :, 0].reshape(-1)        # horizon 0 = 5min

    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_idx = rng.choice(len(X_train_flat), 50000, replace=False)
    X_sample   = X_train_flat[sample_idx]
    y_sample   = y_train_flat[sample_idx]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=8,
        random_state=config.RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_sample, y_sample)
    print("  RF trained.")

    X_test_flat = X_test.reshape(-1, 12)               # (n_test*207, 12)
    y_pred_flat = rf.predict(X_test_flat)

    results = {}
    for name, h in zip(HORIZON_NAMES, HORIZON_IDX):
        true = y_test[:, :, h].reshape(-1)
        mae  = mean_absolute_error(true, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(true, y_pred_flat))
        results[name] = {"MAE": mae, "RMSE": rmse}

    print(f"  {'Horizon':<8} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*26}")
    for name, m in results.items():
        print(f"  {name:<8} {m['MAE']:>8.4f} {m['RMSE']:>8.4f}")
    print("  RF uses last 12 timesteps as features - no graph structure")

    return results


# =============================================================================
# STEP 6 - Comparison table
# =============================================================================
def step6_comparison(persistence_results, hist_avg_results, rf_results):
    print("\n=== STEP 6: Baseline Comparison ===")

    rows = []
    for model_name, res in [("Persistence", persistence_results),
                             ("HistoricalAvg", hist_avg_results),
                             ("RandomForest", rf_results)]:
        for horizon in HORIZON_NAMES:
            rows.append({
                "Model":   model_name,
                "Horizon": horizon,
                "MAE":     round(res[horizon]["MAE"],  4),
                "RMSE":    round(res[horizon]["RMSE"], 4),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out = config.DATA_DIR / 'baseline_results.csv'
    df.to_csv(out, index=False)
    print(f"\n  Saved baseline_results.csv -> {out}")

    return df


# =============================================================================
# STEP 7 - Comparison figure
# =============================================================================
def step7_figure(df):
    models  = ["Persistence", "HistoricalAvg", "RandomForest"]
    colors  = ["steelblue", "darkorange", "green"]
    markers = ["o", "s", "^"]
    horizons = HORIZON_NAMES

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        for model, color, marker in zip(models, colors, markers):
            vals = [df[(df["Model"] == model) & (df["Horizon"] == h)][metric].values[0]
                    for h in horizons]
            ax.plot(horizons, vals, color=color, marker=marker,
                    linewidth=2, markersize=7, label=model)
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Horizon")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Baseline Model Comparison: MAE and RMSE by Forecast Horizon",
                 fontsize=13, fontweight='bold')
    fig.text(0.5, -0.03,
             "Figure 6: All three baselines show increasing error at longer horizons. "
             "Random Forest outperforms persistence, establishing a meaningful benchmark for the graph model.",
             ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    out = config.FIGURES_DIR / 'fig6_baseline_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved fig6 -> {out}")


# =============================================================================
# STEP 8 - Save baseline summary
# =============================================================================
def step8_summary_file(df, persistence_results, hist_avg_results, rf_results):
    out = config.DATA_DIR / 'baseline_summary.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write("METR-LA Baseline Model Summary\n")
        f.write("=" * 50 + "\n\n")

        for model_name, res in [("Persistence",    persistence_results),
                                 ("HistoricalAvg",  hist_avg_results),
                                 ("RandomForest",   rf_results)]:
            f.write(f"{model_name}:\n")
            for h in HORIZON_NAMES:
                f.write(f"  {h:>6}: MAE={res[h]['MAE']:.4f}  RMSE={res[h]['RMSE']:.4f}\n")
            f.write("\n")

        f.write("=" * 50 + "\n")
        f.write("Strongest baseline per horizon (lowest MAE):\n")
        for h in HORIZON_NAMES:
            sub = df[df["Horizon"] == h]
            best = sub.loc[sub["MAE"].idxmin()]
            f.write(f"  {h}: {best['Model']} (MAE={best['MAE']:.4f})\n")

        f.write("\n")
        f.write("Gap: Persistence vs RF (5min MAE):\n")
        diff = persistence_results["5min"]["MAE"] - rf_results["5min"]["MAE"]
        f.write(f"  Persistence MAE - RF MAE = {diff:.4f}\n")
        f.write("\n")
        rf_5 = rf_results["5min"]["MAE"]
        target = rf_5 * 0.9
        f.write(f"Target for graph model: beat RF MAE of {rf_5:.4f} by 10% -> MAE < {target:.4f}\n")

    print(f"  Baseline summary saved -> {out}")


# =============================================================================
# STEP 9 - Final summary box
# =============================================================================
def step9_final_box(df, rf_results):
    print()
    print("+" + "=" * 58 + "+")
    print("|  STAGE 3 COMPLETE -- Baseline Results                  |")
    print("+" + "=" * 58 + "+")
    for h in HORIZON_NAMES:
        sub  = df[df["Horizon"] == h]
        best = sub.loc[sub["MAE"].idxmin()]
        print(f"|  Best @ {h:<5}: {best['Model']:<15} MAE = {best['MAE']:.4f}              |")
    rf_5   = rf_results["5min"]["MAE"]
    target = rf_5 * 0.9
    print(f"|  Graph model target  : beat RF MAE {rf_5:.4f} by 10%          |")
    print(f"|                        -> target MAE < {target:.4f}              |")
    print("+" + "=" * 58 + "+")
    print("\n=== STAGE 3 COMPLETE ===")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    try:
        train_snaps, val_snaps, test_snaps = step1_load_split()

        X_train, y_train, X_val, y_val, X_test, y_test = \
            step2_extract(train_snaps, val_snaps, test_snaps)

        persistence_results = step3_persistence(X_test, y_test)
        hist_avg_results, sensor_mean = step4_hist_avg(X_train, X_test, y_test)
        rf_results = step5_random_forest(X_train, y_train, X_test, y_test)

        df = step6_comparison(persistence_results, hist_avg_results, rf_results)
        step7_figure(df)
        step8_summary_file(df, persistence_results, hist_avg_results, rf_results)
        step9_final_box(df, rf_results)

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Stage 3 failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("\nHint: Ensure stage1 ran successfully and all packages are installed.")
