import sys
import os
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric import utils as pyg_utils
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

matplotlib.style.use("seaborn-v0_8")
matplotlib.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
# STEP 1 — LOAD DATA
# ---------------------------------------------------------------------------
def step1_load_data():
    print("\n=== STEP 1: Loading data ===")

    X_train = torch.FloatTensor(np.load(config.DATA_DIR / "X_train.npy"))
    y_train = torch.FloatTensor(np.load(config.DATA_DIR / "y_train.npy"))
    X_val   = torch.FloatTensor(np.load(config.DATA_DIR / "X_val.npy"))
    y_val   = torch.FloatTensor(np.load(config.DATA_DIR / "y_val.npy"))
    X_test  = torch.FloatTensor(np.load(config.DATA_DIR / "X_test.npy"))
    y_test  = torch.FloatTensor(np.load(config.DATA_DIR / "y_test.npy"))

    from torch_geometric_temporal.dataset import METRLADatasetLoader
    loader  = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=6)
    snap    = next(iter(dataset))
    edge_index = snap.edge_index.to(DEVICE)
    edge_attr  = snap.edge_attr.float().to(DEVICE)

    print(f"X_train    : {X_train.shape}")
    print(f"y_train    : {y_train.shape}")
    print(f"X_val      : {X_val.shape}")
    print(f"y_val      : {y_val.shape}")
    print(f"X_test     : {X_test.shape}")
    print(f"y_test     : {y_test.shape}")
    print(f"edge_index : {edge_index.shape}")
    print(f"edge_attr  : {edge_attr.shape}")
    print("Data loaded")

    return X_train, y_train, X_val, y_val, X_test, y_test, edge_index, edge_attr


# ---------------------------------------------------------------------------
# STEP 2 — DATASET AND DATALOADERS
# ---------------------------------------------------------------------------
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def step2_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    BATCH_SIZE = 64

    train_loader = DataLoader(
        TrafficDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0,
    )
    val_loader = DataLoader(
        TrafficDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0,
    )
    test_loader = DataLoader(
        TrafficDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# STEP 3 — FAST MODEL
# ---------------------------------------------------------------------------
class FastTrafficGNN(nn.Module):
    def __init__(self, num_nodes=207, in_features=1, hidden_dim=64, out_steps=6):
        super().__init__()
        self.num_nodes   = num_nodes
        self.hidden_dim  = hidden_dim
        self.out_steps   = out_steps
        self.in_features = in_features

        from torch_geometric_temporal.nn.recurrent import GConvGRU

        # Single GConvGRU layer — fast and effective
        self.gconv_gru = GConvGRU(in_channels=in_features, out_channels=hidden_dim, K=2)

        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, out_steps),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (batch, num_nodes, 12)
        batch_size = x.shape[0]

        # Build batched edge_index: repeat with node offsets per batch item
        edge_indices = []
        edge_weights = []
        for b in range(batch_size):
            offset = b * self.num_nodes
            edge_indices.append(edge_index + offset)
            edge_weights.append(edge_attr)
        batched_edge_index = torch.cat(edge_indices, dim=1)  # (2, B*E)
        batched_edge_attr  = torch.cat(edge_weights, dim=0)  # (B*E,)

        # (batch, nodes, 12) → (batch*nodes, 12)
        x_flat = x.reshape(batch_size * self.num_nodes, 12)

        h = None
        # 12 GConvGRU calls — all batch items processed together
        for t in range(12):
            x_t = x_flat[:, t].unsqueeze(-1)  # (B*207, 1)
            if h is None:
                h = self.gconv_gru(x_t, batched_edge_index, batched_edge_attr)
            else:
                h = self.gconv_gru(x_t, batched_edge_index, batched_edge_attr, h)

        # h: (B*207, hidden_dim)
        h = self.bn(h)

        # (B*207, hidden_dim) → (B, 207, hidden_dim)
        h = h.reshape(batch_size, self.num_nodes, self.hidden_dim)

        # (B, 207, 6)
        out = self.output_layer(h)
        return out


def step3_build_model():
    print("\n=== STEP 3: Defining Fast Sequential T-GCN ===")

    model = FastTrafficGNN(num_nodes=207, in_features=1, hidden_dim=64, out_steps=6)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fast model parameters: {total_params:,}")
    print("Speed: 12 GConvGRU calls per batch (was 16x12=192 in v2)")
    print(f"Estimated speedup: ~16x faster than v2")

    return model, total_params


# ---------------------------------------------------------------------------
# STEP 4 — TRAINING SETUP
# ---------------------------------------------------------------------------
def step4_training_setup(model):
    print("\n=== STEP 4: Training setup ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )

    NUM_EPOCHS      = 30
    PATIENCE        = 7
    MODEL_SAVE_PATH = config.DATA_DIR / "best_gnn_v3_model.pt"

    print(f"Optimizer     : Adam (lr=0.001, weight_decay=1e-4)")
    print(f"Loss          : HuberLoss (delta=1.0)")
    print(f"Scheduler     : CosineAnnealingLR (T_max=30, eta_min=1e-5)")
    print(f"Epochs        : {NUM_EPOCHS}")
    print(f"Early stop    : patience={PATIENCE}")
    print(f"Batch size    : 64")
    print(f"Save path     : {MODEL_SAVE_PATH}")

    return optimizer, criterion, scheduler, NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH


# ---------------------------------------------------------------------------
# STEP 5 — TRAINING LOOP
# ---------------------------------------------------------------------------
def step5_train(model, train_loader, val_loader, optimizer, criterion,
                scheduler, NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH,
                edge_index, edge_attr, total_params):
    print("\n=== STEP 5: Training Fast T-GCN ===")
    print("Batched processing: all 64 items processed together per timestep")

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []
    val_losses       = []
    best_epoch       = 0
    training_start   = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # — Train —
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch, edge_index, edge_attr)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        # — Validate —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred    = model(X_batch, edge_index, edge_attr)
                loss    = criterion(pred, y_batch)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        scheduler.step()
        epoch_time  = time.time() - epoch_start
        current_lr  = scheduler.get_last_lr()[0]

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch    = epoch
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         best_val_loss,
                    "train_losses":     train_losses,
                    "val_losses":       val_losses,
                    "total_params":     total_params,
                },
                MODEL_SAVE_PATH,
            )
            patience_counter = 0
            marker = " << BEST"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train: {avg_train:.5f} | "
            f"Val: {avg_val:.5f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s{marker}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            print(f"Best epoch: {best_epoch}, Val loss: {best_val_loss:.5f}")
            break

    total_time = time.time() - training_start
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return train_losses, val_losses, best_epoch, total_time


# ---------------------------------------------------------------------------
# STEP 6 — PLOT TRAINING HISTORY
# ---------------------------------------------------------------------------
def step6_plot_history(MODEL_SAVE_PATH):
    print("\n=== STEP 6: Plotting training history ===")

    checkpoint         = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    saved_train_losses = checkpoint["train_losses"]
    saved_val_losses   = checkpoint["val_losses"]
    best_ep            = checkpoint["epoch"]

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(saved_train_losses) + 1)

    ax.plot(epochs_range, saved_train_losses, "b-o", markersize=3,
            label="Training Loss", linewidth=2)
    ax.plot(epochs_range, saved_val_losses, color="orange", linestyle="--",
            marker="s", markersize=3, label="Validation Loss", linewidth=2)
    ax.axvline(x=best_ep, color="green", linestyle=":", linewidth=2,
               label=f"Best epoch ({best_ep})")
    ax.scatter([best_ep], [min(saved_val_losses)], color="green",
               s=150, zorder=5, marker="*")
    ax.annotate(
        f"Best model\nVal Loss={min(saved_val_losses):.4f}",
        xy=(best_ep, min(saved_val_losses)),
        xytext=(best_ep + 1, min(saved_val_losses) + 0.002),
        fontsize=9, color="green",
        arrowprops=dict(arrowstyle="->", color="green"),
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Huber Loss", fontsize=12)
    ax.set_title("Fast T-GCN (V3) Training History — METR-LA", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5, -0.05,
        "Figure 7c: Fast T-GCN (v3) training -- batched processing enables full GPU utilization.",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig7c_training_history_v3.png", dpi=150, bbox_inches="tight")
    print("Saved fig7c")
    plt.close()


# ---------------------------------------------------------------------------
# STEP 7 — EVALUATE ON TEST SET
# ---------------------------------------------------------------------------
def step7_evaluate(model, test_loader, edge_index, edge_attr, MODEL_SAVE_PATH):
    print("\n=== STEP 7: Evaluating Fast T-GCN ===")

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred    = model(X_batch, edge_index, edge_attr)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"all_preds   shape: {all_preds.shape}")
    print(f"all_targets shape: {all_targets.shape}")

    gnn_v3_results = {}
    for horizon_idx, horizon_name in zip([0, 2, 5], ["5min", "15min", "30min"]):
        pred_h = all_preds[:, :, horizon_idx].flatten()
        true_h = all_targets[:, :, horizon_idx].flatten()
        mae    = mean_absolute_error(true_h, pred_h)
        rmse   = np.sqrt(mean_squared_error(true_h, pred_h))
        gnn_v3_results[horizon_name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}
        print(f"V3 GNN {horizon_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    np.save(config.DATA_DIR / "gnn_v3_predictions.npy", all_preds)
    np.save(config.DATA_DIR / "gnn_v3_targets.npy",     all_targets)

    return gnn_v3_results


# ---------------------------------------------------------------------------
# STEP 8 — FULL COMPARISON TABLE
# ---------------------------------------------------------------------------
def step8_compare(gnn_v3_results):
    print("\n=== STEP 8: Full comparison ===")

    import pandas as pd

    existing_df = pd.read_csv(config.DATA_DIR / "all_results.csv")

    v3_rows = []
    for horizon in ["5min", "15min", "30min"]:
        v3_rows.append({
            "Model":   "T-GCN-V3",
            "Horizon": horizon,
            "MAE":     gnn_v3_results[horizon]["MAE"],
            "RMSE":    gnn_v3_results[horizon]["RMSE"],
        })

    full_results = pd.concat([existing_df, pd.DataFrame(v3_rows)], ignore_index=True)
    print(full_results.to_string())
    full_results.to_csv(config.DATA_DIR / "all_results_v3.csv", index=False)

    RF_RESULTS = {"5min": 0.1477, "15min": 0.2005, "30min": 0.2631}
    V1_RESULTS = {"5min": 0.1461, "15min": 0.2164, "30min": 0.2990}

    improvements = {}
    for horizon in ["5min", "15min", "30min"]:
        rf_mae    = RF_RESULTS[horizon]
        v1_mae    = V1_RESULTS[horizon]
        v3_mae    = gnn_v3_results[horizon]["MAE"]
        imp_vs_rf = (rf_mae - v3_mae) / rf_mae * 100
        imp_vs_v1 = (v1_mae - v3_mae) / v1_mae * 100
        improvements[horizon] = imp_vs_rf
        print(
            f"{horizon}: RF={rf_mae:.4f} | V1={v1_mae:.4f} | V3={v3_mae:.4f} | "
            f"vs RF: {imp_vs_rf:+.1f}% | vs V1: {imp_vs_v1:+.1f}%"
        )

    return improvements


# ---------------------------------------------------------------------------
# STEP 9 — FINAL SUMMARY
# ---------------------------------------------------------------------------
def step9_summary(total_params, best_epoch, total_time, gnn_v3_results, improvements):
    print("\n" + "=" * 65)
    print("=== STAGE 4 V3 COMPLETE ===")
    print("=" * 65)
    print("Architecture: Fast batched sequential T-GCN")
    print(f"  - Parameters      : {total_params:,}")
    print(f"  - Best epoch      : {best_epoch}")
    print(f"  - Training time   : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  - Hidden dim      : 64")
    print(f"  - GConvGRU layers : 1 (batched)")
    print(f"  - Batch size      : 64")
    print("─" * 65)
    print("V3 GNN Test Results:")
    for horizon in ["5min", "15min", "30min"]:
        r = gnn_v3_results[horizon]
        print(f"  {horizon}: MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}")
    print("─" * 65)
    print("Improvement vs Random Forest:")
    for horizon in ["5min", "15min", "30min"]:
        print(f"  {horizon}: {improvements[horizon]:+.1f}%")
    print("=" * 65)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # Step 1
    X_train, y_train, X_val, y_val, X_test, y_test, edge_index, edge_attr = step1_load_data()

    # Step 2
    train_loader, val_loader, test_loader = step2_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Step 3
    model, total_params = step3_build_model()

    # Step 4
    optimizer, criterion, scheduler, NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH = step4_training_setup(model)

    # Step 5
    train_losses, val_losses, best_epoch, total_time = step5_train(
        model, train_loader, val_loader,
        optimizer, criterion, scheduler,
        NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH,
        edge_index, edge_attr, total_params,
    )

    # Step 6
    step6_plot_history(MODEL_SAVE_PATH)

    # Step 7
    gnn_v3_results = step7_evaluate(model, test_loader, edge_index, edge_attr, MODEL_SAVE_PATH)

    # Step 8
    improvements = step8_compare(gnn_v3_results)

    # Step 9
    step9_summary(total_params, best_epoch, total_time, gnn_v3_results, improvements)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Stage 4 V3 failed with the following traceback:")
        traceback.print_exc()
        sys.exit(1)
