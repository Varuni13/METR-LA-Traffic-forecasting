import sys
import os
from pathlib import Path

# Allow importing config from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# =============================================================================
# STEP 1 - Import libraries and print versions
# =============================================================================
print("=" * 60)
print("  STEP 1: Importing libraries")
print("=" * 60)

try:
    import torch
    print(f"  torch                    : {torch.__version__}")
except ImportError as e:
    print(f"  [ERROR] torch not found: {e}"); sys.exit(1)

try:
    import torch_geometric_temporal
    print(f"  torch_geometric_temporal : {torch_geometric_temporal.__version__}")
except ImportError as e:
    print(f"  [ERROR] torch_geometric_temporal not found: {e}"); sys.exit(1)

try:
    import sklearn
    print(f"  sklearn                  : {sklearn.__version__}")
except ImportError as e:
    print(f"  [ERROR] sklearn not found: {e}")

try:
    import pandas as pd
    print(f"  pandas                   : {pd.__version__}")
except ImportError as e:
    print(f"  [ERROR] pandas not found: {e}")

try:
    import numpy as np
    print(f"  numpy                    : {np.__version__}")
except ImportError as e:
    print(f"  [ERROR] numpy not found: {e}"); sys.exit(1)

try:
    import matplotlib
    print(f"  matplotlib               : {matplotlib.__version__}")
except ImportError as e:
    print(f"  [ERROR] matplotlib not found: {e}")

print()
config.print_config()

# =============================================================================
# STEP 2 - Load METR-LA dataset
# =============================================================================
print()
print("=" * 60)
print("  STEP 2: Loading METR-LA dataset")
print("=" * 60)

try:
    from torch_geometric_temporal.dataset import METRLADatasetLoader
    loader  = METRLADatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=6)
    print("  Dataset loaded successfully")
except Exception as e:
    print(f"  [ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# =============================================================================
# STEP 3 - Inspect first snapshot
# =============================================================================
print()
print("=" * 60)
print("  STEP 3: Inspecting dataset structure (first snapshot)")
print("=" * 60)

try:
    snapshot = next(iter(dataset))

    num_nodes     = snapshot.x.shape[0]
    num_timesteps = snapshot.x.shape[1] if snapshot.x.dim() >= 2 else 1
    num_features  = snapshot.x.shape[2] if snapshot.x.dim() == 3 else snapshot.x.shape[1]
    num_edges     = snapshot.edge_index.shape[1]
    target_shape  = tuple(snapshot.y.shape)
    x_dtype       = snapshot.x.dtype
    y_dtype       = snapshot.y.dtype

    print(f"  Number of nodes           : {num_nodes}")
    print(f"  Input time steps (SEQ_IN) : {num_timesteps}")
    print(f"  Features per node/step    : {num_features}")
    print(f"  Number of edges           : {num_edges}")
    print(f"  Target shape (y)          : {target_shape}")
    print(f"  x dtype                   : {x_dtype}")
    print(f"  y dtype                   : {y_dtype}")
    print(f"  x full shape              : {tuple(snapshot.x.shape)}")
except Exception as e:
    print(f"  [ERROR] Snapshot inspection failed: {e}")
    sys.exit(1)

# =============================================================================
# STEP 4 - Count total snapshots
# =============================================================================
print()
print("=" * 60)
print("  STEP 4: Counting total snapshots")
print("=" * 60)

try:
    total_snapshots = sum(1 for _ in dataset)
    print(f"  Total snapshots in dataset : {total_snapshots}")
except Exception as e:
    print(f"  [ERROR] Could not count snapshots: {e}")
    total_snapshots = -1

# =============================================================================
# STEP 5 - Sample data values
# =============================================================================
print()
print("=" * 60)
print("  STEP 5: Sample data values (first snapshot)")
print("=" * 60)

try:
    import numpy as np
    snapshot = next(iter(dataset))

    x_np = snapshot.x.numpy()
    y_np = snapshot.y.numpy()

    print(f"  x  ->  min: {x_np.min():.4f}  |  max: {x_np.max():.4f}  |  mean: {x_np.mean():.4f}")
    print(f"  y  ->  min: {y_np.min():.4f}  |  max: {y_np.max():.4f}  |  mean: {y_np.mean():.4f}")
except Exception as e:
    print(f"  [ERROR] Could not compute value summaries: {e}")

# =============================================================================
# STEP 6 - Save summary to data/dataset_summary.txt
# =============================================================================
print()
print("=" * 60)
print("  STEP 6: Saving dataset summary")
print("=" * 60)

try:
    summary_path = config.DATA_DIR / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("METR-LA Dataset Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Number of nodes           : {num_nodes}\n")
        f.write(f"Input time steps (SEQ_IN) : {num_timesteps}\n")
        f.write(f"Features per node/step    : {num_features}\n")
        f.write(f"Number of edges           : {num_edges}\n")
        f.write(f"Target shape (y)          : {target_shape}\n")
        f.write(f"x dtype                   : {x_dtype}\n")
        f.write(f"y dtype                   : {y_dtype}\n")
        f.write(f"Total snapshots           : {total_snapshots}\n")
        f.write("-" * 40 + "\n")
        f.write(f"x  min/max/mean : {x_np.min():.4f} / {x_np.max():.4f} / {x_np.mean():.4f}\n")
        f.write(f"y  min/max/mean : {y_np.min():.4f} / {y_np.max():.4f} / {y_np.mean():.4f}\n")
    print(f"  Summary saved to: {summary_path}")
except Exception as e:
    print(f"  [ERROR] Could not save summary: {e}")

# =============================================================================
# STEP 7 - Success box
# =============================================================================
print()
print("+" + "=" * 58 + "+")
print("|  STAGE 1 COMPLETE -- Key Numbers                         |")
print("+" + "=" * 58 + "+")
print(f"|  Nodes          : {num_nodes:<39}|")
print(f"|  Edges          : {num_edges:<39}|")
print(f"|  Input steps    : {num_timesteps:<39}|")
print(f"|  Features       : {num_features:<39}|")
print(f"|  Target shape   : {str(target_shape):<39}|")
print(f"|  Total snapshots: {total_snapshots:<39}|")
print(f"|  Device         : {str(config.DEVICE):<39}|")
print("+" + "=" * 58 + "+")
