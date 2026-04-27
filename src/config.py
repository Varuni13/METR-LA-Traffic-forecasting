import os
import torch
from pathlib import Path

# -- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FIGURES_DIR  = PROJECT_ROOT / "figures"
SRC_DIR      = PROJECT_ROOT / "src"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -- Reproducibility ----------------------------------------------------------
RANDOM_SEED = 42

# -- Hardware -----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Sequence settings --------------------------------------------------------
SEQ_IN        = 12                          # last 12 steps = last 60 minutes
HORIZONS      = [1, 3, 6]                   # predict 5min, 15min, 30min ahead
HORIZON_NAMES = ["5min", "15min", "30min"]

# -- Train / val / test split -------------------------------------------------
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2

# -- Graph / feature dimensions -----------------------------------------------
NUM_NODES    = 207
NUM_FEATURES = 2


def print_config():
    print("=" * 55)
    print("  METR-LA Assignment - Configuration")
    print("=" * 55)
    print(f"  PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"  DATA_DIR      : {DATA_DIR}")
    print(f"  FIGURES_DIR   : {FIGURES_DIR}")
    print(f"  SRC_DIR       : {SRC_DIR}")
    print("-" * 55)
    print(f"  RANDOM_SEED   : {RANDOM_SEED}")
    print(f"  DEVICE        : {DEVICE}")
    print("-" * 55)
    print(f"  SEQ_IN        : {SEQ_IN}  (input steps)")
    print(f"  HORIZONS      : {HORIZONS}")
    print(f"  HORIZON_NAMES : {HORIZON_NAMES}")
    print("-" * 55)
    print(f"  TRAIN_RATIO   : {TRAIN_RATIO}")
    print(f"  VAL_RATIO     : {VAL_RATIO}")
    print(f"  TEST_RATIO    : {TEST_RATIO}")
    print("-" * 55)
    print(f"  NUM_NODES     : {NUM_NODES}")
    print(f"  NUM_FEATURES  : {NUM_FEATURES}")
    print("=" * 55)


if __name__ == "__main__":
    print_config()
