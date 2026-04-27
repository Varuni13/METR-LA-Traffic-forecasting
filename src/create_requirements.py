import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ---------------------------------------------------------------------------
# STEP 1 — requirements.txt
# ---------------------------------------------------------------------------
def step1_requirements():
    content = """\
torch==2.5.1
torch-geometric==2.7.0
torch-geometric-temporal==0.54.0
numpy==2.4.3
pandas==2.3.1
scikit-learn==1.7.1
xgboost==3.2.0
matplotlib==3.10.8
seaborn==0.13.2
geopandas==1.1.1
folium==0.20.0
contextily==1.6.2
networkx==3.5
scipy==1.16.0
"""
    out = config.PROJECT_ROOT / "requirements.txt"
    out.write_text(content, encoding="utf-8")
    print("Created requirements.txt")


# ---------------------------------------------------------------------------
# STEP 2 — README.txt
# ---------------------------------------------------------------------------
def step2_readme():
    content = """\
============================================================
METR-LA Graph ML Traffic Forecasting
EM627: Spatial Data Science and Applications
============================================================

PROJECT OVERVIEW
----------------
End-to-end graph machine learning pipeline for short-horizon
traffic forecasting on the METR-LA road sensor network.
Supports operational decision-making for ride-hailing platforms.

DATASET
-------
METR-LA: 207 loop detectors, 5-minute intervals, ~4 months
Source: PyTorch Geometric Temporal / DCRNN repository
Auto-downloaded on first run of stage1_setup.py

FOLDER STRUCTURE
----------------
src/              Python source files (run in order)
data/             Dataset, model checkpoints, results
figures/          All exported figures (PNG + HTML)
requirements.txt  Python dependencies
README.txt        This file

HOW TO REPRODUCE RESULTS
--------------------------
Step 1: Create virtual environment
  python -m venv metr_env
  metr_env\\Scripts\\activate  (Windows)

Step 2: Install PyTorch with CUDA (if GPU available)
  pip install torch --index-url https://download.pytorch.org/whl/cu121

Step 3: Install remaining dependencies
  pip install -r requirements.txt

Step 4: Run stages in order
  python src/stage1_setup.py      # Load and inspect dataset
  python src/stage2_eda.py        # Exploratory data analysis (5 figures)
  python src/stage3_baseline.py   # Train baseline models
  python src/stage4_gnn_v3.py     # Train T-GCN graph model (~16 min GPU)
  python src/stage5_evaluate.py   # Evaluate and compare all models
  python src/stage6_business.py   # Generate business maps and recommendations

RANDOM SEED
-----------
All experiments use random seed 42 for reproducibility.
Set in config.py and propagated to all stages.

KEY RESULTS
-----------
T-GCN (Graph ML) vs Random Forest baseline:
  5-minute horizon:  MAE = 0.140  (+5.2% improvement)
  15-minute horizon: MAE = 0.200  (+0.3% improvement)
  30-minute horizon: MAE = 0.275  (-4.6%, RF slightly better)

Spatial autocorrelation: r = 0.672
Training time: 16 minutes on NVIDIA GeForce RTX 3060
Model parameters: 27,750

FIGURES GENERATED
-----------------
fig1_speed_by_hour.png         - Hourly speed patterns
fig2_weekday_weekend.png       - Weekday vs weekend comparison
fig3_heatmap.png               - Sensor x time speed heatmap
fig4_spatial_autocorr.png      - Spatial autocorrelation plot
fig5_network.png               - Sensor network graph
fig6_baseline_comparison.png   - Baseline model comparison
fig7c_training_history_v3.png  - T-GCN training history
fig8_architecture.png          - Model architecture diagram
fig9_final_comparison.png      - All models comparison
fig10_sensor_error.png         - Sensor-level error analysis
fig11_congestion_analysis.png  - Congestion vs free-flow
fig12_sensor_network_map.png   - Sensor network on LA map
fig13_congestion_hotspots.png  - Congestion hotspot map
fig14_prediction_error_map.png - Prediction reliability map
interactive_sensor_map.html    - Interactive folium map

HARDWARE USED
-------------
GPU: NVIDIA GeForce RTX 3060
RAM: 16GB
OS: Windows 11
Python: 3.12.x

CONTACT
-------
Student submission -- EM627 Spatial Data Science
April 2026
============================================================
"""
    out = config.PROJECT_ROOT / "README.txt"
    out.write_text(content, encoding="utf-8")
    print("Created README.txt")


# ---------------------------------------------------------------------------
# STEP 3 — Confirmation
# ---------------------------------------------------------------------------
def step3_confirm():
    print("=" * 50)
    print("FILES CREATED:")
    print("  requirements.txt")
    print("  README.txt")
    print("  executive_memo.html (open in browser -> Ctrl+P -> Save as PDF)")
    print("=" * 50)
    print("NEXT STEP: Open executive_memo.html in Chrome/Edge")
    print("           Press Ctrl+P -> set to A4 -> Save as PDF")
    print("           Save as: executive_memo.pdf")


if __name__ == "__main__":
    step1_requirements()
    step2_readme()
    step3_confirm()
