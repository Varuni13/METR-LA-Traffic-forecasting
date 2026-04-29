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

Graph: 207 nodes, 1722 edges
Input: 12 time steps in, 6 steps out
Features: 12 per node per step

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
  metr_env\Scripts\activate  (Windows)

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
All metrics are on normalized speed units.

MAE by Forecast Horizon:
  Model           5min    15min   30min
  Persistence     0.1443  0.2006  0.2660
  Hist Average    0.7514  0.7515  0.7515
  Random Forest   0.1477  0.2005  0.2631
  T-GCN (V3)      0.1400  0.1998  0.2752

RMSE by Forecast Horizon:
  Model           5min    15min   30min
  Persistence     0.3395  0.4911  0.6335
  Hist Average    1.0985  1.0985  1.0985
  Random Forest   0.3277  0.4740  0.6127
  T-GCN (V3)      0.3250  0.4677  0.5994

T-GCN improvement vs best baseline (lowest MAE per horizon):
  5min:  -5.2% MAE vs RF  (T-GCN best)
  15min: -0.4% MAE vs RF  (T-GCN best)
  30min: +4.6% MAE vs RF  (RF slightly better)

KEY FINDINGS
------------
1. T-GCN outperforms all baselines at 5-minute horizon
2. T-GCN matches Random Forest closely at 15-minute horizon
3. At 30 minutes, temporal patterns dominate over spatial graph structure
4. Spatial autocorrelation r = 0.672 confirms graph structure is meaningful
5. T-GCN improves more during congested conditions (graph captures spillover)

EDA HIGHLIGHTS
--------------
Peak congestion hours: 16:00, 17:00, 18:00 (4-6 PM)
Weekday vs weekend speed difference: 0.028 (normalized units)
Spatial autocorrelation (Moran's r): 0.672
Most congested sensors: #16 (99.5%), #196 (98.8%), #56 (69.2%)

MODEL ARCHITECTURE
------------------
Architecture: FastTrafficGNN (single GConvGRU layer)
  - GConvGRU: in_channels=1, out_channels=64, K=2
  - BatchNorm1d(64)
  - Linear(64 -> 32) + ReLU + Dropout(0.1)
  - Linear(32 -> 6)
Model parameters: 27,750
Training time: ~16 minutes on NVIDIA GeForce RTX 3060
Optimizer: Adam (lr=0.001, weight_decay=1e-4)
Scheduler: CosineAnnealingLR
Loss: HuberLoss (delta=1.0)

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
fig9_final_comparison.png      - All models comparison (MAE + RMSE)
fig10_sensor_error.png         - Sensor-level error distribution
fig11_congestion_analysis.png  - Congested vs free-flow MAE
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
