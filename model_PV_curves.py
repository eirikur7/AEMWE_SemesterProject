# model_PV_curves.py
"""
This script compares predicted PV (E_cell vs I_density) curves from trained surrogate models
against a reference simulated PV curve.

Inputs:
- Simulated PV data: archive/tmp.csv
- Trained models: located in models/ folder
- Model log: data_models_docs.csv (must contain performance metrics and filenames)

Outputs:
- Saves figures in figures/ directory:
  1. All model PV curves + simulated (transparent overlay)
  2. Best models by MAE, R², RMSE, MSE + simulated curve
  3. Bar plots of MAE, R², RMSE across all models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

# --- CONFIG ---
MODELS_FOLDER = "data_models"
LOG_FILE = "data_models_docs.csv"
SIM_DATA_FILE = "archive/tmp.csv"
FIG_FOLDER = "figures"
os.makedirs(FIG_FOLDER, exist_ok=True)

# --- Fixed input parameter values (used to pad E_cell into full input vector) ---
X_val = 0.0004
Y_val = 0.1
T_val = 333.15
c_KOH_val = 1000
W_mem_val = 6e-5

# --- Load simulated PV data ---
sim_df = pd.read_csv(SIM_DATA_FILE)
E_sim = sim_df['E_cell'].values.reshape(-1, 1)
I_sim = sim_df['I_density'].values

# --- Compose full input matrix with constant inputs and varying E_cell ---
X_full = np.column_stack([
    np.full_like(E_sim, X_val),
    np.full_like(E_sim, Y_val),
    np.full_like(E_sim, T_val),
    np.full_like(E_sim, c_KOH_val),
    np.full_like(E_sim, W_mem_val),
    E_sim.flatten()  # Varying E_cell
])

# --- Load model log ---
df_log = pd.read_csv(LOG_FILE)

# --- Plot 1: All predicted PV curves (transparent) + Simulated ---
plt.figure(figsize=(10, 6))
plt.xlim(1.2, 2.5)
plt.ylim(0, 1.5)
plt.xlim(1.2, 2.5)
plt.ylim(0, 1.5)

for model_row in df_log.itertuples():
    model_path = os.path.join(MODELS_FOLDER, model_row.model_name + ".keras")
    scaler_path = os.path.join(MODELS_FOLDER, model_row.model_name + "_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        continue

    model = load_model(model_path, custom_objects={"LeakyReLU": LeakyReLU})
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X_full)
    I_pred = model.predict(X_scaled).flatten()
    plt.plot(E_sim.flatten(), I_pred, alpha=0.3, color='blue')

plt.plot(E_sim.flatten(), I_sim, color='black', linewidth=2.5, label='Simulated')
plt.title("All Model PV Curves vs Simulated")
plt.xlabel("E_cell [V]")
plt.ylabel("I_density [A/cm²]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, "all_models_pv_curves.png"))
plt.show()

# --- Plot 2: Best models per metric ---
metrics = ['mae', 'r2', 'rmse', 'val_loss_best']
plt.figure(figsize=(10, 6))
colors = sns.color_palette("tab10", len(metrics))

for i, metric in enumerate(metrics):
    if metric == 'r2':
        best = df_log.sort_values(by=metric, ascending=False).iloc[0]
    else:
        best = df_log.sort_values(by=metric).iloc[0]

    model_path = os.path.join(MODELS_FOLDER, best.model_name + ".keras")
    scaler_path = os.path.join(MODELS_FOLDER, best.model_name + "_scaler.pkl")
    model = load_model(model_path, custom_objects={"LeakyReLU": LeakyReLU})
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X_full)
    I_pred = model.predict(X_scaled).flatten()
    plt.plot(E_sim.flatten(), I_pred, label=f"Best {metric.upper()}", color=colors[i])

plt.plot(E_sim.flatten(), I_sim, color='black', linewidth=2.5, label='Simulated')
plt.title("Best Models by Metric vs Simulated")
plt.xlabel("E_cell [V]")
plt.ylabel("I_density [A/cm²]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, "best_models_by_metric.png"))
plt.show()

# --- Plot 3: Metrics comparison across all models ---
plot_df = df_log[['model_name', 'mae', 'r2', 'rmse', 'val_loss_best']].copy()
plot_df = plot_df.sort_values(by='mae')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.barplot(data=plot_df, x='model_name', y='mae', ax=axes[0])
sns.barplot(data=plot_df, x='model_name', y='rmse', ax=axes[1])
sns.barplot(data=plot_df, x='model_name', y='r2', ax=axes[2])
sns.barplot(data=plot_df, x='model_name', y='val_loss_best', ax=axes[3])

for ax, title in zip(axes.flatten(), ['MAE', 'RMSE', 'R²', 'Val Loss']):
    ax.set_title(title)
    ax.set_xticks([])
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, "metric_bars_all_models.png"))
plt.show()
