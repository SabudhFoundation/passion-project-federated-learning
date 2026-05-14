"""
KAN-LSTM Hybrid Multi-Step Speed Forecasting
=============================================
Architecture:
  LSTM  →  encodes temporal patterns across LOOKBACK steps (memory, gates)
  KAN   →  maps LSTM hidden state to n-step forecast (interpretable B-splines)

Data format:
  - Row 0 (index)  : Timestamp  "DD-MM-YYYY HH:MM"
  - Columns        : Sensor IDs  e.g. 404356, 402518, ...
  - Values         : Speed / flow integers

Install:
    pip install efficient-kan torch scikit-learn pandas numpy matplotlib
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from efficient_kan import KAN
except ImportError:
    raise ImportError("Run:  pip install efficient-kan")

# ─────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────
CSV_PATH    = "/Users/manavsharma/1_Work/Passion_Project_Sabudh/Code_Model/gba_2019_small(1).csv"
DATE_FORMAT = "%d-%m-%Y %H:%M"

LOOKBACK        = 10    # past timesteps fed to LSTM
FORECAST_STEPS  = 3     # steps to predict
BATCH_SIZE      = 64
EPOCHS          = 30
LR              = 1e-3
WEIGHT_DECAY    = 1e-4

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

# LSTM
LSTM_HIDDEN  = 128   # hidden size per layer
LSTM_LAYERS  = 2     # stacked LSTM depth
LSTM_DROPOUT = 0.2   # dropout between LSTM layers

# KAN  (operates on LSTM output)
KAN_GRID_SIZE    = 5
KAN_SPLINE_ORDER = 3

# Set to a list of sensor IDs to train on a subset; None = all sensors
SENSORS = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format=DATE_FORMAT)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    if SENSORS is not None:
        df = df[[str(s) for s in SENSORS if str(s) in df.columns]]
    df = df.interpolate(method="time").ffill().bfill()
    freq = pd.infer_freq(df.index)
    print(f"Loaded  : {df.shape[0]} timestamps x {df.shape[1]} sensors")
    print(f"Range   : {df.index[0]}  ->  {df.index[-1]}")
    print(f"Freq    : {freq}")
    return df




# ─────────────────────────────────────────────────────────────────
# 3. DATASET  (sequences for LSTM)
# ─────────────────────────────────────────────────────────────────
class SensorDataset(Dataset):
    """
    X : (LOOKBACK, n_sensors)          — 3-D sequence for LSTM
    y : (FORECAST_STEPS * n_sensors,)  — flattened target
    """
    def __init__(self, arr: np.ndarray):
        xs, ys = [], []
        n = len(arr)
        for i in range(n - LOOKBACK - FORECAST_STEPS + 1):
            xs.append(arr[i : i + LOOKBACK])                                   # (L, S)
            ys.append(arr[i + LOOKBACK : i + LOOKBACK + FORECAST_STEPS].flatten())  # (F*S,)
        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32)

    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.y[i]


# ─────────────────────────────────────────────────────────────────
# 4. KAN-LSTM MODEL
# ─────────────────────────────────────────────────────────────────
class KANLSTMForecaster(nn.Module):
    """
    Stage 1 — LSTM encoder:
        Input  : (batch, LOOKBACK, n_sensors)
        Output : final hidden state h_T  shape (batch, LSTM_HIDDEN)
                 (we take only the last layer's last time-step)

    Stage 2 — KAN decoder:
        Input  : h_T  (batch, LSTM_HIDDEN)
        Output : (batch, FORECAST_STEPS * n_sensors)

    Why this split?
      LSTM  → specialised at remembering sequential context; handles
              non-stationarity, gating irrelevant past values.
      KAN   → replaces the plain linear/MLP projection head with
              learnable B-spline activations, giving richer expressivity
              and interpretable per-edge activation functions.
    """
    def __init__(self, n_sensors: int):
        super().__init__()
        self.n_sensors = n_sensors
        out_dim = FORECAST_STEPS * n_sensors

        # ── LSTM ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = n_sensors,
            hidden_size  = LSTM_HIDDEN,
            num_layers   = LSTM_LAYERS,
            batch_first  = True,
            dropout      = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        # Dropout after LSTM (applied to the output h_T before KAN)
        self.drop = nn.Dropout(p=0.1)

        # ── KAN head ─────────────────────────────────
        # Two hidden layers; width scales with output dimensionality
        h1 = min(max(LSTM_HIDDEN, 64), 256)
        h2 = max(h1 // 2, 32)
        self.kan = KAN(
            layers_hidden = [LSTM_HIDDEN, h1, h2, out_dim],
            grid_size     = KAN_GRID_SIZE,
            spline_order  = KAN_SPLINE_ORDER,
        )
        print(f"  LSTM : input={n_sensors}  hidden={LSTM_HIDDEN}  layers={LSTM_LAYERS}")
        print(f"  KAN  : {LSTM_HIDDEN} -> {h1} -> {h2} -> {out_dim}")

    def forward(self, x):
        # x : (batch, LOOKBACK, n_sensors)
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n : (num_layers, batch, hidden) — take last layer
        h_T = h_n[-1]                        # (batch, LSTM_HIDDEN)
        h_T = self.drop(h_T)
        return self.kan(h_T)                 # (batch, FORECAST_STEPS * n_sensors)


# ─────────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────────
def train_model(model, train_dl, val_dl):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None

    for ep in range(1, EPOCHS + 1):
        model.train()
        run = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(xb)
        tr = run / len(train_dl.dataset)

        model.eval()
        run = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                run += loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl = run / len(val_dl.dataset)

        sched.step()
        history["train"].append(tr)
        history["val"].append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{EPOCHS}  train={tr:.5f}  val={vl:.5f}")

    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val:.5f}")
    return history


# ─────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────
def evaluate(model, test_dl, scaler, n_sensors):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            targets.append(yb.numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    def inv(arr):
        return scaler.inverse_transform(
            arr.reshape(-1, n_sensors)
        ).reshape(len(arr), FORECAST_STEPS, n_sensors)

    P = inv(preds)
    T = inv(targets)

    mae  = mean_absolute_error(T.flatten(), P.flatten())
    rmse = math.sqrt(mean_squared_error(T.flatten(), P.flatten()))
    mape = np.mean(np.abs((T - P) / np.clip(np.abs(T), 1, None))) * 100

    print(f"\n{'─'*45}")
    print(f"  Test MAE   : {mae:.3f}")
    print(f"  Test RMSE  : {rmse:.3f}")
    print(f"  Test MAPE  : {mape:.2f}%")
    print(f"{'─'*45}\n")

    # Per-step breakdown
    print("  RMSE per forecast step:")
    for s in range(FORECAST_STEPS):
        step_rmse = math.sqrt(mean_squared_error(T[:, s, :].flatten(), P[:, s, :].flatten()))
        print(f"    t+{s+1}: {step_rmse:.3f}")
    print()
    return P, T


# ─────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────
def plot_all(history, P, T_gt, sensor_names):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("KAN-LSTM Speed Forecaster — Results", fontsize=14, fontweight="bold")

    # (a) Loss curves
    ax = axes[0, 0]
    ax.plot(history["train"], label="Train")
    ax.plot(history["val"],   label="Val")
    ax.set_title("Training / Validation Loss (Huber)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # (b) t+1 forecast — first sensor
    n = min(300, len(P))
    ax = axes[0, 1]
    ax.plot(T_gt[:n, 0, 0], label="Actual", lw=1.5)
    ax.plot(P[:n, 0, 0],    label="Pred t+1", lw=1.5, ls="--")
    ax.set_title(f"Sensor {sensor_names[0]} — t+1")
    ax.set_xlabel("Sample"); ax.set_ylabel("Speed")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (c) t+5 forecast — first sensor
    ax = axes[0, 2]
    ax.plot(T_gt[:n, -1, 0], label="Actual",  lw=1.5)
    ax.plot(P[:n, -1, 0],    label="Pred t+5", lw=1.5, ls="--")
    ax.set_title(f"Sensor {sensor_names[0]} — t+5")
    ax.set_xlabel("Sample"); ax.set_ylabel("Speed")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (d) Scatter
    ax = axes[1, 0]
    ft, fp = T_gt.flatten(), P.flatten()
    ax.scatter(ft, fp, alpha=0.1, s=3, color="steelblue")
    lo, hi = ft.min(), ft.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set_title("Predicted vs Actual — all sensors & steps")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (e) RMSE per step
    ax = axes[1, 1]
    step_rmse = [
        math.sqrt(mean_squared_error(T_gt[:, s, :].flatten(), P[:, s, :].flatten()))
        for s in range(FORECAST_STEPS)
    ]
    ax.bar([f"t+{i+1}" for i in range(FORECAST_STEPS)], step_rmse, color="steelblue", alpha=0.8)
    ax.set_title("RMSE by forecast step")
    ax.set_ylabel("RMSE"); ax.grid(True, alpha=0.3, axis="y")

    # (f) Error distribution
    ax = axes[1, 2]
    errors = (P - T_gt).flatten()
    ax.hist(errors, bins=60, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_title("Prediction error distribution")
    ax.set_xlabel("Error (predicted - actual)"); ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("kan_lstm_results.png", dpi=150, bbox_inches="tight")
    print("  Plots saved -> kan_lstm_results.png")


# ─────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*52}")
    print(" KAN-LSTM Multi-Step Speed Forecaster")
    print(f"{'='*52}\n")
    print(f"Device : {DEVICE}\n")

    # Load
    df = load_csv(CSV_PATH) 
    sensor_names = df.columns.tolist()
    n_sensors    = len(sensor_names)
    values       = df.values.astype(np.float32)

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Split
    T_total = len(scaled)
    n_tr    = int(T_total * TRAIN_FRAC)
    n_val   = int(T_total * (TRAIN_FRAC + VAL_FRAC))

    ds_tr  = SensorDataset(scaled[:n_tr])
    ds_val = SensorDataset(scaled[n_tr:n_val])
    ds_te  = SensorDataset(scaled[n_val:])

    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Samples -> train: {len(ds_tr)} | val: {len(ds_val)} | test: {len(ds_te)}\n")

    # Build
    model = KANLSTMForecaster(n_sensors).to(DEVICE)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total:,}\n")

    # Train
    history = train_model(model, dl_tr, dl_val)
    torch.save(model.state_dict(), "kan_lstm_speed_model.pth")
    print("  Model saved -> kan_lstm_speed_model.pth")

    # Evaluate
    P, T_gt = evaluate(model, dl_te, scaler, n_sensors)

    # Plot
    plot_all(history, P, T_gt, sensor_names)

    # Inference: next 5 steps from last window
    print("-- Next n-step forecast from last known window --")
    last_win = torch.tensor(
        scaled[-LOOKBACK:][np.newaxis, :, :],    # (1, LOOKBACK, n_sensors)
        dtype=torch.float32
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(last_win).cpu().numpy()




if __name__ == "__main__":
    main()
