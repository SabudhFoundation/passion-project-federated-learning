"""
Windowing, chronological split, and scaling for per-district traffic tensors.

This is the missing layer between Revanth's preprocessing and the Flower client.
His pipeline saves each district as flow.npz with a tensor of shape

    (timesteps, nodes, features)     # features = flow + temporal encodings

but stops there. The model needs sliding windows: given the last L time-steps,
predict the next H. This module does exactly that, leakage-free.

Pure numpy on purpose — no torch — so it can be unit-tested anywhere and reused
outside the training stack. The torch DataLoader wrapper lives in district_data.py.

Design decisions (state these if asked):
- Split is CHRONOLOGICAL (train = earliest, then val, then test). Traffic is a time
  series; shuffling would leak the future into the past.
- The scaler is fit on the TRAIN span only, then applied to val/test. Fitting on all
  data is the classic leakage bug.
- Windows are built WITHIN each split, so no single window straddles a split
  boundary.
- Only the flow channel (channel 0) is standardised and predicted. The temporal
  feature channels are already bounded encodings from Revanth's pipeline, so they're
  passed through untouched. Predictions are inverse-transformed before metrics, so
  MAE / RMSE come out in real traffic units and are comparable to Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOOKBACK = 12   # Phase 1 config: 12 time-steps in
HORIZON = 5     # Phase 1 config: 5 time-steps out
TARGET_CHANNEL = 0  # channel 0 is traffic flow, the thing we forecast


@dataclass
class Scaler:
    """Standardises the flow channel using train-set statistics."""

    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    @classmethod
    def fit(cls, flow_channel: np.ndarray) -> "Scaler":
        mean = float(np.mean(flow_channel))
        std = float(np.std(flow_channel)) or 1.0  # guard against a constant span
        return cls(mean=mean, std=std)


def chronological_split(T: int, train=0.7, val=0.15):
    """Return (train_slice, val_slice, test_slice) over the time axis."""
    n_train = int(T * train)
    n_val = int(T * val)
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, T),
    )


def make_windows(tensor: np.ndarray, lookback=LOOKBACK, horizon=HORIZON,
                 target_channel=TARGET_CHANNEL):
    """(T, N, F) -> X (num, lookback, N, F), Y (num, N, horizon).

    Y is the flow channel over the horizon, arranged (nodes, horizon) to match the
    model output (B, N, horizon).
    """
    T, N, F = tensor.shape
    num = T - lookback - horizon + 1
    if num <= 0:
        raise ValueError(
            f"span too short: T={T} needs > lookback+horizon={lookback+horizon}"
        )

    X = np.empty((num, lookback, N, F), dtype=np.float32)
    Y = np.empty((num, N, horizon), dtype=np.float32)
    flow = tensor[:, :, target_channel]  # (T, N)

    for i in range(num):
        X[i] = tensor[i : i + lookback]
        # horizon flow, transposed to (N, horizon)
        Y[i] = flow[i + lookback : i + lookback + horizon].T
    return X, Y


def prepare_district(tensor: np.ndarray, lookback=LOOKBACK, horizon=HORIZON,
                     train=0.7, val=0.15, target_channel=TARGET_CHANNEL):
    """Full pipeline for one district tensor.

    Returns a dict with windowed, scaled X/Y for each split plus the fitted scaler.
    Scaler is fit on the train span's flow channel only.
    """
    T = tensor.shape[0]
    tr, va, te = chronological_split(T, train, val)

    # Fit scaler on train flow only, then standardise the flow channel everywhere.
    scaler = Scaler.fit(tensor[tr][:, :, target_channel])
    scaled = tensor.copy().astype(np.float32)
    scaled[:, :, target_channel] = scaler.transform(scaled[:, :, target_channel])

    out = {"scaler": scaler}
    for name, sl in (("train", tr), ("val", va), ("test", te)):
        X, Y = make_windows(scaled[sl], lookback, horizon, target_channel)
        out[name] = (X, Y)
    return out
