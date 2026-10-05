"""Windowing, chronological split, and scaling for a district traffic tensor."""

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
                     train=0.7, val=0.15, target_channel=TARGET_CHANNEL,
                     scaler: "Scaler | None" = None):
    """Full pipeline for one district tensor.

    Returns a dict with windowed X/Y for each split plus the scaler used.

    If `scaler` is None, the flow channel is standardised here, fitting on the
    train span only (used for raw synthetic data). If a scaler is passed, the flow
    is assumed already standardised upstream (Revanth's pipeline z-scores it and
    stores mean/std in info.json), so we do NOT re-standardise, we just carry the
    scaler for inverse-transform to real traffic units at metric time.
    """
    T = tensor.shape[0]
    tr, va, te = chronological_split(T, train, val)

    scaled = tensor.astype(np.float32)
    if scaler is None:
        scaler = Scaler.fit(tensor[tr][:, :, target_channel])
        scaled = scaled.copy()
        scaled[:, :, target_channel] = scaler.transform(scaled[:, :, target_channel])
    # else: data already standardised upstream, leave values as-is.

    out = {"scaler": scaler}
    for name, sl in (("train", tr), ("val", va), ("test", te)):
        X, Y = make_windows(scaled[sl], lookback, horizon, target_channel)
        out[name] = (X, Y)
    return out
