"""
Load Revanth's per-district NPZ output and turn it into per-client training data.

Revanth's preprocessing saves, per district:
    flow.npz       -> key "tensor",    shape (timesteps, nodes, features)
    adjacency.npz  -> key "adjacency", shape (nodes, nodes)
    metadata.npz, info.json  (not needed for training)

In our federated setup, ONE district = ONE client. That's the clean, consistent
client split we agreed to build fresh (the old team snippets that split into 5
clients were inconsistent, per Revanth's review).

Note on differing node counts: districts have different numbers of sensors. That's
fine here. The model's parameters (GAT/GCN/GRU/head weights) are node-independent —
they're shared across nodes and don't depend on N. Only the adjacency matrix, which
is passed in as data (not a learned parameter), is N x N. So all clients share one
global model under FedAvg while each feeds its own adjacency. Worth stating in the
meeting: it's why district-as-client works without padding every district to the
same size.

torch is imported lazily so the pure-numpy path (windowing/split/scaler) stays
testable without the ML stack installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .windowing import HORIZON, LOOKBACK, prepare_district

FLOW_KEY = "tensor"
ADJ_KEY = "adjacency"


def load_district_arrays(district_dir: str | Path):
    """Read one district's flow tensor and adjacency from Revanth's NPZ files."""
    d = Path(district_dir)
    flow_path = _find(d, ("flow.npz", "flow_tensor.npz"))
    adj_path = _find(d, ("adjacency.npz", "adj.npz"))

    tensor = np.load(flow_path)[FLOW_KEY].astype(np.float32)   # (T, N, F)
    adj = np.load(adj_path)[ADJ_KEY].astype(np.float32)        # (N, N)

    if tensor.ndim != 3:
        raise ValueError(f"{flow_path.name}: expected (T,N,F), got {tensor.shape}")
    if adj.shape[0] != tensor.shape[1]:
        raise ValueError(
            f"adjacency N={adj.shape[0]} != tensor N={tensor.shape[1]} in {d}"
        )
    return tensor, adj


def _find(d: Path, names):
    for n in names:
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"none of {names} found in {d}")


def normalise_adjacency(adj: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2, in numpy. Mirrors the torch helper in stgnn.py."""
    a = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    deg = a.sum(1)
    dinv = np.power(deg, -0.5, where=deg > 0)
    dinv[np.isinf(dinv)] = 0.0
    return dinv[:, None] * a * dinv[None, :]


def prepare_all_districts(district_dirs, lookback=LOOKBACK, horizon=HORIZON):
    """Numpy-only. Returns a list of per-client dicts (no torch needed).

    Each entry:
        {
          "name": str,
          "train": (X, Y), "val": (X, Y), "test": (X, Y),   # windowed + scaled
          "scaler": Scaler,
          "adj_mask": (N,N) 0/1,        # for GAT attention masking
          "adj_norm": (N,N) normalised, # for GCN
          "n_nodes": int, "n_features": int,
        }
    """
    clients = []
    for dd in district_dirs:
        tensor, adj = load_district_arrays(dd)
        prep = prepare_district(tensor, lookback, horizon)
        clients.append(
            {
                "name": Path(dd).name,
                "train": prep["train"],
                "val": prep["val"],
                "test": prep["test"],
                "scaler": prep["scaler"],
                "adj_mask": (adj > 0).astype(np.float32),
                "adj_norm": normalise_adjacency(adj).astype(np.float32),
                "n_nodes": tensor.shape[1],
                "n_features": tensor.shape[2],
            }
        )
    return clients


# ---------------------------------------------------------------------------
# torch layer (lazy import). Turns the numpy arrays above into DataLoaders and
# per-client adjacency tensors for the Flower client.
# ---------------------------------------------------------------------------

def build_torch_partitions(clients, batch_size=64):
    """clients: output of prepare_all_districts().

    Returns:
        partitions  : list of (train_loader, val_loader), one per client
        adjacencies : list of (adj_mask, adj_norm) torch tensors, one per client
    Both lists are index-aligned with `clients`.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def loader(split, shuffle):
        X, Y = split
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    partitions, adjacencies = [], []
    for c in clients:
        partitions.append((loader(c["train"], True), loader(c["val"], False)))
        adjacencies.append(
            (torch.from_numpy(c["adj_mask"]), torch.from_numpy(c["adj_norm"]))
        )
    return partitions, adjacencies
