"""Load per-district NPZ files and build per-client datasets and adjacency.

Reads the LargeST preprocessing output directly:
  flow_temporal.npz  the (timesteps, sensors, 6) tensor (flow + time encodings)
  metadata.npz       static sensor features including lat/lng
  info.json          flow normalization mean/std (flow is already z-scored)

The preprocessing does not output an adjacency matrix, so the graph is built here
from sensor coordinates: sensors that are close on the map are connected. Falls
back to reading an adjacency file if one is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .windowing import HORIZON, LOOKBACK, Scaler, prepare_district

# File-name candidates (handles both Revanth's output and the synthetic test data).
FLOW_FILES = ("flow_temporal.npz", "flow.npz", "flow_tensor.npz")
META_FILES = ("metadata.npz", "meta.npz")
ADJ_FILES = ("adjacency.npz", "adj.npz")
INFO_FILES = ("info.json",)


def _find(d: Path, names, required=True):
    for n in names:
        if (d / n).exists():
            return d / n
    if required:
        raise FileNotFoundError(f"none of {names} found in {d}")
    return None


def _pick_3d_array(npz):
    """Return the first 3D array in an npz (the flow tensor), auto-detecting the key."""
    for k in npz.files:
        a = npz[k]
        if getattr(a, "ndim", 0) == 3:
            return a.astype(np.float32)
    raise ValueError(f"no 3D array found in npz (keys: {list(npz.files)})")


def _read_latlng(meta_npz):
    """Pull lat/lng arrays from metadata.npz, trying common key names."""
    keys = {k.lower(): k for k in meta_npz.files}
    lat_k = next((keys[c] for c in ("lat", "latitude", "y") if c in keys), None)
    lng_k = next((keys[c] for c in ("lng", "lon", "long", "longitude", "x") if c in keys), None)
    if lat_k is None or lng_k is None:
        return None
    return (meta_npz[lat_k].astype(np.float64).ravel(),
            meta_npz[lng_k].astype(np.float64).ravel())


def build_adjacency_from_coords(lat, lng, k=8):
    """k-nearest-neighbour graph from sensor coordinates, Gaussian-weighted.

    Standard construction for traffic GNNs when no explicit road graph is given:
    connect each sensor to its k closest sensors, weight by distance with a
    Gaussian kernel, and symmetrise.
    """
    n = len(lat)
    coords = np.stack([lat, lng], axis=1)
    # pairwise euclidean distance on (lat, lng)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    sigma = dist[dist > 0].std() or 1.0
    A = np.zeros((n, n), dtype=np.float32)
    k = min(k, n - 1)
    for i in range(n):
        nearest = np.argsort(dist[i])[1 : k + 1]  # skip self
        for j in nearest:
            w = float(np.exp(-(dist[i, j] ** 2) / (2 * sigma ** 2)))
            A[i, j] = A[j, i] = max(A[i, j], w)
    return A


def load_district(district_dir: str | Path):
    """Read one district: flow tensor, adjacency, and (if present) the flow scaler.

    Returns (tensor, adj, scaler_or_None).
    """
    d = Path(district_dir)

    tensor = _pick_3d_array(np.load(_find(d, FLOW_FILES)))   # (T, N, F)
    N = tensor.shape[1]

    # Adjacency: use a file if present, else build from metadata coordinates.
    adj_path = _find(d, ADJ_FILES, required=False)
    if adj_path is not None:
        adj = _pick_square_array(np.load(adj_path), N)
    else:
        meta_path = _find(d, META_FILES, required=False)
        coords = _read_latlng(np.load(meta_path, allow_pickle=True)) if meta_path else None
        if coords is None:
            raise FileNotFoundError(
                f"{d}: no adjacency file and no lat/lng in metadata to build one"
            )
        adj = build_adjacency_from_coords(*coords)

    if adj.shape[0] != N:
        raise ValueError(f"adjacency N={adj.shape[0]} != tensor N={N} in {d}")

    # Flow scaler from info.json (flow is already z-scored upstream).
    scaler = None
    info_path = _find(d, INFO_FILES, required=False)
    if info_path is not None:
        info = json.loads(Path(info_path).read_text())
        norm = info.get("flow_normalization") or info.get("normalization") or {}
        if "mean" in norm and "std" in norm:
            scaler = Scaler(mean=float(norm["mean"]), std=float(norm["std"]))

    return tensor, adj.astype(np.float32), scaler


def _pick_square_array(npz, n):
    for k in npz.files:
        a = npz[k]
        if getattr(a, "ndim", 0) == 2 and a.shape[0] == a.shape[1] == n:
            return a.astype(np.float32)
    # fall back to first 2D array
    for k in npz.files:
        a = npz[k]
        if getattr(a, "ndim", 0) == 2:
            return a.astype(np.float32)
    raise ValueError(f"no square adjacency found in npz (keys: {list(npz.files)})")


def normalise_adjacency(adj: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2, in numpy. Mirrors the torch helper in stgnn.py."""
    a = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    deg = a.sum(1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    return dinv[:, None] * a * dinv[None, :]


def prepare_all_districts(district_dirs, lookback=LOOKBACK, horizon=HORIZON):
    """Numpy-only. Returns a list of per-client dicts (no torch needed)."""
    clients = []
    for dd in district_dirs:
        tensor, adj, scaler = load_district(dd)
        prep = prepare_district(tensor, lookback, horizon, scaler=scaler)
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
# torch layer (lazy import). Builds DataLoaders + per-client adjacency tensors.
# ---------------------------------------------------------------------------

def build_torch_partitions(clients, batch_size=64):
    """Returns (partitions, adjacencies), both index-aligned with `clients`."""
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
