"""
Generate synthetic districts in Revanth's exact NPZ format.

Purpose: run and test the whole pipeline (windowing -> split -> scaler -> Flower)
BEFORE the real LargeST data lands. The moment a real district folder is available,
point the loader at it instead and nothing else changes.

Writes, per district, into <out>/district_<id>/ :
    flow.npz        key "tensor",    shape (T, N, F)
    adjacency.npz   key "adjacency", shape (N, N)
    info.json       shape summary (mirrors Revanth's info file)

The synthetic traffic has a daily sine rhythm plus neighbour coupling along a random
sparse road graph, so it is non-trivial to forecast and non-IID across districts
(each district gets a different node count, rhythm, and noise level) — like real
regions. Feature channel 0 is flow; channels 1..F-1 are time-of-day / day-of-week
encodings, matching how Revanth's tensor is laid out.

Run:  python scripts/make_synthetic_districts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("data_synth")
N_DISTRICTS = 4
DAYS = 20
STEPS_PER_DAY = 96      # 15-min sampling -> 96 steps/day
# Match Revanth's real tensor exactly (confirmed from his PART 2B.1):
# ch0 flow, ch1 tod_sin, ch2 tod_cos, ch3 dow_sin, ch4 dow_cos, ch5 month
N_FEATURES = 6
RNG = np.random.default_rng(42)


def random_adjacency(n, avg_degree=4):
    """Sparse symmetric 0/1 adjacency, no self loops."""
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        k = RNG.integers(1, avg_degree + 1)
        for j in RNG.choice(n, size=k, replace=False):
            if i != j:
                A[i, j] = A[j, i] = 1.0
    return A


def synth_district(n_nodes, seed):
    rng = np.random.default_rng(seed)
    T = DAYS * STEPS_PER_DAY
    t = np.arange(T)

    tod = (t % STEPS_PER_DAY) / STEPS_PER_DAY          # time of day in [0,1)
    dow = ((t // STEPS_PER_DAY) % 7) / 6.0             # day of week in [0,1]

    # Base daily rhythm, per node, with a node-specific phase + amplitude.
    phase = rng.uniform(0, 2 * np.pi, size=n_nodes)
    amp = rng.uniform(20, 60, size=n_nodes)
    base = 60 + amp[None, :] * np.sin(2 * np.pi * tod[:, None] + phase[None, :])

    # Neighbour coupling: smooth flow along the road graph.
    A = random_adjacency(n_nodes)
    deg = A.sum(1, keepdims=True)
    deg[deg == 0] = 1
    flow = base.copy()
    for _ in range(2):
        flow = 0.7 * flow + 0.3 * (flow @ A.T) / deg.T
    flow += rng.normal(scale=rng.uniform(2, 6), size=flow.shape)  # district-specific noise
    flow = np.clip(flow, 0, None)

    # Assemble (T, N, 6) matching Revanth's exact channel order:
    # ch0 flow, ch1 tod_sin, ch2 tod_cos, ch3 dow_sin, ch4 dow_cos, ch5 month
    dow7 = ((t // STEPS_PER_DAY) % 7) / 7.0
    month = (((t // (STEPS_PER_DAY * 30)) % 12) + 1) / 12.0  # rough month proxy
    tensor = np.empty((T, n_nodes, N_FEATURES), dtype=np.float32)
    tensor[:, :, 0] = flow
    tensor[:, :, 1] = np.sin(2 * np.pi * tod)[:, None]
    tensor[:, :, 2] = np.cos(2 * np.pi * tod)[:, None]
    tensor[:, :, 3] = np.sin(2 * np.pi * dow7)[:, None]
    tensor[:, :, 4] = np.cos(2 * np.pi * dow7)[:, None]
    tensor[:, :, 5] = month[:, None]
    return tensor, A


def main():
    OUT.mkdir(exist_ok=True)
    node_counts = RNG.integers(15, 35, size=N_DISTRICTS)  # non-uniform, like real districts

    for d in range(N_DISTRICTS):
        n = int(node_counts[d])
        tensor, adj = synth_district(n, seed=100 + d)

        dd = OUT / f"district_{d}"
        dd.mkdir(exist_ok=True)
        np.savez_compressed(dd / "flow.npz", tensor=tensor)
        np.savez_compressed(dd / "adjacency.npz", adjacency=adj)
        info = {
            "district": d,
            "timesteps": int(tensor.shape[0]),
            "nodes": int(tensor.shape[1]),
            "features": int(tensor.shape[2]),
            "synthetic": True,
        }
        (dd / "info.json").write_text(json.dumps(info, indent=2))
        print(f"wrote {dd}  tensor={tensor.shape}  adj={adj.shape}")

    print(f"\n{N_DISTRICTS} synthetic districts written under {OUT}/")
    print("Point the loader at these to run the pipeline before real data arrives.")


if __name__ == "__main__":
    main()
