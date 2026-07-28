# Phase 2 — Flower environment, Taylor-KAN, async aggregation

Work-in-progress scaffold addressing the four items on Dr. Sehra's Miro board.
Phase 1 (FedAvg + STGAT+GCN / GNN-GRU-KAN on PeMS) is unchanged and still lives
in `notebooks/`.

## Two things you can run right now

Neither needs the PeMS dataset. Both need only Python + numpy (+ matplotlib for the plot).

```bash
python demo_taylor_kan.py                    # board item 3 — parameter budget
python -m src.federated.sim_intermittency    # board item 4 — async vs FedAvg under dropout
```

### Result 1 — Taylor-KAN parameter budget

Head layers `[128, 128, 128, 64, 5]`, the Phase 1 KAN configuration.

| Head | Params | int8 | vs B-spline |
|---|---:|---:|---|
| B-spline KAN (Phase 1) | 495,360 | 484 KB | baseline |
| Taylor-KAN (order 1) | 124,288 | 121 KB | 3.99x smaller |
| **Taylor-KAN (order 2)** | **165,568** | **162 KB** | **2.99x smaller** |
| Taylor-KAN (order 3) | 206,848 | 202 KB | 2.39x smaller |
| FC head (Phase 1 winner) | 17,157 | 17 KB | 28.87x smaller |

Per-edge cost drops from `grid_size + spline_order + base = 12` to `(p+1) + base = 4` at order 2.

Across 4 clients x 100 rounds that is ~503 MB less uplink traffic.

**Honest caveat.** Taylor-KAN removes ~2/3 of the KAN head's parameters but is
still ~9.7x heavier than the FC head that beat it in Phase 1. This is not a
guaranteed win — it is a testable hypothesis. Shrinking the KAN hidden layers is
the other untested lever.

### Result 2 — async aggregation under intermittent connectivity

4 clients, 120 rounds, non-IID, mean of 3 seeds. Final test MSE:

| Client availability | 100% | 90% | 75% | 50% | 25% |
|---|---:|---:|---:|---:|---:|
| FedAvg (sync) | 0.0100 | 0.0100 | 0.0100 | 5.2745 | 11.9713 |
| Buffered async | 0.0101 | 0.0100 | 0.0101 | 0.0101 | 0.0100 |
| Buffered + staleness | 0.0101 | 0.0100 | 0.0101 | 0.0102 | 0.0100 |

Successful global aggregations:

| Client availability | 100% | 90% | 75% | 50% | 25% |
|---|---:|---:|---:|---:|---:|
| FedAvg (sync) | 120 | 78 | 39 | 8 | 1 |
| Buffered async | 240 | 217 | 182 | 125 | 61 |

At 50% availability FedAvg completes 8 of 120 rounds. Buffered async completes 125.

**Honest caveats — read before quoting the headline.**

1. The sync baseline is modelled strictly: a round is lost if *any* sampled
   client is offline. That is textbook FedAvg and it is why the gap looks
   enormous. A partial-participation sync implementation would degrade far more
   gracefully. The defensible claim is "strict-participation FedAvg collapses
   under dropout," not "FedAvg is 99% worse."
2. **Staleness weighting bought nothing here** (0.0102 vs 0.0101). The task is
   convex and clients agree closely, so stale updates aren't harmful. It should
   start to matter on the real non-IID PeMS split. Report as untested benefit.
3. This is ridge regression, not the STGNN. It isolates the aggregator only.

## Layout

```
src/
  models/
    kan_params.py               analytic param counts (torch-free, so the
                                comparison runs on any machine)
    taylor_kan.py               TaylorKANLayer / TaylorKAN  — board item 3
    stgnn.py                    GAT -> GCN -> GRU -> LN -> head, dense adjacency
  federated/
    client_app.py               Flower ClientApp            — board item 1
    server_app.py               Flower ServerApp + strategy selection
    strategy_buffered_async.py  buffered + staleness aggregation — board item 4
    sim_intermittency.py        the dropout experiment
demo_taylor_kan.py
results/intermittency.png
```

## Design decisions worth defending

**Dense adjacency instead of PyTorch Geometric in `stgnn.py`.** PyG's
`GATConv`/`GCNConv` use scatter/gather ops with no TFLite Micro equivalent.
Board item 2b wants TinyML. Writing the graph ops as dense matmuls now means the
quantization work later is a conversion rather than a rewrite. At 435 sensors a
dense 435x435 adjacency is ~190K floats — fine. Revisit past a few thousand nodes.

**The aggregator has no Flower import.** `BufferedAsyncAggregator` is pure numpy;
`build_flower_strategy` wraps it. This makes the algorithm unit-testable without
spinning up a federation, and lets the same object drive `sim_intermittency.py`.
Keep them separate.

**Phase 1's parameter clamp to [-2, 2] is carried over verbatim** in
`client_app.py`, flagged as `PARAM_CLAMP`. It's an unusual choice that suggests
the manual FedAvg loop had divergence problems. Kept for like-for-like parity;
worth removing once parity is confirmed, to check whether the divergence was
actually a bug in the manual aggregation.

## The parity gate

`src/federated/server_app.py` defaults to `strategy="fedavg"`. **Do not build on
the async path until stock Flower + FedAvg reproduces Phase 1:**

```
STGAT+GCN   MAE 3.7010   RMSE 4.6809   R² 0.9797
```

If the numbers drift, the bug is in the port — most likely `state_dict` ↔ ndarray
ordering, or the clamp being applied at a different point — not in Flower.

## Known gaps

- `src/data/` is empty. The PeMS loader and the 4-way non-IID partitioner still
  need extracting from `notebooks/Finalized_Model_GRU.ipynb`.
- The Flower apps are written against the modern `ClientApp`/`ServerApp` API but
  have **not been executed** — `flwr` and `torch` were not installable in the
  environment this was drafted in. Treat as reviewed-not-run.
- No TinyML export yet. See the roadmap on why the GNN half is the hard part.

## Install

```bash
pip install torch numpy matplotlib
pip install "flwr[simulation]"      # only needed for the Flower apps
```
