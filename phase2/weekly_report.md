# Weekly Report — Ayuub

**Task assigned:** Jul 21, 2026  |  **Due:** Jul 31, 2026  |  **Status:** In Progress

**My four tasks:**
1. Understand the code created by Revanth's team
2. Deploy the existing framework in a Fed environment
3. Test it to get results using the metrics
4. Taylor-Series to modify the existing KAN model in Fed-KAN

---

## Progress this week

### Task 1 — Understand the existing code — DONE
Went through the existing pipeline: GAT → GCN → GRU → LayerNorm → prediction head,
trained federated across 4 clients with a manual FedAvg loop, 100 rounds, on the
PeMS dataset. Wrote up how it works in `HOW_THE_FRAMEWORK_WORKS.md`. Two open
questions logged for Revanth's team (the [-2,2] weight clamp, and the exact data-split cells).

Key finding I flagged: their KAN model **lost** to the plain FC model
(MAE 5.35 vs 3.70). This is the opposite of the Fed-KAN paper's result, and it's
exactly what task 4 is meant to investigate.

### Task 2 — Deploy in a Fed environment (Flower) — CODE WRITTEN, not yet run
Moved the federated logic out of the notebook into a proper Flower app:
- `src/federated/client_app.py` — the client (local training + eval)
- `src/federated/server_app.py` — the server (FedAvg aggregation)
- `src/models/stgnn.py` — the model, rebuilt without PyTorch Geometric so it can
  later export to edge devices
Kept every Phase 1 setting identical (AdamW, Huber, grad clip, the [-2,2] clamp)
so the port is like-for-like. **Not yet executed** — I need the data-loading cells
from Revanth's notebook to plug in the 4-client split.

### Task 3 — Test and get metrics — GATE DEFINED
The acceptance test is fixed: Flower + FedAvg must reproduce the Phase 1 numbers —
**MAE 3.7010, RMSE 4.6809, R² 0.9797**. If they don't match, the port is wrong.
Metric code (MAE / MSE / RMSE / R²) is written in `client_app.py`. Runs once the
data is wired in.

### Task 4 — Taylor-Series KAN — DONE (implementation + param analysis)
Built the Taylor-series KAN head in `src/models/taylor_kan.py`. It replaces the
B-spline basis with a truncated Taylor expansion.
- Per-edge cost drops from 12 coefficients to 4 (order 2)
- Head shrinks from **495,360 → 165,568 parameters (3x smaller)**
- Runnable demo: `python demo_taylor_kan.py`
Still to do: train it on PeMS and compare accuracy against the Phase 1 KAN baseline.

---

## Next week
- Get the dataset + data-split cells from Revanth's team
- Run the Flower deployment and hit the parity gate (task 3)
- Train Taylor-KAN order 1/2/3 on PeMS, compare metrics vs the old KAN (task 4)

## Blockers
- Need the PeMS dataset (currently only on the team Google Drive) and the exact
  notebook cells that split data across the 4 clients.
