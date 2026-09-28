"""
End-to-end federated run over districts: FedAvg across clients, FC head vs Taylor-KAN.

This is the script that ties everything together and actually TRAINS. It needs torch,
so it runs on your machine, not in the notebook sandbox. It uses the synthetic
districts by default so it runs today; swap --data for a real LargeST folder when
Revanth's data lands and nothing else changes.

What it does:
  1. Loads every district as a client (data_synth/district_* by default).
  2. Builds one global STGNN model with the chosen head (fc or taylor).
  3. Runs FedAvg: each client trains locally on its own district, the server averages
     the weights, repeat for --rounds rounds. Each client passes its OWN adjacency.
  4. Reports MAE / RMSE / R2 per round on each client's val split, in real units.

Why it matters for the tasks:
  - Task 2 (Flower deployment): proves the client/server/data wiring runs end to end.
  - Task 3 (testing): the same metrics harness that will check parity on real data.
  - Task 4 (Taylor-KAN): `--head taylor` vs `--head fc` is the head-to-head that tests
    whether the smaller Taylor head trains competitively under FedAvg.

Run:
    python run_federated_local.py --head fc     --rounds 15
    python run_federated_local.py --head taylor --rounds 15
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_synth", help="folder of district_* subfolders")
    p.add_argument("--head", choices=["fc", "taylor"], default="fc")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--taylor-order", type=int, default=2)
    return p.parse_args()


def main():
    args = get_args()

    import torch
    import torch.nn as nn

    from src.data.district_data import build_torch_partitions, prepare_all_districts
    from src.models.stgnn import STGNN

    dirs = sorted(str(p) for p in Path(args.data).glob("district_*"))
    if not dirs:
        raise SystemExit(f"no district_* folders under {args.data} "
                         f"(run scripts/make_synthetic_districts.py first)")

    clients_np = prepare_all_districts(dirs)
    partitions, adjacencies = build_torch_partitions(clients_np, args.batch_size)
    n_features = clients_np[0]["n_features"]

    # One global model. Head-independent params are what FedAvg averages.
    def new_model():
        return STGNN(
            n_nodes=clients_np[0]["n_nodes"],   # only used to size head input; N-agnostic otherwise
            in_features=n_features,
            head=args.head,
            taylor_order=args.taylor_order,
            horizon=5,
        )

    global_model = new_model()
    print(f"Head={args.head}  |  head params: {global_model.head_params():,}  "
          f"|  total params: {global_model.n_params():,}")
    print(f"Clients: {len(partitions)}  |  rounds: {args.rounds}\n")

    def get_state(m):
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    def set_state(m, sd):
        m.load_state_dict(sd)

    def evaluate(model, loader, adj_mask, adj_norm, scaler):
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb, adj_mask, adj_norm)
                P.append(out.numpy())
                T.append(yb.numpy())
        p = scaler.inverse_transform(np.concatenate(P).ravel())
        t = scaler.inverse_transform(np.concatenate(T).ravel())
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return mae, rmse, r2

    global_sd = get_state(global_model)

    for rnd in range(1, args.rounds + 1):
        local_states, sizes = [], []

        for ci, (train_loader, _) in enumerate(partitions):
            model = new_model()
            set_state(model, deepcopy(global_sd))
            adj_mask, adj_norm = adjacencies[ci]

            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.HuberLoss()
            model.train()
            for _ in range(args.local_epochs):
                for xb, yb in train_loader:
                    opt.zero_grad()
                    loss = crit(model(xb, adj_mask, adj_norm), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            local_states.append(get_state(model))
            sizes.append(len(train_loader.dataset))

        # FedAvg: example-weighted average of client weights.
        sizes = np.array(sizes, dtype=float)
        w = sizes / sizes.sum()
        new_sd = {}
        for k in global_sd:
            stacked = torch.stack([ls[k].float() for ls in local_states], dim=0)
            shape = [-1] + [1] * (stacked.dim() - 1)
            new_sd[k] = (stacked * torch.tensor(w, dtype=torch.float32).reshape(shape)).sum(0)
            new_sd[k] = new_sd[k].to(global_sd[k].dtype)
        global_sd = new_sd
        set_state(global_model, global_sd)

        # Evaluate the new global model on each client's val split.
        maes, rmses, r2s = [], [], []
        for ci, (_, val_loader) in enumerate(partitions):
            adj_mask, adj_norm = adjacencies[ci]
            mae, rmse, r2 = evaluate(global_model, val_loader, adj_mask, adj_norm,
                                     clients_np[ci]["scaler"])
            maes.append(mae); rmses.append(rmse); r2s.append(r2)
        print(f"round {rnd:2d} | MAE {np.mean(maes):7.3f} | "
              f"RMSE {np.mean(rmses):7.3f} | R2 {np.mean(r2s):+.3f}")

    print("\nDone. This is the harness that will check parity on the real data:")
    print("target to reproduce from Phase 1 -> MAE 3.70, R2 0.98")


if __name__ == "__main__":
    main()
