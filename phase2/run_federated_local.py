"""Run FedAvg across districts and compare the FC head vs the Taylor KAN head."""

from __future__ import annotations

import argparse
import json
import time
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
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--out", default="", help="optional path to save round results as json")
    p.add_argument("--resample", type=int, default=1,
                   help="merge every k steps (3 = 15-min data, like Phase 1)")
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap train batches per client per epoch (0 = all)")
    p.add_argument("--max-val-batches", type=int, default=0,
                   help="cap val batches per client when scoring (0 = all)")
    return p.parse_args()

def main():
    args = get_args()

    import torch
    import torch.nn as nn

    from src.data.district_data import build_torch_partitions, prepare_all_districts
    from src.models.stgnn import STGNN

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dirs = sorted(str(p) for p in Path(args.data).glob("district_*"))
    if not dirs:
        raise SystemExit(f"no district_* folders under {args.data} "
                         f"(run scripts/make_synthetic_districts.py first)")

    clients_np = prepare_all_districts(dirs, lazy=True, resample=args.resample)  # lazy keeps memory flat on real data
    partitions, adjacencies = build_torch_partitions(clients_np, args.batch_size)
    adjacencies = [(m.to(device), n.to(device)) for m, n in adjacencies]
    n_features = clients_np[0]["n_features"]

    # One global model. Head-independent params are what FedAvg averages.
    def new_model():
        return STGNN(
            n_nodes=clients_np[0]["n_nodes"],   # only used to size head input; N-agnostic otherwise
            in_features=n_features,
            head=args.head,
            taylor_order=args.taylor_order,
            horizon=5,
        ).to(device)

    global_model = new_model()
    print(f"Head={args.head}  |  head params: {global_model.head_params():,}  "
          f"|  total params: {global_model.n_params():,}")
    print(f"Clients: {len(partitions)}  |  rounds: {args.rounds}  |  "
          f"resample: {args.resample}  |  batch: {args.batch_size}  |  device: {device}\n")

    def get_state(m):
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    def set_state(m, sd):
        m.load_state_dict(sd)

    def evaluate(model, loader, adj_mask, adj_norm, scaler):
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for bi, (xb, yb) in enumerate(loader):
                if args.max_val_batches and bi >= args.max_val_batches:
                    break
                out = model(xb.to(device), adj_mask, adj_norm)
                P.append(out.cpu().numpy())
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
    history = []
    t_start = time.time()

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
                for bi, (xb, yb) in enumerate(train_loader):
                    if args.max_batches and bi >= args.max_batches:
                        break
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    loss = crit(model(xb, adj_mask, adj_norm), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            print(f"  round {rnd} client {ci + 1}/{len(partitions)} done", flush=True)
            local_states.append(get_state(model))
            sizes.append(len(train_loader.dataset))

        # FedAvg: example-weighted average of client weights.
        sizes = np.array(sizes, dtype=float)
        w = sizes / sizes.sum()
        new_sd = {}
        for k in global_sd:
            stacked = torch.stack([ls[k].float() for ls in local_states], dim=0)
            shape = [-1] + [1] * (stacked.dim() - 1)
            weights = torch.tensor(w, dtype=torch.float32, device=stacked.device).reshape(shape)
            new_sd[k] = (stacked * weights).sum(0)
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
        mins = (time.time() - t_start) / 60
        print(f"round {rnd:2d} | MAE {np.mean(maes):7.3f} | "
              f"RMSE {np.mean(rmses):7.3f} | R2 {np.mean(r2s):+.3f} | {mins:.1f} min", flush=True)
        history.append({"round": rnd, "mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
                        "r2": float(np.mean(r2s)), "minutes": round(mins, 2)})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "clients": [c["name"] for c in clients_np],
                       "head_params": global_model.head_params(),
                       "total_params": global_model.n_params(), "rounds": history}, f, indent=2)
        print(f"saved results to {args.out}")

    print("\nDone. This is the harness that will check parity on the real data:")
    print("target to reproduce from Phase 1 -> MAE 3.70, R2 0.98")

if __name__ == "__main__":
    main()
