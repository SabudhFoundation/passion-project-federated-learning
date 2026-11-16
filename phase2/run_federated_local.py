"""Run FedAvg across districts and compare the FC, B-spline KAN and Taylor KAN heads."""

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
    p.add_argument("--head", choices=["fc", "taylor", "bspline"], default="fc")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--taylor-order", type=int, default=2)
    p.add_argument("--grid-size", type=int, default=8, help="B-spline head only")
    p.add_argument("--spline-order", type=int, default=3, help="B-spline head only")
    p.add_argument("--kan-hidden", type=int, default=64, help="hidden width of every head")
    p.add_argument("--eval-split", choices=["val", "test"], default="val",
                   help="split scored after the final round (val is scored every round)")
    p.add_argument("--mape-min", type=float, default=1.0,
                   help="MAPE skips targets below this many vehicles (avoids divide by ~0)")
    p.add_argument("--seed", type=int, default=0)
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    partitions, adjacencies, test_loaders = build_torch_partitions(
        clients_np, args.batch_size, with_test=True)
    adjacencies = [(m.to(device), n.to(device)) for m, n in adjacencies]
    n_features = clients_np[0]["n_features"]

    # One global model. Head-independent params are what FedAvg averages.
    def new_model():
        return STGNN(
            n_nodes=clients_np[0]["n_nodes"],   # only used to size head input; N-agnostic otherwise
            in_features=n_features,
            head=args.head,
            taylor_order=args.taylor_order,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            horizon=5,
            kan_hidden=(args.kan_hidden,),
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
        # (windows, sensors, horizon) in vehicle counts
        p = scaler.inverse_transform(np.concatenate(P))
        t = scaler.inverse_transform(np.concatenate(T))
        overall = metrics(p.ravel(), t.ravel())
        steps = []
        for h in range(p.shape[-1]):
            steps.append(metrics(p[..., h].ravel(), t[..., h].ravel()))
        overall["steps"] = steps
        return overall

    def metrics(p, t):
        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        keep = t >= args.mape_min
        mape = float(np.mean(np.abs(err[keep]) / t[keep]) * 100)
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = 0.0
        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    def average(results):
        """Plain average over clients, overall and per forecast step."""
        out = {}
        for k in ("mae", "rmse", "mape", "r2"):
            out[k] = float(np.mean([r[k] for r in results]))
        steps = []
        for h in range(len(results[0]["steps"])):
            step = {}
            for k in ("mae", "rmse", "mape", "r2"):
                step[k] = float(np.mean([r["steps"][h][k] for r in results]))
            steps.append(step)
        out["steps"] = steps
        return out

    def score(model, loaders):
        results = []
        for ci, loader in enumerate(loaders):
            adj_mask, adj_norm = adjacencies[ci]
            results.append(evaluate(model, loader, adj_mask, adj_norm, clients_np[ci]["scaler"]))
        return average(results)

    def quantized_copy(model):
        """Copy with the head's weights rounded to int8 (symmetric, per tensor) and back.
        Simulates sending an 8-bit head; the backbone stays fp32."""
        q = deepcopy(model)
        with torch.no_grad():
            for prm in q.head.parameters():
                scale = prm.abs().max() / 127.0
                if scale > 0:
                    prm.copy_(torch.round(prm / scale).clamp(-127, 127) * scale)
        return q

    global_sd = get_state(global_model)
    history = []
    t_start = time.time()

    for rnd in range(1, args.rounds + 1):
        local_states, sizes = [], []

        for ci, (train_loader, _val) in enumerate(partitions):
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
        val_loaders = [v for _t, v in partitions]
        r = score(global_model, val_loaders)
        mins = (time.time() - t_start) / 60
        print(f"round {rnd:2d} | MAE {r['mae']:7.3f} | RMSE {r['rmse']:7.3f} | "
              f"MAPE {r['mape']:6.2f}% | R2 {r['r2']:+.3f} | {mins:.1f} min", flush=True)
        r["round"] = rnd
        r["minutes"] = round(mins, 2)
        history.append(r)

    # Final scoring: chosen split, full precision and with an int8 head.
    if args.eval_split == "test":
        final_loaders = test_loaders
    else:
        final_loaders = [v for _t, v in partitions]
    final = score(global_model, final_loaders)
    final_int8 = score(quantized_copy(global_model), final_loaders)
    print(f"\nfinal ({args.eval_split}) fp32 | MAE {final['mae']:.3f} | RMSE {final['rmse']:.3f} | "
          f"MAPE {final['mape']:.2f}% | R2 {final['r2']:+.3f}")
    print(f"final ({args.eval_split}) int8 head | MAE {final_int8['mae']:.3f} | RMSE {final_int8['rmse']:.3f} | "
          f"MAPE {final_int8['mape']:.2f}% | R2 {final_int8['r2']:+.3f}")
    for h, s in enumerate(final["steps"]):
        print(f"  {15 * (h + 1):3d} min | MAE {s['mae']:.3f} | RMSE {s['rmse']:.3f} | "
              f"MAPE {s['mape']:.2f}% | R2 {s['r2']:+.3f}")

    # Communication: every client uploads the whole model each round.
    n_total = global_model.n_params()
    n_head = global_model.head_params()
    n_clients = len(partitions)
    comm = {
        "model_bytes_fp32": n_total * 4,
        "head_bytes_fp32": n_head * 4,
        "head_bytes_int8": n_head,
        "upload_bytes_per_round_fp32": n_total * 4 * n_clients,
        "upload_bytes_per_round_int8_head": ((n_total - n_head) * 4 + n_head) * n_clients,
    }
    print(f"comm | model {comm['model_bytes_fp32'] / 1024:.1f} KB fp32 per client | head "
          f"{comm['head_bytes_fp32'] / 1024:.1f} KB fp32, {comm['head_bytes_int8'] / 1024:.1f} KB int8")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "clients": [c["name"] for c in clients_np],
                       "head_params": n_head, "total_params": n_total,
                       "comm": comm, "rounds": history,
                       "final": {"split": args.eval_split, "fp32": final, "int8_head": final_int8}},
                      f, indent=2)
        print(f"saved results to {args.out}")

    print("\nDone. This is the harness that will check parity on the real data:")
    print("target to reproduce from Phase 1 -> MAE 3.70, R2 0.98")

if __name__ == "__main__":
    main()
