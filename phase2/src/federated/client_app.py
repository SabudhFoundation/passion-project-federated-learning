"""
Flower ClientApp  —  Board item 1: "Develop the Fed environment - Flower.AI"

This replaces the hand-rolled FedAvg loop currently living inside
notebooks/Finalized_Model_GRU.ipynb.

What carries over from Phase 1, deliberately
--------------------------------------------
  - AdamW, lr 1e-3, weight_decay 1e-4
  - Huber loss
  - gradient clipping at 1.0
  - parameter clamping to [-2, 2]

The clamp is unusual. It is in the Phase 1 config and it strongly suggests the
manual loop had divergence problems. It is kept here so the Flower port is a
true like-for-like, but it is flagged as `PARAM_CLAMP` below: once Flower parity
is confirmed, try removing it and see whether the divergence was actually a bug
in the manual aggregation rather than a property of the model.

Parity acceptance test
----------------------
Flower + FedAvg must reproduce Phase 1 test metrics within noise:
    STGAT+GCN -> MAE 3.7010, RMSE 4.6809, R^2 0.9797
If it does not, the port is wrong. Do not move on to async or Taylor-KAN until
this passes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.stgnn import STGNN

GRAD_CLIP = 1.0
PARAM_CLAMP = 2.0  # Phase 1 carry-over; revisit after parity is confirmed


# ---------------------------------------------------------------------------
# Parameter <-> ndarray plumbing. Flower speaks lists of numpy arrays.
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [p.cpu().numpy() for p in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    )
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Local training / evaluation.
# ---------------------------------------------------------------------------

def train_one_client(
    model: nn.Module,
    loader,
    adj_mask: torch.Tensor,
    adj_norm: torch.Tensor,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> float:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.HuberLoss()

    total, n_batches = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, adj_mask, adj_norm)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                for p in model.parameters():
                    p.clamp_(-PARAM_CLAMP, PARAM_CLAMP)
            total += loss.item()
            n_batches += 1
    return total / max(1, n_batches)


@torch.no_grad()
def evaluate_one_client(
    model: nn.Module,
    loader,
    adj_mask: torch.Tensor,
    adj_norm: torch.Tensor,
    device: str = "cpu",
) -> Tuple[float, Dict[str, float]]:
    model.to(device).eval()
    crit = nn.HuberLoss()
    preds, trues, total, n = [], [], 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb, adj_mask, adj_norm)
        total += crit(out, yb).item()
        n += 1
        preds.append(out.cpu().numpy())
        trues.append(yb.cpu().numpy())

    p = np.concatenate(preds).ravel()
    t = np.concatenate(trues).ravel()
    mae = float(np.mean(np.abs(p - t)))
    mse = float(np.mean((p - t) ** 2))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return total / max(1, n), {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Flower ClientApp. Lazy import so this module is usable without flwr.
# ---------------------------------------------------------------------------

def build_client_app(partitions, adj_mask, adj_norm, model_cfg: dict, local_epochs: int = 1):
    """partitions: list of (train_loader, val_loader), one per client."""
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context

    class STGNNClient(NumPyClient):
        def __init__(self, cid: int):
            self.cid = cid
            self.model = STGNN(**model_cfg)
            self.train_loader, self.val_loader = partitions[cid]

        def get_parameters(self, config):
            return get_parameters(self.model)

        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            loss = train_one_client(
                self.model,
                self.train_loader,
                adj_mask,
                adj_norm,
                epochs=config.get("local_epochs", local_epochs),
            )
            n = len(self.train_loader.dataset)
            return get_parameters(self.model), n, {"train_loss": loss, "cid": self.cid}

        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            loss, metrics = evaluate_one_client(
                self.model, self.val_loader, adj_mask, adj_norm
            )
            return loss, len(self.val_loader.dataset), metrics

    def client_fn(context: Context):
        cid = int(context.node_config.get("partition-id", 0))
        return STGNNClient(cid).to_client()

    return ClientApp(client_fn=client_fn)
