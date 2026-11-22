"""Traffic-aware aggregation weights, after Kaur, Sehra et al. (MLWA 2026), Sec. 3.5.

Each client k is described by six metrics (Eq. 6):
    z_k = (V_k, mu_k, sigma_k, R_k, Phi_k, Psi_k)
The server min-max normalises every metric across clients (Eq. 13), scores each
client with a weighted sum of the normalised metrics (Eq. 14) and divides by the
total so the weights sum to 1 (Algorithm 1, step 11).

Substitutions for our data (LargeST has flow but no speed column):
  Phi_k  paper: mean speed.         here: mean |c_{t+1} - c_t|, the mean absolute
                                     step-to-step change in flow.
  Psi_k  paper: std of speed.       here: std of (c_{t+1} - c_t), the std of the
                                     step-to-step change in flow.
  R_k    paper: t_now - t_max,k.    here: in time steps, with t_now the latest
                                     training step over all clients. Every district
                                     covers the same period, so R_k = 0 for all k and
                                     the normalised value is 0 (the term drops out).
The paper inverts R during normalisation so fresher clients score higher. We do
that by normalising -R, i.e. (max R - R_k) / (max R - min R + eps).

Two additions that are not in the paper:
  * If every score is 0 (all clients identical, so every normalised metric is 0)
    the traffic weights fall back to uniform instead of dividing by zero.
  * A floor mixes in FedAvg weights so no client is zeroed out:
        w_k = (1 - floor) * traffic_k + floor * fedavg_k
    floor = 1 gives FedAvg exactly; floor = 0 gives the paper's weights.

All metrics come from the training split only.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8

# Column order of the metric matrix Z, as in Eq. 6.
METRICS = ("V", "mu", "sigma", "R", "phi", "psi")
V, MU, SIGMA, R, PHI, PSI = range(6)

# Eq. 14 pairs the coefficients (alpha, beta, gamma, delta, epsilon, eta) with
# (V, sigma, mu, R, Phi, Psi). Note beta goes with sigma and gamma with mu.
COEF_COLUMNS = (V, SIGMA, MU, R, PHI, PSI)


def client_metrics(train, scaler):
    """Raw metrics for one client from its training split (a LazyWindows).

    Flow is converted back to vehicle counts first, since each district has its
    own scaler and the metrics have to be comparable across clients.
    Returns V, mu, sigma, phi, psi and t_max (last training step index); R needs
    every client's t_max, so metric_matrix computes it.
    """
    if not hasattr(train, "tensor"):
        raise TypeError("client_metrics expects the LazyWindows training split (lazy=True)")
    flow = scaler.inverse_transform(
        np.asarray(train.tensor[:, :, train.target_channel], dtype=np.float64))  # (T_train, N)
    step = np.diff(flow, axis=0)
    return {
        "V": float(len(train)),
        "mu": float(flow.mean()),
        "sigma": float(flow.std()),
        "t_max": int(flow.shape[0] - 1),
        "phi": float(np.abs(step).mean()),
        "psi": float(step.std()),
    }


def metric_matrix(metrics):
    """Stack per-client metric dicts into Z (K, 6), filling in R = t_now - t_max."""
    t_now = max(m["t_max"] for m in metrics)
    return np.array([[m["V"], m["mu"], m["sigma"], t_now - m["t_max"], m["phi"], m["psi"]]
                     for m in metrics], dtype=np.float64)


def minmax(Z, eps=EPS):
    """Eq. 13: (z - min_u z) / (max_u z - min_u z + eps), column by column."""
    Z = np.asarray(Z, dtype=np.float64)
    lo = Z.min(axis=0)
    hi = Z.max(axis=0)
    return (Z - lo) / (hi - lo + eps)


def aggregation_weights(Z, sizes, coef=(1.0,) * 6, floor=0.1, eps=EPS):
    """Final per-client weights plus the pieces they were built from.

    Z      (K, 6) raw metrics in METRICS order
    sizes  training examples per client (what FedAvg weights by)
    coef   (alpha, beta, gamma, delta, epsilon, eta) of Eq. 14, all >= 0
    floor  share of the FedAvg weights mixed in, in [0, 1]
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] != 6:
        raise ValueError(f"Z must be (K, 6), got {Z.shape}")
    coef = np.asarray(coef, dtype=np.float64)
    if coef.shape != (6,) or (coef < 0).any():
        raise ValueError("coef needs 6 non-negative values")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")

    sizes = np.array(sizes, dtype=float)
    fedavg = sizes / sizes.sum()   # same expression as the FedAvg step in the runner

    signed = Z.copy()
    signed[:, R] = -signed[:, R]   # invert recency: fresher data -> higher value
    Zn = minmax(signed, eps)

    score = Zn[:, list(COEF_COLUMNS)] @ coef                      # Eq. 14
    total = score.sum()
    if total > 0:
        traffic = score / total                                    # Algorithm 1, step 11
    else:
        traffic = np.full(len(Z), 1.0 / len(Z))

    weights = (1.0 - floor) * traffic + floor * fedavg
    return weights, {"normalized": Zn, "score": score, "traffic": traffic, "fedavg": fedavg}


def parse_coef(text):
    """'a,b,g,d,e,h' -> list of 6 floats (for argparse)."""
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 6 or any(v < 0 for v in vals):
        raise ValueError("--agg-coef needs 6 non-negative comma-separated numbers")
    return vals
