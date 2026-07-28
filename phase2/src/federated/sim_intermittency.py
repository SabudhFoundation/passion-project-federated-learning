"""
Intermittent-connectivity simulation: FedAvg vs buffered async vs staleness-weighted.

Board item 4. This is the experiment that produces the plot.

Method
------
IoT roadside units drop off the network. We do not need real flaky hardware to
study that - we need a controlled availability trace. Each client is available
in each round with probability p, independently. We sweep p from 1.0 down to
0.25 and measure how each aggregation mode degrades.

  SYNC (FedAvg, Phase 1 behaviour)
      The server waits for every sampled client. If any client is offline the
      round cannot close, so the round is lost. This is the cost of synchrony
      under intermittency and it compounds fast: with 4 clients at p=0.7, all
      four are present only 0.7^4 = 24% of the time.

  BUFFERED (FedBuff-style)
      Aggregate as soon as K of N updates arrive. Offline clients no longer
      block progress; their updates land later and roll into a later
      aggregation.

  STALENESS
      As BUFFERED, but an update computed against global version v and applied
      at version v' is discounted by (1 + v' - v)^(-alpha), so stale gradients
      do not drag the global model backwards.

Task
----
Deliberately a small federated ridge-regression problem, not the STGNN. The
question here is "does the aggregation scheme survive dropout", which is a
property of the aggregator, not of the model. Keeping the task tiny means this
runs in seconds on a laptop, has no dataset dependency, and isolates the one
variable we care about. The same aggregator object is what plugs into Flower
for the real STGNN runs.

Non-IID split: each client's feature distribution is shifted and rescaled, so
client optima genuinely disagree and naive aggregation can hurt.

Run:  python -m src.federated.sim_intermittency
"""

from __future__ import annotations

import numpy as np

from .strategy_buffered_async import (
    BUFFERED,
    STALENESS,
    SYNC,
    BufferedAsyncAggregator,
    ClientUpdate,
)

RNG = np.random.default_rng(0)

N_CLIENTS = 4
N_FEATURES = 20
N_PER_CLIENT = 400
N_TEST = 1000
N_ROUNDS = 120
LOCAL_STEPS = 5
LOCAL_LR = 0.05


def make_federation(seed: int = 0):
    """Non-IID clients sharing one ground-truth mapping."""
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=N_FEATURES)

    clients = []
    for c in range(N_CLIENTS):
        shift = rng.normal(scale=0.8, size=N_FEATURES)
        scale = rng.uniform(0.5, 1.5)
        X = rng.normal(size=(N_PER_CLIENT, N_FEATURES)) * scale + shift
        y = X @ w_true + rng.normal(scale=0.1, size=N_PER_CLIENT)
        clients.append((X, y))

    Xt = rng.normal(size=(N_TEST, N_FEATURES))
    yt = Xt @ w_true + rng.normal(scale=0.1, size=N_TEST)
    return clients, (Xt, yt), w_true


def local_train(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A few SGD steps on local data — the analogue of client.fit()."""
    w = w.copy()
    n = len(y)
    for _ in range(LOCAL_STEPS):
        idx = RNG.choice(n, size=min(64, n), replace=False)
        Xb, yb = X[idx], y[idx]
        grad = 2.0 * Xb.T @ (Xb @ w - yb) / len(yb)
        w -= LOCAL_LR * grad / (1.0 + np.linalg.norm(grad))  # normalised step
    return w


def test_mse(w: np.ndarray, test) -> float:
    Xt, yt = test
    return float(np.mean((Xt @ w - yt) ** 2))


def run(mode: str, availability: float, seed: int = 0) -> dict:
    """One full federated run under a given availability probability."""
    clients, test, _ = make_federation(seed)
    rng = np.random.default_rng(seed + 1000)

    agg = BufferedAsyncAggregator(
        mode=mode,
        buffer_size=2 if mode != SYNC else N_CLIENTS,
        staleness_alpha=0.5,
    )
    agg._sync_threshold = N_CLIENTS
    agg.initialise([np.zeros(N_FEATURES)])

    # Each client remembers which global version it last pulled.
    client_version = [0] * N_CLIENTS
    client_params = [agg.get_global()[0].copy() for _ in range(N_CLIENTS)]

    lost_rounds = 0
    history = []

    for rnd in range(N_ROUNDS):
        online = [c for c in range(N_CLIENTS) if rng.random() < availability]

        if mode == SYNC:
            # FedAvg semantics: the round only closes if everyone reports.
            if len(online) < N_CLIENTS:
                lost_rounds += 1
                history.append(test_mse(agg.get_global()[0], test))
                continue

        for c in online:
            base = agg.version
            w = agg.get_global()[0]                     # pull current global
            client_params[c] = local_train(w, *clients[c])
            client_version[c] = base
            agg.submit(
                ClientUpdate(
                    client_id=c,
                    parameters=[client_params[c]],
                    num_examples=N_PER_CLIENT,
                    base_version=base,
                )
            )

        history.append(test_mse(agg.get_global()[0], test))

    agg.flush()
    final = test_mse(agg.get_global()[0], test)

    s = agg.summary()
    s.update(
        {
            "availability": availability,
            "final_mse": final,
            "lost_rounds": lost_rounds,
            "history": history,
        }
    )
    return s


def sweep(seeds=(0, 1, 2)):
    availabilities = [1.0, 0.9, 0.75, 0.5, 0.25]
    modes = [SYNC, BUFFERED, STALENESS]
    results = {}

    for mode in modes:
        for p in availabilities:
            runs = [run(mode, p, seed=s) for s in seeds]
            results[(mode, p)] = {
                "final_mse": float(np.mean([r["final_mse"] for r in runs])),
                "std": float(np.std([r["final_mse"] for r in runs])),
                "aggregations": float(np.mean([r["aggregations"] for r in runs])),
                "lost_rounds": float(np.mean([r["lost_rounds"] for r in runs])),
                "mean_staleness": float(np.mean([r["mean_staleness"] for r in runs])),
            }
    return availabilities, modes, results


LABEL = {
    SYNC: "FedAvg (sync)",
    BUFFERED: "Buffered async",
    STALENESS: "Buffered + staleness",
}


def main():
    availabilities, modes, res = sweep()

    print()
    print("=" * 78)
    print("INTERMITTENT CONNECTIVITY SWEEP".center(78))
    print(f"{N_CLIENTS} clients, {N_ROUNDS} rounds, non-IID, mean of 3 seeds".center(78))
    print("=" * 78)
    print()
    print("Final test MSE (lower is better)")
    print()
    header = f"{'Client availability':<24}" + "".join(f"{int(p*100):>10}%" for p in availabilities)
    print(header)
    print("-" * len(header))
    for mode in modes:
        row = f"{LABEL[mode]:<24}"
        for p in availabilities:
            row += f"{res[(mode, p)]['final_mse']:>11.4f}"
        print(row)

    print()
    print("Successful global aggregations")
    print()
    print(header)
    print("-" * len(header))
    for mode in modes:
        row = f"{LABEL[mode]:<24}"
        for p in availabilities:
            row += f"{res[(mode, p)]['aggregations']:>11.0f}"
        print(row)

    print()
    print("Rounds lost to stragglers (sync only — async never stalls)")
    print()
    print(header)
    print("-" * len(header))
    for mode in modes:
        row = f"{LABEL[mode]:<24}"
        for p in availabilities:
            row += f"{res[(mode, p)]['lost_rounds']:>11.0f}"
        print(row)

    # Headline number for the meeting.
    p_bad = 0.5
    sync_bad = res[(SYNC, p_bad)]["final_mse"]
    stale_bad = res[(STALENESS, p_bad)]["final_mse"]
    print()
    print("=" * 78)
    print(f"At {int(p_bad*100)}% client availability:")
    print(f"  FedAvg (sync)        final MSE = {sync_bad:.4f}")
    print(f"  Buffered + staleness final MSE = {stale_bad:.4f}")
    if sync_bad > 0:
        print(f"  -> {(1 - stale_bad / sync_bad) * 100:.1f}% lower error from async aggregation")
    print("=" * 78)
    print()
    print("READ THE CAVEATS BEFORE QUOTING THE HEADLINE NUMBER:")
    print()
    print("  1. The sync baseline is modelled strictly: a round is LOST if any")
    print("     sampled client is offline. That is textbook FedAvg, and it is why")
    print("     the gap looks enormous. A sync implementation that tolerates")
    print("     partial participation would degrade far more gracefully. The")
    print("     honest claim is 'strict-participation FedAvg collapses under")
    print("     dropout', not 'FedAvg is 99% worse'.")
    print()
    buf_bad = res[(BUFFERED, p_bad)]["final_mse"]
    print(f"  2. Staleness weighting bought us nothing here ({stale_bad:.4f} vs")
    print(f"     {buf_bad:.4f} for plain buffered). Expected: this task is convex")
    print("     and the clients agree closely enough that stale updates are not")
    print("     harmful. Staleness discounting should start to matter on the real")
    print("     non-IID PeMS split with the STGNN. Report it as untested-benefit,")
    print("     not as a win.")
    print()
    print("  3. This is ridge regression, not the STGNN. It isolates the")
    print("     aggregator. It does not tell us the STGNN will behave the same.")
    print("=" * 78)
    print()

    try:
        _plot(availabilities, modes, res)
    except Exception as exc:  # matplotlib absent is not a failure
        print(f"(plot skipped: {exc})")

    return res


def _plot(availabilities, modes, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [p * 100 for p in availabilities]

    # Buffered and staleness overlap almost exactly on this task, so give them
    # distinct linestyles/widths or one hides the other completely.
    style = {
        SYNC: dict(marker="o", ls="-", lw=2.0),
        BUFFERED: dict(marker="s", ls="-", lw=3.5, alpha=0.5),
        STALENESS: dict(marker="^", ls="--", lw=2.0),
    }

    for mode in modes:
        ax1.plot(x, [res[(mode, p)]["final_mse"] for p in availabilities],
                 label=LABEL[mode], **style[mode])
        ax2.plot(x, [res[(mode, p)]["aggregations"] for p in availabilities],
                 label=LABEL[mode], **style[mode])

    ax1.set_xlabel("Client availability (%)")
    ax1.set_ylabel("Final test MSE (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Accuracy under intermittent connectivity")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.set_xlabel("Client availability (%)")
    ax2.set_ylabel("Successful global aggregations")
    ax2.set_title("Training progress under intermittent connectivity")
    ax2.invert_xaxis()
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = "results/intermittency.png"
    import os
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Plot written to {out}")


if __name__ == "__main__":
    main()
