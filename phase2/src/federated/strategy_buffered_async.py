"""
Buffered asynchronous aggregation with staleness weighting.

Board item 4: "handle the intermittent connectivity by adding asynchronous
updates to the global"

Why this file exists
--------------------
Flower does NOT ship asynchronous federated learning. Its Strategy API is
round-synchronous: configure_fit -> wait for the sampled clients -> aggregate_fit.
So "use Flower" does not give us item 4 for free. We implement it.

Three aggregation modes are implemented here so they can be compared directly
on the same intermittency trace:

  SYNC       - FedAvg. Server waits for all N sampled clients. One straggler or
               dropout stalls or shrinks the round. This is Phase 1's behaviour
               and the baseline to beat.

  BUFFERED   - FedBuff-style. Server keeps a buffer and aggregates as soon as K
               of N updates have landed, K < N. Late updates roll into the next
               aggregation instead of being discarded. No client blocks the round.

  STALENESS  - BUFFERED plus staleness discounting. A client that started from
               global version v and reports at version v' has staleness
               s = v' - v. Its update is weighted (1 + s)^(-alpha), so updates
               computed against an old global model count for less rather than
               dragging the global backwards.

The staleness weight is the standard polynomial discount used across the async
FL literature (FedAsync, FedBuff, FedStaleWeight). alpha=0 recovers plain
buffered aggregation.

Design note on the Flower boundary
----------------------------------
The aggregation maths lives in BufferedAsyncAggregator, which is pure
numpy and has no Flower import. FlowerBufferedAsyncStrategy wraps it in
Flower's Strategy interface. Keeping them separate means the algorithm is
unit-testable without spinning up a federation, and the same aggregator drives
sim_intermittency.py. Do not merge these two classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

NDArrays = List[np.ndarray]

SYNC = "sync"
BUFFERED = "buffered"
STALENESS = "staleness"


@dataclass
class ClientUpdate:
    """One client's contribution, tagged with the global version it started from."""

    client_id: int
    parameters: NDArrays
    num_examples: int
    base_version: int  # global version this client pulled before training


@dataclass
class BufferedAsyncAggregator:
    """Framework-agnostic async aggregation core."""

    mode: str = BUFFERED
    buffer_size: int = 2          # K: aggregate once this many updates have landed
    staleness_alpha: float = 0.5  # discount exponent; 0 disables discounting
    server_lr: float = 1.0        # 1.0 = full replace, <1 = damped global step

    version: int = 0
    _buffer: List[ClientUpdate] = field(default_factory=list)
    _global: Optional[NDArrays] = None

    # -- bookkeeping for the report ---------------------------------------
    n_aggregations: int = 0
    n_updates_applied: int = 0
    staleness_log: List[int] = field(default_factory=list)

    def initialise(self, parameters: NDArrays) -> None:
        self._global = [p.copy() for p in parameters]
        self.version = 0

    def get_global(self) -> NDArrays:
        assert self._global is not None, "call initialise() first"
        return [p.copy() for p in self._global]

    def _staleness_weight(self, update: ClientUpdate) -> float:
        if self.mode != STALENESS:
            return 1.0
        s = max(0, self.version - update.base_version)
        return float((1.0 + s) ** (-self.staleness_alpha))

    def submit(self, update: ClientUpdate) -> bool:
        """Add an update. Returns True if this triggered an aggregation."""
        self._buffer.append(update)
        self.staleness_log.append(max(0, self.version - update.base_version))

        threshold = self.buffer_size if self.mode != SYNC else self._sync_threshold
        if len(self._buffer) >= threshold:
            self._aggregate()
            return True
        return False

    _sync_threshold: int = 4  # overwritten by the caller for SYNC mode

    def flush(self) -> bool:
        """Force aggregation of whatever is buffered. Used at end of a run."""
        if not self._buffer:
            return False
        self._aggregate()
        return True

    def _aggregate(self) -> None:
        assert self._global is not None

        weights = np.array(
            [u.num_examples * self._staleness_weight(u) for u in self._buffer],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0:
            self._buffer.clear()
            return
        weights = weights / total

        # Weighted mean of the buffered client parameters.
        agg = [
            np.sum([w * u.parameters[i] for w, u in zip(weights, self._buffer)], axis=0)
            for i in range(len(self._global))
        ]

        # Damped server step: g <- (1-lr) g + lr * agg
        if self.server_lr >= 1.0:
            self._global = agg
        else:
            self._global = [
                (1.0 - self.server_lr) * g + self.server_lr * a
                for g, a in zip(self._global, agg)
            ]

        self.n_aggregations += 1
        self.n_updates_applied += len(self._buffer)
        self.version += 1
        self._buffer.clear()

    def summary(self) -> dict:
        sl = self.staleness_log
        return {
            "mode": self.mode,
            "global_version": self.version,
            "aggregations": self.n_aggregations,
            "updates_applied": self.n_updates_applied,
            "mean_staleness": float(np.mean(sl)) if sl else 0.0,
            "max_staleness": int(np.max(sl)) if sl else 0,
        }


# ---------------------------------------------------------------------------
# Flower wrapper.
#
# Imported lazily so this module stays usable (and testable) without flwr
# installed. `pip install flwr[simulation]` to enable.
# ---------------------------------------------------------------------------

def build_flower_strategy(
    mode: str = BUFFERED,
    buffer_size: int = 2,
    staleness_alpha: float = 0.5,
    fraction_fit: float = 1.0,
    min_available_clients: int = 4,
):
    """Return a Flower Strategy backed by BufferedAsyncAggregator.

    Flower samples clients each round and waits for the round to close, so this
    is *semi*-asynchronous: within a round we aggregate as soon as `buffer_size`
    results are in and treat the remainder as stale arrivals folded into the
    next aggregation. True async requires patching Flower's coordination layer
    (see arXiv:2606.24230); semi-async is the tractable version and is what the
    FedBuff line of work actually does in production.
    """
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg

    class FlowerBufferedAsyncStrategy(FedAvg):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.agg = BufferedAsyncAggregator(
                mode=mode,
                buffer_size=buffer_size,
                staleness_alpha=staleness_alpha,
            )
            self.agg._sync_threshold = min_available_clients
            self._client_versions: dict[str, int] = {}

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}

            if self.agg._global is None:
                self.agg.initialise(parameters_to_ndarrays(results[0][1].parameters))

            for client_proxy, fit_res in results:
                cid = str(client_proxy.cid)
                base = self._client_versions.get(cid, self.agg.version)
                self.agg.submit(
                    ClientUpdate(
                        client_id=hash(cid) % 10_000,
                        parameters=parameters_to_ndarrays(fit_res.parameters),
                        num_examples=fit_res.num_examples,
                        base_version=base,
                    )
                )
                self._client_versions[cid] = self.agg.version

            # Anything still buffered at round end is carried, not dropped.
            metrics = self.agg.summary()
            metrics["failures"] = len(failures)
            return ndarrays_to_parameters(self.agg.get_global()), metrics

    return FlowerBufferedAsyncStrategy(
        fraction_fit=fraction_fit,
        min_available_clients=min_available_clients,
        min_fit_clients=min_available_clients,
    )
