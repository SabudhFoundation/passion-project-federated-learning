"""Flower server: FedAvg or buffered-async strategy selection."""

from __future__ import annotations

from typing import Dict, List, Tuple

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate client eval metrics weighted by example count."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    keys = [k for k in metrics[0][1] if isinstance(metrics[0][1][k], (int, float))]
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}

def build_server_app(
    num_rounds: int = 100,
    num_clients: int = 4,
    strategy: str = "fedavg",
    buffer_size: int = 2,
    staleness_alpha: float = 0.5,
):
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg

    from .strategy_buffered_async import STALENESS, build_flower_strategy

    def server_fn(context):
        if strategy == "fedavg":
            strat = FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
            )
        elif strategy == "buffered":
            strat = build_flower_strategy(
                mode=STALENESS,
                buffer_size=buffer_size,
                staleness_alpha=staleness_alpha,
                min_available_clients=num_clients,
            )
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")

        return ServerAppComponents(
            strategy=strat,
            config=ServerConfig(num_rounds=num_rounds),
        )

    return ServerApp(server_fn=server_fn)
