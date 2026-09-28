# Phase 2

Federated traffic forecasting, continuing from Phase 1 (FedAvg + STGNN on PeMS).
This folder adds the Flower port, the Taylor KAN head, the data pipeline, and an
async aggregation experiment.

## Run

```bash
pip install numpy
python verify_data_pipeline.py     # checks the data pipeline on synthetic data
python demo_taylor_kan.py          # parameter counts: B-spline KAN vs Taylor KAN

pip install torch
python run_federated_local.py --head fc       # FedAvg across districts
python run_federated_local.py --head taylor    # same, Taylor KAN head
```

## Layout

```
src/
  models/
    kan_params.py     param-count helpers (no torch)
    taylor_kan.py     Taylor KAN layer
    stgnn.py          GAT -> GCN -> GRU -> LayerNorm -> head
  federated/
    client_app.py     Flower client
    server_app.py     Flower server (FedAvg or buffered async)
    strategy_buffered_async.py   buffered/staleness aggregation
    sim_intermittency.py         dropout experiment
  data/
    windowing.py      windows, split, scaler
    district_data.py  load per-district NPZ, build loaders
scripts/make_synthetic_districts.py
demo_taylor_kan.py
run_federated_local.py
verify_data_pipeline.py
```

## Notes

- The model uses dense adjacency instead of torch-geometric, so it runs without
  the PyG build and can be quantized for edge devices later.
- Taylor KAN order 2 uses ~4 params per edge vs ~12 for the B-spline KAN, so the
  head drops from 495K to 166K params. Whether that helps accuracy under FedAvg
  is the thing to test once real data is in.
- Flower has no built-in async, so the buffered/staleness aggregator is a custom
  strategy.
- The parity check: Flower + FedAvg should reproduce the Phase 1 numbers
  (MAE 3.70, R2 0.98) before trusting the port.
