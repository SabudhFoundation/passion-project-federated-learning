# Weekly report - Ayuub

My four tasks:
1. Understand the existing code
2. Deploy the framework in Flower
3. Test it against the metrics
4. Taylor-series KAN

## Progress

Task 1 - done. Read through the Phase 1 pipeline (GAT, GCN, GRU, head, FedAvg
across 4 clients). Notes in HOW_THE_FRAMEWORK_WORKS.md. The KAN model did worse
than the plain FC model in Phase 1, which is what Task 4 looks at.

Task 4 - done. Wrote the Taylor KAN head in src/models/taylor_kan.py. It uses a
Taylor expansion instead of B-splines, so the head drops from ~495K to ~166K
params. Demo: python demo_taylor_kan.py.

Task 2 - Flower client/server written (src/federated/). This week I added the
data pipeline (src/data/) that turns the per-district tensors into training
batches, so it runs end to end on synthetic data. Real run needs the dataset.

Task 3 - metrics code (MAE/RMSE/R2) is in run_federated_local.py. Target is to
reproduce the Phase 1 numbers (MAE 3.70, R2 0.98) once real data is in.

## Next

- get a real district file from Revanth's LargeST output
- run the parity check on real data (Task 3)
- compare FC vs Taylor KAN on real data (Task 4)
