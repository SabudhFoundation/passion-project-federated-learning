# Notes on the existing framework (Task 1)

Quick notes from reading the Phase 1 code so I know how the existing pipeline works.

## Model

Per client the model is:

```
GAT   -> graph attention over the road network (each sensor weighs its neighbours)
GCN   -> graph convolution, smooths features across connected sensors
GRU   -> reads the last 12 timesteps, captures how traffic changes over time
LayerNorm
head  -> KAN (GNN-GRU-KAN) or Linear/SiLU/Linear (STGAT+GCN)
-> forecast the next 5 timesteps
```

## Federated loop

Right now it's a manual FedAvg loop in the notebook:

1. server sends the global model to all 4 clients
2. each client trains on its own local data
3. each client sends back its weights (not its data)
4. server averages the 4 weight sets (FedAvg) into a new global model
5. repeat for 100 rounds

## Settings (from the README)

- 4 clients, FedAvg, 100 rounds
- AdamW, lr 1e-3, weight_decay 1e-4
- Huber loss, grad clip 1.0
- weight clamp to [-2, 2] (unusual, probably added to stop divergence, need to ask about this)
- KAN: grid 8, order 3, hidden [128, 128, 64] (this is what makes the KAN head ~495K params)

## Phase 1 results

| Metric | GNN-GRU-KAN | STGAT+GCN |
|---|---|---|
| MAE | 5.35 | 3.70 |
| RMSE | 6.66 | 4.68 |
| R2 | 0.959 | 0.980 |

The KAN model did worse than the plain FC model on every metric. That's the
opposite of the Fed-KAN paper, and it's what Task 4 (Taylor KAN) is aimed at.

## Open questions

- why the [-2, 2] weight clamp
- which cells split the data across clients
- the dataset (on Google Drive, not in the repo)
