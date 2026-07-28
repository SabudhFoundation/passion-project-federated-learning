# Task 1 — How Revanth's team's framework works

My notes from reading the existing code + README. Written so I can explain it in
the meeting without hand-waving. This is the "understand the code" deliverable.

## The pipeline, end to end

```
PeMS traffic data (435 sensors, 15-min samples)
        │
        │  split across 4 clients — each client holds its own sensors' data,
        │  data never leaves the client (this is the "privacy-preserving" claim)
        ▼
For each client, the model is:
        GAT   — graph attention: each sensor looks at its road-network neighbours
         │      and weighs which neighbours matter for its own prediction
        GCN   — graph convolution: smooths features across connected sensors
         │
        GRU   — the temporal part: reads the last 12 timesteps in order and
         │      builds a hidden state summarising "what traffic has been doing"
        LayerNorm
         │
        Prediction head  ── one of two:
             • KAN  (GNN-GRU-KAN model)     ← this is the one I modify in task 4
             • Linear→SiLU→Linear (STGAT+GCN model) ← the Phase 1 winner
        ▼
        Forecast: next 5 timesteps of traffic
```

## The federated loop (this is the part task 2 replaces)

Right now it's a **manual FedAvg loop inside the notebook**, not a framework.
The logic is:

1. Server starts with one global model, copies it to all 4 clients.
2. Each client trains on **its own** data for a bit (local training).
3. Each client sends its **updated weights** back — not its data.
4. Server **averages** the 4 sets of weights (this is FedAvg) → new global model.
5. Repeat for 100 communication rounds.

That's the whole idea of federated learning: the data stays put, only the model
weights travel, and averaging the weights gives you a model that learned from
everyone's data without anyone sharing it.

## The exact settings (from the README)

| Thing | Value | Why it matters |
|---|---|---|
| Clients | 4 | matches the 4 boxes on the professor's diagram |
| Aggregation | FedAvg | the plain "average the weights" method |
| Rounds | 100 | how many times they sync |
| Optimizer | AdamW, lr 1e-3 | standard |
| Loss | Huber | robust to outlier traffic spikes |
| Grad clip | 1.0 | stops exploding gradients |
| **Param clamp [-2, 2]** | **unusual** | **every weight forced into ±2 — a sign training was unstable. Ask Revanth's team why.** |
| KAN config | grid 8, order 3, hidden [128,128,64] | this is what makes the KAN head huge — 495K params |

## The result they got (the baseline I have to preserve)

| Metric | GNN-GRU-KAN | STGAT+GCN |
|---|---|---|
| MAE | 5.35 | **3.70** |
| RMSE | 6.66 | **4.68** |
| R² | 0.959 | **0.980** |

**The KAN model LOST to the plain FC model on everything.** That's the key fact
for my task 4 — and it's the opposite of what the Fed-KAN paper found. My job is
to shrink the KAN with a Taylor-series head and see if that flips the result.

## What I still need from Revanth's team

- Why the [-2, 2] weight clamp? (Suggests they hit divergence.)
- Which exact cells load the data and split it across the 4 clients — I need those
  to feed my Flower version.
- The dataset itself (it's on their Google Drive, not in the repo).
