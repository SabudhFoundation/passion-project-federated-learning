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
python run_federated_local.py --head taylor --agg traffic   # traffic-aware aggregation

python -m unittest discover -s tests -v        # aggregation weight tests (numpy only)
```

Traffic-aware flags: `--agg-coef a,b,g,d,e,h` sets (α, β, γ, δ, ε, η) of Eq. 14
below (default all 1.0), and `--agg-floor` sets the FedAvg share (default 0.1).
`--agg-penalize-instability` scores σ and Ψ as (1 − normalised value) (see below).
The per-client weights for every round are saved under `rounds[i].agg_weights` in
the `--out` JSON. The raw and normalised metrics and the weights before the floor
are saved under `agg`. Every score (each round's val score and the final score)
also has `per_client`: MAE, RMSE, MAPE and R2 for each district.

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
    traffic_aggregation.py       traffic-aware aggregation weights
  data/
    windowing.py      windows, split, scaler
    district_data.py  load per-district NPZ, build loaders
tests/test_traffic_aggregation.py
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

## Aggregation methods

Notation: $K$ clients (districts), client $k$ has $n_k$ training windows, and
$w_k^{r+1}$ is its model after local training in round $r$. Both methods
average the full model, backbone and head.

### FedAvg (`--agg fedavg`, default)

$$
w^{r+1} = \sum_{k=1}^{K} p_k \, w_k^{r+1}, \qquad p_k = \frac{n_k}{\sum_{u=1}^{K} n_u}
$$

### Traffic-aware aggregation (`--agg traffic`)

From Kaur, Sehra et al., "A traffic-aware federated learning prediction framework
with custom aggregation", *Machine Learning with Applications* 23 (2026) 100829,
Sec. 3.5, Eqs. 6-15 and Algorithm 1. Code: `src/federated/traffic_aggregation.py`.

**Client metrics (Eq. 6).** Each client sends a metric vector with its update:

$$
\mathbf{z}_k = \{V_k,\ \mu_k,\ \sigma_k,\ R_k,\ \Phi_k,\ \Psi_k\}
$$

All six come from the client's **training split only**. Flow $c_{k,i}$ is first
converted back to vehicle counts with that district's scaler, since the districts
are z-scored separately. Here $i$ runs over every (time step, sensor) value in the
training span, and $T_k$ is the number of training time steps.

| Metric | Paper | Here |
|---|---|---|
| $V_k$ (Eq. 7) | number of records, $\lvert D_k \rvert$ | number of training windows, $n_k$ |
| $\mu_k$ (Eq. 8) | mean flow | $\mu_k = \frac{1}{M_k}\sum_i c_{k,i}$, over the $M_k$ flow values |
| $\sigma_k$ (Eq. 9) | std of flow | $\sigma_k = \sqrt{\frac{1}{M_k}\sum_i (c_{k,i}-\mu_k)^2}$ |
| $R_k$ (Eq. 10) | $t_{now} - t^{k}_{max}$ | same, in time steps, with $t_{now} = \max_u t^{u}_{max}$ |
| $\Phi_k$ (Eq. 11) | mean speed $\frac{1}{V_k}\sum_i s_{k,i}$ | **substituted:** mean absolute step-to-step change in flow |
| $\Psi_k$ (Eq. 12) | std of speed | **substituted:** std of the step-to-step change in flow |

**Substitutions.** LargeST gives flow only, with no speed column, so $\Phi_k$ and
$\Psi_k$ use flow-change stand-ins. For sensor $j$ at training step $t$, let
$\Delta_{k,t,j} = c_{k,t+1,j} - c_{k,t,j}$. Then

$$
\Phi_k = \frac{1}{(T_k-1)N_k}\sum_{t,j} \lvert \Delta_{k,t,j} \rvert, \qquad
\Psi_k = \sqrt{\frac{1}{(T_k-1)N_k}\sum_{t,j} \left(\Delta_{k,t,j} - \bar\Delta_k\right)^2}
$$

where $\bar\Delta_k$ is the mean of $\Delta_{k,t,j}$. Like speed, these describe
how smoothly traffic moves, but they come from flow, so they are correlated with
$\sigma_k$. Every district covers the same period, so $R_k = 0$ for all $k$. The
term is kept in the code but normalises to 0 and drops out. On the four real
districts $V_k$ is also equal for every client, so it drops out too.

**Normalisation (Eq. 13).** Min-max across clients for each metric $j$, with
$\varepsilon = 10^{-8}$:

$$
\tilde z_{k,j} = \frac{z_{k,j} - \min_u z_{u,j}}{\max_u z_{u,j} - \min_u z_{u,j} + \varepsilon}
$$

The paper inverts $R$ during normalisation so fresher data scores higher. We do
this by normalising $-R_k$, which gives
$\tilde R_k = (\max_u R_u - R_k)/(\max_u R_u - \min_u R_u + \varepsilon)$.

**Score (Eq. 14).** Note the order: $\beta$ goes with $\sigma$ and $\gamma$ with
$\mu$, and `--agg-coef` follows the same order.

$$
\omega_k = \alpha\,\tilde V_k + \beta\,\tilde\sigma_k + \gamma\,\tilde\mu_k
         + \delta\,\tilde R_k + \epsilon\,\tilde\Phi_k + \eta\,\tilde\Psi_k
$$

The coefficients are non-negative. The paper samples them from $[0.1, 1.5]$ and
grid-searches. The week 2 notebook does 8 random draws scored on the validation
split. All coefficients are positive, so a higher $\sigma$ or $\Psi$ raises a
client's weight.

**Penalize-instability variant (`--agg-penalize-instability`, our addition).**
The paper's text says $\beta$ and $\eta$ "penalize instability", but Eq. 14 with
positive coefficients rewards it. This flag follows the text: $\sigma$ and $\Psi$
enter the score as one minus their normalised value, so the most variable client
contributes 0 on those terms and the most stable contributes 1:

$$
\omega_k^{\text{pen}} = \alpha\,\tilde V_k + \beta\,(1 - \tilde\sigma_k) + \gamma\,\tilde\mu_k
         + \delta\,\tilde R_k + \epsilon\,\tilde\Phi_k + \eta\,(1 - \tilde\Psi_k)
$$

with the same Eq. 13 normalisation. Everything else is unchanged: normalise to sum 1, apply the floor, then Eq. 15. If $\sigma$ (or
$\Psi$) is the same for every client, its normalised value is 0 for all, so the
term adds the same constant $\beta$ (or $\eta$) to every client.

**Normalised weights (Algorithm 1, step 11).**

$$
\bar\omega_k = \frac{\omega_k}{\sum_{u=1}^{K} \omega_u}
$$

If every $\omega_k = 0$ (all clients identical, so every $\tilde z = 0$), we use
$\bar\omega_k = 1/K$. The paper does not cover this case.

**Floor (our addition).** Min-max sets the lowest client to 0 on every metric, so
a client that is lowest on all of them gets $\bar\omega_k = 0$ and is dropped.
To prevent that, we mix in the FedAvg weights:

$$
\hat\omega_k = (1 - f)\,\bar\omega_k + f\,p_k, \qquad f = \texttt{--agg-floor} \in [0, 1]
$$

With $f > 0$, every client keeps at least $f\,p_k$. $f = 1$ is exactly FedAvg
and $f = 0$ is the paper's rule. The default is $f = 0.1$.

**Global update (Eq. 15).**

$$
w^{r+1} = \sum_{k=1}^{K} \hat\omega_k \, w_k^{r+1}
$$

The paper recomputes the metrics every round. Our training splits don't change,
so the metrics and weights are the same in every round. They are still logged
each round.
