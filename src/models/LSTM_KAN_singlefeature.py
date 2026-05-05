# =============================================================================
# FINAL CLEAN VERSION: FL + LSTM-KAN + METRICS
# =============================================================================
import pandas as pd
import argparse, warnings
import numpy as np
import h5py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import flwr as fl
from flwr.common import Context
warnings.filterwarnings("ignore")

# =============================================================================
# SCALER
# =============================================================================

class TrafficScaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    def fit(self, X):
        self.scaler.fit(X)
        return self
    def transform(self, X):
        return self.scaler.transform(X).astype(np.float32)

# =============================================================================
# LOADERS
# =============================================================================

def clean_data(data):
    data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])
    return np.nan_to_num(data, nan=0.0)

def load_h5(path, max_stations=200):
    print(f"\n[Data] Loading {path}")
    with h5py.File(path, 'r') as f:
        data = f['t/block0_values'][:]
        sensors = f['t/axis0'][:]
    sensors = [s.decode('utf-8') for s in sensors]
    data = data[:, :max_stations]
    sensors = sensors[:max_stations]
    data = clean_data(data)
    data = data[..., np.newaxis].astype(np.float32)
    print("Shape:", data.shape)
    return data, np.array(sensors)

def load_csv(path, max_stations=200):
    print(f"\n[Data] Loading CSV: {path}")
    df = pd.read_csv(path)
    df = df.drop(columns=[df.columns[0]])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.iloc[:, :max_stations]
    data = clean_data(df.values.astype(np.float32))
    data = data[..., np.newaxis]
    print("Shape:", data.shape)
    return data, df.columns.astype(str).values
# =============================================================================
# WINDOWS
# =============================================================================
def make_windows(data, seq_len, horizon):
    T, N, F = data.shape
    Xs, ys = [], []
    for t in range(T - seq_len - horizon):
        Xs.append(data[t:t+seq_len])              # (seq_len, N, 1)
        ys.append(data[t+seq_len+horizon, :, 0])  # (N,)
    X = np.stack(Xs)   # (samples, seq_len, N, 1)
    y = np.stack(ys)   # (samples, N)
    # reshape for model (sensor-wise samples)
    X = X.transpose(0, 2, 1, 3).reshape(-1, seq_len, 1)
    y = y.reshape(-1)
    # remove NaNs safely
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    return X, y
# =============================================================================
# MODEL
# =============================================================================
class LSTM_KAN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# =============================================================================
# DATASET
# =============================================================================

class ClientDataset:
    def __init__(self, data, seq_len, horizon, batch_size=64):

        T, N, _ = data.shape

        scaler = TrafficScaler()
        flat = data.reshape(-1, 1)

        scaler.fit(flat)
        data = scaler.transform(flat).reshape(T, N, 1)

        X, y = make_windows(data, seq_len, horizon)

        split = int(0.8 * len(X))

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split])),
            batch_size=batch_size, shuffle=True
        )

        self.test_loader = DataLoader(
            TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:])),
            batch_size=batch_size
        )

# =============================================================================
# CLIENT
# =============================================================================

class FLClient(fl.client.NumPyClient):
    def __init__(self, dataset, seq_len):
        self.model = LSTM_KAN(seq_len)
        self.dataset = dataset
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)})
        self.model.load_state_dict(sd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for xb, yb in self.dataset.train_loader:
            pred = self.model(xb).squeeze()
            loss = ((pred - yb)**2).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return self.get_parameters({}), len(self.dataset.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        preds, trues = [], []

        with torch.no_grad():
            for xb, yb in self.dataset.test_loader:
                pred = torch.nan_to_num(self.model(xb).squeeze())
                preds.append(pred.numpy())
                trues.append(yb.numpy())

        p = np.concatenate(preds)
        t = np.concatenate(trues)

        mask = np.isfinite(p) & np.isfinite(t)
        p, t = p[mask], t[mask]

        mae = mean_absolute_error(t, p)
        mse = mean_squared_error(t, p)
        rmse = np.sqrt(mse)

        return mae, len(t), {"mae": mae, "mse": mse, "rmse": rmse}

# =============================================================================
# METRIC AGGREGATION
# =============================================================================

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)

    mae = sum(n*m["mae"] for n, m in metrics) / total
    mse = sum(n*m["mse"] for n, m in metrics) / total
    rmse = sum(n*m["rmse"] for n, m in metrics) / total

    print("\n📊 GLOBAL METRICS")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return {"mae": mae, "mse": mse, "rmse": rmse}
# =============================================================================
# MAIN
# =============================================================================

def main(args):

    if args.data_path.endswith(".h5"):
        data, _ = load_h5(args.data_path, args.max_stations)
    else:
        data, _ = load_csv(args.data_path, args.max_stations)

    parts = np.array_split(range(data.shape[1]), args.n_clients)

    clients = []
    for part in parts:
        ds = ClientDataset(data[:, part, :], args.seq_len, args.horizon)
        clients.append(FLClient(ds, args.seq_len))

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        return clients[cid].to_client()

    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.n_clients,
        config=fl.server.ServerConfig(num_rounds=args.n_rounds),
        strategy=strategy,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/Users/manavsharma/Downloads/gba_2019_small(1).csv")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--max_stations", type=int, default=200)

    main(parser.parse_args())

