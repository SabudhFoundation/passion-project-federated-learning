# Federated Learning for Intelligent Transportation Systems

## Overview

This project implements a privacy-preserving traffic forecasting framework using **Federated Learning (FL)** and **Spatio-Temporal Graph Neural Networks (STGNNs)**. The goal is to predict future traffic flow while ensuring that raw traffic sensor data remains localized on client devices.

The study compares two federated architectures:

1. **GNN-GRU-KAN**

   * Graph Attention Network (GAT)
   * Graph Convolutional Network (GCN)
   * Gated Recurrent Unit (GRU)
   * Kolmogorov-Arnold Network (KAN) prediction head

2. **STGAT+GCN**

   * Graph Attention Network (GAT)
   * Graph Convolutional Network (GCN)
   * Gated Recurrent Unit (GRU)
   * Fully Connected prediction head

Experiments were conducted on the California PeMS traffic dataset using the FedAvg aggregation algorithm.

---

## Features

* Federated Learning using FedAvg
* Privacy-preserving traffic forecasting
* Graph-based spatial feature extraction
* GRU-based temporal modeling
* KAN and Fully Connected prediction heads
* Multi-step traffic flow forecasting
* Performance comparison between architectures
* Evaluation using multiple forecasting metrics

---

## Dataset

### California PeMS Dataset

* Real-world traffic sensor data
* 15-minute sampling interval
* 435 traffic sensors
* Multi-city California traffic network

### Data Split

| Split      | Percentage |
| ---------- | ---------- |
| Training   | 70%        |
| Validation | 15%        |
| Testing    | 15%        |

### Forecasting Setup

| Parameter         | Value        |
| ----------------- | ------------ |
| Lookback Window   | 12 timesteps |
| Forecast Horizon  | 5 timesteps  |
| Sampling Interval | 15 minutes   |

---

## Model Architecture

### Shared Spatio-Temporal Encoder

```text
Traffic Data
     │
     ▼
Graph Attention Network (GAT)
     │
     ▼
Graph Convolution Network (GCN)
     │
     ▼
GRU Encoder
     │
     ▼
Layer Normalization
     │
     ▼
Prediction Head
```

### GNN-GRU-KAN

```text
GAT → GCN → GRU → LayerNorm → KAN
```

KAN Configuration:

* Hidden Layers: [128, 128, 64]
* Grid Size: 8
* Spline Order: 3

### STGAT+GCN

```text
GAT → GCN → GRU → LayerNorm
                    │
                    ▼
      Linear → SiLU → Dropout → Linear
```

---

## Federated Learning Configuration

| Parameter            | Value      |
| -------------------- | ---------- |
| Clients              | 4          |
| Aggregation          | FedAvg     |
| Communication Rounds | 100        |
| Optimizer            | AdamW      |
| Loss Function        | Huber Loss |
| Batch Size           | 64         |
| Gradient Clipping    | 1.0        |
| Parameter Clamping   | [-2, 2]    |

---

## Training Hyperparameters

| Hyperparameter   | Value |
| ---------------- | ----- |
| Learning Rate    | 1e-3  |
| Weight Decay     | 1e-4  |
| GRU Hidden Size  | 128   |
| GRU Layers       | 2     |
| GRU Dropout      | 0.2   |
| FC Dropout       | 0.1   |
| KAN Grid Size    | 8     |
| KAN Spline Order | 3     |

---

## Results

### Final Test Performance

| Metric    | GNN-GRU-KAN | STGAT+GCN   |
| --------- | ----------- | ----------- |
| MAE       | 5.3474      | **3.7010**  |
| MSE       | 44.3993     | **21.9106** |
| RMSE      | 6.6633      | **4.6809**  |
| MAPE (%)  | 12.4639     | **7.9356**  |
| SMAPE (%) | 10.9100     | **7.5799**  |
| R² Score  | 0.9588      | **0.9797**  |

### Key Observation

The **STGAT+GCN** architecture consistently outperformed the **GNN-GRU-KAN** model across all evaluation metrics.

Potential reasons include:

* Instability of spline coefficients during FedAvg aggregation
* Higher parameter complexity of KAN layers
* Better compatibility of fully connected layers with federated averaging

---

## Project Structure

```text
.
├── data/
│   └── final_data.npz
│
├── notebooks/
│   └── Finalized_Model_GRU_1.ipynb
│
├── models/
│   ├── gnn_gru_kan.py
│   ├── stgat_gcn.py
│   └── federated_utils.py
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── checkpoints/
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/federated-traffic-forecasting.git
cd federated-traffic-forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dependencies

* Python 3.10+
* PyTorch
* PyTorch Geometric
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* efficient-kan

Install manually:

```bash
pip install torch torch-geometric numpy pandas scikit-learn matplotlib efficient-kan
```

---

## Running the Project

Launch the notebook:

```bash
jupyter notebook Finalized_Model_GRU_1.ipynb
```

or

```bash
python train.py
```

---

## Evaluation Metrics

The following metrics are used:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Symmetric Mean Absolute Percentage Error (SMAPE)
* Coefficient of Determination (R²)

---

## Future Work

* Traffic-aware federated aggregation
* FedProx implementation
* Asynchronous Federated Learning
* Non-IID client partitioning using clustering
* Experiments on METR-LA and other traffic datasets
* Improved aggregation strategies for KAN layers
* Real-world deployment on edge devices

---

## Authors

* Manav Sharma
* Justin Johnson
* Aryan Raj
* Revanth Badithabonu
* Gajavada Sanjeevkumar

### Mentors

* Dr. Sukhjit Singh Sehra
* Mr. Sanchit Umate

---

## License

This project is developed for academic and research purposes under the Sabudh Foundation Data Science Internship Program.
