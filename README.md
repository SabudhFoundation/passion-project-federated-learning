# Federated Learning for Intelligent Transportation Systems

## Overview

This project presents a privacy-preserving traffic forecasting framework using **Federated Learning (FL)** and **Spatio-Temporal Graph Neural Networks (STGNNs)**. The objective is to accurately predict future traffic flow while preserving data privacy by ensuring that raw traffic sensor data remains localized on client devices and is never shared with a central server.

Two federated deep learning architectures are implemented and compared:

### 1. GNN-GRU-KAN

* Graph Attention Network (GAT)
* Graph Convolutional Network (GCN)
* Gated Recurrent Unit (GRU)
* Kolmogorov-Arnold Network (KAN) Prediction Head

### 2. STGAT+GCN

* Graph Attention Network (GAT)
* Graph Convolutional Network (GCN)
* Gated Recurrent Unit (GRU)
* Fully Connected Prediction Head

Experiments were conducted using the **California PeMS Traffic Dataset** under a federated learning environment utilizing the **FedAvg aggregation algorithm**.

---

# Key Features

* Federated Learning using FedAvg
* Privacy-preserving traffic forecasting
* Graph-based spatial feature extraction
* GRU-based temporal sequence modeling
* KAN and Fully Connected prediction heads
* Multi-step traffic flow forecasting
* Comparative evaluation of federated architectures
* Comprehensive performance analysis using multiple forecasting metrics

---

# Dataset

## California PeMS Traffic Dataset

The experiments utilize the California Performance Measurement System (PeMS) dataset:

* Real-world traffic sensor data
* 15-minute sampling interval
* 435 traffic sensors
* Multi-city California traffic network
* Large-scale spatio-temporal traffic observations

## Dataset Access

Due to GitHub file size limitations, the complete dataset and project resources are hosted on Google Drive.

### Download Link

🔗 **Project Data Archive**

https://drive.google.com/file/d/1iTa07LPK1uC2051IQJ6UFwdVm_B4M9jy/view?usp=drive_link

Please download and extract the archive before running the notebooks or training scripts.

## Data Split

| Dataset Split | Percentage |
| ------------- | ---------- |
| Training      | 70%        |
| Validation    | 15%        |
| Testing       | 15%        |

## Forecasting Configuration

| Parameter         | Value        |
| ----------------- | ------------ |
| Lookback Window   | 12 Timesteps |
| Forecast Horizon  | 5 Timesteps  |
| Sampling Interval | 15 Minutes   |

---

# Model Architecture

## Shared Spatio-Temporal Encoder

```text
Traffic Data
     │
     ▼
Graph Attention Network (GAT)
     │
     ▼
Graph Convolutional Network (GCN)
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

## GNN-GRU-KAN

```text
GAT → GCN → GRU → LayerNorm → KAN
```

### KAN Configuration

* Hidden Layers: [128, 128, 64]
* Grid Size: 8
* Spline Order: 3

## STGAT+GCN

```text
GAT → GCN → GRU → LayerNorm
                    │
                    ▼
      Linear → SiLU → Dropout → Linear
```

---

# Federated Learning Configuration

| Parameter             | Value      |
| --------------------- | ---------- |
| Clients               | 4          |
| Aggregation Algorithm | FedAvg     |
| Communication Rounds  | 100        |
| Optimizer             | AdamW      |
| Loss Function         | Huber Loss |
| Batch Size            | 64         |
| Gradient Clipping     | 1.0        |
| Parameter Clamping    | [-2, 2]    |

---

# Training Hyperparameters

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

# Experimental Results

## Final Test Performance

| Metric    | GNN-GRU-KAN | STGAT+GCN   |
| --------- | ----------- | ----------- |
| MAE       | 5.3474      | **3.7010**  |
| MSE       | 44.3993     | **21.9106** |
| RMSE      | 6.6633      | **4.6809**  |
| MAPE (%)  | 12.4639     | **7.9356**  |
| SMAPE (%) | 10.9100     | **7.5799**  |
| R² Score  | 0.9588      | **0.9797**  |

## Key Findings

The **STGAT+GCN** model consistently outperformed the **GNN-GRU-KAN** architecture across all evaluation metrics.

Possible reasons include:

* Greater stability during FedAvg aggregation
* Lower parameter complexity
* Better compatibility with federated averaging
* Reduced sensitivity to client-side model divergence
* Improved convergence behavior across communication rounds

---

# Project Structure

```text
.
├── notebooks/
│   └── Finalized_Model_GRU_1.ipynb
│
├── results/
│   ├── metrics/
│   └──plots/
│
├── requirements.txt
└── README.md
```

---

# Installation

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

# Dependencies

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

# Running the Project

Launch the notebook:

```bash
jupyter notebook Finalized_Model_GRU_1.ipynb
```

---

# Evaluation Metrics

The following metrics are used to evaluate forecasting performance:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Symmetric Mean Absolute Percentage Error (SMAPE)
* Coefficient of Determination (R²)

---

# Future Work

* Traffic-aware federated aggregation strategies
* FedProx implementation
* Asynchronous Federated Learning
* Non-IID client partitioning using clustering
* Experiments on METR-LA and additional traffic datasets
* Enhanced aggregation methods for KAN layers
* Real-world deployment on edge and IoT devices

---

# Authors

* Manav Sharma
* Justin Johnson
* Aryan Raj
* Revanth Badithabonu
* Gajavada Sanjeevkumar

## Mentors

* Dr. Sukhjit Singh Sehra
* Mr. Sanchit Umate

---

# License

This project was developed as part of the **Sabudh Foundation Data Science Internship Program** and is intended for academic and research purposes.
