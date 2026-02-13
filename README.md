# Wind Power Forecasting with Spatio-Temporal Graph Neural Networks

Research project on wind power forecasting for **179 wind farms** using spatio-temporal Graph Neural Networks (GNNs). The graph structure is constructed based on the geographical proximity of wind farms, with edge weights inversely proportional to distance.

## Models

The following spatio-temporal GNN architectures are explored:

- **A3TGCN2** — Attention-based Temporal Graph Convolutional Network
- **DCRNN** — Diffusion Convolutional Recurrent Neural Network
- **ASTGNN** — Attention-based Spatial-Temporal Graph Neural Network
- **DSTAGNN** — Discrete Spatial-Temporal Attention Graph Neural Network
- **STGCN** — Spatio-Temporal Graph Convolutional Network
- **Spacetimeformer** — Spatial-Temporal Transformer

## Main Entry Points

- `a3tgcn_for_traffic_forecasting.py` — Main training script (A3TGCN2 / DCRNN)
- `pytorch_geometric_temporal/notebooks/` — Notebook-based experiments

## Setup

The project uses **PyTorch** and **PyTorch Geometric Temporal**. A local fork of `pytorch_geometric_temporal` is included in the repository.

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Input window | 48 time steps |
| Forecast horizon | 36 time steps |
| Lag | 12 time steps |
| Batch size | 64 |
| Optimizer | Adam (lr=0.005) |
| Epochs | 50 |

## Dataset

> **⚠️ The dataset is confidential and not included in this repository.**

The data covers 179 wind farms and includes power output along with associated features. Wind farms are grouped and connected via a spatial graph where edges link farms within a distance threshold.

## Project Structure

```
├── a3tgcn_for_traffic_forecasting.py   # Main training script
├── data_utils.py                       # Data utilities
├── dataset_utils.py                    # Dataset utilities
├── graph.py                            # Graph construction
├── prepare_data.py                     # Data preparation
├── read_nwp.py                         # NWP data reader
├── pytorch_geometric_temporal/         # Local fork of PyG Temporal
├── ASTGNN-Traffic Flow/                # ASTGNN experiments
├── DSTAGNN-Traffic Flow/               # DSTAGNN experiments
├── stgcn-TrafficPred/                  # STGCN experiments
├── spacetimeformer-main/               # Spacetimeformer experiments
└── PASSAT_5p625-main/                  # PASSAT model experiments
```
