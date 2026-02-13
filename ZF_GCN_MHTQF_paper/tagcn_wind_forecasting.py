"""
Spatial-Temporal Wind Power Probabilistic Forecasting
Based on Time-Aware Graph Convolutional Network (TAGCN)

This implementation includes:
1. Graph construction from geographical coordinates
2. Time-Aware GCN for spatial-temporal feature extraction
3. Quantile regression for probabilistic forecasting

Reference: Spatial-Temporal Wind Power Probabilistic Forecasting Based on 
Time-Aware Graph Convolutional Network (IEEE)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional, Dict
import warnings
import os

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. GRAPH CONSTRUCTION MODULE
# ==============================================================================

class GraphConstructor:
    """
    Constructs adjacency matrix for wind farm network based on:
    - Geographical distance (Gaussian kernel)
    - Learned correlation (optional)
    """
    
    def __init__(self, sigma: float = 10.0, threshold: float = 0.1):
        """
        Args:
            sigma: Gaussian kernel bandwidth for distance weighting
            threshold: Minimum edge weight to include (sparsity control)
        """
        self.sigma = sigma
        self.threshold = threshold
    
    def build_distance_adjacency(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Build adjacency matrix based on geographical distance.
        
        Args:
            coordinates: (N, 2) array of [latitude, longitude] for N wind farms
            
        Returns:
            Adjacency matrix (N, N)
        """
        # Compute pairwise distances
        dist_matrix = cdist(coordinates, coordinates, metric='euclidean')
        
        # Apply Gaussian kernel
        adj_matrix = np.exp(-dist_matrix ** 2 / (2 * self.sigma ** 2))
        
        # Apply threshold for sparsity
        adj_matrix[adj_matrix < self.threshold] = 0
        
        # Remove self-loops
        np.fill_diagonal(adj_matrix, 0)
        
        return adj_matrix
    
    def build_correlation_adjacency(self, power_data: np.ndarray) -> np.ndarray:
        """
        Build adjacency matrix based on Pearson correlation of power outputs.
        
        Args:
            power_data: (T, N) array of power outputs for N farms over T timesteps
            
        Returns:
            Adjacency matrix (N, N)
        """
        corr_matrix = np.corrcoef(power_data.T)
        
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Keep only positive correlations
        corr_matrix = np.maximum(corr_matrix, 0)
        
        # Apply threshold
        corr_matrix[corr_matrix < self.threshold] = 0
        
        # Remove self-loops
        np.fill_diagonal(corr_matrix, 0)
        
        return corr_matrix
    
    def build_combined_adjacency(self, coordinates: np.ndarray, 
                                  power_data: np.ndarray,
                                  alpha: float = 0.5) -> np.ndarray:
        """
        Combine distance-based and correlation-based adjacency matrices.
        
        Args:
            coordinates: (N, 2) geographical coordinates
            power_data: (T, N) historical power data
            alpha: Weight for distance adjacency (1-alpha for correlation)
            
        Returns:
            Combined adjacency matrix (N, N)
        """
        adj_dist = self.build_distance_adjacency(coordinates)
        adj_corr = self.build_correlation_adjacency(power_data)
        
        # Normalize each matrix
        adj_dist = adj_dist / (adj_dist.sum(axis=1, keepdims=True) + 1e-8)
        adj_corr = adj_corr / (adj_corr.sum(axis=1, keepdims=True) + 1e-8)
        
        return alpha * adj_dist + (1 - alpha) * adj_corr


# ==============================================================================
# 2. TIME-AWARE ATTENTION MODULE
# ==============================================================================

class TimeAwareAttention(nn.Module):
    """
    Time-aware attention mechanism that learns dynamic temporal weights.
    Incorporates positional encoding for time awareness.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 max_seq_len: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Time embedding for temporal awareness (learnable positional encoding)
        self.time_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            seq_len: Sequence length for positional encoding
            
        Returns:
            Attention output (batch, seq_len, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Add time embedding
        time_emb = self.time_embedding[:, :seq_len, :]
        x_with_time = x + time_emb
        
        # Multi-head attention
        Q = self.query(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_proj(attn_output)
        output = self.layer_norm(x + self.dropout(output))
        
        return output


# ==============================================================================
# 3. GRAPH CONVOLUTIONAL NETWORK MODULE
# ==============================================================================

class SpatialGCN(nn.Module):
    """
    Graph Convolutional Network for capturing spatial dependencies
    between wind farms.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
            self.layer_norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            
        Returns:
            Updated node features (num_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = self.layer_norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


# ==============================================================================
# 4. TEMPORAL LSTM MODULE
# ==============================================================================

class TemporalLSTM(nn.Module):
    """
    LSTM module for capturing temporal dependencies in wind power sequences.
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.bidirectional = bidirectional
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            
        Returns:
            outputs: All hidden states (batch, seq_len, hidden_size*2)
            final_hidden: Final hidden state (batch, hidden_size*2)
        """
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Concatenate final forward and backward hidden states
        if self.bidirectional:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_hidden = h_n[-1]
        
        return outputs, final_hidden


# ==============================================================================
# 5. TIME-AWARE GCN (TAGCN) MAIN MODEL
# ==============================================================================

class TimeAwareGCN(nn.Module):
    """
    Time-Aware Graph Convolutional Network for Spatial-Temporal
    Wind Power Probabilistic Forecasting.
    
    Architecture:
    1. Input embedding layer
    2. Spatial GCN for inter-farm dependencies
    3. Time-aware attention for dynamic temporal weighting
    4. Temporal LSTM for sequential patterns
    5. Quantile regression output for probabilistic forecasting
    """
    
    def __init__(self, 
                 num_nodes: int,
                 input_dim: int,
                 hidden_dim: int = 64,
                 gcn_hidden: int = 32,
                 lstm_hidden: int = 64,
                 num_gcn_layers: int = 2,
                 num_lstm_layers: int = 2,
                 num_attention_heads: int = 4,
                 forecast_horizon: int = 24,
                 quantiles: List[float] = None,
                 dropout: float = 0.1):
        """
        Args:
            num_nodes: Number of wind farms
            input_dim: Number of input features per node
            hidden_dim: Hidden dimension for attention
            gcn_hidden: Hidden dimension for GCN layers
            lstm_hidden: Hidden dimension for LSTM
            num_gcn_layers: Number of GCN layers
            num_lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            forecast_horizon: Number of future timesteps to predict
            quantiles: Quantile levels for probabilistic forecasting
            dropout: Dropout rate
        """
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Spatial GCN
        self.spatial_gcn = SpatialGCN(
            in_channels=hidden_dim,
            hidden_channels=gcn_hidden,
            out_channels=hidden_dim,
            num_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # Time-aware attention
        self.time_attention = TimeAwareAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Temporal LSTM
        self.temporal_lstm = TemporalLSTM(
            input_size=hidden_dim * num_nodes,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Output layers for quantile regression
        lstm_output_dim = lstm_hidden * 2  # bidirectional
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_nodes * forecast_horizon)
            )
            for _ in quantiles
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for TAGCN.
        
        Args:
            x: Input features (batch, seq_len, num_nodes, input_dim)
            edge_index: Graph edge connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            
        Returns:
            Quantile predictions (batch, num_quantiles, num_nodes, forecast_horizon)
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # 1. Input embedding
        x = self.input_embedding(x)  # (batch, seq_len, num_nodes, hidden_dim)
        
        # 2. Apply Spatial GCN at each timestep
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, num_nodes, hidden_dim)
            
            # Process each sample in batch
            batch_gcn = []
            for b in range(batch_size):
                gcn_out = self.spatial_gcn(x_t[b], edge_index, edge_weight)
                batch_gcn.append(gcn_out)
            
            gcn_outputs.append(torch.stack(batch_gcn, dim=0))
        
        x = torch.stack(gcn_outputs, dim=1)  # (batch, seq_len, num_nodes, hidden_dim)
        
        # 3. Apply Time-aware attention
        # Reshape for attention: treat each node's sequence separately
        x = x.permute(0, 2, 1, 3)  # (batch, num_nodes, seq_len, hidden_dim)
        x = x.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
        x = self.time_attention(x, seq_len)
        x = x.reshape(batch_size, num_nodes, seq_len, self.hidden_dim)
        x = x.permute(0, 2, 1, 3)  # (batch, seq_len, num_nodes, hidden_dim)
        
        # 4. Flatten spatial dimension and apply Temporal LSTM
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, num_nodes*hidden_dim)
        _, final_hidden = self.temporal_lstm(x)  # (batch, lstm_hidden*2)
        
        # 5. Generate quantile predictions
        quantile_outputs = []
        for output_layer in self.output_layers:
            q_out = output_layer(final_hidden)  # (batch, num_nodes*forecast_horizon)
            q_out = q_out.reshape(batch_size, num_nodes, self.forecast_horizon)
            quantile_outputs.append(q_out)
        
        output = torch.stack(quantile_outputs, dim=1)  # (batch, num_quantiles, num_nodes, horizon)
        
        return output


# ==============================================================================
# 6. LOSS FUNCTIONS
# ==============================================================================

class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    """
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_quantiles, num_nodes, horizon)
            targets: (batch, num_nodes, horizon)
            
        Returns:
            Quantile loss scalar
        """
        losses = []
        targets = targets.unsqueeze(1)  # (batch, 1, num_nodes, horizon)
        
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, i:i+1, :, :]
            errors = targets - pred_q
            loss_q = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss_q.mean())
        
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Quantile loss + Temporal smoothness regularization.
    """
    
    def __init__(self, quantiles: List[float], smoothness_weight: float = 0.1):
        super().__init__()
        self.quantile_loss = QuantileLoss(quantiles)
        self.smoothness_weight = smoothness_weight
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Quantile loss
        q_loss = self.quantile_loss(predictions, targets)
        
        # Temporal smoothness: penalize large differences between consecutive predictions
        median_idx = len(self.quantiles) // 2
        median_pred = predictions[:, median_idx, :, :]  # Use median quantile
        temporal_diff = median_pred[:, :, 1:] - median_pred[:, :, :-1]
        smoothness_loss = (temporal_diff ** 2).mean()
        
        return q_loss + self.smoothness_weight * smoothness_loss


# ==============================================================================
# 7. DATA PREPARATION UTILITIES
# ==============================================================================

class WindPowerDataset(Dataset):
    """
    Dataset for wind power forecasting with sliding window approach.
    """
    
    def __init__(self, power_data: np.ndarray, features: np.ndarray,
                 seq_len: int = 24, forecast_horizon: int = 24):
        """
        Args:
            power_data: (T, N) wind power for N farms over T timesteps
            features: (T, N, F) additional features (NWP, etc.)
            seq_len: Input sequence length
            forecast_horizon: Forecast horizon
        """
        self.power_data = torch.FloatTensor(power_data)
        self.features = torch.FloatTensor(features)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        self.num_samples = len(power_data) - seq_len - forecast_horizon + 1
    
    def __len__(self):
        return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        # Input: features from t to t+seq_len
        x = self.features[idx:idx + self.seq_len]  # (seq_len, N, F)
        
        # Target: power from t+seq_len to t+seq_len+horizon
        y = self.power_data[idx + self.seq_len:idx + self.seq_len + self.forecast_horizon].T
        # (N, horizon)
        
        return x, y


def adjacency_to_edge_index(adj_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert adjacency matrix to edge_index and edge_weight format.
    """
    edge_list = []
    edge_weights = []
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:
                edge_list.append([i, j])
                edge_weights.append(adj_matrix[i, j])
    
    if len(edge_list) == 0:
        # No edges - create self-loops as fallback
        num_nodes = adj_matrix.shape[0]
        edge_list = [[i, i] for i in range(num_nodes)]
        edge_weights = [1.0] * num_nodes
    
    edge_index = torch.LongTensor(edge_list).T
    edge_weight = torch.FloatTensor(edge_weights)
    
    return edge_index, edge_weight


# ==============================================================================
# 8. TRAINING AND EVALUATION
# ==============================================================================

class TAGCNTrainer:
    """
    Trainer class for Time-Aware GCN model.
    """
    
    def __init__(self, model: TimeAwareGCN, device: str = 'cuda',
                 save_path: str = 'best_tagcn_model.pt'):
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              edge_index: torch.Tensor, edge_weight: torch.Tensor,
              epochs: int = 100, lr: float = 1e-3, patience: int = 10,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the TAGCN model.
        
        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = CombinedLoss(self.model.quantiles)
        
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(x, edge_index, edge_weight)
                loss = criterion(predictions, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x, edge_index, edge_weight)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path))
        
        return history
    
    def predict(self, data_loader: DataLoader, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Returns:
            predictions: (N_samples, num_quantiles, num_nodes, horizon)
            targets: (N_samples, num_nodes, horizon)
        """
        self.model.eval()
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                predictions = self.model(x, edge_index, edge_weight)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y.numpy())
        
        return np.concatenate(all_preds), np.concatenate(all_targets)


# ==============================================================================
# 9. EVALUATION METRICS
# ==============================================================================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray, 
                    quantiles: List[float]) -> Dict[str, float]:
    """
    Compute evaluation metrics for probabilistic forecasting.
    
    Args:
        predictions: (N, num_quantiles, num_nodes, horizon)
        targets: (N, num_nodes, horizon)
        quantiles: List of quantile levels
        
    Returns:
        Dictionary of metrics
    """
    # Find median index
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_pred = predictions[:, median_idx, :, :]
    
    # Point prediction metrics
    mae = np.mean(np.abs(median_pred - targets))
    rmse = np.sqrt(np.mean((median_pred - targets) ** 2))
    
    # Normalized metrics
    target_std = targets.std()
    target_mean = np.abs(targets).mean()
    nrmse = rmse / (target_std + 1e-8) if target_std > 0 else rmse
    nmae = mae / (target_mean + 1e-8) if target_mean > 0 else mae
    
    # Probabilistic metrics - Pinball loss
    pinball_losses = []
    for i, q in enumerate(quantiles):
        pred_q = predictions[:, i, :, :]
        errors = targets - pred_q
        pinball = np.mean(np.maximum(q * errors, (q - 1) * errors))
        pinball_losses.append(pinball)
    
    # Coverage (for 80% prediction interval if 0.1 and 0.9 quantiles exist)
    coverage = None
    interval_width = None
    if 0.1 in quantiles and 0.9 in quantiles:
        lower_idx = quantiles.index(0.1)
        upper_idx = quantiles.index(0.9)
        lower = predictions[:, lower_idx, :, :]
        upper = predictions[:, upper_idx, :, :]
        coverage = np.mean((targets >= lower) & (targets <= upper))
        interval_width = np.mean(upper - lower)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'NMAE': nmae,
        'Pinball_Loss': np.mean(pinball_losses),
        'Coverage_80': coverage,
        'Interval_Width': interval_width
    }


def compute_per_horizon_metrics(predictions: np.ndarray, targets: np.ndarray,
                                 quantiles: List[float]) -> pd.DataFrame:
    """
    Compute metrics for each forecast horizon.
    
    Args:
        predictions: (N, num_quantiles, num_nodes, horizon)
        targets: (N, num_nodes, horizon)
        quantiles: List of quantile levels
        
    Returns:
        DataFrame with metrics per horizon
    """
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_pred = predictions[:, median_idx, :, :]
    
    horizon = predictions.shape[-1]
    results = []
    
    for h in range(horizon):
        pred_h = median_pred[:, :, h]
        target_h = targets[:, :, h]
        
        mae_h = np.mean(np.abs(pred_h - target_h))
        rmse_h = np.sqrt(np.mean((pred_h - target_h) ** 2))
        
        results.append({
            'horizon': h + 1,
            'MAE': mae_h,
            'RMSE': rmse_h
        })
    
    return pd.DataFrame(results)


# ==============================================================================
# 10. MAIN EXECUTION EXAMPLE
# ==============================================================================

def run_example():
    """
    Example usage of the TAGCN model for wind power forecasting.
    """
    # Configuration
    NUM_FARMS = 10
    SEQ_LEN = 24  # 24 hours of historical data
    FORECAST_HORIZON = 24  # Predict next 24 hours
    INPUT_DIM = 5  # Features: power, wind_speed, wind_dir, temperature, pressure
    HIDDEN_DIM = 64
    BATCH_SIZE = 32
    EPOCHS = 100
    QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    T = 8760  # One year of hourly data
    
    # Synthetic wind farm coordinates
    coordinates = np.random.rand(NUM_FARMS, 2) * 100  # 100km x 100km region
    
    # Synthetic power data with spatial correlation
    base_power = np.sin(np.linspace(0, 50*np.pi, T))[:, None] * 0.5 + 0.5
    spatial_noise = np.random.randn(T, NUM_FARMS) * 0.1
    power_data = np.clip(base_power + spatial_noise, 0, 1)
    
    # Synthetic features (power + NWP features)
    features = np.random.randn(T, NUM_FARMS, INPUT_DIM) * 0.5
    features[:, :, 0] = power_data  # First feature is power
    
    # Build adjacency matrix
    graph_constructor = GraphConstructor(sigma=20.0, threshold=0.1)
    adj_matrix = graph_constructor.build_combined_adjacency(coordinates, power_data)
    edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
    
    print(f"Graph: {NUM_FARMS} nodes, {edge_index.shape[1]} edges")
    
    # Create datasets
    train_size = int(0.7 * T)
    val_size = int(0.15 * T)
    
    train_dataset = WindPowerDataset(
        power_data[:train_size], 
        features[:train_size],
        seq_len=SEQ_LEN, 
        forecast_horizon=FORECAST_HORIZON
    )
    
    val_dataset = WindPowerDataset(
        power_data[train_size:train_size+val_size],
        features[train_size:train_size+val_size],
        seq_len=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON
    )
    
    test_dataset = WindPowerDataset(
        power_data[train_size+val_size:],
        features[train_size+val_size:],
        seq_len=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = TimeAwareGCN(
        num_nodes=NUM_FARMS,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        gcn_hidden=32,
        lstm_hidden=64,
        num_gcn_layers=2,
        num_lstm_layers=2,
        num_attention_heads=4,
        forecast_horizon=FORECAST_HORIZON,
        quantiles=QUANTILES,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = TAGCNTrainer(model, device=device)
    history = trainer.train(
        train_loader, val_loader, edge_index, edge_weight,
        epochs=EPOCHS, lr=1e-3, patience=15
    )
    
    # Evaluate on test set
    predictions, targets = trainer.predict(test_loader, edge_index, edge_weight)
    metrics = compute_metrics(predictions, targets, QUANTILES)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
    
    # Per-horizon metrics
    horizon_metrics = compute_per_horizon_metrics(predictions, targets, QUANTILES)
    print("\nPer-horizon RMSE (first 6 hours):")
    print(horizon_metrics.head(6).to_string(index=False))
    
    return model, metrics, history


if __name__ == "__main__":
    model, metrics, history = run_example()
