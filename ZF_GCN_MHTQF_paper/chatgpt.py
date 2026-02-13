# Below is a PyTorch implementation of the ZF-GCN-MHTQF model architecture described in "Spatial-Temporal Wind Power
# Probabilistic Forecasting Based on Time-Aware Graph Convolutional Network". The code is organized into modular classes
# corresponding to each component of the model, with inline comments and citations for clarity:
#
# ZigzagPersistenceImage – Stub for computing the Zigzag Persistence Image (ZPI) from a sequence of graph features
# (uses a placeholder tensor of fixed shape as full TDA is complex).
#
# FlexConv – Flexible-size continuous kernel convolution using multiplicative anisotropic Gabor filters and Gaussian
# masks. Implemented as a custom 2D convolution where the kernel weights are modulated by a learnable Gaussian mask.
#
# SpatialGraphConvolution – Spatial graph convolution layer using Laplacian link (incorporating adjacency matrix powers)
# to mix node features based on graph structure.
#
# TemporalGraphConvolution – Temporal convolution layer using a learnable projection vector to collapse P time steps
# (window) into one, followed by a linear transform.
#
# STGCNBlock – Combines the spatial and temporal graph convolution outputs, and integrates topological features.
# The spatial and temporal outputs (each of dimension d_out/2) are each scaled (elementwise multiplied) by the
# topological feature vector from ZPI/FlexConv, then concatenated.
#
# HTQFOutput – Outputs the Heavy-Tailed Quantile Function parameters (μ, σ, u, v) for each node, using a linear layer.
# A softplus activation is applied to σ to ensure positivity.
#
# ZF_GCN_MHTQF_Model – The full model that ties everything together, including a GRU to capture temporal dependencies
# across multiple time windows. The GRU processes the sequence of STGCN outputs and its final hidden state
# is used to produce the HTQF parameters for each node.
#
# Finally, synthetic input data is created to demonstrate a forward pass, and the output shape is printed to verify
# the architecture.
# In this example, the output of the forward pass is a tensor of shape (5, 4) (for 5 nodes, each with 4 parameters μ, σ, u, v),
# confirming the model architecture produces the expected output shape. Each component of the model corresponds to
# the descriptions in the paper: the FlexConv layer uses a Gaussian mask on the convolution kernel,
# the spatial and temporal graph convolutions leverage Laplacian adjacency powers and a learnable time projection respectively,
# and the final output layer produces the heavy-tailed quantile function parameters.
# All these pieces are integrated into the ZF-GCN-MHTQF model, and the code can be extended or
# refined (e.g. replacing the adjacency placeholder or the ZPI stub with real computations) for a full implementation.

import os
os.environ["OMP_NUM_THREADS"] = "1"  # limit threads for reproducibility
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)

# Zigzag Persistence Image (ZPI) processing stub
class ZigzagPersistenceImage(nn.Module):
    def __init__(self, out_height: int = 6, out_width: int = 6):
        """
        Placeholder for Zigzag Persistence Image (ZPI) computation.
        This stub simulates a persistence image of fixed size (out_height x out_width).
        In practice, ZPI would be computed from topological features of a sequence.
        """
        super(ZigzagPersistenceImage, self).__init__()
        self.out_height = out_height
        self.out_width = out_width

    def forward(self, X_window: torch.Tensor) -> torch.Tensor:
        """
        Compute a dummy Zigzag Persistence Image from a sequence of feature matrices.
        X_window: Tensor of shape (P, N, F) representing features of N nodes over P time steps.
        Returns: Tensor of shape (1, out_height, out_width) representing a persistence image.
        """
        # For demonstration, fill the image with the average value of the input window.
        avg_val = X_window.mean().item()
        zpi_image = torch.full((1, self.out_height, self.out_width), avg_val, dtype=X_window.dtype)
        return zpi_image

# FlexConv: Flexible-size continuous kernel convolution with anisotropic Gabor filters and Gaussian masks
class FlexConv(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 8, kernel_size: int = 3):
        """
        FlexConv: continuous kernel convolution using multiplicative anisotropic Gabor filters and Gaussian masks.
        Implemented as a 2D convolution with a learnable kernel, where the kernel weights are element-wise
        multiplied by a Gaussian mask parameter matrix (simulating an anisotropic Gaussian mask):contentReference[oaicite:7]{index=7}.
        """
        super(FlexConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Learnable Gabor filter weights and mask parameters for the kernel
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.theta_mask = nn.Parameter(torch.randn(kernel_size, kernel_size))
        # (No bias for simplicity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FlexConv to input image x.
        x: Tensor of shape (batch, in_channels, H, W) or (in_channels, H, W) for a single image.
        Returns: Tensor of shape (batch, out_channels, H_out, W_out) after convolution.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # add batch dimension if missing
        # Create an anisotropic Gaussian mask from theta_mask parameters
        mask = torch.exp(-(self.theta_mask ** 2))
        mask_expanded = mask.expand(self.out_channels, self.in_channels, -1, -1)  # shape like weight
        # Element-wise modulate the convolution weights by the mask:contentReference[oaicite:8]{index=8}
        effective_weight = self.weight * mask_expanded
        # 2D convolution with 'same' padding so output size ~ input size
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad))
        out = F.conv2d(x_padded, effective_weight, bias=None, stride=1)
        return out

# Spatial Graph Convolution layer using Laplacian link (adjacency power series)
class SpatialGraphConvolution(nn.Module):
    def __init__(self, d_in: int, d_out: int, adjacency: torch.Tensor, M: int = 1):
        """
        Spatial Graph Convolution using a Laplacian link with power series of adjacency:contentReference[oaicite:9]{index=9}.
        d_in: input feature dimension per node.
        d_out: output feature dimension per node.
        adjacency: (N x N) adjacency matrix of the graph.
        M: power level (use adjacency powers up to M).
        """
        super(SpatialGraphConvolution, self).__init__()
        self.N = adjacency.shape[0]
        self.d_in = d_in
        self.d_out = d_out
        self.M = M
        # Pre-compute [I, A, A^2, ..., A^M] (Laplacian link powers)
        I = torch.eye(self.N, dtype=adjacency.dtype)
        A_powers = [I]
        for k in range(1, M + 1):
            A_powers.append(torch.matrix_power(adjacency, k))
        # Register as buffer (non-parameter constant)
        self.register_buffer('adjacency_powers', torch.stack(A_powers))  # shape (M+1, N, N)
        # Learnable weight for each power in adjacency_powers: shape (M+1, d_in, d_out)
        self.W = nn.Parameter(torch.randn(M + 1, d_in, d_out))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: Tensor of shape (N, d_in) - node features at current time.
        Returns: Tensor of shape (N, d_out) - transformed node features.
        """
        # Ensure correct input shape
        assert X.shape[-2:] == (self.N, self.d_in), f"Expected input shape (_, {self.N}, {self.d_in}), got {X.shape}"
        # Sum contributions from each power of adjacency
        out = torch.zeros((self.N, self.d_out), dtype=X.dtype, device=X.device)
        for k in range(self.M + 1):
            # (N x N) @ (N x d_in) -> (N x d_in)
            X_k = torch.matmul(self.adjacency_powers[k], X)
            # Multiply by weight for power k: (N x d_in) @ (d_in x d_out) -> (N x d_out)
            out += X_k @ self.W[k]
        return out

# Temporal Graph Convolution layer using projection
class TemporalGraphConvolution(nn.Module):
    def __init__(self, d_in: int, d_out: int, P: int):
        """
        Temporal Graph Convolution using a learnable projection vector to reduce P time steps:contentReference[oaicite:10]{index=10}.
        d_in: input feature dimension per node.
        d_out: output feature dimension per node.
        P: window size (number of time steps in the temporal convolution input).
        """
        super(TemporalGraphConvolution, self).__init__()
        self.P = P
        self.d_in = d_in
        self.d_out = d_out
        # Learnable projection vector (size P) to collapse P time steps:contentReference[oaicite:11]{index=11}
        self.proj_vector = nn.Parameter(torch.randn(P))
        # Learnable weight to transform projected features (d_in -> d_out)
        self.W = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, X_window: torch.Tensor) -> torch.Tensor:
        """
        X_window: Tensor of shape (P, N, d_in) - features of N nodes over P past time steps (inclusive of current).
        Returns: Tensor of shape (N, d_out) - transformed node features at current time.
        """
        # Use proj_vector to weight each time step in the window
        # Expand proj_vector to shape (P, 1, 1) and multiply with X_window
        weighted = X_window * self.proj_vector.view(self.P, 1, 1)
        # Sum over the P dimension to get an N x d_in matrix (aggregated features):contentReference[oaicite:12]{index=12}
        X_proj = weighted.sum(dim=0)  # shape (N, d_in)
        # Linear transform to get output features of size d_out
        out = X_proj @ self.W  # (N x d_in) @ (d_in x d_out) -> (N, d_out)
        return out

# Spatial-Temporal Graph Convolution block (STGCN) combining spatial, temporal, and topological features
class STGCNBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, adjacency: torch.Tensor, P: int, M: int = 1):
        """
        Combines SpatialGraphConvolution and TemporalGraphConvolution layers, integrating topological features.
        d_in: input feature dimension per node.
        d_out: total output feature dimension per node (split equally into spatial and temporal parts).
        adjacency: adjacency matrix for spatial convolution.
        P: window size for temporal convolution.
        M: power level for spatial Laplacian link.
        """
        super(STGCNBlock, self).__init__()
        assert d_out % 2 == 0, "d_out must be even."
        # Define spatial and temporal convolution sub-layers
        self.spatial_conv = SpatialGraphConvolution(d_in, d_out // 2, adjacency, M=M)
        self.temporal_conv = TemporalGraphConvolution(d_in, d_out // 2, P=P)

    def forward(self, X_current: torch.Tensor, X_window: torch.Tensor, F_topo: torch.Tensor) -> torch.Tensor:
        """
        X_current: (N, d_in) features at current time t.
        X_window: (P, N, d_in) features from time t-P+1 to t.
        F_topo: (d_out/2,) topological feature vector from ZPI/FlexConv.
        Returns: (N, d_out) combined spatial-temporal-topological features.
        """
        # Apply spatial graph convolution on current features -> (N, d_out/2)
        S_out = self.spatial_conv(X_current)
        # Apply temporal graph convolution on window -> (N, d_out/2)
        T_out = self.temporal_conv(X_window)
        # Scale (modulate) spatial and temporal outputs by topological features:contentReference[oaicite:13]{index=13}
        # F_topo is a vector of length d_out/2; broadcast it across N nodes for elementwise multiplication
        S_scaled = S_out * F_topo  # (N, d_out/2)
        T_scaled = T_out * F_topo  # (N, d_out/2)
        # Concatenate scaled spatial and temporal features -> (N, d_out)
        C_out = torch.cat([S_scaled, T_scaled], dim=1)
        return C_out

# Heavy-Tailed Quantile Function (HTQF) output layer
class HTQFOutput(nn.Module):
    def __init__(self, in_dim: int, N: int):
        """
        Outputs heavy-tailed quantile function parameters (μ, σ, u, v) for each node:contentReference[oaicite:14]{index=14}.
        in_dim: input feature dimension (e.g., GRU hidden state size).
        N: number of nodes (to output 4 parameters per node).
        """
        super(HTQFOutput, self).__init__()
        self.N = N
        # Linear layer to produce N*4 outputs (μ, σ, u, v for each node)
        self.linear = nn.Linear(in_dim, N * 4)
        # Constant A in HTQF formula (could be tuned; using A=1 for simplicity)
        self.A = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (in_dim,) or (1, in_dim) – input features (e.g., last GRU hidden state).
        Returns: Tensor of shape (N, 4) – columns are [μ, σ, u, v] for each node.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # shape (1, in_dim) for linear layer
        params = self.linear(x)  # shape (1, N*4)
        params = params.view(-1, self.N, 4)      # reshape to (1, N, 4)
        params = params.squeeze(0)               # shape (N, 4)
        # Ensure σ is positive (apply softplus activation to second column)
        params[:, 1] = F.softplus(params[:, 1])
        return params

# Full ZF-GCN-MHTQF model architecture
class ZF_GCN_MHTQF_Model(nn.Module):
    def __init__(self, N: int, F_in: int, P: int, d_out: int, M: int = 1):
        """
        ZF-GCN-MHTQF model combining all components.
        N: number of nodes.
        F_in: number of features per node.
        P: window size for STGCN (number of time steps in each input window).
        d_out: total output feature dimension per node from STGCN block (must be even).
        M: power level for spatial graph convolution.
        """
        super(ZF_GCN_MHTQF_Model, self).__init__()
        assert d_out % 2 == 0, "d_out must be even."
        # Example adjacency (identity matrix as placeholder – in practice, use actual or learned adjacency)
        A = torch.eye(N)
        # Initialize sub-modules
        self.zpi = ZigzagPersistenceImage(out_height=6, out_width=6)
        self.flexconv = FlexConv(in_channels=1, out_channels=d_out // 2, kernel_size=3)
        self.stgcn = STGCNBlock(d_in=F_in, d_out=d_out, adjacency=A, P=P, M=M)
        # GRU for temporal dependence across multiple windows; input and hidden size = N * d_out (flattened features)
        self.gru = nn.GRU(input_size=N * d_out, hidden_size=N * d_out, batch_first=True)
        self.htqf = HTQFOutput(in_dim=N * d_out, N=N)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: Tensor of shape (T, N, F_in) – input feature sequence for N nodes over T time steps.
        Returns: Tensor of shape (N, 4) – heavy-tailed quantile function parameters [μ, σ, u, v] for each node.
        """
        T = X.shape[0]
        assert T >= self.stgcn.P, "Sequence length T must be at least P."
        st_outputs = []  # collect STGCN outputs for each time window
        # Slide a window of length P from t-P+1 to t for t from P-1 to T-1
        for t in range(self.stgcn.P - 1, T):
            X_window = X[t - self.stgcn.P + 1 : t + 1]  # shape (P, N, F_in)
            X_current = X[t]                            # shape (N, F_in)
            # Topological features via ZPI and FlexConv
            zpi_image = self.zpi(X_window)              # (1, H, W)
            topo_feat_map = self.flexconv(zpi_image.unsqueeze(0))  # add channel & batch dims: (1,1,H,W)
            topo_feat = topo_feat_map.mean(dim=[2, 3]).squeeze(0)  # global average pool -> (d_out/2,)
            # Spatial-temporal graph convolution block output (N, d_out)
            C_out = self.stgcn(X_current, X_window, topo_feat)
            st_outputs.append(C_out)
        # Prepare sequence of STGCN outputs for GRU (seq_len x (N*d_out)), using batch_size=1
        ST_sequence = torch.stack(st_outputs, dim=0)             # shape (seq_len, N, d_out)
        ST_sequence_flat = ST_sequence.view(1, ST_sequence.size(0), -1)  # (1, seq_len, N*d_out)
        # GRU to model temporal dependencies across windows
        _, h_last = self.gru(ST_sequence_flat)                   # h_last: (num_layers, 1, N*d_out)
        h_last = h_last[-1].flatten()                            # final hidden state, shape (N*d_out,)
        # Output heavy-tailed quantile function parameters for each node
        out_params = self.htqf(h_last)  # shape (N, 4)
        return out_params

# --- Demonstration of the model with synthetic data ---
if __name__ == "__main__":
    # Define dimensions
    N = 5            # number of nodes (e.g., wind farms)
    F_in = 3         # number of features per node
    T = 10           # number of time steps in the input sequence
    P = 3            # window size for STGCN (e.g., use last 3 time steps to predict next)
    d_out = 16       # output feature dimension from STGCN block (must be even, e.g., 16 -> 8 spatial + 8 temporal)
    # Initialize model
    model = ZF_GCN_MHTQF_Model(N=N, F_in=F_in, P=P, d_out=d_out, M=1)
    # Create a dummy input tensor of shape (T, N, F_in)
    X_dummy = torch.randn(T, N, F_in)
    # Forward pass
    output = model(X_dummy)
    print("Output shape:", output.shape)  # expected shape: (N, 4)
