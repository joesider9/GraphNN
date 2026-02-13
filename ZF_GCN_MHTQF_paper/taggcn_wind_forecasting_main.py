from tagcn_wind_forecasting import (
    TimeAwareGCN, TAGCNTrainer, GraphConstructor,
    WindPowerDataset, adjacency_to_edge_index
)

# 1. Build graph from farm coordinates
graph_constructor = GraphConstructor(sigma=20.0)
adj_matrix = graph_constructor.build_combined_adjacency(coordinates, power_data)
edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)

# 2. Create datasets
dataset = WindPowerDataset(power_data, features, seq_len=24, forecast_horizon=24)

# 3. Initialize and train model
model = TimeAwareGCN(num_nodes=N, input_dim=F, forecast_horizon=24)
trainer = TAGCNTrainer(model, device='cuda')
trainer.train(train_loader, val_loader, edge_index, edge_weight)