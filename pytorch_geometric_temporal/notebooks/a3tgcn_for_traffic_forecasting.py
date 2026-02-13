import os
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN as DCRNN

from torch_geometric_temporal.nn.recurrent import A3TGCN2, GConvLSTM, DyGrEncoder, MPNNLSTM
from torch.utils.data import DataLoader
from ADMIE.configuration.config_input_data import (variables, TARGET_VARIABLE)
from torch_geometric_temporal.dataset.admie_ds import ADMIEDataset


class TemporalGNN(torch.nn.Module):
    def __init__(self, nn_method='A3TGCN2'):
        super(TemporalGNN, self).__init__()
        if nn_method == 'A3TGCN2':
            self.layer = A3TGCN2(in_channels=2,  out_channels=256, periods=36, batch_size=28) # node_features=2, periods=12
            self.linear = torch.nn.Linear(2 * 256, 36)
        elif nn_method == 'DCRNN':
            self.layer = DCRNN(2, 128, 3)
            self.linear = torch.nn.Linear(2 * 128, 36)
        else:
            raise ValueError(f"Method {nn_method} not supported")
        # Equals single-shot prediction


    def forward(self, x, temp_edge_index, temp_edge_weight, edge_index, edge_weights):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.layer(x, temp_edge_index, temp_edge_weight, edge_index, edge_weights) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)
        return h

# GPU support
DEVICE = torch.device('cuda') # cuda
shuffle=False
batch_size = 28
park_group_name = 'group1'
#%%
path_sys = '/media/sider/data' if os.path.exists('/media/sider/data') else '/home/smartrue'
path_data = os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')

parks = joblib.load(os.path.join(path_data, 'parks_static_info.pickle'))
park_group = joblib.load(os.path.join(path_data, 'park_group_availability.pickle'))
#%%
spatial = []
for i, park in enumerate(parks):
    df = pd.DataFrame([[park['name'], i, park['lat'], park['long']]],
                      columns=['name', 'id', 'latitude', 'longitude'])
    spatial.append(df)
spatial = pd.concat(spatial, ignore_index=True)
spatial.set_index('name', inplace=True)
edges = []
edges_weights = []
for i, park in enumerate(parks):
    coordinates = spatial.loc[park['name']]
    variables_names =  park['features'].index.tolist()
    check = []
    for feature in variables_names:
        tag = feature.split('_')[-1]
        name = '_'.join(feature.split('_')[:-1])
        coord_temp = spatial.loc[name]
        dist = np.sqrt((coordinates['latitude'] - coord_temp['latitude']) ** 2 + (
                    coordinates['longitude'] - coord_temp['longitude']) ** 2)
        if dist < 1 and name not in check:
            edges.append([i, coord_temp['id']])
            edges_weights.append(5 * (1 - dist))
            check.append(name)
edges = np.array(edges).T
edges_weights = np.array(edges_weights).T



rated = np.array([p['rated'] for p in parks])
park_names = [p['name'] for p in parks]

variables_template = variables()


def collate_fn(batch):
    fn = list(filter(lambda x: x[0] is not None if isinstance(x, tuple) else x is not None, batch))
    if len(fn) == 0:
        return None, None, None, None
    past, target, lag_edge_index, lag_edge_weight = zip(*fn)
    return (torch.utils.data.default_collate(past),
            torch.utils.data.default_collate(target),
            torch.utils.data.default_collate(lag_edge_index),
            torch.utils.data.default_collate(lag_edge_weight))



loss_fn = torch.nn.MSELoss()
edges = torch.from_numpy(edges).type(torch.LongTensor).to(DEVICE)
edges_weights = torch.from_numpy(edges_weights).type(torch.FloatTensor).to(torch.float32).to(DEVICE)
methods = ['A3TGCN2', 'DCRNN']
for method in methods:
    data_struct = 'lstm' if method not in ['DCRNN', 'DyGrEncoder'] else 'graph'
    dataset_train = ADMIEDataset(
        path_data,
        48,
        36,
        12,
        rated,
        park_group[park_group_name],
        task='train', data_struct=data_struct)
    dataset_val = ADMIEDataset(
        path_data,
        48,
        36,
        12,
        rated,
        park_group[park_group_name],
        task='val', data_struct=data_struct)
    dataset_test = ADMIEDataset(
        path_data,
        48,
        36,
        12,
        rated,
        park_group[park_group_name],
        task='test', data_struct=data_struct)
    dataloader_train = DataLoader(dataset_train, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)
    model = TemporalGNN(nn_method=method).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    validation = []
    best = np.nan
    for epoch in tqdm(range(80)):
        model.train()
        loss_list = 0
        n = 0
        for past, labels, lag_edge_index, lag_edge_weight in tqdm(dataloader_train):
            if past is None:
                continue
            past = past.to(torch.float32).to(DEVICE)
            lag_edge_index = lag_edge_index.to(DEVICE)
            lag_edge_weight = lag_edge_weight.to(DEVICE)
            labels = labels.to(torch.float32).to(DEVICE)
            y_hat = model(past, lag_edge_index, lag_edge_weight, edges, edges_weights)         # Get model predictions
            loss = loss_fn(y_hat, labels)
            # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.2)
            optimizer.step()
            optimizer.zero_grad()
            loss_list += loss.item()
            n += 1
        print("Epoch {} train RMSE: {:.4f}".format(epoch, loss_list / n))
        model.eval()
        val_loss = []
        for past, labels, lag_edge_index, lag_edge_weight in tqdm(dataloader_val):
            if past is None:
                continue
            past = past.to(torch.float32).to(DEVICE)
            lag_edge_index = lag_edge_index.to(DEVICE)
            lag_edge_weight = lag_edge_weight.to(DEVICE)
            labels = labels.to(torch.float32).to(DEVICE)
            y_hat = model(past, lag_edge_index, lag_edge_weight, edges, edges_weights)
            loss = torch.abs(y_hat - labels).mean(dim=0)
            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.concatenate(val_loss, axis=-1).mean(axis=1)
        validation.append(pd.DataFrame(val_loss, index=park_names, columns=['val_mae']))
        print("val_loss MAPE: {:.4f}".format(np.mean(val_loss)))
        if best > np.sum(val_loss):
            best_weights = model.state_dict()
            best_tot_iteration = epoch
            best = np.sum(val_loss)
            torch.save(best_weights, f'{method}_weights.pt')
    pd.concat(validation, axis=1).to_csv(f'val_loss_{method}.csv')


    model.eval()
    step = 0
    # Store for analysis
    test_loss = []
    for past, labels, lag_edge_index, lag_edge_weight in tqdm(dataloader_test):
        if past is None:
            continue
        past = past.to(torch.float32).to(DEVICE)
        lag_edge_index = lag_edge_index.to(DEVICE)
        lag_edge_weight = lag_edge_weight.to(DEVICE)
        labels = labels.to(torch.float32).to(DEVICE)
        y_hat = model(past, lag_edge_index, lag_edge_weight, edges, edges_weights)
        loss = torch.abs(y_hat - labels).mean(dim=0)
        test_loss.append(loss.detach().cpu().numpy())
    test_loss = np.concatenate(test_loss, axis=-1).mean(axis=1)
    print("test_loss MAPE: {:.4f}".format(np.mean(test_loss)))
    pd.DataFrame(test_loss, index=park_names, columns=['val_mae']).to_csv(f'test_loss_{method}.csv')