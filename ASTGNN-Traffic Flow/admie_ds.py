import os
import numpy as np
import joblib
from einops import rearrange, repeat
import torch
from torch.utils.data import Dataset


class ADMIEDataset(Dataset):
    def __init__(self, path_data, variables, targ_variable, rated, task='train', data_struct='lstm'):
        self.path_data = path_data
        data = joblib.load(os.path.join(path_data, 'data_parks_with_nan.pickle'))
        self.park_names = data['park_names']
        self.dates = data['dates']
        self.X = dict()
        norm_x = np.array([30, 360])[np.newaxis, np.newaxis, :]
        self.X = torch.from_numpy(np.clip(np.nan_to_num(data['X'], 0) / norm_x, 0, 1)).float()
        norm_y = np.array(rated)[np.newaxis, :, np.newaxis]
        self.y = torch.from_numpy(np.clip(np.nan_to_num(data['y'],0) / norm_y, 0, 1)).float()
        self.spatial = data['spatial']
        n_lags = 48
        n_pred_lags = 36
        self.indices = [i for i in range(n_lags + 1, self.X.shape[0] - n_pred_lags - 1) if
                        not torch.isnan(self.X[i - n_lags:i + n_pred_lags]).any()
                        and not torch.isnan(self.y[i - n_lags:i + n_pred_lags]).any()]
        self.targ_variable = targ_variable
        self.variables = variables
        self.data_struct = data_struct
        if task == 'train':
            self.indices = self.indices[:int(len(self.indices) * 0.7)]
        elif task == 'val':
            self.indices = self.indices[int(len(self.indices) * 0.7):int(len(self.indices) * 0.85)]
        elif task == 'test':
            self.indices = self.indices[int(len(self.indices) * 0.85):]

    def __len__(self):
        return len(self.indices)

    def get_item(self, variable_temp, idx):
        data_arma = []
        for var_name in variable_temp:
            lag = var_name['lags']
            data1 = [torch.unsqueeze(self.X[l + idx], dim=-1) for l in lag]
            if var_name['transformer'] is not None:
                data1 = torch.cat(data1, dim=-1).mean(dim=-1, keepdim=True)
                data_arma += [data1]
            else:
                data_arma += data1
        return torch.cat(data_arma, dim=-1)


    def __getitem__(self, i):
        idx = self.indices[i]
        variable_past = [n for n in self.variables if ('beh' in n['name'] or '-' in n['name']) and 'SWIN' in n['name']]
        past = self.get_item(variable_past, idx)
        ind = [idx + l for l in self.targ_variable['lags']]
        target = rearrange(self.y[ind].squeeze(-1), 'N l -> l N')
        ind = [idx + l - 1 for l in self.targ_variable['lags']]
        decoder_input = rearrange(self.y[ind].squeeze(-1), 'N l -> l N')
        if torch.isnan(target).any() or torch.isnan(decoder_input).any() or torch.isnan(past).any():
            return None, None, None
        return past.flip(dims=(0,)), decoder_input, target

