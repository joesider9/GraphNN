import os
import joblib
import numpy as np
import pandas as pd
import torch

path_sys = '/media/sider/data' if os.path.exists('/media/sider/data') else '/home/smartrue'
path_data = os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')

data = joblib.load(os.path.join(path_data, 'data_parks_imputed.pickle'))
X = torch.from_numpy(data['X'][..., :1]).float()
farms  = X.shape[1]
indices = data['indices']

spatial = data['spatial']
park_names = data['park_names']
nlags = 48
n_per_hour = 12

correlation = torch.zeros((X.shape[0], nlags - n_per_hour, farms, farms))
for idx in indices:
    for i in range(farms):
        for j in range(i, farms):
            for h in range(nlags - n_per_hour):
                correlation[idx, h, i, j] = torch.corrcoef(torch.stack([X[idx:idx - n_per_hour, i, 0],
                                                           X[idx - h:idx - h - n_per_hour, j, 0]]))[0, 1]
