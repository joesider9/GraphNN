import os
import joblib
import pandas as pd
import numpy as np
from boruta import BorutaPy
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler





file_x = '/media/smartrue/HHD1/George/models/PPC/PPC_sat_ver0/pv/Ptolemaida/multi-output/model_ver0/DATA/dataset_row_data.pickle'
file_y = '/media/smartrue/HHD1/George/models/PPC/PPC_sat_ver0/pv/Ptolemaida/multi-output/model_ver0/DATA/dataset_target_data.csv'
if __name__ == '__main__':
    data_x = joblib.load(file_x)['row_stats'].iloc[:, 45:]
    data_y = pd.read_csv(file_y, index_col=0, header=0, parse_dates=True)
    #%%

    dates = data_x.dropna(axis='index', how='any').index.intersection(data_y.dropna(axis='index', how='any').index)
    x = data_x.loc[dates].values
    y = data_y.loc[dates].values
    #%%

    scaler = MinMaxScaler()
    x_ = scaler.fit_transform(x)
    scaler_y = MinMaxScaler()
    y_ = scaler_y.fit_transform(y)

    #%%
    lgb = LGBMRegressor(n_estimators=100, max_depth=6)
    boruta =  BorutaPy(lgb, n_estimators='auto', verbose=0)
    boruta.fit(x_, y_[:, 0])
    print(data_x.columns[boruta.support_])