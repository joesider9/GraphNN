import numpy as np
import pandas as pd
import os
import joblib
from einops import rearrange
from read_nwp import impute_missing_values
from tqdm import tqdm
import torch
from ADMIE.configuration.config_input_data import (variables, TARGET_VARIABLE)

path_sys = '/media/sider/data' if os.path.exists('/media/sider/data') else '/home/smartrue'
path_data = os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')

parks = joblib.load(os.path.join(path_data, 'parks_static_info.pickle'))
dates_curt = pd.read_csv(os.path.join(path_data, 'quarters_with_curtailment.csv'), parse_dates=True)
dates_curt = pd.to_datetime(dates_curt['START_OF_15MIN_EET'])
dates_curt = pd.DatetimeIndex(dates_curt.values)

for i in range(1, 15):
    dates_curt = dates_curt.union(dates_curt + pd.DateOffset(minutes=i))
dates_curt_min = dates_curt.sort_values()

file_curt = os.path.join(path_data, 'curt_per_min_202501_202508.csv')
file_curt = pd.read_csv(file_curt, index_col=0, parse_dates=True)
dates_curt_min = dates_curt_min.union(file_curt[file_curt['CURTAILED'] == 1].index)
dates_curt_min = dates_curt_min.sort_values()

def read_data():
    if os.path.exists(os.path.join(path_data, 'swin.csv')):
        swin = pd.read_csv(os.path.join(path_data, 'swin.csv'), index_col=0, parse_dates=True)
        dwin = pd.read_csv(os.path.join(path_data, 'dwin.csv'), index_col=0, parse_dates=True)
        y = pd.read_csv(os.path.join(path_data, 'y.csv'), index_col=0, parse_dates=True)
    else:
        X = []
        y = []
        park_names = []
        for i in range(len(parks)):
            park = parks[i]
            print(park['name'])
            name = park['name']
            file = os.path.join(path_data, name, f'{name}.csv')
            data1 = pd.read_csv(file, index_col=0, header=0, parse_dates=True)
            x = data1[['SWIN', 'DWIN']]
            x.columns = [f'{name}_{col}' for col in x.columns]
            X.append(x)
            y.append(data1['MW'].to_frame(name))
            park_names.append(name)
        X = pd.concat(X, axis=1)
        dates_drop = dates_curt_min.intersection(X.index)
        X = X.drop(dates_drop)
        y = pd.concat(y, axis=1)
        dates_drop = dates_curt_min.intersection(y.index)
        y = y.drop(dates_drop)

        dates = X.index.intersection(y.index)
        X = X.loc[dates].sort_index()
        y = y.loc[dates].sort_index()

        cols_swin = [col for col in X.columns if 'SWIN' in col]
        cols_dwin = [col for col in X.columns if 'DWIN' in col]
        swin, dwin = X[cols_swin], X[cols_dwin]
        swin.columns = ['_'.join(col.split('_')[:-1]) for col in swin.columns]
        dwin.columns = ['_'.join(col.split('_')[:-1]) for col in dwin.columns]
        swin.to_csv(os.path.join(path_data, 'swin.csv'))
        dwin.to_csv(os.path.join(path_data, 'dwin.csv'))
        y.to_csv(os.path.join(path_data, 'y.csv'))
    return swin, dwin, y


def get_item(variable_temp, idx, x_):
    data_arma = []
    for var_name in variable_temp:
        lag = var_name['lags']
        data1 = [torch.unsqueeze(x_[l + idx], dim=-1) for l in lag]
        if var_name['transformer'] is not None:
            data1 = torch.cat(data1, dim=-1).mean(dim=-1, keepdim=True)
            data_arma += [data1]
        else:
            data_arma += data1
    return torch.cat(data_arma, dim=-1)


def get(i, x_, y_, variables):
    try:
        idx = i
        variable_past = [n for n in variables if ('beh' in n['name'] or '-' in n['name']) and 'SWIN' in n['name']]
        past = get_item(variable_past, idx, x_)
        variable_future = [n for n in variables if ('beh' not in n['name'] and '-' not in n['name']) and 'SWIN' in n['name']]
        future = get_item(variable_future, idx, x_)
        target = y_[idx]
    except:
        return None
    if torch.isnan(target).any() or torch.isnan(past).any() or torch.isnan(future).any():
        return None
    return i

if __name__ == '__main__':
    spatial = []
    for i, park in enumerate(parks):
        df = pd.DataFrame([[park['name'], i, park['lat'], park['long']]],
                          columns=['tag', 'id', 'latitude', 'longitude'])
        spatial.append(df)
    spatial = pd.concat(spatial, ignore_index=True)
    spatial.set_index('id', inplace=True)
    swin, dwin, y = read_data()
    if  os.path.exists(os.path.join(path_data, 'data_parks_with_nan.pickle')):
        data = joblib.load(os.path.join(path_data, 'data_parks_with_nan.pickle'))
        y_parks = data['y']
    else:
        park_names = swin.columns
        dates = swin.index.intersection(y.index).intersection(dwin.index).sort_values()
        X_parks = np.stack([swin.loc[dates].values, dwin.loc[dates][park_names].values], axis=0)
        X_parks = rearrange(X_parks, 'fobs B N -> B N fobs')
        y_parks = np.expand_dims(y.loc[dates, park_names].values, axis=-1)
        data = {'dates': dates, 'park_names': park_names, 'X': X_parks, 'y': y_parks, 'spatial': spatial}
        joblib.dump(data, os.path.join(path_data, 'data_parks_with_nan.pickle'))
    # if not  os.path.exists(os.path.join(path_data, 'data_parks_imputed.pickle')):
    #     coordinates = pd.concat([pd.DataFrame([(p['lat'], p['long'])], index=[p['name']], columns=['lat', 'long'])
    #                              for p in parks])
    #
    #     if (not os.path.exists(os.path.join(path_data, 'SWIN_nwp_2023_2025.pickle')) and
    #             not os.path.exists(os.path.join(path_data, 'DWIN_nwp_2023_2025.pickle'))):
    #         swin, dwin = impute_missing_values(swin, dwin, coordinates)
    #     else:
    #         swin = pd.read_csv(os.path.join(path_data, 'SWIN_nwp_2023_2025.csv'), index_col=0, parse_dates=True)
    #         dwin = pd.read_csv(os.path.join(path_data, 'DWIN_nwp_2023_2025.csv'), index_col=0, parse_dates=True)
    #     park_names = swin.columns
    #     dates = swin.index.intersection(y.index).intersection(dwin.index).sort_values()
    #     X_parks = np.stack([swin[park_names].values, dwin[park_names].values], axis=0)
    #     X_parks = rearrange(X_parks, 'fobs B N -> B N fobs')
    #     data = {'dates': dates, 'park_names': park_names, 'X': X_parks, 'y': y_parks, 'spatial': spatial}
    #     joblib.dump(data, os.path.join(path_data, 'data_parks_imputed.pickle'))
    # else:
    #     data = joblib.load(os.path.join(path_data, 'data_parks_imputed.pickle'))
    #     park_names = data['park_names']
    #     dates = data['dates']
    #     X_parks = data['X']
    #     y_parks = data['y']
    #     # for i in range(10):
    #     #     file = os.path.join(path_data, f'predictions_2023_2025_{i}.csv')
    #     #     df = pd.read_csv(file, index_col=0, parse_dates=True)
    #     #     df = df[park_names]
    #     #     dates_join = dates.intersection(df.index)
    #     #     indices = dates.get_indexer(dates_join)
    #     #     if len(dates_join) > 0:
    #     #         df_indices = df.index.get_indexer(dates_join)
    #     #         nan_mask = np.isnan(y_parks[indices])
    #     #         for park_idx in range(y_parks.shape[1]):
    #     #             park_feature_nan_mask = nan_mask[:, park_idx, 0]
    #     #
    #     #             if np.any(park_feature_nan_mask):
    #     #                 df_values = df.iloc[df_indices, park_idx].values
    #     #                 valid_df_mask = ~np.isnan(df_values)
    #     #
    #     #                 # Only replace where both X_parks has NaN and df has valid values
    #     #                 replace_mask = park_feature_nan_mask & valid_df_mask
    #     #                 if np.any(replace_mask):
    #     #                     y_parks[indices[replace_mask], park_idx, 0] = df_values[replace_mask]
    #     variables_template = variables()
    #     dates_nan = pd.date_range(dates[0], dates[-1], freq='5min')
    #     X_new = np.empty([dates_nan.shape[0], X_parks.shape[1], X_parks.shape[2]])
    #     indices = dates_nan.get_indexer(dates)
    #     X_new[indices] = X_parks
    #     y_new = np.empty([dates_nan.shape[0], y_parks.shape[1], y_parks.shape[2]])
    #     y_new[indices] = y_parks
    #     X_new = torch.from_numpy(X_new)
    #     y_new = torch.from_numpy(y_new)
    #     indices_new = [i for i in tqdm(range(X_new.shape[0])) if get(i, X_new, y_new, variables_template) is not None]
    #     data = {'dates': dates, 'indices': indices_new, 'park_names': park_names, 'X': X_new.numpy(), 'y': y_new.numpy(), 'spatial': spatial}
    #     joblib.dump(data, os.path.join(path_data, 'data_parks_imputed.pickle'))




