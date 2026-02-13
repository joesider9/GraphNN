import os
import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm

from eforecast.nwp_extraction.ecmwf_extractor import EcmwfExtractor
from eforecast.common_utils.dataset_utils import fix_timeseries_dates

def interpolate_nwp(nwp, coords, tag):
    mat = []
    for n in range(coords.shape[0]):
        row_x, row_y = coords.iloc[n]['id_lat'], coords.iloc[n]['id_long']
        values = np.array([nwp[tag][x, y] for x in row_x for y in row_y]).reshape(1, -1)
        row_x, row_y = coords.iloc[n]['diff_lat'], coords.iloc[n]['diff_long']
        coef = np.array([x * y for x in row_x for y in row_y]).reshape(1, -1)
        mat.append((coef * values).sum())

    return np.array(mat)

def upsample_dataset(df):
    dates = pd.DatetimeIndex(pd.DataFrame([df.index + pd.DateOffset(minutes=i) for i in range(0, 60, 5)]).melt(value_name='value')[
        'value'].values)
    df_temp = pd.DataFrame(index=dates, columns=df.columns)
    df_temp.loc[df.index] = df
    df_temp = df_temp.bfill(axis=0, limit=6)
    df_temp = df_temp.ffill(axis=0, limit=6)
    return df_temp

def create_wind_ts(t, res, coords):
    ws = []
    wd = []
    if res is None or len(res) == 0:
        for dt in pd.date_range(t, periods=24, freq='h'):
            ws.append(pd.DataFrame(index=[dt], columns=[name for name in coords.index]))
            wd.append(pd.DataFrame(index=[dt], columns=[name for name in coords.index]))
    else:

        print(f'nwp downloaded for {t}')
        nwp = res[list(res.keys())[0]]
        coords['id_lat'] = coords.apply(lambda x: np.argsort(np.abs(nwp['lat'][:, 0] - x['lat']))[:2], axis=1)
        coords['id_long'] = coords.apply(lambda x: np.argsort(np.abs(nwp['long'][0, :] - x['long']))[:2], axis=1)
        coords['diff_lat'] = coords.apply(
            lambda x: [1 - np.abs(nwp['lat'][i, 0] - x['lat']) / 0.1 for i in x['id_lat']], axis=1)
        coords['diff_long'] = coords.apply(
            lambda x: [1 - np.abs(nwp['long'][0, i] - x['long']) / 0.1 for i in x['id_long']], axis=1)
        for dt in pd.date_range(t, periods=24, freq='h'):
            nwp = res[dt.strftime('%d%m%y%H%M')]
            ws_values = interpolate_nwp(nwp, coords, 'WS')
            wd_values = interpolate_nwp(nwp, coords, 'WD')
            ws.append(pd.DataFrame(ws_values.reshape(1, -1), index=[dt], columns=[name for name in coords.index]))
            wd.append(pd.DataFrame(wd_values.reshape(1, -1), index=[dt], columns=[name for name in coords.index]))
    return pd.concat(ws, axis=0), pd.concat(wd, axis=0)

def nwp_extraction(dates, coords, path_nwp):
    if not os.path.exists(os.path.join(path_nwp, 'ecmwf_2023_2024_08_2025.pickle')):
        gfs_extractor = EcmwfExtractor(dates, path_nwp)
        results = gfs_extractor.extract_nwp()
    else:
        results = joblib.load(os.path.join(path_nwp, 'ecmwf_2023_2024_08_2025.pickle'))
    wind = Parallel(n_jobs=10)(delayed(create_wind_ts)(t, res, coords) for t, res in tqdm(results))
    wind_s, wind_d = [], []
    for s, d in wind:
        wind_s.append(s)
        wind_d.append(d)
    return pd.concat(wind_s, axis=0), pd.concat(wind_d, axis=0)

def impute_with_nwp(df_, wind):
    if not isinstance(df_, pd.DataFrame):
        df_ = df_.to_frame()
    if not isinstance(wind, pd.DataFrame):
        wind = wind.to_frame()
    wind = upsample_dataset(wind[~wind.index.duplicated(keep='last')])
    dates_nan = df_.index[df_.isna().any(axis=1)]
    dates_nan = dates_nan.intersection(wind.index)
    wind_nan = wind.loc[dates_nan]
    return wind_nan

def impute_with_neighbors(data, nn_data, dist):
    if not isinstance(data, pd.DataFrame):
        data = data.to_frame()
    dates_nan = data.index[data.isna().any(axis=1)]
    dates_nan = dates_nan.intersection(nn_data.index)
    data.loc[dates_nan] = nn_data.loc[dates_nan].apply(lambda x: (x * dist).dropna().mean(), axis=1).to_frame(data.columns[0])
    return data

def clean_data(data, rated):
    data_new = []
    for tag in data.columns:
        df_orig = data[tag].to_frame(tag)
        df = fix_timeseries_dates(df_orig, freq='1min')
        df[tag] = df[tag].astype(float).interpolate(method='linear', limit_direction='both')
        df15 = df[tag].resample('15min').mean().to_frame(tag)
        df15 = df15.round(3)

        df15_new = df15.copy()
        df15_new = df15_new.rolling(window=4).mean()
        zeros_period = {'MW': 4 * 48, 'SWIN': 4*12, 'DWIN':4*18}
        non_zeros_period = {'MW': 4 * 4, 'SWIN': 4*4, 'DWIN':4*8}
        g = df15_new[tag].ne(df15_new[tag].shift()).cumsum()
        consecutive_dates = df15_new[[tag]][df15_new[tag].groupby(g).transform('count') > zeros_period[tag]].index
        consecutive_dates = consecutive_dates.union(df15_new[[tag]][df15_new[tag].groupby(g).transform('count') >
                                                              non_zeros_period[tag]].index)
        if tag == 'MW':
            index = np.where(df15_new[tag] > rated)[0]
            if len(index) > 5:
                q_max = np.quantile(df15_new[tag].fillna(0).values, 0.99999)
                index = np.where(df15_new[tag] > q_max)[0]
                new_rated = 1.05 * df15_new[tag].max()
            consecutive_dates = consecutive_dates.union(df15_new.index[index])
        for i in range(1, 15):
            consecutive_dates = consecutive_dates.union(consecutive_dates + pd.DateOffset(minutes=i))
        consecutive_dates = consecutive_dates.sort_values()
        df.loc[consecutive_dates.intersection(df.index)] = np.nan
        data_new.append(df[tag])
    data_new = pd.concat(data_new,  axis=1).astype(float)
    return data_new