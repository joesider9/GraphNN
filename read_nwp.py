import pandas as pd
from joblib import Parallel, delayed
import os
from tqdm import tqdm
from data_utils import nwp_extraction
from data_utils import impute_with_nwp

path_sys = '/media/sider/data' if os.path.exists('/media/sider/data') else '/home/smartrue'
path_data = os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')
path_nwp = '/media/sider/data/ECMWF' if os.path.exists('/media/sider') else '/media/smartrue/HHD2/ECMWF'

def read_nwp(dates, coords):
    file_s = os.path.join(path_data, 'wind_speed.csv')
    file_d = os.path.join(path_data, 'wind_direction.csv')
    if not os.path.exists(file_d) and not os.path.exists(file_s):
        ws, wd = nwp_extraction(dates, coords, path_nwp)
        ws.to_csv(file_s)
        wd.to_csv(file_d)
    else:
        ws = pd.read_csv(file_s, index_col=0, parse_dates=True)
        wd = pd.read_csv(file_d, index_col=0, parse_dates=True)
    return ws, wd

def impute_func(name, Swin, Dwin, wsd, wdir):
    swin = impute_with_nwp(Swin, wsd)
    dwin = impute_with_nwp(Dwin, wdir)
    return name, swin, dwin

def impute_missing_values(SWIN, DWIN, coords):
    wind_speed, wind_dir = read_nwp(SWIN.index.union(DWIN.index), coords)
    swin_new = []
    dwin_new = []
    wind_speed = wind_speed[wind_speed.index >= SWIN.index[0] - pd.DateOffset(hours=1)]
    wind_dir = wind_dir[wind_dir.index >= DWIN.index[0] - pd.DateOffset(hours=1)]
    wind_speed = wind_speed[wind_speed.index <= SWIN.index[-1] + pd.DateOffset(hours=1)]
    wind_dir = wind_dir[wind_dir.index <= DWIN.index[-1] + pd.DateOffset(hours=1)]
    results = Parallel(n_jobs=10)(delayed(impute_func)(name, SWIN[name], DWIN[name],
                                   wind_speed[name], wind_dir[name]) for name in tqdm(list(coords.index)))
    # results =[impute_func(name, SWIN[name], DWIN[name],
    #                                    wind_speed[name], wind_dir[name]) for name in tqdm(list(coords.index))]
    for res in results:
        name = res[0]
        ws = res[1]
        wd = res[2]
        SWIN.loc[ws.index, name] = ws[name]
        DWIN.loc[wd.index, name] = wd[name]

    return SWIN, DWIN