import os
import numpy as np
from ADMIE.configuration.config_project import config_project
from ADMIE.configuration.config_utils import *

static_data = config_project()
path_owner = os.path.join(static_data['sys_folder'], static_data['project_owner'])
path_data = os.path.join(path_owner, f"{static_data['projects_group']}_ver{static_data['version_group']}", 'DATA')

NWP_MODELS = static_data['NWP']
NWP = NWP_MODELS is None

TYPE = static_data['type']
ts_resolution = 0.25 if static_data['ts_resolution'] == '15min' else 1

NWP_DATA_MERGE = ['all']  # 'all', 'by_area', 'by_area_variable', #! THIS IS NOT EMPTY. SET [None] INSTEAD
# 'by_variable',#! THIS IS NOT EMPTY. SET [None] INSTEAD
# by_nwp_provider#! THIS IS NOT EMPTY. SET [None] INSTEAD
#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_COMPRESS = ['dense']  # dense or semi_full or full or load #! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_IMG = [None]#! THIS IS NOT EMPTY. SET [None] INSTEAD

DATA_IMG_SCALE = [None]#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_NWP_SCALE = ['minmax'] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_ROW_SCALE = ['minmax'] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD

DATA_TARGET_SCALE = 'maxabs' #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD

USE_DATA_BEFORE_AND_AFTER_TARGET = False

REMOVE_NIGHT_HOURS = False

USE_POLARS = True

HORIZON = static_data['horizon']
HORIZON_TYPE = static_data['horizon_type']

## TRANSFORMER FEATURES

GLOBAL_PAST_LAGS = None
GLOBAL_FUTURE_LAGS = None

TIME_MERGE_VARIABLES = {}
if HORIZON_TYPE == 'multi-output':
    targ_lags = [int(i) for i in range(int(HORIZON / ts_resolution))]
else:
    targ_lags = [0]
targ_tag = 'Step' if ts_resolution == 0.25 else 'Hour'

TARGET_VARIABLE = {'name': 'MW',
                   'source': 'MW',
                   'lags': targ_lags if HORIZON_TYPE == 'multi-output' else [0],
                   'columns': [f'{targ_tag}_{i}' for i in targ_lags]
                                                       if HORIZON_TYPE == 'multi-output'else ['target'],
                   'transformer': None,
                   'transformer_params': None
                   }
## LAGs for NWP and Images are hourly steps

def variables():
    if TYPE == 'pv':
        pass

    elif TYPE == 'wind':
        # Labels for NWP variables: Uwind, Vwind, WS, WD, Temperature

        variable_list = []
        for tag in ['SWIN', 'DWIN']:
            var_obs = variable_wrapper(f'{tag}_ahead', input_type='timeseries', source='target',
                                       lags=[i for i in range(1, 12)],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
            var_obs = variable_wrapper(f'{tag}_behind', input_type='timeseries', source='target',
                                       lags=[-i for i in range(12)],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
            for stst in ['mean']:
                var_obs = variable_wrapper(f'{tag}_{stst}-15', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(12, 14)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)

                var_obs = variable_wrapper(f'{tag}_{stst}-30', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(14, 16)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-45', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(16, 18)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-1h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(18, 20)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-1h15', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(20, 22)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-1h30', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(22, 24)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-2h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(24, 26)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-3h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(26, 29)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}-4h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[-i for i in range(29, 32)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+15', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(12, 14)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+30', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(14, 16)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+45', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(16, 18)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+1h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(18, 20)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+1h15', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(20, 22)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+1h30', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(22, 24)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+2h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(24, 26)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+3h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(26, 29)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
                var_obs = variable_wrapper(f'{tag}_{stst}+4h', input_type='timeseries', source='target',
                                           transformer=f'{stst}',
                                           lags=[i for i in range(29, 32)],
                                           timezone=static_data['local_timezone'])
                variable_list.append(var_obs)
        # if HORIZON > 0:
        #     var_obs = variable_wrapper('Obs', input_type='timeseries', source='target', lags=3,
        #                                timezone=static_data['local_timezone'])
        #     variable_list.append(var_obs)
    elif TYPE == 'FA':
        lags1 = [-int(i) for i in range(1, 7)]
        lags2 = [-int(i) for i in range(7, 11)] + [-int(i) for i in range(14, 16)] + [-int(i) for i in range(21, 23)]

        lags_pred = [int(i) for i in range(2)]
        variable_list = [
            variable_wrapper('Final/Ζητούμενο', input_type='timeseries', source='target', lags=lags2,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Athens_24', input_type='timeseries', source='target', lags=lags1,
                             timezone=static_data['local_timezone'], use_diff_between_lags=True),
            # variable_wrapper('Athens_6', input_type='timeseries', source='target',
            #                  timezone=static_data['local_timezone']),
            variable_wrapper('temp_max', input_type='timeseries', source='target', lags=lags_pred + lags1,
                             timezone=static_data['local_timezone']),
            variable_wrapper('temp_min', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('temp_mean', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('rh', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('precip', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('hdd_h', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('hdd_h2', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_max', input_type='timeseries', source='nwp_dataset',
                             lags=[1, 0, -1, -2, -3],
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_min', input_type='timeseries', source='nwp_dataset',
                             lags=[1, 0, -1, -2, -3],
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temperature', nwp_provider='ALL'),
            variable_wrapper('Cloud', nwp_provider='ALL'),
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ALL'),
            variable_wrapper('dayweek', input_type='calendar', source='index', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('sp_index', input_type='calendar', source='index', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('month', input_type='calendar', source='index', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            # variable_wrapper('dayofyear', input_type='calendar', source='index', lags=lags_pred,
            #                  timezone=static_data['local_timezone']),
        ]
    elif TYPE == 'load':
        if HORIZON > 0:
            lags = [-int(i) for i in range(1, 13)] + [-int(i) for i in range(22, 28)] + [-int(i) for i in range(47, 53)] + \
                   [-int(i) for i in range(166, 176)] + [-192]

            lags_days = [-int(24 * i)  for i in range(0, 8)]
        else:
            if HORIZON_TYPE == 'day-ahead':
                lags = [-int(i) for i in range(48, 60)] + [-int(i) for i in range(72, 77)] + [-int(i) for i in range(96, 100)] + \
                       [-int(i) for i in range(120, 122)] + [-int(i) for i in range(144, 146)] + [-int(i) for i in range(166, 176)] + \
                       [-int(i) for i in range(190, 192)] + [-216]  # + ['last_year_lags']
            else:
                lags = [-int(i) for i in range(24, 36)] + [-int(i) for i in range(48, 54)] + [-int(i) for i in range(72, 77)] + \
                       [-int(i) for i in range(96, 100)] + \
                       [-int(i) for i in range(120, 122)] + [-int(i) for i in range(144, 146)] + [-int(i) for i in range(166, 176)] + \
                       [-int(i) for i in range(190, 192)] + [-216]  # + ['last_year_lags']

            lags_days = [-int(24 * i)  for i in range(0, 8)]

        variable_list = [
            variable_wrapper('load', input_type='timeseries', source='target', lags=lags,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_max', input_type='timeseries', source='nwp_dataset', lags=lags_days,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp', input_type='timeseries', source='nwp_dataset',
                             lags=[0, -1, -2, -3, -24, -48, -168],
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temperature', nwp_provider='ALL'),
            variable_wrapper('Cloud', nwp_provider='ALL'),
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ALL'),
            variable_wrapper('dayweek', input_type='calendar', source='index',
                             timezone=static_data['local_timezone']),
            variable_wrapper('sp_index', input_type='calendar', source='index', lags=lags_days,
                             timezone=static_data['local_timezone']),
            variable_wrapper('hour', input_type='calendar', source='index',
                             timezone=static_data['local_timezone']),
            variable_wrapper('month', input_type='calendar', source='index',
                             timezone=static_data['local_timezone'])
        ]
    else:
        raise NotImplementedError(f'Define variables for type {TYPE}')
    return variable_list


def variable_wrapper(name, input_type='nwp', source='grib', lags=None, timezone='UTC', nwp_provider=None,
                     transformer=None, transformer_params=None, bands=None, use_diff_between_lags=False):
    if nwp_provider is not None:
        if nwp_provider == 'ALL':
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS]
        else:
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS if nwp_model['model'] == nwp_provider]
    else:
        providers = None

    return {'name': name,
            'type': input_type,  # nwp or timeseries or calendar
            'source': source,  # use 'target' for the main timeseries otherwise 'grib', 'database' for nwps,
            # 'nwp_dataset' to get data from created nwp dataset,
            # a column label of input file csv or a csv file extra, 'index' for calendar variables,
            # 'astral' for zenith, azimuth
            'lags': define_variable_lags(name, input_type, lags),
            'timezone': timezone,
            'transformer': transformer,
            'transformer_params': transformer_params,
            'bands': bands,
            'nwp_provider': providers,
            'use_diff_between_lags': use_diff_between_lags
            }


def define_variable_lags(name, input_type, lags):
    if lags is None or lags == 0:
        lags = [0] if HORIZON_TYPE != 'multi-output' else [int(i) for i in range(int(HORIZON / ts_resolution))]
    elif isinstance(lags, int):
        lags = [-int(i) for i in range(int(lags / ts_resolution))]
    elif isinstance(lags, list):
        pass
    else:
        raise ValueError(f'lags should be None or int or list')
    if name in {'Flux', 'wind'}:
        if USE_DATA_BEFORE_AND_AFTER_TARGET:
            if HORIZON == 0:
                max_lag = np.max(lags)
                min_lag = np.min(lags)
                lags = [min_lag - 1] + lags + [max_lag + 1]
    return lags


def config_data():
    static_input_data = {'nwp_data_merge': NWP_DATA_MERGE,
                         'compress_data': DATA_COMPRESS,
                         'img_data': DATA_IMG,
                         'use_data_before_and_after_target': USE_DATA_BEFORE_AND_AFTER_TARGET,
                         'remove_night_hours': REMOVE_NIGHT_HOURS,
                         'variables': variables(),
                         'target_variable': TARGET_VARIABLE,
                         'time_merge_variables': TIME_MERGE_VARIABLES,
                         'global_past_lags': GLOBAL_PAST_LAGS,
                         'global_future_lags': GLOBAL_FUTURE_LAGS,
                         'scale_row_method': DATA_ROW_SCALE,
                         'scale_img_method': DATA_IMG_SCALE,
                         'scale_nwp_method': DATA_NWP_SCALE,
                         'scale_target_method': DATA_TARGET_SCALE,
                         'use_polars': USE_POLARS
                         }
    return static_input_data
