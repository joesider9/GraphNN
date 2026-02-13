import numpy as np


def get_data(lag, var_data, data, data_pl):
    data_temp = []

    freq = '5min'
    if isinstance(lag, int) or isinstance(lag, np.integer):
        if static_data['use_polars']:
            data_temp = data_pl.select(col).shift(-lag)
            data_temp.columns = [f'{var_name}_lag_{lag}']
        else:
            data_temp = data[col].shift(-lag).to_frame()
            data_temp.columns = [f'{var_name}_lag_{lag}']

        data_temp = lylags
    return data_temp


def concat_lagged_data(data, var_name, var_data, lstm_lags=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.sort_index()
    data = fix_timeseries_dates(data, freq='5min')
    dates = pd.date_range(data.index[0], data.index[-1],
                          freq='5min')
    data_temp = pd.DataFrame(index=dates)
    if lstm_lags is not None:
        var_lags = []
        for l in lstm_lags:
            if isinstance(l, int) or isinstance(l, np.integer):
                if l < 0 or l in var_data['lags']:
                    var_lags.append(l)
            else:
                var_lags.append(l)
    else:
        var_lags = var_data['lags']
    if len(var_lags) == 0:
        return None
    data_pl = pl.from_pandas(data)
    results = [get_data(lag, var_data, data, data_pl) for lag in var_lags]
    # with Pool(5) as pool:
    #     results = pool.map(partial(get_data, var_data=var_data, data=data, data_pl=data_pl), var_lags)
    data_temp = pd.concat(results, axis=1)
    data_temp.index = data.index

    if var_data['use_diff_between_lags']:
        diff_df = []
        for lag1 in var_lags:
            for lag2 in var_lags:
                if isinstance(lag1, str) or isinstance(lag2, str):
                    continue
                if np.abs(lag1) > 3 or np.abs(lag2) > 3:
                    continue
                if lag1 > lag2:
                    diff = data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}']
                    diff = diff.to_frame(f'Diff_{var_name}_lag{lag1}_lag{lag2}')
                    diff2 = np.square(diff)
                    diff2.columns = [f'Diff2_{var_name}_lag{lag1}_lag{lag2}']
                    diff_df.append(pd.concat([diff, diff2], axis=1))
        data_temp = pd.concat([data_temp] + diff_df, axis=1)
    if var_data['transformer'] == 'mean':
        data_temp = data_temp.mean(axis=1).to_frame(var_name)
    elif var_data['transformer'] == 'max':
        data_temp = data_temp.max(axis=1).to_frame(var_name)
    elif var_data['transformer'] == 'min':
        data_temp = data_temp.min(axis=1).to_frame(var_name)
    elif var_data['transformer'] == 'median':
        data_temp = data_temp.median(axis=1).to_frame(var_name)
    elif var_data['transformer'] == 'std':
        data_temp = data_temp.std(axis=1).to_frame(var_name)
    elif var_data['transformer'] == 'sum':
        data_temp = data_temp.sum(axis=1).to_frame(var_name)
    return data_temp


def wrap_lagged_data(var_name, var_data, data, lag_lstm):
    data_temp = concat_lagged_data(data, var_name, var_data[var_name], lstm_lags=lag_lstm)
    if data_temp is None:
        return None
    data_temp = data_temp.dropna(axis='index', how='any')
    return data_temp


def create_autoregressive_dataset(data, variables, time_lags='past'):
    lags = []
    for var_name in variables:
        for l in var_name['lags']:
            if l <= 0 and time_lags == 'past':
                lags.append(np.abs(l))
            elif l > 0 and time_lags == 'future':
                lags.append(l)
            else:
                raise ValueError(f'Invalid time lag: {l} and time_lags: {time_lags}')
    mlag = max(lags) + 1
    data_arma = []
    for var_name in variables:
        lag = var_name['lags']
        if var_name['transformer'] is not None:
            data1 = ([np.expand_dims(data[mlag + l: l - 1], axis=-1) for l in lag if l <= 0] +
                     [np.expand_dims(data[l: -(mlag - l)], axis=-1) for l in lag if l > 0])
            data1 = np.concatenate(data1, axis=-1).mean(axis=-1, keepdims=True)
            data_arma += [data1]
        else:
            data1 = ([np.expand_dims(data[mlag + l: l - 1], axis=-1) for l in lag if l <= 0] +
                     [np.expand_dims(data[l: -(mlag - l)], axis=-1) for l in lag if l > 0])
            data_arma += data1
    return np.concatenate(data_arma, axis=-1)