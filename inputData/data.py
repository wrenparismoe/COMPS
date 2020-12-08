from System import *
from pandas_datareader.data import DataReader as DR
from technicalAnalysis.wrapper import add_all_ta_features
from fredapi import Fred
from statsmodels.datasets.macrodata import load_pandas as load_macro

def get_min(d: pd.DataFrame):
    des = d.describe()
    minimum = min(des.loc['min'])
    return minimum

def get_max(d :pd.DataFrame):
    des = d.describe()
    maximum = max((des.loc['max']))
    return maximum

def get_close_df(ticker, new_data=False):
    startDate = '2012-1-01'
    if not new_data:
        endDate = '2020-08-31'
    else:
        endDate = '2020-11-27'

    data = pd.DataFrame(DR(ticker, start=startDate, end=endDate, data_source='yahoo'))

    data = pd.DataFrame(data['Close'])
    data.name = ticker

    return data


def get_ohlc_df(ticker, new_data=False):
    startDate = '2012-1-01'
    #startDate = '2015-01-01'
    if not new_data:
        endDate = '2020-08-31'
    else:
        endDate = '2020-11-27'

    data = pd.DataFrame(DR(ticker, start=startDate, end=endDate, data_source='yahoo'))

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.name = ticker

    return data

def get_ta_df(data: pd.DataFrame):
    close = 'Close'
    open = 'Open'
    volume = 'Volume'
    high = 'High'
    low = 'Low'

    indicator_df = pd.DataFrame(add_all_ta_features(data, open, high, low, close, volume, fillna=True))

    return indicator_df


def get_mv_df(indices, new_data=False):
    api_key = '9ab1f8cf73b430491cb394ccc8ef7af3'
    startDate = '2012-1-01'
    #startDate = '2015-01-01'
    if not new_data:
        endDate = '2020-08-31'
    else:
        endDate = '2020-11-27'

    mv_df = pd.DataFrame(np.nan, columns=['FFR', 'DXY', 'CPI'], index=indices)

    fred = Fred(api_key=api_key)
    mv_df['FFR'] = fred.get_series('FEDFUNDS', observation_start=startDate, observation_end=endDate)
    mv_df['DXY'] = fred.get_series('DTWEXBGS', observation_start=startDate, observation_end=endDate)
    mv_df['CPI'] = fred.get_series('CPIAUCSL', observation_start=startDate, observation_end=endDate)

    mv_df['FFR'].iloc[0] = 0.12
    mv_df['CPI'].iloc[0] = 236.222

    current_val = 0.12
    for ind, val in mv_df['FFR'].iteritems():
        if val >= 0:
            current_val = val
        else:
            mv_df['FFR'].loc[ind] = current_val

    for ind, val in mv_df['DXY'].iteritems():
        if pd.isnull(val):
            loc = mv_df.index.get_loc(ind)
            mv_df['DXY'].loc[ind] = mv_df['DXY'].iloc[loc-1]

    current_val = 236.222
    for ind, val in mv_df['CPI'].iteritems():
        if val >= 0:
            current_val = val
        else:
            mv_df['CPI'].loc[ind] = current_val

    return mv_df



def get_advanced_df(ticker, new_data=False):
    data = get_ohlc_df(ticker, new_data)
    ind = data.index.values

    ta = get_ta_df(data)
    mv = get_mv_df(ind, new_data)

    advanced_df = pd.merge(ta, mv, left_index=True, right_index=True)

    advanced_df.name = ticker
    return advanced_df


def get_data(t, system: SystemComponents, new_data=False):
    df = None
    if system.feature_space == 'C':
        while df is None:
            try:
                df = get_close_df(t, new_data)
            except:
                pass
        df['y'] = df['Close'].shift(-forecast_out)
        y = df[['Close']].shift(-1)
        days = np.arange(start=1, stop=len(df) +1, step=1)
        df['day'] = days
        x = df[['day']][:-1]
        return x, y, df
    else:
        if system.feature_space == 'OHLC':
            while df is None:
                try:
                    df = get_ohlc_df(t, new_data)
                except:
                    pass
        elif system.feature_space == 'OHLCTAMV':
            while df is None:
                try:
                    df = get_advanced_df(t, new_data)
                except:
                    pass
        df['y'] = df['Close'].shift(-forecast_out)
        x = df.drop(['y'], 1)[:-forecast_out].copy()

    y = df['y'][:-forecast_out]
    df.name = t
    x.name = t
    y.name = t

    return x, y, df


def create_labels(data: pd.DataFrame):
    pct_chg = data['Close'].pct_change()
    for ind, p in pct_chg.iteritems():
        if p < 0:
            pct_chg.loc[ind] = 0
        if p > 0:
            pct_chg.loc[ind] = 1

    return pct_chg

def get_class_data(t, system: SystemComponents, new_data=False):
    df = None
    if system.feature_space == 'C':
        while df is None:
            try:
                df = get_close_df(t, new_data)
            except:
                pass
        labels = create_labels(df)
        df['y'] = labels.shift(-forecast_out)


        days = np.array([x for x in range(1, len(df.index.values) + 1)])
        df.insert(loc=0, column='Day', value=days)
        x = df[['Day']][:-forecast_out]
    else:
        if system.feature_space == 'OHLC':
            while df is None:
                try:
                    df = get_ohlc_df(t, new_data)
                except:
                    pass
        elif system.feature_space == 'OHLCTAMV':
            while df is None:
                try:
                    df = get_advanced_df(t, new_data)
                except:
                    pass
        labels = create_labels(df)
        df['y'] = labels.shift(-forecast_out)
        days = np.array([x for x in range(1, len(df.index.values) + 1)])
        df.insert(loc=0, column='Day', value=days)
        x = df.drop(['y'], 1)[:-forecast_out].copy()

    y = df['y'][:-forecast_out]
    df.name = t
    x.name = t
    y.name = t

    return x, y, df









def save_data_files():
    sys = SystemComponents()

    for t in market_etfs:
        for f in sys.input_list:
            sys.feature_space = f

            x, y, data = get_data(t, sys)

            data_name = r'dataFiles\{}'.format(t + '_' + f + '.csv')
            data.to_csv(data_name)




