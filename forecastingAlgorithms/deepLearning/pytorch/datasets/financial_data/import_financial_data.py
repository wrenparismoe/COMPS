from config import fred_api_key
from pandas_datareader.data import DataReader as DR
from System import *
from technicalAnalysis.wrapper import add_all_ta_features
from datetime import datetime
from time import time_ns
import requests


startDate = '2012-01-01'
#startDate = '2015-01-01'
endDate = '2021-01-01'
Dates = None

def get_date_index():
    return Dates

def parse_date(date_str: str, format='%Y-%m-%d'):
    rv = datetime.strptime(date_str, format)
    rv = np.datetime64(rv)
    return rv

def close_data(ticker):
    data = DR(ticker, start=startDate, end=endDate, data_source='yahoo')
    data = data['Close']
    return data

def ohlc_data(ticker):
    data = DR(ticker, start=startDate, end=endDate, data_source='yahoo')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def technical_analysis_data(ticker):
    data = DR(ticker, start=startDate, end=endDate, data_source='yahoo')
    global Dates
    Dates = data.index
    open, high, low, close, volume = 'Open', 'High', 'Low', 'Close', 'Volume'
    indicator_df = add_all_ta_features(data, open, high, low, close, volume, fillna=True).drop('Adj Close', axis=1)
    return indicator_df

def get_MV(series_id: str):
    params = {"api_key": fred_api_key,
              "file_type": 'json',
              "series_id": series_id,
              "observation_start": startDate,
              "observation_end": endDate,
              }

    root_url = 'https://api.stlouisfed.org/fred'
    url = root_url + '/series/observations'

    req = requests.get(url, params=params)
    decoded_req = req.json()
    observations = decoded_req['observations']

    empty_series = pd.Series(np.nan, index=Dates, name='E', dtype='float32')
    quarterly = pd.Series(data=[obs.get('value') for obs in observations],
                              index=[parse_date(obs.get('date')) for obs in observations], name=series_id)

    quarterly = quarterly.replace(to_replace=[np.nan, '.'], method='ffill').replace(to_replace=[np.nan, '.'],
                                                                                    method='bfill').astype('float32')

    mv_df = pd.concat([empty_series, quarterly], axis=1, join='outer').drop(labels='E', axis=1).replace(to_replace=np.nan, method='ffill')
    mv_df = pd.Series(mv_df[series_id], name=series_id, index=empty_series.index, dtype='float32')

    return mv_df

def macroeconomic_data():
    FFR = get_MV('FEDFUNDS')
    DXY = get_MV('DTWEXBGS')
    CPI = get_MV('CPIAUCSL')

    macro = pd.concat([FFR, DXY, CPI], axis=1, join='outer')
    return macro

def get_advanced_df(ticker, start_date=startDate):
    global startDate
    startDate = start_date
    ta = technical_analysis_data(ticker)
    mv = macroeconomic_data()
    advanced_df = pd.concat([ta, mv], axis=1, join='outer')
    advanced_df.name = ticker
    return advanced_df


def save_data_files():
    market_etfs = ['SPY', 'QQQ', 'IWM', 'DJI']
    for t in market_etfs:
        x, y, data = get_advanced_df(t)

        data_name = r'C:\Users\wrenp\Documents\COMPS\dataFiles\{}'.format(t + '.csv')
        data.to_csv(data_name, columns=data.columns)






