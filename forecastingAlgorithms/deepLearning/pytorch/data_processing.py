import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series):
    adtest = adfuller(series, autolag='AIC')
    test_val = adtest[0]
    for key,value in adtest[4].items():
       if value < test_val:
           return False

    return True

def feature_difference(data: pd.DataFrame):
    for f in data:
        series = data[f]
        stationary = adf_test(series)
        while not stationary:
            series = difference(series)
            series = series.dropna()
            stationary = adf_test(series)
        data[f] = series
    return data.dropna(axis=0)

def difference(data:pd.DataFrame):
    # todays price is now today - yesterday
    return data - data.shift(1)

