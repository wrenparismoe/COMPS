from fredapi import Fred
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', None)


fred_api_key = '08961e24c1992580d5de16becb5865e7'
api_key = '9ab1f8cf73b430491cb394ccc8ef7af3'




fred = Fred(api_key=api_key)
ffr = fred.get_series('FEDFUNDS', observation_start='2015-01-01', observation_end='2020-11-03')
dxy = fred.get_series('DTWEXBGS', observation_start='2015-01-01', observation_end='2020-11-03')
cpi = fred.get_series('CPIAUCSL', observation_start='2015-01-01', observation_end='2020-11-03')

