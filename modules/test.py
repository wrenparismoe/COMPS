from System import *

from inputData.data import get_close_df, get_data

system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='MutualInfo', processor='PT')

data = get_data('SPY', system, True)



print(data)

