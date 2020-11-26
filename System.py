import pandas as pd
import numpy as np
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

warnings.filterwarnings("ignore")

np.set_printoptions(edgeitems=100, linewidth=1000)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', None)


market_etfs = ['SPY', 'QQQ', 'IWM', 'DIA']
#market_etfs = ['SPY', 'QQQ']
#market_etfs = ['QQQ']

train_index_df = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/train_index.csv', index_col='Date')
test_index_df = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/test_index.csv', index_col='Date')

forecast_out = 7

train_index = pd.to_datetime(train_index_df.index.values, format='%Y/%m/%d')
test_index = pd.to_datetime(test_index_df.index.values, format='%Y/%m/%d')


######################################################

class SystemComponents:
    def __init__(self):
        #self.input_list = ['C', 'OHLC', 'OHLCTAMV']
        self.input_list = ['OHLC', 'OHLCTAMV']
        #self.input_list = ['OHLCTAMV']

        self.feature_engineering_list = ['Pearson', 'MutualInfo', 'PCA', 'SAE']
        #self.feature_engineering_list = ['MutualInfo', 'PCA', 'SAE']
        #self.feature_engineering_list = ['PCA', 'SAE']
        #self.feature_engineering_list = ['MutualInfo']
        #self.feature_engineering_list = ['SAE']

        self.processor_list = ['', 'MMS', 'PT']
        #self.processor_list = ['MMS', 'PT'] # cannot use no raw data with SVR
        #self.processor_list = ['', 'PT'] # MMS does nothing for LR-OLS
        #self.processor_list = ['PT']
        #self.processor_list = ['']

        self.feature_space = ''
        self.feature_engineering = ''
        self.processor = ''

        self.distribution_list = [None, 'normal']
        self.distribution = self.distribution_list[0]


run_list = ['basic', 'derived', 'custom']
run = run_list[2]

dimension = 32

include_pred_errors = True

create_plot = True
show_plot = True
save_plot = True
etf_to_save = market_etfs[1]


######################################################


if include_pred_errors:
    errors = pd.DataFrame(np.zeros((len(market_etfs), 5)), columns=["ME", "MAE", "MAPE", "RMSE", "Prediction %"],
                          index=market_etfs)
else:
    errors = pd.DataFrame(np.zeros((len(market_etfs), 4)), columns=["ME", "MAE", "MAPE", "RMSE"], index=market_etfs)
errors.index.name = 'Market'
