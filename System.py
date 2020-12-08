import pandas as pd
import numpy as np
import warnings
import os
from config import *
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from preprocessing.transform_data.transformer import PowerTransformer
import math

studio_params = {'username': username, 'api_key': api_key}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

#warnings.filterwarnings("ignore")

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

forecast_out = 1

train_index = pd.to_datetime(train_index_df.index.values, format='%Y/%m/%d')
test_index = pd.to_datetime(test_index_df.index.values, format='%Y/%m/%d')


######################################################

class SystemComponents:
    def __init__(self, feature_space='', feature_engineering='', processor='', distribution=None):
        self.input_list = ['OHLC', 'OHLCTAMV']
        #self.input_list = ['OHLCTAMV']

        self.feature_engineering_list = ['Pearson', 'Spearman', 'MutualInfo', 'PCA']
        #self.feature_engineering_list = ['Spearman', 'MutualInfo', 'PCA']
        #self.feature_engineering_list = ['PCA', 'SAE']
        #self.feature_engineering_list = ['PCA']
        #self.feature_engineering_list = ['SAE']

        self.model_type = None

        f_e = []

        self.processor_list = ['', 'SS', 'MMS', 'PT']
        #self.processor_list = ['SS', 'MMS', 'PT']
        #self.processor_list = ['', 'MMS', 'PT']
        #self.processor_list = ['MMS', 'PT'] # cannot use no raw data with SVR
        #self.processor_list = ['', 'PT'] # MMS does nothing for LR-OLS
        #self.processor_list = ['PT']
        #self.processor_list = ['']

        self.feature_space = feature_space
        self.feature_engineering = feature_engineering
        self.processor = processor

        self.p = None

        self.distribution_list = [None, 'normal']
        self.distribution = distribution

        self.fitted_processor = None
        self.selected_features = None
        self.fitted_pca = None

    def get_processor(self):
        if self.processor == '':
            return None
        if self.processor == 'SS':
            return StandardScaler()
        if self.processor == 'MMS':
             return MinMaxScaler()
        elif self.processor == 'PT':
            return PowerTransformer()



run_list = ['basic', 'derived', 'custom']
run = run_list[2]

dimension = 32

include_pred_errors = True

create_plot = True
show_plot = True
save_plot = False
etf_to_save = market_etfs[0]


######################################################


if include_pred_errors:
    # errors = pd.DataFrame(np.zeros((len(market_etfs), 5)), columns=["MSE", "MAPE", "sMAPE", "r2", "Prediction %"],
    #                       index=market_etfs)
    errors = pd.DataFrame(np.zeros((len(market_etfs), 6)), columns=["ME", "MAE", "MAPE", "RMSE", "Prediction %", 'Pred Incr %'],
                          index=market_etfs)
else:
    errors = pd.DataFrame(np.zeros((len(market_etfs), 4)), columns=["MSE", "MAPE", "sMAPE", "r2"], index=market_etfs)
errors.index.name = 'Market'


classifier_errors = pd.DataFrame(np.zeros((len(market_etfs), 5)), columns=["Accuracy", "Precision", "Recall", "f1", "fbeta"],
                          index=market_etfs)
