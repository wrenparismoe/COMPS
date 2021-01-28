from System import *

from math import floor
from sklearn.preprocessing import MinMaxScaler, maxabs_scale
from preprocessing.transform_data.transformer import PowerTransformer
from statsmodels.tsa.stattools import adfuller
from inputData.data import get_data

def get_df_min(d: pd.DataFrame):
    des = d.describe()
    minimum = min(des.loc['min'])
    return minimum

def get_df_max(d :pd.DataFrame):
    des = d.describe()
    maximum = max((des.loc['max']))
    return maximum

def get_min(series: pd.Series) -> np.float32:
    return series.min().astype(np.float32)

def get_max(series: pd.Series) -> np.float32:
    return series.max().astype(np.float32)



def train_test_split(x: pd.DataFrame, y: pd.Series):


    x_train = x.loc[train_index]
    x_test = x.loc[test_index]

    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    return x_train, x_test, y_train, y_test

def make_positive(data):
    des = data.describe()
    for f in des:
        min = des[f].loc['min']
        if min <= 0:
            data[f] += abs(min)
            data[f] += 1e-5
    return data


def preprocess(x, system: SystemComponents):
    x_transformed = x.copy()
    if system.processor == '':
        system.fitted_processor = None
        return x_transformed
    if system.processor == 'SS':
        p = StandardScaler()
        p = p.fit(x)
        system.fitted_processor = p
        x_transformed = p.transform(x)
    elif system.processor == 'MMS':
        p = MinMaxScaler()
        p = p.fit(x)
        system.fitted_processor = p
        x_transformed = p.transform(x)
    elif system.processor == 'PT':
        if system.distribution is None:
            p = PowerTransformer()
            p = p.fit(x)
            system.fitted_processor = p
            x_transformed = p.transform(x)

        elif system.distribution == 'normal':
            p = PowerTransformer()
            p = p.fit(x)
            system.fitted_processor = p
            x_transformed = p.transform(x)
            x_transformed = maxabs_scale(x_transformed)

    if isinstance(x, pd.DataFrame):
        x_transformed = pd.DataFrame(x_transformed, columns=x.columns, index=x.index)
    return x_transformed

def add_lags(df : pd.DataFrame, lags : int=7):
    for i in range(1, lags+1):
        col_name = 'Close_Lag_{}'.format(i)
        df[col_name] = df['Close'].shift(i)

    df = df.fillna(0)
    return df

def get_model_name(model: str, system: SystemComponents):
    model_name = model + '_' + system.feature_space
    if system.feature_engineering != '':
        model_name = model_name + '_' + system.feature_engineering
    if system.processor != '':
        model_name = model_name + '_' + system.processor

    if system.distribution is not None and system.processor == 'PT':
        model_name = model_name + '-' + system.distribution + ''


    return model_name



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
    return data

def difference(data:pd.DataFrame):
    return data - data.shift(1)

def invert_difference(raw_data, differenced):
    return differenced + raw_data.shift(1)

system = SystemComponents(feature_space='OHLCTAMV')






















