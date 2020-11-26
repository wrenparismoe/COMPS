from System import *

from math import floor
from sklearn.preprocessing import MinMaxScaler, maxabs_scale
from preprocessing.transform_data.transformer import PowerTransformer

def get_min(d: pd.DataFrame):
    des = d.describe()
    minimum = min(des.loc['min'])
    return minimum

def get_max(d :pd.DataFrame):
    des = d.describe()
    maximum = max((des.loc['max']))
    return maximum


def train_test_split(x: pd.DataFrame, y: pd.DataFrame):


    x_train = x.loc[train_index]
    x_test = x.loc[test_index]

    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    return x_train, x_test, y_train, y_test

def scale(x, p):
    if p == None:
        return x
    x_transformed = p.fit_transform(x)
    return x_transformed


def preprocess(x, system: SystemComponents):
    x_transformed = x.copy()
    if system.processor == '':
        return x_transformed
    else:
        if system.processor == 'MMS':
            p = MinMaxScaler()
            x_transformed = scale(x, p)
        elif system.processor == 'PT':
            if system.distribution is None:
                p = PowerTransformer()
                x_transformed = scale(x, p)

            elif system.distribution == 'normal':
                p = PowerTransformer()
                x_transformed = scale(x, p)
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





























