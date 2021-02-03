from System import *

from inputData.data import get_class_data
from preprocessing.process_data import get_model_name, preprocess
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, LeakyReLU, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import tensorflow.keras.backend as K

import pickle

"""
LSTM model for classification in TensorFlow under development
"""

def create_data_seq(x, y):
    x_seq, y_seq = [], []
    for i in range(len(x), - time_steps - 1):
        x.append(x[i: i + time_steps,:])
        y.append(y[i + time_steps,:])
    return (np.array(x), np.array(y))


class DataGenerater:
    def __init__(self, time_steps=3, batch_size = 32):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.dim0 = df.shape[0] - forecast_out
        self.dim1 = df.shape[1] - 1
        self.x_seq = None
        self.y_seq = None

    def dictionary_creator(self):
        x_seq = []
        y_seq = []
        for i in range(self.time_steps - 1, self.dim0 - 1):
            x_step = np.full(shape=(self.time_steps, self.dim1), fill_value=np.nan, dtype=np.float)


            for j in range(self.time_steps):
                x_step[j] = x.iloc[i + j - (self.time_steps - 1)]

            x_seq.append(x_step)
            y_seq.append(y[i])

        self.x_seq = np.array(x_seq)
        self.y_seq = np.array(y_seq)

system = SystemComponents()

time_steps = 5


if run == 'custom':

    #################################################################

    system = SystemComponents(feature_space='OHLC', feature_engineering='', processor='MMS', distribution=None)
    system.model_type = 'class'
    #################################################################

    model_name = 'LSTMC'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        print(t)
        x, y, df = get_class_data(t, system)
        x = x.to_numpy()
        y = np.reshape(y.to_numpy(), (-1, 1))

        x_transformed = preprocess(x, system)

        if not system.feature_engineering == '':
            x_transformed = select_features(x_transformed, y, system)


        DG = DataGenerater()
