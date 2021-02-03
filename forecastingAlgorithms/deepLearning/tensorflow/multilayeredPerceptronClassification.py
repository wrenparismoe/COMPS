from System import *

from inputData.data import get_class_data
from preprocessing.process_data import get_model_name, preprocess
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import class_metrics_list, format_results
from modules.plot import plot_class_results

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.activations import relu, swish, sigmoid, tanh
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tensorflow.python.keras import backend as K
import math
import sys

"""
MLP model for classification in TensorFlow/Keras under development
"""

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

#####################################################

epochs = 100

batch_size = 32

#####################################################

class MLP:
    def __init__(self):
        self.network_name = model_name + '-' + t
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_pred = None
        self.test_pred = None
        self.model = None
        self.modelTimer = None


    def create_model(self):
        dim = self.x_train.shape[1]

        lrelu = lambda x: LeakyReLU(alpha=0.001)(x)

        inputs = Input(shape=(dim, ), name='Input')
        d1 = Dense(64, activation=relu, name='Dense_1')(inputs)
        d2 = Dense(32, activation=relu, name='Dense_2')(d1)
        outputs = Dense(1, activation=sigmoid, name='Output')(d2)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.network_name)

        # inputs = Input(shape=(dim,), name='Input')
        # d = Dense(dim, activation=swish)(inputs)
        # outputs = Dense(1)(d)

    def plot_model(self):
        plot_model(self.model, show_shapes=True)

    def train_model(self):
        global epochs
        global batch_size
        self.modelTimer = TimingCallback()
        # stopper = EarlyStopping(monitor='mean_absolute_percentage_error', min_delta=0.0001, patience=20, mode='min',
        #                         verbose=1, restore_best_weights=True)


        self.model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                callbacks=[self.modelTimer, marketTimer], validation_data=(self.x_test, self.y_test))

        self.epochs = len(history.history['accuracy'])
        #plot_accuracy(history, self.network_name)

    def evaluate_model(self):
        print()
        print('Epochs:', self.epochs)
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print(t, 'training time:', round(sum(self.modelTimer.logs), 4))
        print(f'test_loss: {loss} --> test_acc: {acc}' + '\n')


    def predict_model(self):
        test_pred = self.model.predict(x_test)


        preds = np.full(len(test_pred), np.nan, dtype=np.float)
        i = 0
        for p in test_pred:
            preds[i] = np.float(p)
            i += 1

        self.test_pred = pd.Series(preds, index=self.x_test.index, name='pred')



marketTimer = TimingCallback()

if run == 'custom':
    #################################################################
    system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='', processor='MMS',
                              distribution=None)
    system.model_type = 'class'
    #################################################################

    model_name = 'MLPC'
    model_name = get_model_name(model_name, system)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    cb = TimingCallback()
    epochs_list = []
    for t in market_etfs:
        x, y,  df = get_class_data(t, system)



        x_transformed = preprocess(x, system)

        if system.feature_engineering != '':
            x_transformed = select_features(x_transformed, y, system)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, shuffle=False, train_size=0.75)

        y_train = pd.DataFrame.to_numpy(y_train).astype(int)
        y_test = pd.DataFrame.to_numpy(y_test).astype(int)

        multilayered_perceptron = MLP()
        multilayered_perceptron.create_model()
        multilayered_perceptron.train_model()
        multilayered_perceptron.evaluate_model()
        multilayered_perceptron.predict_model()

        y_pred = multilayered_perceptron.test_pred
        y_test = pd.Series(y_test, index=x_test.index.values)
        print(y_test)
        print(y_pred)

        results = format_results(df, y_test, y_pred)
        metrics = class_metrics_list(y_test, y_pred)
        classifier_errors.loc[t] = metrics
        epochs_list.append(multilayered_perceptron.epochs)

        # if create_plot:
        #     plot_class_results(results, model_name)

    print('Total training time:', round(sum(marketTimer.logs), 4), 'seconds')
    print('Epochs:', epochs_list)
    print(model_name + '          features:', x_train.shape[1])
    print(classifier_errors)
    classifier_errors.to_clipboard(excel=True, index=False, header=False)

