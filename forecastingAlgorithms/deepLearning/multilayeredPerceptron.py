from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.activations import relu, swish
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tensorflow.python.keras import backend as K
import math




class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

system = SystemComponents()


def plot_accuracy(history, model_title):
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title(model_title)
    plt.ylabel('MAPE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def max_absolute(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred), axis=0)

#####################################################

epochs = 500

batch_size = 16

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
        d1 = Dense(64, activation=lrelu, name='Dense_1')(inputs)
        d2 = Dense(32, activation=lrelu, name='Dense_2')(d1)
        outputs = Dense(1, activation=lrelu, name='Output')(d2)

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
        stopper = EarlyStopping(monitor='mean_absolute_percentage_error', min_delta=0.0001, patience=20, mode='min',
                                verbose=1, restore_best_weights=True)

        self.model.compile(optimizer=Adam(0.001), loss='mean_absolute_percentage_error', metrics=['mean_absolute_percentage_error'])
        self.model.summary()

        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                callbacks=[self.modelTimer, marketTimer, stopper], validation_data=(self.x_test, self.y_test))

        self.epochs = len(history.history['mean_absolute_percentage_error'])
        plot_accuracy(history, self.network_name)

    def evaluate_model(self):
        print()
        print('Epochs:', self.epochs)
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print(t, 'training time:', round(sum(self.modelTimer.logs), 4))
        print(f'test_loss: {loss} --> test_acc: {acc}' + '\n')


    def predict_model(self):
        train_pred = self.model.predict(x_train)
        test_pred = self.model.predict(x_test)

        predictions_2d = [train_pred, test_pred]
        predictions_1d = []

        for preds in predictions_2d:
            preds_temp = np.full(len(preds), np.nan, dtype=np.float64)
            i = 0
            for p in preds:
                preds_temp[i] = np.float64(p)
                i += 1
            predictions_1d.append(preds_temp)

        train_pred = predictions_1d[0]
        test_pred = predictions_1d[1]

        self.train_pred = pd.Series(train_pred, index=self.x_train.index, name='pred')
        self.test_pred = pd.Series(test_pred, index=self.x_test.index, name='pred')


if __name__ == '__main__':
    marketTimer = TimingCallback()

    if run == 'custom':
        #################################################################
        system.feature_space = 'OHLCTAMV'

        system.feature_engineering = ''

        system.processor = 'MMS'

        system.distribution = None
        #################################################################

        model_name = 'MLP'
        model_name = get_model_name(model_name, system)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        cb = TimingCallback()
        epochs_list = []
        for t in market_etfs:
            x, y,  df = get_data(t, system)

            x_transformed = preprocess(x, system)

            if system.feature_engineering != '':
                x_transformed = select_features(x_transformed, y, system)

            x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

            y_train = pd.DataFrame.to_numpy(y_train)
            y_test = pd.DataFrame.to_numpy(y_test)

            multilayered_perceptron = MLP()
            multilayered_perceptron.create_model()
            multilayered_perceptron.train_model()
            multilayered_perceptron.evaluate_model()
            multilayered_perceptron.predict_model()

            train_pred = multilayered_perceptron.train_pred
            test_pred = multilayered_perceptron.test_pred

            results = format_results(df, train_pred, test_pred, include_pred_errors)
            forecast_metrics = metrics_list(results.loc[x_test.index.values], include_pred_errors)
            errors.loc[t] = forecast_metrics
            epochs_list.append(multilayered_perceptron.epochs)

            if create_plot:
                plot_results(results, model_name)

        print('Total training time:', round(sum(marketTimer.logs), 4), 'seconds')
        print('Epochs:', epochs_list)
        print(model_name + '          features:', x_train.shape[1])
        print(errors)
        errors.to_clipboard(excel=True, index=False, header=False)



    if run == 'basic':
        for data in system.input_list:
            system.feature_space = data
            for proc in system.processor_list:
                system.processor = proc
                model_name = 'MLP'
                model_name = get_model_name(model_name, system)
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                cb = TimingCallback()
                epochs = []
                for t in market_etfs:
                    x, df = get_data(t, system)
                    features = x.columns
                    label = 'Close_Next'

                    x_transformed = preprocess(x, system)
                    y = pd.DataFrame.to_numpy(df['Close_Next'][:-1])

                    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, shuffle=True, train_size=0.75)

                    multilayered_perceptron = MLP()
                    multilayered_perceptron.create_model()
                    multilayered_perceptron.train_model()

                    multilayered_perceptron.evaluate_model()
                    multilayered_perceptron.predict_model()

                    train_pred = multilayered_perceptron.train_pred
                    test_pred = multilayered_perceptron.test_pred

                    results = format_results(df, train_pred, test_pred, include_pred_errors)

                    forecast_metrics = metrics_list(results.loc[x_test.index.values], include_pred_errors)
                    errors.loc[t] = forecast_metrics
                    epochs.append(multilayered_perceptron.epochs)

                    if create_plot and t == etf_to_save:
                        plot_results(results, model_name)

                print('Total training time:', round(sum(marketTimer.logs), 4), 'seconds')
                print('Epochs:', epochs)
                print(model_name + '          features:', x_train.shape[1])
                print(errors)
                errors.to_clipboard(excel=True, index=False, header=False)

                print()
                cont = input('continue? - type y:  ')
                if cont == 'y':
                    print()
                    continue
                else:
                    exit()

    if run == 'derived':
        system.feature_space = 'OHLCTAMV'
        for f_e in system.feature_engineering_list:
            system.feature_engineering = f_e
            for proc in system.processor_list:
                system.processor = proc
                model_name = 'MLP'
                model_name = get_model_name(model_name, system)
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                cb = TimingCallback()
                epochs = []
                for t in market_etfs:
                    x, df = get_data(t, system)
                    features = x.columns
                    label = 'Close_Next'

                    x_transformed = preprocess(x, system)
                    y = pd.DataFrame.to_numpy(df['Close_Next'][:-1])

                    x_transformed = select_features(x_transformed, y, system)

                    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)
                    y_train = pd.DataFrame.to_numpy(y_train)
                    y_test = pd.DataFrame.to_numpy(y_test)

                    multilayered_perceptron = MLP()
                    multilayered_perceptron.create_model()
                    multilayered_perceptron.train_model()

                    multilayered_perceptron.evaluate_model()
                    multilayered_perceptron.predict_model()

                    train_pred = multilayered_perceptron.train_pred
                    test_pred = multilayered_perceptron.test_pred

                    results = format_results(df, train_pred, test_pred, include_pred_errors)

                    forecast_metrics = metrics_list(results[train_index], include_pred_errors)
                    errors.loc[t] = forecast_metrics
                    epochs.append(multilayered_perceptron.epochs)

                    if create_plot:
                        if t == etf_to_save:
                            if save_plot:
                                plot_results(results, model_name)
                                break
                        plot_results(results, model_name)

                print('Total training time:', round(sum(marketTimer.logs), 4), 'seconds')
                print('Epochs:', epochs)
                print(model_name + '          features:', x_train.shape[1])
                print(errors)
                errors.to_clipboard(excel=True, index=False, header=False)

                print()
                cont = input('continue? - type y:  ')
                if cont == 'y':
                    print()
                    continue
                else:
                    exit()


