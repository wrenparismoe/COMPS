from System import *

from inputData.data import get_data
from preprocessing.process_data import train_test_split, get_model_name, preprocess
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from tensorflow.keras.layers import Input, Dense, LeakyReLU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from timeit import default_timer as timer


system = SystemComponents()


class DataSequenceGenerator(object):

    def __init__(self,x, y, batch_size, time_steps):
        print()


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


class LSTM():
    def __init__(self):
        self.batch_size = 32
        self.time_steps = 3
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_pred = None
        self.test_pred = None
        self.model = None



    def create_model(self):
        print()


    def plot_model(self):
        plot_model(self.model, show_shapes=True)

    def train_model(self, epochs, optimizer, batch_size):

        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])
        self.model.summary()
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, callbacks=[cb])

    def evaluate_model(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print('training time:', round(sum(cb.logs), 4))
        print(f'test_loss: {loss} --> test_acc: {acc}')
        print(self.model.summary(), '\n')

    def predict_model(self):
        self.train_pred = self.model.predict(self.x_train)
        self.test_pred = self.model.predict(self.x_test)









if __name__ == '__main__':

    if run == 'custom':

        #################################################################

        system.feature_space = 'OHLC'

        system.feature_engineering = 'S'

        system.processor = 'MMS'

        #################################################################

        model_name = 'LSTM'
        model_name = get_model_name(model_name, system)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        cb = TimingCallback()
        for t in market_etfs:
            x, df = get_data(t, system)
            df.name = t
            y = df['Close_Next'][:-1]

            x_transformed = preprocess(x, system)
            df_transformed = pd.merge(x_transformed, y, left_index=True, right_index=True)

            if system.feature_engineering != '':
                x_transformed = select_features(df_transformed, system)

            x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)
            train_size = len(x_train)

            # DataGenerate for training data

            long_short_term_memory = LSTM()
            long_short_term_memory.create_model()
            long_short_term_memory.train_model(250, 'Adam', 32)

            long_short_term_memory.evaluate_model()
            long_short_term_memory.predict_model()

            train_pred = long_short_term_memory.train_pred
            test_pred = long_short_term_memory.test_pred

            results = format_results(df, x_train, x_test, train_pred, test_pred, include_pred_errors)

            forecast_metrics = metrics_list(results[train_size:], include_pred_errors)
            errors.loc[t] = forecast_metrics

            if create_plot:
                if t == etf_to_save:
                    if save_plot:
                        plot_results(results, model_name)
                        break
                plot_results(results, model_name)

        print(model_name + '          features:', len(x_train.columns))
        print(errors)
        errors.to_clipboard(excel=True, index=False, header=False)



    if run == 'basic':
        for data in system.input_list:
            system.feature_space = data
            for proc in system.processor_list:
                system.processor = proc
                model_name = 'LSTM'
                model_name = get_model_name(model_name, system)
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                cb = TimingCallback()
                for t in market_etfs:
                    x, df = get_data(t, system)
                    df.name = t
                    y = df['Close_Next'][:-1]

                    x_transformed = preprocess(x, system)
                    df_transformed = pd.merge(x_transformed, y, left_index=True, right_index=True)

                    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)
                    train_size = len(x_train)


                    long_short_term_memory = LSTM()
                    long_short_term_memory.create_model()
                    long_short_term_memory.train_model(250, 'Adam', 32)

                    long_short_term_memory.evaluate_model()
                    long_short_term_memory.predict_model()

                    train_pred = long_short_term_memory.train_pred
                    test_pred = long_short_term_memory.test_pred

                    results = format_results(df, x_train, x_test, train_pred, test_pred, include_pred_errors)

                    forecast_metrics = metrics_list(results[train_size:], include_pred_errors)
                    errors.loc[t] = forecast_metrics

                    if create_plot:
                        if t == etf_to_save:
                            if save_plot:
                                plot_results(results, model_name)
                                break
                        plot_results(results, model_name)

                print(model_name + '          features:', len(x_train.columns))
                print(errors)
                errors.to_clipboard(excel=True, index=False, header=False)

                print()
                cont = input('continue? - type y:  ')
                if cont == 'y':
                    print()
                    continue
                else:
                    exit()