from System import *

from inputData.data import get_data
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


system = SystemComponents()


tf.config.run_functions_eagerly(True)

def directional_loss(y_true, y_pred):
    # extract the "next day's price" of tensor
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    # extract the "today's price" of tensor
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    # print('Shape of y_pred_back -', y_pred_tdy.get_shape())

    # substract to get up/down movement of the two tensors
    y_true_diff = tf.subtract(y_true_next, y_true_tdy)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)

    # create a standard tensor with zero value for comparison
    standard = tf.zeros_like(y_pred_diff)

    # compare with the standard; if true, UP; else DOWN
    y_true_move = tf.greater_equal(y_true_diff, standard)
    y_pred_move = tf.greater_equal(y_pred_diff, standard)
    y_true_move = tf.reshape(y_true_move, [-1])
    y_pred_move = tf.reshape(y_pred_move, [-1])

    # find indices where the directions are not the same
    condition = tf.not_equal(y_true_move, y_pred_move)
    indices = tf.where(condition)

    # move one position later
    ones = tf.ones_like(indices)
    indices = tf.add(indices, ones)
    indices = K.cast(indices, dtype='int32')

    # create a tensor to store directional loss and put it into custom loss output
    direction_loss = tf.Variable(tf.ones_like(y_pred), dtype='float32')
    updates = K.cast(tf.ones_like(indices), dtype='float32')
    alpha = 1000
    direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices, alpha * updates)

    custom_loss = K.mean(tf.multiply(K.square(y_true - y_pred), direction_loss), axis=-1)

    return custom_loss


# class LongSortTermMemory:
#     def __init__(self):
#         self.x_train = x_t
#         self.y_train = np.array(y_t).transpose()
#         self.x_test = x_test_t
#         self.y_test = np.array(y_test_t).transpose()
#         self.train_pred = None
#         self.test_pred = None
#         self.model = None
#
#     def create_model(self):
#
#         inputs = Input(batch_shape=batch_size, batch_input_shape=(batch_size, time_steps, self.x_train.shape[1]))
#         x = LSTM(100, dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
#                  kernel_initializer='random_uniform')(inputs)
#         x = Dropout(0.4)(x)
#         x = LSTM(60, dropout=0.0)(x)
#         x = Dropout(0.4)(x)
#         x = Dense(20, activation=swish)(x)
#         outputs = Dense(1, activation=swish)(x)
#
#         self.model = Model(inputs, outputs)
#
#     def plot_model(self):
#         plot_model(self.model, show_shapes=True)
#
#     def train_model(self):
#
#         self.model.compile(optimizer=Adam(0.001), loss=directional_loss, metrics=['mean_absolute_percentage_error'])
#         self.model.summary()
#         self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, callbacks=[cb], shuffle=False)
#
#     def evaluate_model(self):
#
#         loss, acc = self.model.evaluate(self.x_test, self.y_test)
#         print('training time:', round(sum(cb.logs), 4))
#         print(f'test_loss: {loss} --> test_acc: {acc}')
#         print(self.model.summary(), '\n')
#
#     def predict_model(self):
#         train_pred = self.model.predict(self.x_train, batch_size=batch_size)
#         train_pred = train_pred.flatten()
#         test_pred = self.model.predict(self.x_test, batch_size=batch_size)
#         test_pred = test_pred.flatten()
#
#         print(train_pred)
#
#         print(test_pred)
#
#         # self.train_pred = pd.Series(train_pred, index=self.x_train.index, name='pred')
#         # self.test_pred = pd.Series(test_pred, index=self.x_test.index, name='pred')


class DataGenerater:
    def __init__(self, time_steps=3, batch_size = 32):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.dict_seq = dict()
        self.dim0 = df.shape[0] - forecast_out
        self.dim1 = df.shape[1] - 1
        self.x_seq = None
        self.y_seq = None
        self.indices = df.index.values

    def dictionary_creator(self):
        x_seq_step = np.full(shape=(self.dim0 - self.time_steps, self.time_steps, self.dim1), fill_value=np.nan, dtype=np.float)
        y_seq_step = y[i] ### do samefor y
        for i in range(self.time_steps - 1, self.dim0 - 1):
            x_step = np.full(shape=(self.time_steps, self.dim1), fill_value=np.nan, dtype=np.float)
            ind = self.indices[i]

            for j in range(self.time_steps):
                x_step[j] = x.iloc[i + j - (self.time_steps - 1)]

            self.dict_seq[ind] = [x_step, y[i]]

            x_seq_step[i - self.time_steps + 1] = x_step

        self.sequence_creator()

    def sequence_creator(self):
        x_seq_temp = 0
        for date, values in self.dict_seq.items():
            exit()




if __name__ == '__main__':
    if run == 'custom':

        #################################################################

        system.feature_space = 'OHLC'

        system.feature_engineering = ''

        system.processor = 'MMS'

        #################################################################

        model_name = 'LSTM'
        model_name = get_model_name(model_name, system)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        for t in market_etfs:
            x, y, df = get_data(t, system)

            data = DataGenerater()
            data.dictionary_creator()

            exit()

"""
            x_transformed = preprocess(x, system)

            if system.feature_engineering != '':
                x_transformed = select_features(x_transformed, y, system)

            x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)



            long_short_term_memory = LongSortTermMemory()
            long_short_term_memory.create_model()
            long_short_term_memory.train_model()

            #long_short_term_memory.evaluate_model()
            long_short_term_memory.predict_model()

            train_pred = long_short_term_memory.train_pred
            test_pred = long_short_term_memory.test_pred

            results = format_results(df, train_pred, test_pred, include_pred_errors)

            forecast_metrics = metrics_list(results[test_index], include_pred_errors)
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
                    
"""