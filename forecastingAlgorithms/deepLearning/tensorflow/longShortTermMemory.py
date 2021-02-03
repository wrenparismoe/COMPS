from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results


from tensorflow.keras.layers import Input, Dense, LeakyReLU, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import pickle

tf.random.set_seed(11)
np.random.seed(11)

"""
LSTM model for regression in TensorFlow under development
"""

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




class DataGenerater:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, time_steps=3, batch_size=32, train_ratio=0.75):
        self.x = x
        self.y = y
        self.x_seq = None
        self.y_seq = None
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.train_size = None
        self.processor = system.get_processor()
        self.sequence_creator()

    def sequence_creator(self):
        rows = self.x.shape[0]
        cols = self.x.shape[1]
        x_seq = []
        y_seq = []
        for i in range(self.time_steps, rows):
            x_step = np.full(shape=(self.time_steps, cols), fill_value=np.nan)
            for j in range(self.time_steps):
                x_step[j] = self.x.iloc[i - self.time_steps + j].values
            x_seq.append(x_step)
            y_seq.append(self.y[i])

        if not self.seq_match(x_seq, y_seq):
            print('Data generated incorrectly')
            exit()


        self.x_seq = np.array(x_seq, dtype='float32')
        self.y_seq = np.reshape(np.array(y_seq, dtype='float32'), newshape=(rows-self.time_steps,))

    def seq_match(self, x, y):
        return len(x) == len(y)

    def process_training(self):
        x_train = self.x_seq[:self.train_size].copy()
        if self.processor is None:
            return x_train
        else:
            processors = {}
            dim = x_train.shape[2]
            for i in range(dim):
                col = self.x_seq[:, :, i]
                processors[i] = self.processor.fit(col)
                x_train[:, :, i] = processors[i].transform(col)
            return x_train

    def get_step(self, step):
        x_train = self.x_seq[:self.train_size + step].copy()
        x_test = self.x_seq[self.train_size + step].copy()
        print('train')
        print(x_train)
        print('test')
        print(x_test)
        print(len(self.x))
        exit()
        if self.processor is None:
            return x_train

        else:
            processors = {}
            dim = x_train.shape[2]
            for i in range(dim):
                train_col = x_train[:, :, i]
                test_col = x_test[:, :, i]
                processors[i] = self.processor.fit(train_col)
                x_train[:, :, i] = processors[i].transform(train_col)
                x_test[:, :, i] = processors[i].transform(test_col)

            return x_train, x_test


class LongSortTermMemory:
    def __init__(self, time_steps=3, batch_size=32):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.model = None

    def create_model(self):
        inputs = Input(batch_shape=self.batch_size, batch_input_shape=(self.batch_size, x_train.shape[1], x_train.shape[2]))
        x = LSTM(100, activation=relu)(inputs)
        x = Dropout(0.2)(x)
        # x = LSTM(100, activation=relu)(x)
        # x = Dropout(0.2)(x)
        # x = Dense(100, activation=swish)(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs, outputs)

    def plot_model(self):
        plot_model(self.model, show_shapes=True)

    def train_model(self):
        file_path = 'saved_models/model_epoch_{epoch:02d}.hdf5'
        cp = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.compile(optimizer=Adam(0.001), loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])
        self.model.summary()
        self.model.fit(x_train, y_train, epochs=100, batch_size=self.batch_size, callbacks=[cp],
                       validation_data=(x_test, y_test), verbose=1, shuffle=False)

    def evaluate_model(self):

        loss, acc = self.model.evaluate(x_test, y_test)
        print(f'test_loss: {loss} --> test_acc: {acc}')
        print(self.model.summary(), '\n')

    def predict_model(self):
        return self.model.predict(x_test)





if run == 'custom':

    #################################################################

    system = SystemComponents(feature_space='OHLC', feature_engineering='', processor='MMS', distribution=None)

    #################################################################

    model_name = 'LSTM'
    model_name = get_model_name(model_name, system)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    for t in market_etfs:
        x, y, df = get_data(t, system)

        DG = DataGenerater(x, y)
        x = DG.process_training()
        y = DG.y_seq
        train_size = math.floor(x.shape[0] * 0.75)
        train_start = df.index.get_loc('2018-02-26')

        long_short_term_memory = LongSortTermMemory()
        long_short_term_memory.create_model()


        for i in range(train_start, x_test.shape[0]):
            x_train, x_test = DG.get_step(i)

            if system.feature_engineering != '':
                x_transformed = select_features(x_transformed, y, system)

            x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)


            long_short_term_memory.train_model()

            long_short_term_memory.evaluate_model()
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
                    
