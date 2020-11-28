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
import sys




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



tf.config.run_functions_eagerly(True)

@tf.function
def print(tensor):
    tf.print(tensor, summarize=-1)

@tf.autograph.experimental.do_not_convert
def pred_loss(y_true, y_pred):

    y_true_last = y_true[:-1,]
    y_true = y_true[1:]
    y_pred = y_pred[1:]

    print(tf.concat([tf.concat([y_pred, y_true], axis=1), y_true_last], axis=1))

    true_diff = tf.subtract(y_true, y_true_last)
    pred_diff = tf.subtract(y_pred, y_true_last)

    true_chg = tf.divide(true_diff, y_true_last)
    pred_chg = tf.divide(pred_diff, y_true_last)

    true_chg_sign = tf.sign(true_chg)
    pred_chg_sign = tf.sign(pred_chg)

    sign_tensor = tf.multiply(true_chg_sign, pred_chg_sign)
    one_n = tf.constant(-1, dtype=tf.float32)

    failed_pred_bools = tf.equal(sign_tensor, one_n)
    failed_pred_ints = tf.cast(failed_pred_bools, dtype=tf.float32)

    print(failed_pred_ints)

    failed_preds = tf.math.multiply(failed_pred_ints, pred_chg)
    failed_preds = tf.abs(failed_preds)
    failed_preds = tf.multiply(failed_preds, tf.constant(100, dtype=tf.float32))

    #####################################################################

    failed_true = tf.multiply(failed_pred_ints, y_true)
    failed_predictions = tf.multiply(failed_pred_ints, y_pred)

    index_nonzero = tf.where(tf.not_equal(failed_true, tf.constant(0, dtype=tf.float32)), None, None)
    index_nonzero = tf.squeeze(index_nonzero)

    failed_true = tf.gather(failed_true, index_nonzero)
    failed_predictions = tf.gather(failed_predictions, index_nonzero)

    failed_mse = K.mean(tf.square(tf.subtract(failed_true, failed_predictions)))

    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    mape = 100. * K.mean(diff)

    failed_diff = K.abs((failed_true - failed_predictions) / K.clip(K.abs(failed_true), K.epsilon(), None))
    failed_mape = 100. * K.mean(failed_diff, axis=1)

    exit()
    return failed_mape

    # return K.math_ops.multiply(failed_mape, mape)


def directional_loss(y_true, y_pred):
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]
    # extract the "today's price" of tensor
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    # tf.print('Shape of y_pred_back -', y_pred_tdy.get_shape())

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

    # print(direction_loss)
    # tf.print(direction_loss.get_shape(), end='\n')
    # print(indices)
    # tf.print(indices.get_shape(), end='\n')
    # print(alpha * updates)
    #
    # exit()

    direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices, alpha*updates)

    return K.mean(tf.multiply(K.square(y_true - y_pred), direction_loss), axis=-1)

#####################################################

epochs = 500

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
        d1 = Dense(32, activation=swish, name='Dense_1')(inputs)
        d2 = Dense(16, activation=swish, name='Dense_2')(d1)
        outputs = Dense(1, activation=swish, name='Output')(d2)

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


        self.model.compile(optimizer=Adam(0.01), loss=directional_loss, metrics=['mean_absolute_percentage_error'])
        self.model.summary()

        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                callbacks=[self.modelTimer, marketTimer, stopper], validation_data=(self.x_test, self.y_test))

        self.epochs = len(history.history['mean_absolute_percentage_error'])
        #plot_accuracy(history, self.network_name)

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
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
                    x, y, df = get_data(t, system)


                    x_transformed = preprocess(x, system)

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


