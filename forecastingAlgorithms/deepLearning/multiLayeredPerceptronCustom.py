from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Lambda, Layer
from tensorflow.keras.activations import relu, swish
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredError, MeanSquaredLogarithmicError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import tensorflow_probability as tfp
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.keras import activations, initializers, optimizers
import math

np.random.seed(7)


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


@tf.autograph.experimental.do_not_convert
def successful_pred_pct(y_true, y_pred):
    y_true_last = y_true[:-1,]
    y_true = y_true[1:]
    y_pred = y_pred[1:]

    size = tf.math.count_nonzero(tf.squeeze(y_true), dtype=tf.float32)

    true_diff = tf.math.subtract(y_true, y_true_last)
    pred_diff = tf.math.subtract(y_pred, y_true_last)

    true_chg = tf.math.divide(true_diff, y_true_last)
    pred_chg = tf.math.divide(pred_diff, y_true_last)

    true_chg_sign = tf.math.sign(true_chg)
    pred_chg_sign = tf.math.sign(pred_chg)

    sign_tensor = tf.math.multiply(true_chg_sign, pred_chg_sign)
    one = tf.constant(1, dtype=tf.float32)

    successful_pred_bools = tf.squeeze(tf.math.equal(sign_tensor, one))
    successful_pred_ints = tf.cast(successful_pred_bools, dtype=tf.float32)



    successful_preds_count = tf.math.count_nonzero(successful_pred_ints, dtype=tf.float32)

    percent_successful = tf.math.divide(successful_preds_count, size)

    percent_successful = tf.math.multiply(percent_successful, tf.constant(100, dtype=tf.float32))

    return percent_successful

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def max_absolute(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred), axis=0)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    mape = 100. * K.mean(diff, axis=-1)
    return mape

@tf.autograph.experimental.do_not_convert
def pred_loss(y_true, y_pred):

    y_true_last = y_true[:-1,]
    y_true = y_true[1:]
    y_pred = y_pred[1:]

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

    return failed_mape

    # return K.math_ops.multiply(failed_mape, mape)



class DenseVariational(Layer):
    def __init__(self, units, kl_weight, activation=None, prior_sigma_1=1.5, prior_sigma_2=0.1, prior_pi=0.5, **kwargs):
        self.units = units
# Below lines follow hypertuning stats, edit when needed for prior sigma, prior sigma 2, and prior pi
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 + self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

# Computes the output shape of layer
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

# Defines the kernals and biases
    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu', shape=(input_shape[1], self.units),
                                         initializer=initializers.RandomNormal(stddev=self.init_sigma), trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', shape=(self.units,),
                                       initializer=initializers.RandomNormal(stddev=self.init_sigma), trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', shape=(input_shape[1], self.units),
                                          initializer=initializers.Constant(0.0), trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', shape=(self.units,),
                                        initializer=initializers.Constant(0.0), trainable=True)
        super().build(input_shape)

# Calculations :)
    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

# The variational distribution we will be using, don't change!
    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

# The more calculations! Can change depending on how well things go, if so, becareful. Mostly for scaling if necessary
    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) + self.prior_pi_2 * comp_2_dist.prob(w))



#####################################################

epochs = 1000

batch_size = 100

num_batches = 1100 / batch_size
noise = 1

# !!! Hypertuning variables (not k1_weight) !!!
kl_weight = 1.0 / num_batches
prior_params = {'prior_sigma_1': 1.5, 'prior_sigma_2': 0.1, 'prior_pi': 0.5 }

#####################################################


def neg_log_likelihood(y_true, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_true))

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

        # lrelu = lambda x: LeakyReLU(alpha=0.01)(x)

        inputs = Input(shape=(dim, ), name='Input')
        d1 = Dense(64, activation=swish, name='Dense_1')(inputs)
        d2 = Dense(32, activation=swish, name='Dense_2')(d1)

        # d1 = DenseVariational(5, kl_weight, **prior_params, activation=swish)(inputs)
        # d2 = DenseVariational(3, kl_weight, **prior_params, activation=swish)(d1)

        outputs = Dense(1, name='Output')(d2)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.network_name)

        # inputs = Input(shape=(dim,), name='Input')
        # d = Dense(21, activation=swish)(inputs)
        # outputs = Dense(1)(d)

    def plot_model(self):
        plot_model(self.model, show_shapes=True)

    def train_model(self):
        global epochs
        global batch_size
        self.modelTimer = TimingCallback()
        # stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, mode='min',
        #                         verbose=1, restore_best_weights=True)

        stopper = EarlyStopping(monitor='val_mean_absolute_percentage_error', min_delta=0.0001, patience=30, mode='min',
                                verbose=1, restore_best_weights=True)

        output = self.model.get_layer(index=3)

        # lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.7)

        self.model.compile(optimizer=Adam(), loss=neg_log_likelihood, metrics=['mean_absolute_percentage_error', root_mean_squared_error])
        self.model.summary()

        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                callbacks=[self.modelTimer, marketTimer, stopper], validation_data=(self.x_test, self.y_test))

        self.epochs = len(history.history['mean_absolute_percentage_error'])
        #plot_accuracy(history, self.network_name)

    def evaluate_model(self):
        print()
        print('Epochs:', self.epochs)
        loss, mape, rmse = self.model.evaluate(self.x_test, self.y_test)
        print(t, 'training time:', round(sum(self.modelTimer.logs), 4))
        print(f'test_loss: {loss} --> test_mape: {mape} --> test_rmse: {rmse}' + '\n')


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



marketTimer = TimingCallback()

if run == 'custom':
    #################################################################
    system.feature_space = 'OHLC'

    system.feature_engineering = ''

    system.processor = 'PT'

    system.distribution = 'normal'
    #################################################################

    model_name = 'MLPC'
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

        if create_plot and t == 'SPY':
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
            model_name = 'MLPC'
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
            model_name = 'MLPC'
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





