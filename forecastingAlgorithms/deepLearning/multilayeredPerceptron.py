from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, difference, feature_difference, get_min, get_max
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.evaluation import mean_absolute_percentage_error as mape
from modules.plot import plot_results
from modules.time_process import Timer

from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.activations import relu, swish, linear
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Reduction
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import plot_model, normalize
import tensorflow as tf
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
gpu = tf.config.list_physical_devices('GPU')[0]

tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpu, 'GPU')
#tf.config.run_functions_eagerly(True)
tf.keras.backend.set_floatx('float32')


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

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
# @tf.function
# def print_tensor(tensor):
#     tf.print(tensor, summarize=-1)
#
# @tf.autograph.experimental.do_not_convert
# def pred_loss(y_true, y_pred):
#
#     y_true_last = y_true[:-1,]
#     y_true = y_true[1:]
#     y_pred = y_pred[1:]
#
#     print(tf.concat([tf.concat([y_pred, y_true], axis=1), y_true_last], axis=1))
#
#     true_diff = tf.subtract(y_true, y_true_last)
#     pred_diff = tf.subtract(y_pred, y_true_last)
#
#     true_chg = tf.divide(true_diff, y_true_last)
#     pred_chg = tf.divide(pred_diff, y_true_last)
#
#     true_chg_sign = tf.sign(true_chg)
#     pred_chg_sign = tf.sign(pred_chg)
#
#     sign_tensor = tf.multiply(true_chg_sign, pred_chg_sign)
#     one_n = tf.constant(-1, dtype=tf.float32)
#
#     failed_pred_bools = tf.equal(sign_tensor, one_n)
#     failed_pred_ints = tf.cast(failed_pred_bools, dtype=tf.float32)
#
#     print(failed_pred_ints)
#
#     failed_preds = tf.math.multiply(failed_pred_ints, pred_chg)
#     failed_preds = tf.abs(failed_preds)
#     failed_preds = tf.multiply(failed_preds, tf.constant(100, dtype=tf.float32))
#
#     #####################################################################
#
#     failed_true = tf.multiply(failed_pred_ints, y_true)
#     failed_predictions = tf.multiply(failed_pred_ints, y_pred)
#
#     index_nonzero = tf.where(tf.not_equal(failed_true, tf.constant(0, dtype=tf.float32)), None, None)
#     index_nonzero = tf.squeeze(index_nonzero)
#
#     failed_true = tf.gather(failed_true, index_nonzero)
#     failed_predictions = tf.gather(failed_predictions, index_nonzero)
#
#     failed_mse = K.mean(tf.square(tf.subtract(failed_true, failed_predictions)))
#
#     diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
#     mape = 100. * K.mean(diff)
#
#     failed_diff = K.abs((failed_true - failed_predictions) / K.clip(K.abs(failed_true), K.epsilon(), None))
#     failed_mape = 100. * K.mean(failed_diff, axis=1)
#
#     exit()
#     return failed_mape

    # return K.math_ops.multiply(failed_mape, mape)



# tf.config.run_functions_eagerly(True)

@tf.function
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
    indices = tf.cast(indices, dtype='int32')

    # create a tensor to store directional loss and put it into custom loss output
    # direction_loss = tf.Variable(tf.ones_like(y_pred), dtype='float32')
    direction_loss = tf.ones_like(y_pred, dtype=tf.float32)
    updates = tf.cast(tf.ones_like(indices), dtype='float32')
    alpha = 1000

    # print(direction_loss)
    # tf.print(direction_loss.get_shape(), end='\n')
    # print(indices)
    # tf.print(indices.get_shape(), end='\n')
    # print(alpha * updates)
    #
    # exit()tf.multiply(tf.cast(alpha, dtype=tf.float32), updates)

    direction_loss = tf.tensor_scatter_nd_update(direction_loss, indices, alpha*updates)

    return tf.math.reduce_mean(tf.multiply(tf.square(y_true - y_pred), direction_loss), axis=-1)

################################################################################################3

#####################################################
@tf.function
def smape(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)

    num = tf.reduce_sum(tf.multiply(tf.constant(2, dtype=tf.float32), tf.abs(tf.subtract(y_true, y_pred))))
    denom = tf.add(tf.abs(y_true), tf.abs(y_pred))
    frac = tf.divide(num, denom)
    coefficient = tf.divide(tf.constant(1, dtype=tf.float32), tf.cast(tf.size(y_true, out_type=tf.int32), dtype=tf.float32))
    return tf.multiply(tf.multiply(coefficient, frac), tf.constant(100, dtype=tf.float32))


class WindowGenerator():
  def __init__(self, train, val, test, input_width, window=100):
    # Store the raw data.
    self.train = train
    self.val = val
    self.test = test

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(train.shape[1])}

    # Work out the window parameters.
    self.input_width = input_width
    self.window = window

    self.total_window_size = input_width + window

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - 1
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


def MinMaxTensor(input: np.ndarray, current_range, feature_range = (0,1)) -> tf.Tensor:
    x = tf.convert_to_tensor(input, dtype=tf.float32, name='input')
    mins = tf.convert_to_tensor(current_range[0], dtype=tf.float32, name='mins')
    maxs = tf.convert_to_tensor(current_range[1], dtype=tf.float32, name='maxs')
    x_std = (x - mins) / (maxs - mins)
    x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return x_scaled




class MultilayeredPerceptron:
    def __init__(self):
        self.network_name = model_name + '-' + t
        self.model = None
        self.y_pred = None
    def create_model(self):
        dim = cols
        # lrelu = lambda x: LeakyReLU(alpha=0.001)(x)
        # inputs = Input(shape=(dim, ), name='Input')
        # d1 = Dense(32, activation=swish, name='Dense_1')(inputs)
        # d2 = Dense(16, activation=swish, name='Dense_2')(d1)
        # outputs = Dense(1, name='Output')(d2)
        #self.model = Model(inputs=inputs, outputs=outputs, name=self.network_name)
        inputs = Input(shape=(dim,), name='Input')
        d = Dense(dim//2, activation=relu)(inputs)
        outputs = Dense(1, activation=linear)(d)
        self.model = Model(inputs=inputs, outputs=outputs, name=self.network_name)
    def plot(self):
        plot_model(self.model, show_shapes=True)

    def train(self, lr=0.001, epochs=100, batch_size=1):
        self.epochs = epochs
        self.modelTimer = TimingCallback()
        self.stopper = EarlyStopping(monitor='loss', min_delta=0.01, baseline=None, patience=10, mode='min', verbose=0,
                                      restore_best_weights=True)

        self.model.compile(optimizer=Adam(lr), loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])


        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[self.stopper], verbose=0)

        MLP.stopper.model.save('my_model.h5')
        self.epochs = len(history.history['mean_absolute_percentage_error'])
        #plot_accuracy(history, self.network_name)

    def evaluate(self):
        print('Epochs:', str(self.stopper.stopped_epoch) + '---' + str(self.stopper.best))
        loss, acc = self.model.evaluate(x_test, y_test)
        print(f'test_loss: {loss} --> test_mape: {acc}')
    def predict(self):
        pred = self.model.predict_step(x_test)
        self.y_pred = np.float64(pred[0])
        return self.y_pred




#################################################################
system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='', processor='MMS')
#################################################################

model_name = 'MLP'
model_name = get_model_name(model_name, system)
marketTimer = TimingCallback()
timer_list = []
minMax_list = [[]]
for t in market_etfs:
    clear_session()
    timer = Timer()
    x, y, df = get_data(t, system)
    x = pd.DataFrame.to_numpy(x).astype('float32')
    y = pd.Series.to_numpy(y).astype('float32')
    cols = x.shape[1]

    MLP = MultilayeredPerceptron()
    MLP.create_model()

    y_pred, y_test, test_index, pred_chg_list = [], [], [], []
    train_start = df.index.get_loc('2018-02-26')

    training_window =100

    for i in range(train_start, x.shape[0]):
        if i == len(x[train_start:]) // 2:
            print('50%%%%%%%%%%%%%%%')


        x_train = x[i - training_window:i]
        x_train = preprocess(x_train, system)
        MAS = MaxAbsScaler()
        MAS_fit = MAS.fit(x_train)
        x_train = MAS.transform(x_train)
        y_train = y[i - training_window:i]

        x_test = x[i:i+1]
        x_test = system.fitted_processor.transform(x_test)
        x_test = MAS_fit.transform(x_test)

        date = df.index[i]
        ts = pd.to_datetime(str(date))
        d = ts.strftime('%Y-%m-%d')
        test_index.append(date)

        # if not i == train_start:
        #     model = load_model('my_model.h5', custom_objects={'loss': MLP.stopper.model.loss}, compile=False)

        cb = TimingCallback()
        MLP.train()
        pred = MLP.predict()
        true = float(y[i])

        y_test.append(true)
        y_pred.append(pred)
        pred_chg_list.append(pred -float(y[i-1]))

        print(d, '| Best Loss:', round(MLP.stopper.best, 5), ': ', round(true, 3), '--', round(pred, 3))



    y_pred = pd.Series(y_pred, index=test_index, name='pred')
    y_test = pd.Series(y_test, index=test_index, name='y_test')
    results = format_results(df, y_test, y_pred)
    pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
    pred_up_str = str(pred_up) + '%'
    forecast_metrics = metrics_list(results.loc[test_index])
    forecast_metrics.append(pred_up_str)
    print(forecast_metrics)
    errors.loc[t] = forecast_metrics
    timer_list.append(round(timer.end_timer(), 4))

    if create_plot:
        plot_results(results, model_name)

print(model_name + '          features:', x.shape[1])
print(errors)
errors.to_clipboard(excel=True, index=False, header=False)
print(timer_list)










"""
model_name = 'MLP'
model_name = get_model_name(model_name, system)
marketTimer = TimingCallback()
timer_list = []
for t in market_etfs:
    clear_session()
    timer = Timer()
    x, y, df = get_data(t, system)
    x = pd.DataFrame.to_numpy(x).astype('float32')
    y = pd.Series.to_numpy(y).astype('float32')
    MLP = MultilayeredPerceptron()
    MLP.create_model()
    y_pred, y_test_list, test_index, pred_chg_list = [], [], [], []
    train_start = df.index.get_loc('2019-02-26')
    training_window =100
    for i in range(train_start, x.shape[0]):
        if i == len(x[train_start:]) // 2:
            print('50%%%%%%%%%%%%%%%')
        x_train, y_train = preprocess(x[i - training_window-3:i-3], system), y[i - training_window-3:i-3]
        MAS = MaxAbsScaler()
        MAS_fit = MAS.fit(x_train)
        x_train = MAS.transform(x_train)
        x_val, y_val = system.fitted_processor.transform(x[i-3:i]), y[i-3:i]
        x_test, y_test = system.fitted_processor.transform(x[i:i+1]), y[i:i+1]
        x_test = MAS_fit.transform(x_test)

        date = df.index[i]
        ts = pd.to_datetime(str(date))
        d = ts.strftime('%Y-%m-%d')
        test_index.append(date)
        if not i == train_start:
            model = load_model('my_model.h5', custom_objects={'loss': MLP.stopper.model.loss}, compile=False)

        cb = TimingCallback()
        MLP.train()
        pred = MLP.predict()

        y_test_list.append(float(y[i]))
        y_pred.append(pred)
        pred_chg_list.append(pred -y[i-1])
        print(d, '| Best Loss:', round(MLP.stopper.best, 5), ': ', round(float(y_test), 3), '--', round(pred, 3), '   MAPE:', round(mape(y_test, pred), 3), '%')



    y_pred = pd.Series(y_pred, index=test_index, name='pred')
    y_test = pd.Series(y_test_list, index=test_index, name='y_test')
    results = format_results(df, y_test, y_pred)
    pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
    pred_up_str = str(pred_up) + '%'
    forecast_metrics = metrics_list(results.loc[test_index])
    forecast_metrics.append(pred_up_str)
    print(forecast_metrics)
    errors.loc[t] = forecast_metrics
    timer_list.append(round(timer.end_timer(), 4))

    if create_plot:
        plot_results(results, model_name)

print(model_name + '          features:', x.shape[1])
print(errors)
errors.to_clipboard(excel=True, index=False, header=False)
print(timer_list)


"""








