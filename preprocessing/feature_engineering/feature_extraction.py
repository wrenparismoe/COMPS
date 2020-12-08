from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from System import *
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import plot_model
from timeit import default_timer as timer
from tensorflow_addons.layers import WeightNormalization


def principal_component_analysis(x: pd.DataFrame, system: SystemComponents, dim: int=dimension) -> pd.DataFrame:
    if dim is None:
        dim = 16
    pca = PCA(n_components=16, svd_solver='auto')
    pca = pca.fit(x)
    system.fitted_pca = pca
    extracted = pca.transform(x)
    x_extracted = pd.DataFrame(extracted, index=x.index)

    return x_extracted

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


class SAE:
    def __init__(self, x):
        self.x = x
        self.x_train = x.loc[train_index]
        self.x_test = x.loc[test_index]
        self.encoder = keras.models.Model
        self.autoencoder = keras.models.Model

    def create_model(self):
        dim = self.x.shape[1]

        # lrelu = lambda x: LeakyReLU(alpha=0.01)(x)



        # e = Dense(64, activation=swish)(inputs)
        # e = Dense(32, activation=swish)(e)

        inputs = Input(shape=(dim,), name='Input')

        encoded = Dense(32, activation=swish)(inputs)

        outputs = Dense(dim)(encoded)

        # d = Dense(32, activation=swish)(encoded)
        # d = Dense(64, activation=swish)(d)


        self.autoencoder = Model(inputs, outputs)
        self.encoder = Model(inputs, encoded)

    # def plot_model(self):
    #     plot_model(self.encoder, show_shapes=True)

    def train_model(self, epochs, batch_size):
        self.autoencoder.compile(optimizer=Adam(0.001), loss='mean_squared_error', metrics=["mean_squared_error"])
        global cb
        # stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min',
        #                         verbose=1, restore_best_weights=True)

        self.autoencoder.fit(x=self.x_train, y=self.x_train, epochs=epochs, batch_size=batch_size, callbacks=[cb],
                             validation_data=(self.x_test, self.x_test))

    def evaluate_model(self):
        loss, acc = self.autoencoder.evaluate(x=self.x_test, y=self.x_test)
        print('training time:', round(sum(cb.logs), 4))
        print(f'test_loss: {loss} --> test_acc: {acc}')
        print(self.encoder.summary(), '\n')

    def predict_model(self):
        self.x_train_extracted = self.encoder.predict(x=self.x_train)
        self.x_test_extracted = self.encoder.predict(x=self.x_test)



def stacked_auto_encoders(x):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.compat.v1.enable_eager_execution()

    stacked_auto_encoder = SAE(x)
    stacked_auto_encoder.create_model()
    stacked_auto_encoder.train_model(epochs=150, batch_size=32)
    stacked_auto_encoder.evaluate_model()
    stacked_auto_encoder.predict_model()

    x_train = stacked_auto_encoder.x_train_extracted
    x_test = stacked_auto_encoder.x_test_extracted

    x_train = pd.DataFrame(x_train, index=train_index)
    x_test = pd.DataFrame(x_test, index=test_index)

    x_d = x_train.append(x_test)
    x_d = x_d.sort_index()

    return x_d


cb = TimingCallback()


