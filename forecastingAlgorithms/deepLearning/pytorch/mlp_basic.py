from System import *
from datasets.financial_data.import_financial_data import get_advanced_df, ohlc_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


class Sliding_Window(object):
    def __init__(self, data: pd.DataFrame, window_size=200):
        self.window_size = window_size
        self.rows = data.shape[0]
        self.cols = data.shape[1]
        self.features = data.columns.values
        x = data.values[:-1, :]
        y = data.values[1:, 3]
        MMS = MinMaxScaler()
        fitted = MMS.fit(x)
        x_t = MMS.transform(x)
        #y_t = MMS.transform()
        ### PREPROCESS
        self.x = torch.tensor(x_t, requires_grad=False, dtype=torch.float32)
        self.y = torch.tensor(y, requires_grad=False, dtype=torch.float32)

    def split(self):
        train_out = []
        test_out = []

        for start_ind in range(self.rows - self.window_size - 1):
            end_ind = start_ind + self.window_size
            x_train = self.x[start_ind: end_ind, :]
            y_train = self.y[start_ind: end_ind]
            train_add = [x_train, y_train]
            train_out.append(train_add)

            x_test = self.x[end_ind: end_ind + 1, :]
            y_test = self.y[end_ind: end_ind + 1]
            test_add = [x_test, y_test]
            test_out.append(test_add)

        window_data = [(train, test) for train, test in zip(train_out, test_out)]
        return window_data




class MultilayeredPerceptron(nn.Module):
    def __init__(self, n_inputs):
        super(MultilayeredPerceptron, self).__init__()

        self.INPUT_LAYER_SIZE = n_inputs
        self.HIDDEN_LAYER_SIZE = round((2/3) * n_inputs)
        self.OUTPUT_LAYER_SIZE = 1

        self.wh = nn.Parameter(torch.randn(self.INPUT_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, requires_grad=True, dtype=torch.float32))
        self.bh = nn.Parameter(torch.randn(1, self.HIDDEN_LAYER_SIZE, requires_grad=True, dtype=torch.float32))

        self.wo = nn.Parameter(torch.randn(self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_SIZE, requires_grad=True, dtype=torch.float32))
        self.bo = nn.Parameter(torch.randn(1, self.OUTPUT_LAYER_SIZE, requires_grad=True, dtype=torch.float32))

    def forward(self, x):
        zh = torch.matmul(x, self.wh) + self.bh
        h = F.relu(zh)
        zo = torch.matmul(h, self.wo) + self.bo
        # o = F.relu(zo)
        return zo


def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

def get_sequential_batch(data: [torch.Tensor, torch.Tensor], batch_count, batch_size=8):
    start = batch_count * batch_size
    x_batch, y_batch = data[0][start: start+batch_size], data[1][start: start+batch_size]

    return x_batch, y_batch

def train_model(train, model, epochs, batch_size=8, lr=0.01):
    # loss_val_prev = 2
    # loss_threshold = 0.000099

    length = train[0].size()[0]
    iterations = length // batch_size

    for e in range(epochs):
        batch_count = 0
        while batch_count <= iterations:
            x, y_true = get_sequential_batch(train, batch_count)
            batch_count += 1

            y_pred = model.forward(x)
            loss = mse(y_true, y_pred)
            loss.backward()

            for f in model.parameters():
                f.data.sub_(-lr * f.grad.data)
                f.grad.zero_()

        loss_val = round(abs(loss.item()), 4)
        # loss_diff = loss_val_prev - loss_val
        # if loss_diff < loss_threshold:
        #     print('\nEarly Stopping at Epoch #{}'.format(e), ' --->  Final Loss: {}'.format(loss_val))
        #     break
        print('Epoch #{}'.format(e), ' --->  Loss: {}'.format(loss_val))
        loss_val_prev = loss_val

#df = get_advanced_df('SPY')
df = ohlc_data('SPY')
window = Sliding_Window(df)

epochs = 200

for train, test in window.split():
    # x_train, y_train = train[0], train[1]
    # x_test, y_test = test[0], test[1]

    model = MultilayeredPerceptron(window.cols)

    train_model(train, model, epochs, batch_size=8, lr=0.01)

    exit()







