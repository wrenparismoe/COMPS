from System import *
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from time import time_ns

torch.manual_seed(0)
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# Basic model for learning low level pytorch

"""
Task --> Predicting if a structure exists in the ionosphere based off collected radar signals

    "Good" radar returns are those showing evidence of some type of structure 
    in the ionosphere.  "Bad" returns are those that do not; their signals pass
    through the ionosphere
"""

def prepare_data(path):
    data = pd.read_csv(path, sep=',', header=None)
    x = data.values[:, :-1].astype('float32')

    y = LabelEncoder().fit_transform(data.values[:, -1]).astype('float32').reshape(len(x), 1)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # get train/test split
    test_ratio = 0.33
    test_size = int(test_ratio * len(x))
    train_size = len(x) - test_size
    train, test = [x[:train_size], y[:train_size]], [x[train_size:], y[train_size:]]
    # create train/test DataLoaders
    return train, test

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs):
        super(LogisticRegression, self).__init__()

        self.INPUT_LAYER_SIZE = n_inputs
        self.HIDDEN_LAYER_SIZE = 1
        self.OUTPUT_LAYER_SIZE = 1

        self.w = nn.Parameter(torch.randn(self.INPUT_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, requires_grad=True,
                                          dtype=torch.float32))
        self.b = nn.Parameter(torch.randn(1, self.OUTPUT_LAYER_SIZE, requires_grad=True, dtype=torch.float32))

    def forward(self, x):
        z = torch.matmul(x, self.w) + self.b
        a = self.sigmoid(z)
        return a

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

def log_likelihood(y_true, y_pred):
    return 1 / len(y_true) * (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).sum()

def train_model(train, model, epochs, lr=0.01):
    x, y = train[0], train[1]
    loss_val_prev = 2
    loss_threshold = 0.000099
    for e in range(epochs):
        y_pred = model.forward(x)

        loss = log_likelihood(y, y_pred)
        loss.backward()

        for f in model.parameters():
            f.data.sub_(-lr * f.grad.data)
            f.grad.zero_()

        loss_val = round(abs(loss.item()), 4)
        loss_diff = loss_val_prev - loss_val
        if loss_diff < loss_threshold:
            print('\nEarly Stopping at Epoch #{}'.format(e), ' --->  Final Loss: {}'.format(loss_val))
            break
        print('Epoch #{}'.format(e), ' --->  Loss: {}'.format(loss_val))
        loss_val_prev = loss_val

def evaluate_model(test, model):
    x, y_true = test[0], test[1]
    y_pred = model(x)
    y_pred = y_pred.detach().numpy()
    y_pred = y_pred.round()

    y_true = y_true.numpy()
    y_true = y_true.reshape(len(y_true), 1)

    y_true, y_pred = np.vstack(y_true), np.vstack(y_pred)

    acc = round(accuracy_score(y_true, y_pred)*100, 2)

    return acc


start = time_ns()

path = 'C:/Users/wrenp/pytorch/datasets/ionosphere.csv'
train, test = prepare_data(path)
print('Train size: {}'.format(len(train[0])), ' |  Test size: {}'.format(len(test[0])))
n_inputs = train[0].size()[1]

model = LogisticRegression(n_inputs)

train_model(train, model, epochs=1000, lr=0.1)

acc = evaluate_model(test, model)
print("Accuracy:", str(acc) + '%')

end = time_ns()
runtime = end - start
print('Runtime: {} ns'.format(runtime))

