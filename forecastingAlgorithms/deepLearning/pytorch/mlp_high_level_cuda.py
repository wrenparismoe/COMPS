from datasets.financial_data.import_financial_data import get_advanced_df, ohlc_data
from data_classes.sliding_window_iterable_dataset import Sliding_Window_IterableDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from feature_engineering import pearson, spearman, mutual_info
from data_processing import feature_difference
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from time import time
import torch
from System import *

SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda")

class Metrics():
    def __init__(self, y_true, y_pred):
        self.true = y_true
        self.pred = y_pred
        self.MA_()
        self.MAE_()
        self.RMSE_()
        self.MAPE_()
        self.HR_IR_()

    def get_metrics(self):
        return [self.ma, self.mae, self.rmse, self.mape, self.hr, self.ir]

    def MA_(self):
        self.ma = torch.max(torch.abs(self.true - self.pred))

    def MAE_(self):
        self.mae = torch.mean(torch.abs(self.true - self.pred))

    def MAPE_(self):
        self.mape = torch.mean(torch.abs((self.true - self.pred) / self.true)) * 100

    def RMSE_(self):
        self.rmse = torch.sqrt(torch.mean(torch.square(self.true - self.pred)))

    def HR_IR_(self):
        hit_count = 0
        inc_count = 0
        for i in range(1, self.true.size(0)):
            true_chg = self.true[i] - self.true[i - 1]
            pred_chg = self.pred[i] - self.true[i - 1]
            if pred_chg > 0:
                inc_count += 1
            if true_chg * pred_chg > 0:
                hit_count += 1
        self.hr = hit_count / (self.true.size(0) - 1) * 100
        self.ir = inc_count / (self.true.size(0) - 1) * 100


class Rolling_Metrics():
    def __init__(self):
        self.last = None

        self.residuals = torch.zeros(model_iterations, dtype=torch.float32)  # = []
        self.max_error = 0

        self.mae_list = torch.zeros(model_iterations, dtype=torch.float32)
        self.mae_avg = 0

        self.rmse_list = torch.zeros(model_iterations, dtype=torch.float32)
        self.rmse_avg = 0

        self.mape_list = torch.zeros(model_iterations, dtype=torch.float32)
        self.mape_avg = 0

        # self.hits = np.full(model_iterations, False, dtype=np.bool_)
        self.hits = torch.zeros(model_iterations, dtype=torch.bool)
        self.hit_ratio = 0

        # self.inc = np.full(model_iterations, False, dtype=np.bool_)
        self.inc = torch.zeros(model_iterations, dtype=torch.bool)
        self.inc_ratio = 0

    def update(self, true, pred):
        diff = abs(true - pred)
        self.residuals[ind] = diff  # .append(diff)
        if diff > self.max_error:
            self.max_error = diff

        self.mae_list[ind] = MAE(true, pred)
        self.mae_avg = torch.mean(self.mae_list)

        self.rmse_list[ind] = RMSE(true, pred)
        self.rmse_avg = torch.mean(self.rmse_list)

        self.mape_list[ind] = MAPE(true, pred)
        self.mape_avg = torch.sum(self.mape_list) / torch.count_nonzero(self.mape_list)

        if self.last is None:
            self.last = true
            return
        true_chg = true - self.last
        pred_chg = pred - self.last
        if pred_chg > 0:
            self.inc[ind] = True
            self.inc_ratio = torch.nansum(self.inc) / ind
        if true_chg * pred_chg > 0:
            self.hits[ind] = True
            self.hit_ratio = torch.nansum(self.hits) / ind
        self.last = true


def MAE(true, pred):
    return torch.mean(torch.abs(true - pred))

def RMSE(true, pred):
    return torch.sqrt(torch.mean(torch.square(true - pred)))

def MAPE(true, pred):
    return torch.mean(torch.abs((true - pred) / true)) * 100

def Loss():
    def MAPE_Loss(y_true, y_pred):
        val = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        return val

    return MAPE_Loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, min_delta=9e-4, verbose=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.last = 1e10
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = min_delta

    def __call__(self, val_loss, model):
        score = val_loss
        diff = abs(score - self.last)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score and diff > self.min_delta:
            self.counter += 1
            if self.verbose >= 1:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        self.last = score

    def save_checkpoint(self, val_loss, model):
        if self.verbose >= 2:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class MLP(nn.Module):
    def __init__(self, input_neurons, hidden1_neurons, hidden2_neurons):
        super(MLP, self).__init__()
        self.h1 = torch.nn.Linear(input_neurons, hidden1_neurons)  # hidden layer 1
        self.h2 = torch.nn.Linear(hidden1_neurons, hidden2_neurons)  # hidden layer
        self.predict = torch.nn.Linear(hidden2_neurons, 1)  # output layer

    def forward(self, x):
        x = F.relu(self.h1(x))  # activation function for hidden layer
        x = F.relu(self.h2(x))
        x = self.predict(x)  # linear output
        return x


def train(train_dataset, batch_size, epochs=500, learning_rate=0.01, verbose=0):
    iterations_per_epoch = round(window_size / batch_size)
    early_stopper = EarlyStopping(patience=10, verbose=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = Loss()
    e = 0
    while e < epochs:
        train = DataLoader(train_dataset.reset(), batch_size=batch_size, shuffle=False)
        running_loss = 0
        for i, (x, y) in enumerate(train):
            y_pred = model.forward(x)
            loss = loss_fn(y, y_pred)
            with torch.no_grad():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.detach() * x.size(0)
            if verbose >= 3:
                print('    Iteration {}/{}:'.format(i + 1, iterations_per_epoch), loss)
        epoch_loss = running_loss / window_size
        if verbose >= 2:
            print('  Epoch #{}'.format(e), ' --->  Loss: {}'.format(epoch_loss))
        early_stopper(epoch_loss, model)
        if early_stopper.early_stop:
            break
        e += 1
    if verbose >= 1:
        if early_stopper.early_stop:
            print('{}/{} Early Stopping at Epoch #{}'.format(ind + 1, model_iterations, e), ' --->  Best Loss: {}'.format(early_stopper.best_score))
        else:
            print('{}/{} Final Epoch #{}'.format(ind + 1, model_iterations, e), ' --->  Final Loss: {}'.format(epoch_loss))

def eval_train(train_dataset, batch_size, verbose=0):
    train = DataLoader(train_dataset.reset(), batch_size=batch_size, shuffle=False)
    pred = torch.zeros(window_size, device=device, dtype=torch.float32)
    true = torch.zeros(window_size, device=device, dtype=torch.float32)
    for i, (x, y) in enumerate(train):
        y_pred = model.forward(x)
        pred[i * x.size(0): (i + 1) * x.size(0)] = y_pred.detach().flatten()
        true[i * x.size(0): (i + 1) * x.size(0)] = y.flatten()
    m = Metrics(true, pred)
    if verbose == 4:
        print('Training:')
        print('     ma:', m.ma, ' ----  mae:', m.mae)
        print('     mape:', m.mape, ' ----  rmse:', m.rmse)
        print('     HR:', m.hr, '  ----  IR:', m.ir)
    elif verbose == 3:
        print('Training:')
        print('     mape:', m.mape, ' ----  rmse:', m.rmse)
        print('     HR:', m.hr, '  ----  IR:', m.ir)
    elif verbose == 2:
        print('     HR:', m.hr, '  ----  IR:', m.ir)
    elif verbose == 1:
        print(m.get_metrics())
    train_metrics.append(m.get_metrics())

def predict(test_dataset, verbose=0):
    for x, y in iter(test_dataset):
        y_pred = model(x)
        pred = y_pred.detach()
        true = y
        m_rolling.update(true, pred)
        if verbose >= 1:
            print('      True: {},  Pred: {}   ----->  MAPE: {},  Hit: {}'.format(true, pred, MAPE(true, pred),
                                                                                m_rolling.hits[ind]))
        if verbose >= 2:
            print('      Rolling Metrics | MAPE: {}  ====>  HR: {}'.format(m_rolling.mape_avg, m_rolling.hit_ratio))
        actual[ind], predictions[ind] = true, pred
        return

def get_data():
    df = get_advanced_df('QQQ', start_date='2018-01-01')

    df = df.iloc[:170]

    df = pearson(df)

    # df_diff = feature_difference(df.copy())
    # close_raw = df['Close'].shift(2).dropna().values

    return df

df = get_data()

window_size = 128
model_params = {
    'input_neurons': df.shape[1],
    'hidden1_neurons': 16,
    'hidden2_neurons': 8
}
train_params = {
    'batch_size': 16,
    'epochs': 300,
    'learning_rate': 0.01,
    'verbose': 1
}

start = time()
model_iterations = df.shape[0] - window_size - 2
window = Sliding_Window_IterableDataset(df, device)

with torch.no_grad():
    model = MLP(**model_params)
    model.cuda(device=device)

# loss_fn = nn.MSELoss()
ind, hit_count, hit_ratio = 0, 0, 0
actual = torch.zeros(model_iterations, dtype=torch.float32)
predictions = torch.zeros(model_iterations, dtype=torch.float32)
train_metrics = []
m_rolling = Rolling_Metrics()

cudnn.benchmark = True



for trainSet, testSet in window.split(window_size=window_size):

    train(trainSet, **train_params)
    eval_train(trainSet, batch_size=train_params.get('batch_size'), verbose=0)
    model.load_state_dict(torch.load('checkpoint.pt'))
    predict(testSet, verbose=2)

    ind += 1
    print()

assert not torch.isnan(predictions).any()

print(predictions)

results = pd.DataFrame({'True': actual, 'Pred': predictions, 'Hit': m_rolling.hits})
print(results, '\n')

train_avgs = np.average(np.array(train_metrics), axis=0)
print('Training Results:', train_avgs)

m = Metrics(np.array(actual), np.array(predictions))
metrics = m.get_metrics()
print('Prediction Results:', np.round(metrics))

end = time()
elapsed = end - start
print('Elapsed Time: {}'.format(elapsed))
