from datasets.financial_data.import_financial_data import get_advanced_df, ohlc_data
from data_classes.sliding_window_iterable_dataset import Sliding_Window_IterableDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from feature_engineering import pearson, spearman, mutual_info
from data_processing import feature_difference
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from time import time
import torch
from System import *

SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")
torch.set_default_tensor_type('torch.FloatTensor')

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
        return [self.ma, self.mae, self.mape, self.rmse, self.hr, self.ir]

    def MA_(self):
        self.ma = np.max(np.abs(self.true - self.pred))

    def MAE_(self):
        self.mae = np.mean(np.abs(self.true - self.pred))

    def MAPE_(self):
        self.mape = np.mean(np.abs((self.true - self.pred) / self.true)) * 100

    def RMSE_(self):
        self.rmse = np.sqrt(np.mean(np.square(self.true - self.pred)))

    def HR_IR_(self):
        hit_count = 0
        inc_count = 0
        for i in range(1, len(self.true)):
            true_chg = self.true[i] - self.true[i-1]
            pred_chg = self.pred[i] - self.true[i-1]
            if pred_chg > 0:
                inc_count += 1
            if true_chg * pred_chg > 0:
                hit_count += 1
        self.hr = hit_count / (len(self.true) - 1) * 100
        self.ir = inc_count / (len(self.true) - 1) * 100

class Rolling_Metrics():
    def __init__(self):
        self.last = None
        self.record = False
        self.pre_record_count = 0

        self.residuals = torch.zeros(model_iterations, dtype=torch.float32)  # = []
        self.max_error = 0

        self.mae_list = torch.zeros(model_iterations, dtype=torch.float32)
        self.mae_avg = 0

        self.rmse_list = torch.zeros(model_iterations, dtype=torch.float32)
        self.rmse_avg = 0

        self.mape_list = torch.zeros(model_iterations, dtype=torch.float32)
        self.mape_avg = 0

        self.hits = torch.zeros(model_iterations, dtype=torch.bool)
        self.hit_ratio = 0

        self.inc = torch.zeros(model_iterations, dtype=torch.bool)
        self.inc_ratio = 0

    def get_metrics(self):
        return [self.max_error, self.mae_avg, self.mape_avg, self.rmse_avg, self.hit_ratio, self.inc_ratio]

    def update(self, true, pred):
        if not self.record:
            self.pre_record_count += 1
            return
        diff = abs(true - pred)
        self.residuals[ind] = diff
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
        self.inc_ratio = (torch.sum(self.inc[self.pre_record_count:]) / (ind-self.pre_record_count+1))
        if true_chg * pred_chg > 0:
            self.hits[ind] = True
        self.hit_ratio = (torch.sum(self.hits[self.pre_record_count:]) / (ind-self.pre_record_count+1))
        self.last = true


def MAE(true, pred):
    return torch.mean(torch.abs(true - pred))

def RMSE(true, pred):
    return torch.sqrt(torch.mean(torch.square(true - pred)))

def MAPE(true, pred):
    return torch.mean(torch.abs((true - pred) / true)) * 100

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, min_delta=9e-3, verbose=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 1000e10
        self.last = 1e10
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = min_delta

    def __call__(self, val_loss, model):
        score = val_loss
        diff = abs(score - self.best_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score and diff < self.min_delta:
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
        self.h1 = torch.nn.Linear(input_neurons, hidden1_neurons)   # hidden layer 1
        self.h2 = torch.nn.Linear(hidden1_neurons, hidden2_neurons)  # hidden layer
        self.predict = torch.nn.Linear(hidden2_neurons, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.h1(x))      # activation function for hidden layer
        x = F.relu(self.h2(x))
        x = self.predict(x)             # linear output
        return x

def Loss():
    def MAPE_Loss(y_true, y_pred):
        val = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        return val

    return MAPE_Loss


def train(train_dataset, batch_size, epochs=500, learning_rate=0.001, verbose=0):
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
        early_stopper(epoch_loss, model)
        if verbose >= 2:
            print('  Epoch #{}'.format(e), ' --->  Loss: {}'.format(epoch_loss))
        if early_stopper.early_stop:
            m_rolling.record = True
            print('___________________________________________________________________________________________________')
            break
        e += 1
    if verbose >= 1:
        if early_stopper.early_stop:
            print('{}/{} Early Stopping at Epoch #{}'.format(ind + 1, model_iterations, e),
                  ' --->  Best Loss: {}'.format(early_stopper.best_score))
        else:
            print('{}/{} Final Epoch #{}'.format(ind + 1, model_iterations, e),
                  ' --->  Final Loss: {}'.format(epoch_loss))

def eval_train(train_dataset, batch_size, verbose=0):
    train = DataLoader(train_dataset.reset(), batch_size=batch_size, shuffle=False)
    pred = np.full(window_size, np.nan, dtype=np.float32)
    true = np.full(window_size, np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(train):
        y_pred = model.forward(x)
        pred[i*x.size(0): (i+1)*x.size(0)] = y_pred.detach().numpy().flatten()
        true[i*x.size(0): (i+1)*x.size(0)] = y.flatten().numpy().flatten()
    m = Metrics(true, pred)
    if verbose == 1:
        print(m.get_metrics())
    train_metrics.append(m.get_metrics())

def predict(test_dataset, verbose=0):
    for x, y in iter(test_dataset):
        y_pred = model(x)
        pred = y_pred.detach()
        true = y
        m_rolling.update(true, pred)
        if verbose >= 1:
            print('    True: {}  Pred: {}  ---->  MAPE: {},  Hit: {}'.format(true,pred, MAPE(true, pred), m_rolling.hits[ind]))
        if verbose >= 2:
            print('    Rolling Metrics | MAPE: {}  ====>  HR: {}'.format(m_rolling.mape_avg, m_rolling.hit_ratio))
        if m_rolling.record:
            actual[ind], predictions[ind] = true, pred
        return

def get_data(t):
    df = get_advanced_df(t, start_date='2018-01-01')
    df = df.iloc[:160]
    df = pearson(df)

    # df_diff = feature_difference(df.copy())
    # close_raw = df['Close'].shift(2).dropna().values

    return df

t = 'SPY'
df = get_data(t)
window_size = 64
model_params = {
    'input_neurons': df.shape[1],
    'hidden1_neurons': 16,
    'hidden2_neurons': 8
}
train_params = {
    'batch_size': 32,
    'epochs': 500,
    'learning_rate': 0.01,
    'verbose': 1
}
start = time()
model_iterations = df.shape[0] - window_size - 2
window = Sliding_Window_IterableDataset(df, device)

with torch.no_grad():
    model = MLP(**model_params)

ind, hit_count, hit_ratio = 0, 0, 0
actual = torch.zeros(model_iterations, dtype=torch.float32)
predictions = torch.zeros(model_iterations, dtype=torch.float32)
train_metrics = []
m_rolling = Rolling_Metrics()

for trainSet, testSet in window.split(window_size=window_size):
    train(trainSet, **train_params)
    eval_train(trainSet, batch_size=train_params.get('batch_size'), verbose=0)
    model.load_state_dict(torch.load('checkpoint.pt'))
    predict(testSet, verbose=2)

    ind += 1

print(type(torch.count_nonzero(predictions).detach().item()), type(model_iterations))
assert torch.count_nonzero(predictions).detach().item() == model_iterations

results = pd.DataFrame({'True': actual, 'Pred': predictions, 'Hit': m_rolling.hits})
print(results, '\n')
train_avgs = np.average(np.array(train_metrics), axis=0)
print('Training Results:', train_avgs)

errors = pd.DataFrame(np.zeros((1, 6)), columns=["ME", "MAE", "MAPE", "RMSE", "Prediction %", 'Pred Incr %'], index=[t])
errors.index.name = 'Market'
errors.loc[t] = m_rolling.get_metrics()
print('Prediction Results: \n', errors)
errors.to_clipboard(excel=True, index=False, header=False)

end = time()
elapsed = end - start
print('Elapsed Time: {}'.format(elapsed))
