from System import *
import torch
import torch.nn as nn
import plotly.graph_objects as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import math

# Basic model for learning low level pytorch

device = torch.device('cuda')

def set_display_options():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(edgeitems=100, linewidth=1000)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    torch.random.seed = 2

set_display_options()

data_df = pd.read_csv('/datasets/New_York_Hourly.csv', index_col='date')
data_df = data_df.rename(columns={'TemperatureF': 'Temp', 'Dew PointF': 'Dew', 'Wind SpeedMPH': 'Wind'})

""" Task: Predict when the weather is clear based on available hourly based time series data"""

# first with raw data

def preprocess(df):
    """
    Method to Clean Data, Engineer Features, and Prepare the Target Variable
    """
    inds = range(len(df))
    hour, ampm = np.zeros(len(df), dtype=np.int16), np.full(len(df), np.nan,  dtype='<U6')
    count = 0

    calms = [i for i in inds if df['Wind'].iat[i] == 'Calm']
    negs = [i for i in inds if i not in calms and float(df['Wind'].iat[i]) < -100]

    for i in inds:
        time_split = df['TimeEST'].iat[i].split(':')
        hour[i] = int(time_split[0])
        ampm[i] = time_split[1][len(time_split[1])-2:]
        if df['Temp'].iat[i] < -100:
            df['Temp'].iat[i] = (df['Temp'].iat[i - 1] + df['Temp'].iat[i + 1]) / 2

        if i in calms:
            last = [b for b in inds[i-4:i] if b not in negs]
            next = [b for b in inds[i:i+4] if b not in negs and b not in calms]
            last += next
            avg = sum(float(x) for x in df['Wind'].iloc[last]) / len(last)
            df['Wind'].iat[i] = avg
        elif i in negs:
            last = [b for b in inds[i - 4:i]]
            next = [b for b in inds[i:i + 4] if b not in negs and b not in calms]
            last += next
            avg = sum(float(x) for x in df.Wind[last]) / len(last)
            df['Wind'].iat[i] = avg

        label = df['Conditions'].iat[i]
        # if label == 'Heavy Rain' or label == 'Light Rain' or label == 'Rain':
        if label == 'Clear':
            df['Conditions'].iat[i] = 1
            count += 1
        else:
            df['Conditions'].iat[i] = 0

    encoder = LabelEncoder()
    period = encoder.fit_transform(ampm)
    df['Hour'] = hour.astype(np.float)
    df['Period'] = period.astype(np.float)
    df['Wind'] = df['Wind'].astype(np.float)
    df.insert(df.shape[1], column='Clear', value=df['Conditions'].astype(np.float))
    df = df.drop(['TimeEST', 'Conditions'], axis=1)
    return df


def train_test_split(train_ratio = 0.8):
    if isinstance(prepared_data, pd.DataFrame):
        data = prepared_data.to_numpy()
    else:
        data = prepared_data
    x = data[:, :data.shape[1] - 1]
    y = data[:, data.shape[1] - 1:]
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    train_size = math.floor(len(x) * train_ratio)

    x_train, y_train = x[train_size:], y[train_size:]
    x_test, y_test = x[:train_size], y[:train_size]

    return x_train, x_test, y_train, y_test

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

def accuracy(y_true, y_pred):
    preds = torch.argmax(y_pred, dim=1)
    return (preds == y_true).float().mean()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Dimensions
        self.input_dim = 6
        self.hidden_dim = 4
        self.output_dim = 1

        # Weights
        self.w1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim, dtype=torch.float, device=device))
        self.w1.requires_grad = False
        self.b1 = nn.Parameter(torch.randn(1, self.hidden_dim, dtype=torch.float, device=device))
        self.b1.requires_grad = False
        self.w2 = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim, dtype=torch.float, device=device))
        self.w2.requires_grad = False
        self.b2 = nn.Parameter(torch.randn(1, self.output_dim, dtype=torch.float, device=device))
        self.b2.requires_grad = False

    def forward(self, X):
        self.z1 = torch.matmul(X, self.w1) + self.b1 # multiply inputs by weights connecting hidden layer
        self.z2 = self.sigmoid(self.z1) # sigmoid activation function
        self.z3 = torch.matmul(self.z2, self.w2) + self.b2 # multiply outputs of hidden layer by weights connecting final output layer
        o = self.sigmoid(self.z3) # final sigmoid activation function - produces output value [0,1]
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def backward(self, X, y, o, lr=0.01):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sigmoid w.r.t error
        self.z2_error = torch.matmul(self.o_delta, self.w2.t())
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        # update weights and biases
        self.w1 += torch.matmul(X.t(), self.z2_delta) * lr
        self.w2 += torch.matmul(self.z2.t(), self.o_delta) * lr

        self.b1 += self.z2_delta.sum() * lr
        self.b2 += self.o_delta.sum() * lr

    def train_model(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        return o

    def save_weights(self, model):
        torch.save(model, "MLP")



prepared_data = preprocess(data_df)
features = prepared_data.drop('Clear', axis=1).columns.values
""" ['Temp' 'Dew' 'Humidity' 'Wind' 'Hour' 'Period'] --> ['Clear'] (1 or 0)"""

MMS = MinMaxScaler()
prepared_data = MMS.fit_transform(prepared_data)

x_train, x_test, y_train, y_test = train_test_split()
x_train, y_train = x_train.to(device), y_train.to(device)

model = MLP().to(device)
bce = torch.nn.BCELoss()

epochs = 500
#lr = 1e-5
for _ in range(epochs):

    y_pred = model.train_model(x_train, y_train)

    acc = accuracy(y_train, y_pred)
    #loss = mse(y_train, y_pred)
    loss = bce(y_train, y_pred)

    print('Loss: {}'.format(round(loss.item(), 4)), '  Accuracy: {}'.format(round(acc.item(), 4)))

y_pred = y_pred.cpu().numpy()
y_true = y_train.cpu().numpy()

results = pd.DataFrame(data=y_pred, columns=['Pred'])
results['True'] = y_true

results['Pred_01'] = (results['Pred'] > 0.5).astype(float)

correct = 0
for i in range(len(results)):
    if results['True'].iat[i] == results['Pred_01'].iat[i]:
        correct += 1

acc = correct / len(results) * 100

print(acc)





