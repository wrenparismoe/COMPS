from System import *
import torch.nn as nn
import torch
import plotly.graph_objects as plt

# Basic model for learning low level pytorch

torch.random.seed = 2
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('/datasets/New_York_Hourly.csv', index_col='date')
df = df.rename(columns={'TemperatureF': 'temp'})
data = df['temp']
for i in range(len(data)):
    if data.iat[i] < -100:
        data.iat[i] = (data.iat[i - 1] + data.iat[i + 1]) / 2
x = np.array([i for i in range(1, len(data))])
y_true = np.array(data.iloc[:-1])
x = torch.tensor(x, dtype=torch.float32)
y_true = torch.tensor(y_true, dtype=torch.float32)

class PolyUnit(nn.Module):
    def __init__(self):
        super(PolyUnit, self).__init__()
        # initial slope/weight and intercept
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    # create output function for model: y = w * x + b
    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

    def backward(self):
        with torch.no_grad():
            self.a -= lr * self.b.grad
            self.b -= lr * self.b.grad
            self.c -= lr * self.c.grad

            self.a.grad.zero_()
            self.b.grad.zero_()
            self.c.grad.zero_()

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

model = PolyUnit()

epochs = 100
lr = 1e-12
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(epochs):
    y_pred = model(x)

    loss = mse(y_true, y_pred)
    print('loss:', loss)

    # automatically computes derivative of y with respect to w and b
    loss.backward()

    model.backward()

    # optimizer.step()
    # optimizer.zero_grad()





eq = 'y = ' + str(round(model.a.item(), 6)) + 'x**2 + ' + str(round(model.b.item(), 5)) + 'x + ' + str(round(model.c.item(), 3))
title = 'Polynomial Regression:  ' + eq + '  |  Loss: ' + str(round(loss.detach().item(), 3))
index = data.index[:-1]

plot = plt.Figure()
plot.add_trace(plt.Scatter(x=index, y=y_true.numpy(), mode='markers'))
plot.add_trace(plt.Scatter(x=index, y=y_pred.detach().numpy(), mode='lines'))
plot.update_layout(title=title)

plot.show()