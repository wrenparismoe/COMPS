from System import *
import torch
import torch.nn as nn
import plotly.graph_objects as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
torch.random.seed = 2

# Basic model for learning low level pytorch

df = pd.read_csv('/datasets/New_York_Hourly.csv', index_col='date')
df = df.rename(columns={'TemperatureF': 'temp'})
data = df['temp']
for i in range(len(data)):
    if data.iat[i] < -100:
        data.iat[i] = (data.iat[i-1] + data.iat[i+1])/2
x = np.array([i for i in range(1, len(data))])
y_true = np.array(data.iloc[:-1])
x = torch.tensor(x, dtype=torch.float32)
y_true = torch.tensor(y_true, dtype=torch.float32)

def mse(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

class LinearUnit(nn.Module):
    def __init__(self):
        super(LinearUnit, self).__init__()
        # initial slope/weight and intercept
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.w * x + self.b

    # Stochastic gradient descent (Backpropagation)
    def backward(self, lr=1e-8):
        with torch.no_grad():
            self.w -= lr * self.w.grad
            self.b -= lr * self.b.grad

            self.w.grad.zero_()
            self.b.grad.zero_()

model = LinearUnit()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for _ in range(epochs):
    y_pred = model(x)

    loss = mse(y_true, y_pred)
    print('loss:', loss)

    # automatically computes derivative of y with respect to w and b
    loss.backward()

    # SGD
    model.backward()

    # Adam (not very good for single unit LR
    # optimizer.step()
    # optimizer.zero_grad()







eq = 'y = ' + str(round(model.w.item(), 3)) + 'x + ' + str(round(model.b.item(), 3))
title = 'Linear Regression:  ' + eq + '  |  Loss: ' + str(round(loss.detach().item(), 3))
index = data.index[:-1]

plot = plt.Figure()
plot.add_trace(plt.Scatter(x=index, y=y_true.numpy(), mode='markers'))
plot.add_trace(plt.Scatter(x=index, y=y_pred.detach().numpy(), mode='lines'))
plot.update_layout(title=title)

plot.show()

