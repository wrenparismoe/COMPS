import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Sliding_Window_Dataset(object):
    def __init__(self, data, window_size=200):

        self.rows = data.shape[0] - 1
        self.cols = data.shape[1]
        self.features = data.columns.values
        self.x = data.values[:-1, :]
        self.y = data['Close'].shift(-1).dropna().values.reshape(self.rows, 1)


    def split(self, window_size, scaler=MinMaxScaler()):
        self.window_size = window_size
        train_out = []
        test_out = []

        for start_ind in range(self.rows - self.window_size - 1):
            end_ind = start_ind + self.window_size
            x_train = self.x[start_ind: end_ind, :]
            y_train = self.y[start_ind: end_ind, :]

            scaler.fit(x_train)
            x_train = scaler.transform(x_train)

            x_train = torch.tensor(x_train, requires_grad=False,dtype=torch.float32)
            y_train = torch.tensor(y_train, requires_grad=False, dtype=torch.float32)

            train_ds = TimeSeriesDataset(x_train, y_train)
            train_out.append(train_ds)

            x_test = self.x[end_ind: end_ind + 1, :]
            y_test = self.y[end_ind: end_ind + 1, :]
            x_test = scaler.transform(x_test)

            x_test = torch.tensor(x_test, requires_grad=False, dtype=torch.float32)
            y_test = torch.tensor(y_test, requires_grad=False, dtype=torch.float32)

            test_ds = TimeSeriesDataset(x_test, y_test)
            test_out.append(test_ds)

        window_data = zip(train_out, test_out)
        return window_data


