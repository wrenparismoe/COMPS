import torch
from sklearn.preprocessing import MinMaxScaler

class FastDataLoader(object):
    def __init__(self, *tensors, batch_size):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size


        # Calculate the number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class Sliding_Window_Dataloader(object):
    def __init__(self, data):
        self.rows = data.shape[0] - 1
        self.cols = data.shape[1]
        self.features = data.columns.values
        self.x = data.values[:-1, :]
        self.y = data['Close'].shift(-1).dropna().values.reshape(self.rows, 1)


    def split(self, window_size, batch_size, scaler=MinMaxScaler()):
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

            train_dl = FastDataLoader(x_train, y_train, batch_size=batch_size)
            train_out.append(train_dl)

            x_test = self.x[end_ind: end_ind + 1, :]
            y_test = self.y[end_ind: end_ind + 1, :]
            x_test = scaler.transform(x_test)

            x_test = torch.tensor(x_test, requires_grad=False, dtype=torch.float32)
            y_test = torch.tensor(y_test, requires_grad=False, dtype=torch.float32)

            test_dl = FastDataLoader(x_test, y_test, batch_size=batch_size)
            test_out.append(test_dl)

        window_data = zip(train_out, test_out)
        return window_data