from robustness.timeseries_robust import add_timeseries_noise
import datetime
from posixpath import split
import numpy as np
import pandas as pd
import pandas_datareader
import torch
from torch.utils.data import DataLoader

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))


def get_dataloader(stocks, input_stocks, output_stocks, batch_size=16, train_shuffle=True, start_date=datetime.datetime(2000, 6, 1), end_date=datetime.datetime(2021, 2, 28), window_size=500, val_split=3200, test_split=3700, modality_first=True, cuda=True):
    stocks = np.array(stocks)

    def fetch_finance_data(symbol, start, end):
        return pandas_datareader.data.DataReader(symbol, 'yahoo', start, end)

    data = []
    for stock in stocks:
        fetch = fetch_finance_data(stock, start_date, end_date)
        print(stock + ' length: ' + str(len(fetch)))
        fetch.insert(0, 'Symbol', stock)
        data.append(fetch)

    data = pd.concat(data)
    data = data.sort_values(by=['Date', 'Symbol'])

    input_stocks = np.array(
        [np.where(data['Symbol'] == x)[0][0] for x in input_stocks])
    output_stocks = np.array(
        [np.where(data['Symbol'] == x)[0][0] for x in output_stocks])

    X = torch.tensor(list(data['Open'])).view(-1, len(stocks))
    RX = torch.log(X[1:] / X[:-1])
    Y = RX[window_size:, output_stocks]
    Y = Y * Y

    RX = RX / torch.std(RX[:window_size + val_split])
    Y = Y / torch.std(Y[:val_split])

    X = [RX[i:i + window_size, input_stocks].reshape(
        1, window_size, -1) for i in range(len(RX) - window_size)]
    X = torch.cat(X)

    if cuda:
        X = X.cuda()
        Y = Y.cuda()

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y, modality_first):
            self.X, self.Y = X, Y
            self.modality_first = modality_first

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            # Data augmentation
            def quantize(x, y):
                hi = torch.max(x)
                lo = torch.min(x)
                x = (x - lo) * 25 / (hi - lo)
                x = torch.round(x)
                x = x * (hi - lo) / 25 + lo
                return x, y

            x, y = quantize(self.X[index], self.Y[index])

            if not modality_first:
                return x, y
            else:
                if len(x.shape) == 2:
                    x = x.permute([1, 0])
                    x = list(x)
                    x.append(y)
                    return x
                else:
                    x = x.permute([0, 2, 1])
                    res = []
                    for data, label in zip(x, y):
                        data = list(data)
                        data.append(label)
                        res.append(data)
                    return res

    train_ds = MyDataset(X[:val_split], Y[:val_split], modality_first)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=train_shuffle, batch_size=batch_size)
    val_ds = MyDataset(X[val_split:test_split],
                       Y[val_split:test_split], modality_first)
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    test_loader = []
    for noise_level in range(10):
        X_robust = torch.tensor(add_timeseries_noise(
            X[test_split:].cpu().numpy(), noise_level=noise_level/10), dtype=torch.float32)
        if cuda:
            X_robust = X_robust.cuda()
        test_ds = MyDataset(X_robust, Y[test_split:], modality_first)
        test_loader.append(torch.utils.data.DataLoader(
            test_ds, shuffle=False, batch_size=batch_size, drop_last=False))

    return train_loader, val_loader, test_loader
