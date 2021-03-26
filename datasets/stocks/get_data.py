import datetime
import numpy as np
import pandas as pd
import pandas_datareader
import torch
from torch.utils.data import DataLoader

def get_dataloader(stocks, input_stocks, output_stocks, batch_size=16, train_shuffle=True, start_date=datetime.datetime(2000, 6, 1), end_date=datetime.datetime(2021, 2, 28), window_size=500, val_split=3200, test_split=3700):
    stocks = np.array(stocks)

    input_stocks = np.array([np.where(stocks == x)[0][0] for x in input_stocks])
    output_stocks = np.array([np.where(stocks == x)[0][0] for x in output_stocks])

    def fetch_finance_data(symbol, start, end):
        return pandas_datareader.data.DataReader(symbol, 'yahoo', start, end)

    data = []
    for stock in stocks:
        fetch = fetch_finance_data(stock, start_date, end_date)
        fetch.insert(0, 'Symbol', stock)
        data.append(fetch)

    data = pd.concat(data)
    data = data.sort_values(by=['Date', 'Symbol'])

    X = torch.tensor(list(data['Open'])).view(-1, len(stocks))
    RX = torch.log(X[1:] / X[:-1])
    SRX = RX * RX
    RX = RX / torch.std(RX[:window_size + val_split])
    SRX = SRX / torch.std(SRX[:window_size + val_split])

    Y = SRX[window_size:, output_stocks]
    X = [RX[i:i + window_size, input_stocks].reshape(1, window_size, -1) for i in range(len(RX) - window_size)]
    X = torch.cat(X)

    X = X.cuda()
    Y = Y.cuda()

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X, self.Y = X, Y

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

            return quantize(self.X[index], self.Y[index])

    train_ds = MyDataset(X[:val_split], Y[:val_split])
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=train_shuffle, batch_size=batch_size)
    val_ds = MyDataset(X[val_split:test_split], Y[val_split:test_split])
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    test_ds = MyDataset(X[test_split:], Y[test_split:])
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, drop_last=False)

    return train_loader, val_loader, test_loader
