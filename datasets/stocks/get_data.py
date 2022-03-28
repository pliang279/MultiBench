"""Implements dataloaders for the robotics datasets."""

from robustness.timeseries_robust import add_timeseries_noise
import copy
import datetime
from posixpath import split
import io
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader
from torch import nn


def get_dataloader(stocks, input_stocks, output_stocks, batch_size=16, train_shuffle=True, start_date=datetime.datetime(2000, 6, 1), end_date=datetime.datetime(2021, 2, 28), window_size=500, val_split=3200, test_split=3700, modality_first=True, cuda=True):
    """Generate dataloader for stock data.

    Args:
        stocks (list): List of strings of stocks to grab data for.
        input_stocks (list): List of strings of input stocks
        output_stocks (list): List of strings of output stocks
        batch_size (int, optional): Batchsize. Defaults to 16.
        train_shuffle (bool, optional): Whether to shuffle training dataloader or not. Defaults to True.
        start_date (datetime, optional): Start-date to grab data from. Defaults to datetime.datetime(2000, 6, 1).
        end_date (datetime, optional): End-date to grab data from. Defaults to datetime.datetime(2021, 2, 28).
        window_size (int, optional): Window size. Defaults to 500.
        val_split (int, optional): Number of samples in validation split. Defaults to 3200.
        test_split (int, optional): Number of samples in test split. Defaults to 3700.
        modality_first (bool, optional): Whether to make modality the first index or not. Defaults to True.
        cuda (bool, optional): Whether to load data to cuda objects or not. Defaults to True.

    Returns:
        tuple: Tuple of training data-loader, test data-loader, and validation data-loader.
    """
    stocks = np.array(stocks)

    def _fetch_finance_data(symbol, start, end):
        url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start.strftime("%s")}&period2={end.strftime("%s")}&interval=1d&events=history&includeAdjustedClose=true'
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        text = requests.get(url, headers={'User-Agent': user_agent}).text
        return pd.read_csv(io.StringIO(text), encoding='utf8', parse_dates=True, index_col=0)

    data = []
    for stock in stocks:
        fetch = _fetch_finance_data(stock, start_date, end_date)
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
        X = X.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        Y = Y.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    class _MyDataset(torch.utils.data.Dataset):
        """"""
        def __init__(self, X, Y, modality_first):
            """Initialize Dataset Class"""
            self.X, self.Y = X, Y
            self.modality_first = modality_first

        def __len__(self):
            """Get length of dataset."""
            return len(self.X)

        def __getitem__(self, index):
            """Get item from dataset."""
            # Data augmentation
            def _quantize(x, y):
                hi = torch.max(x)
                lo = torch.min(x)
                x = (x - lo) * 25 / (hi - lo)
                x = torch.round(x)
                x = x * (hi - lo) / 25 + lo
                return x, y

            x, y = _quantize(self.X[index], self.Y[index])

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

    train_ds = _MyDataset(X[:val_split], Y[:val_split], modality_first)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=train_shuffle, batch_size=batch_size)
    val_ds = _MyDataset(X[val_split:test_split],
                       Y[val_split:test_split], modality_first)
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    test_loader = dict()
    test_loader['timeseries'] = []
    for noise_level in range(9):
        X_robust = copy.deepcopy(X[test_split:].cpu().numpy())
        X_robust = torch.tensor(add_timeseries_noise(
            X_robust, noise_level=noise_level/10), dtype=torch.float32)
        if cuda:
            X_robust = X_robust.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        test_ds = _MyDataset(X_robust, Y[test_split:], modality_first)
        test_loader['timeseries'].append(torch.utils.data.DataLoader(
            test_ds, shuffle=False, batch_size=batch_size, drop_last=False))
    print(len(test_loader))
    return train_loader, val_loader, test_loader


class Grouping(nn.Module):
    """Module to collate stock data."""
    
    def __init__(self, n_groups):
        """Instantiate grouper. n_groups determines the number of groups."""
        super().__init__()
        self.n_groups = n_groups

    def forward(self, x):
        """Apply grouper to input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            list: List of outputs
        """
        x = x.permute(2, 0, 1)

        n_modalities = len(x)
        out = []
        for i in range(self.n_groups):
            start_modality = n_modalities * i // self.n_groups
            end_modality = n_modalities * (i + 1) // self.n_groups
            sel = list(x[start_modality:end_modality])
            sel = torch.stack(sel, dim=len(sel[0].size()))
            out.append(sel)

        return out
