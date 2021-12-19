import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
# from pl_bolts.datasets import DummyDataset


import numpy as np
import pandas as pd
from functools import reduce
import bs4 as bs
import requests
import yfinance as yf
import datetime
import time
# import pandas_datareader as web
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


class MyDataset():

    def __init__(self):
        self.url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.stocks_fname = "sp500_closefull.csv"
        self.start = datetime.datetime(2010, 1, 1)
        self.stop = datetime.datetime.now()
        self.Ntest = 1000
        self.now = time.time()

    def get_train_test(self) -> pd.DataFrame:
        start = self.start
        end = self.stop

        if not os.path.isfile(self.stocks_fname):
            resp = requests.get(self.url)
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})
            tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            tickers = [s.replace('\n', '') for s in tickers]
            data = yf.download(tickers, start=start, end=end)
            data['Adj Close'].to_csv(self.stocks_fname)

        df0 = pd.read_csv(self.stocks_fname, index_col=0, parse_dates=True)
        df_spy = yf.download("SPY", start=start, end=end)
        df_spy = df_spy.loc[:, ['Adj Close']]
        df_spy.columns = ['SPY']

        df0 = pd.concat([df0, df_spy], axis=1)
        df0.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:",
              df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)
        df0 = df0.drop(df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns, 1)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())

        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        df_returns.dropna(axis=0, how='any', inplace=True)
        df_returns.SPY = [1 if spy > 0 else 0 for spy in df_returns.SPY]

        train_data = df_returns.iloc[:-self.Ntest]
        train_labels = torch.tensor(train_data.SPY.values).float()
        train_features = train_data.drop('SPY', axis=1).values
        train_features = torch.tensor(train_features).float()

        test_data = df_returns.iloc[len(df_returns) - self.Ntest:(len(df_returns)) - int(self.Ntest / 2)]
        test_labels = torch.tensor(test_data.SPY.values).float()
        test_features = test_data.drop('SPY', axis=1).values
        test_features = torch.tensor(test_features).float()

        val_data = df_returns.iloc[int(self.Ntest / 2):]
        val_labels = torch.tensor(val_data.SPY.values).float()
        val_features = val_data.drop('SPY', axis=1).values
        val_features = torch.tensor(val_features).float()

        return train_labels, train_features, test_labels, test_features, val_labels, val_features

dataset = MyDataset()
train_labels, train_features, test_labels, test_features , val_labels, val_features = dataset.get_train_test()

print("test")