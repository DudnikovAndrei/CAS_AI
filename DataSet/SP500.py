import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import bs4 as bs
import requests
import yfinance as yf
import datetime
import time

class SP500DataSet():
    def __init__(self, batch_size=32, train_val_test_split=[80, 10, 10]):
        self.url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.stocks_fname = "sp500_closefull.csv"
        self.start = datetime.datetime(2010, 1, 1)
        self.stop = datetime.datetime.now()
        self.Ntest = 1000
        self.now = time.time()
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split

    def load_data(self):
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
        print("Dropping columns due to nans > 50%:", df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)
        df0 = df0.drop(df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns, axis=1)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())
        return df0

    def transform_to_dataset(self, df):
        # Split labels and features
        labels = df.SPY.values
        features = df.iloc[:, :-1].values

        # Convert to tensor
        tensor_labels = torch.tensor(labels).unsqueeze(1).float()
        tensor_features = torch.tensor(features).float()

        # Create tensor dataset
        return TensorDataset(tensor_features, tensor_labels)

    def get_returns(self, df0):
        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        df_returns.dropna(axis=0, how='any', inplace=True)
        df_returns.SPY = [1 if spy > 0 else 0 for spy in df_returns.SPY]
        return df_returns

    def get_train_data(self, nTest):
        df = self.load_data()
        df_returns = self.get_returns(df)
        return df_returns.iloc[: -nTest]

    def get_test_data(self, nTest):
        df = self.load_data()
        df_returns = self.get_returns(df)
        return df_returns.iloc[-nTest:]

    def get_rows(self, rows, percent):
        rows = int(percent * rows / 100)
        return rows

    def get_data_loader(self, df_returns, start, end):
        data = df_returns.iloc[start:end]
        dataset = self.transform_to_dataset(data)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        return data_loader

    def get_train_loader(self):
        train_data_loader, _, _ = self.get_data_loaders()
        return train_data_loader

    def get_val_loader(self):
        _, val_data_loader, _ = self.get_data_loaders()
        return val_data_loader

    def get_test_loader(self):
        _, _, test_data_loader = self.get_data_loaders()
        return test_data_loader

    def get_data_loaders(self):
        df = self.load_data()
        df_returns = self.get_returns(df)
        train_percent, val_percent, test_percent = self.train_val_test_split
        rows = df_returns.shape[0]

        val_rows = self.get_rows(rows, val_percent)
        test_rows = self.get_rows(rows, test_percent)
        train_rows = rows - test_rows - val_rows

        train_data_loader = self.get_data_loader(df_returns, 0, train_rows)
        val_data_loader = self.get_data_loader(df_returns, train_rows, train_rows + val_rows)
        test_data_loader = self.get_data_loader(df_returns, train_rows + val_rows, train_rows + val_rows + test_rows)
        return train_data_loader, val_data_loader, test_data_loader

