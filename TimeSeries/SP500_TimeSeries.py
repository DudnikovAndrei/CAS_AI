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

    def transform_to_timeseries_df(self, df):
        # create time index
        df['date'] = df.index.to_pydatetime()
        df["time_idx"] = df["date"].dt.year * 12 * 31 + df["date"].dt.month * 31 + df["date"].dt.day
        df["time_idx"] -= int(df["time_idx"].min())
        # create group (At least one groupt is used for the TimeSeriesDataSet)
        df['group_id'] = 0
        return df

    def get_train_dataset(self, nTest):
        df_return = self.get_train_data(nTest)
        return self.transform_to_timeseries_df(df_return)

    def get_returns(self, df0):
        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = df0[name]

        df_returns.dropna(axis=0, how='any', inplace=True)
        return df_returns

    def get_train_data(self, nTest):
        df = self.load_data()
        df_returns = self.get_returns(df)
        return df_returns.iloc[: -nTest]

    def get_test_data(self, nTest):
        df = self.load_data()
        df_returns = self.get_returns(df)
        return df_returns.iloc[-nTest:]


