
import requests
import pandas as pd
import numpy as np
import os.path

import warnings
warnings.filterwarnings('ignore')

class MyDataset():

    def __init__(self):
        self.Ntest = 1000
        self.url = "https://lazyprogrammer.me/course_files/sp500_closefull.csv"
        self.fname = "sp500_closefull.csv"

        if not os.path.isfile(self.fname):
            r = requests.get(self.url)
            open(self.fname, 'wb').write(r.content)

    def get_train_test(self):
        df0 = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
        df0.dropna(axis=0, how='all', inplace=True)
        df0.dropna(axis=1, how='any', inplace=True)

        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        # split into train and test
        df_returns.dropna(axis=0, how='all', inplace=True)
        train_data = df_returns.iloc[:-self.Ntest]
        test_data = df_returns.iloc[-self.Ntest:]

        return train_data, test_data


if __name__  == "__main__":
   dataset = MyDataset()
   train, test = dataset.get_train_test()
  # print(train.iloc[0, :])