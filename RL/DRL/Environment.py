import gym
import numpy as np

class MyEnv(gym.Env):
    def __init__(self, df_train, df_test, play=False):
        self.df_train = df_train
        self.df_test = df_test
        self.play = play
        if not self.play:
            self.df = self.df_train
        else:
            self.df = self.df_test
        # self.config = EnvConfig()
        self.current_idx = 0
        self.stocks = ['AAPL', 'MSFT', 'AMZN', 'NFLX', 'XOM', 'JPM', 'T']  # target stocks
        # self.stocks_adj_close_names = [stock + '_Adj_Close' for stock in self.stocks] # AAPL_Adj_Close so heissen die Spalten bei mir ?

        # self.weights = np.full((len(self.config.stocks)), 1 / len(self.config.stocks), dtype=float)

        # cash per stock_values
        self.initial_cash = 10_000  # self.weights * self.config.initial_cash
        self.cash = self.initial_cash

        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.stocks))

        self.states = self.df.loc[:, ~self.df.columns.isin(self.stocks)].to_numpy()  # aufpassen, tatsächlich existierende Spaltennamen
        # self.rewards = self.df[self.stocks].to_numpy()
        self.n = len(self.df)

        self.nA = 3

    def reset(self):

        self.cash = self.initial_cash
        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.stocks))

        self.current_idx = 0

        next_state = self.states[self.current_idx]
        next_state = np.array(next_state).reshape(1, -1)
        # next_state = torch.tensor(next_state).float().to(self.config.device)
        return next_state

    def step(self, action):
        reward = 0
        if self.current_idx >= self.n:
            raise Exception("Episode already done")

        # state transition
        done = (self.current_idx == self.n - 1)

        # apply action TODO
        probs = [1 / 3, 1 / 3, 1 / 3]
        action = np.random.choice(np.arange(self.nA), p=probs)
        print('Action', action)
        self.current_idx += 1

        if not done:
            # compute reward TODO
            previous = np.sum(self.states[self.current_idx - 1])
            next = np.sum(self.states[self.current_idx])
            reward = (next - previous) / previous

            next_state = self.states[self.current_idx]
            next_state = np.array(next_state).reshape(1, -1)
            # next_state = torch.tensor(next_state).reshape(1, -1).float().to(self.config.device)

        else:
            next_state = None
            reward = 0
            # self.portfolio_value = np.dot(self.stock_values, current_prices)

        return next_state, reward, done

