import gym
from gym import spaces
import numpy as np
import torch

from dataset import MyDataset


class MyEnv(gym.Env):

    def __init__(self, df):
        self.df = df
        self.current_idx = 0
        self.invested = 0 # Bestand am Wertpapier
        #self.states = df.values
        #self.action_space = [0, 1, 2] # BUY, SELL, HOLD
        self.feats = ['AAPL', 'MSFT', 'AMZN'] # alle möglich ausser SPY, target, oder reword column
        self.states = self.df[self.feats].to_numpy()
        self.rewards = self.df['SPY'].to_numpy()
        self.n = len(self.states)
        self.bins = 40 # jede Zahl, aber nicht all zu gross
        self.discretized = np.linspace(0., 1000., self.bins).astype(int)

        self.action_space = spaces.Box(
            low=0,
            high=10_000, shape=(2,),
            dtype=int
        )
        self.observation_space = spaces.Box( # TODO was ist min max der Daten
            low=0,
            high=np.nan,
            shape=(self.states.shape[1] + 1,), # + 1 durch invested Feld
            dtype=np.float32
        )

    def reset(self):
        self.current_idx = 0
        # TODO auch hier invested nicht vergessen
        next_state = self.states[self.current_idx].reshape(1, -1)
        next_state = torch.tensor(next_state).float()
        return next_state


    def step(self, action):

        if self.current_idx >= self.n:
            raise Exception("Episode already done")

        action = np.array([self.discretized[aa] for aa in action])
        action = action.ravel()
        # TODO Kaufen unbeschränkt, Verkaufen max den Bestand


        #TODO
        '''
        if action == 0:  # BUY
            self.invested = 1
        elif action == 1:  # SELL
            self.invested = 0

            # compute reward
        if self.invested:
            reward = self.rewards[self.current_idx] # reward ist log reward (np.exp - > reward), mal die Anzahl - gesamt reward
        else:
            reward = 0
            
        '''

        reward = None # TODO

        # state transition
        done = (self.current_idx == self.n - 1)

        self.current_idx += 1

        if not done:

            if action is None:
                raise Exception("NaNs detected!")
            next_state = self.states[self.current_idx]
            # TODO invested auch als state anhängen
            # next_state = list(next_state).append(self.invested)
            next_state = np.array(next_state).reshape(1, -1)
            next_state = torch.tensor(next_state).reshape(1, -1).float()
            # print(next_state)
        else:
            next_state = None # TODO das abfangen, wen man trainiert, done wird dabei true sein

        return next_state, reward, done


if __name__ == "__main__":
    dataset = MyDataset()
    train, test = dataset.get_train_test()

    train_env = MyEnv(train)
    test_env = MyEnv(test)
    num_states = train_env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = train_env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = train_env.action_space.high[0]
    lower_bound = train_env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    action = [np.random.randint(0, 39, 3)]
    print(action)

    first_state = train_env.reset()
    next_state, reward, done = train_env.step(action)
    print(first_state, next_state, reward, done)