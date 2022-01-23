import warnings
warnings.filterwarnings('ignore')

from DataSet.SP500 import SP500DataSet
from Environment import MyEnv

# DataSet
nTests = 1000
sp500_dataset = SP500DataSet()
train = sp500_dataset.get_train_data(nTests)
test = sp500_dataset.get_test_data(nTests)

train_env = MyEnv(train, test)
train_env.reset()

next_state, reward, done = train_env.step(1)
next_state, reward, done = train_env.step(1)