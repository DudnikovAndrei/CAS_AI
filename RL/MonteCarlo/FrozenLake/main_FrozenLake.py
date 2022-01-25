
import gym
import numpy as np

from Policy import Policy
from Episode import Episode
from Control import Control
from Prediction import Prediction

env = gym.make('FrozenLake-v1', is_slippery=True)
env.render()

policy = Policy()
episode = Episode()
control = Control()
prediction = Prediction()

# Obtain the action-value function
# ======== ADJUST HERE AS APPROPRIATE ========
Q = prediction.mc_prediction_q(env, 5000, episode.generate_episode_from_limit_stochastic, policy)
V = dict((k,np.max(v)) for k, v in Q.items())

gen_policy, Q = control.mc_control(env, episode, control, 50000, 0.02)

# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())
print(V)
print(gen_policy)

wins = 0
losses = 0
for i_episode in range(5000):
  state = env.reset()
  while state < 15:
    action = gen_policy.get(state)
    state, reward, done, info = env.step(action)
    if done:
      if reward > 0:
        wins += 1
      else:
        losses += 1
      break
print(f'Success rate (wins): {wins/(wins+losses) *100}%')


