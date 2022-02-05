
import gym
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

# Set plotting options
try:
    get_ipython().magic("matplotlib inline")
except:
    plt.ion()

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
plt.interactive(True)

class Environment():
    def __init__(self, envName):
        self.env = gym.make(envName)
        self.env.seed(505);

    def show_environment(self):
        self.env.reset()
        img = plt.imshow(self.env.render(mode='rgb_array'))
        for t in range(100):
            action = self.env.action_space.sample()
            img.set_data(self.env.render(mode='rgb_array'))
            plt.axis('off')
            display.clear_output(wait=True)
            state, reward, done, _ = self.env.step(action)
            if done & t > 100:
                print('Score: ', t + 1)
                break

        self.env.close()

    def show_environment_with_qagent(self, q_agent_new):
        state = self.env.reset()
        score = 0
        img = plt.imshow(self.env.render(mode='rgb_array'))
        for t in range(1000):
            action = q_agent_new.act(state, mode='test')
            img.set_data(self.env.render(mode='rgb_array'))
            plt.axis('off')
            display.clear_output(wait=True)
            state, reward, done, _ = self.env.step(action)
            score += reward
            if done:
                print('Score: ', score)
                break

        self.env.close()

    def get_env(self):
        return self.env