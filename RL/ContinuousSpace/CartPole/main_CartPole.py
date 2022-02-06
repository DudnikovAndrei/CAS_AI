
import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment
from Discretization import Discretization
from QLearningAgent import QLearningAgent
from Runner import Runner
from Visualizer import Visualizer

visualizer = Visualizer()

# Create an environment and set random seed
environment = Environment('CartPole-v0')
environment.show_environment()
env = environment.get_env()

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

# Generate some samples from the state space
print("State space samples:")
print(np.array([env.observation_space.sample() for i in range(10)]))

# Explore the action space
print("Action space:", env.action_space)

# Generate some samples from the action space
print("Action space samples:")
print(np.array([env.action_space.sample() for i in range(10)]))

low = env.observation_space.low
high = env.observation_space.high
low[1] = -3.0
low[3] = -2.0
high[1] = 3.0
high[3] = 2.0
bins = (10, 10, 10, 10)

discretization = Discretization()
grids = discretization.create_uniform_grid(low, high, bins)  # [test]
print(grids)

samples = np.array([env.observation_space.sample() for i in range(5)] )
discretized_samples = np.array([discretization.discretize(sample, grids) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")
print("\nGrid:", repr(grids), sep="\n")

cart_samples = discretization.transform_cart_simples_to_array(samples)
pole_samples = discretization.transform_pole_simples_to_array(samples)
cart_discretized = discretization.transform_cart_discretized_to_array(discretized_samples)
pole_discretized = discretization.transform_cart_discretized_to_array(discretized_samples)

visualizer.visualize_samples(cart_samples, cart_discretized, grids[0:2])
visualizer.visualize_samples(pole_samples, pole_discretized, grids[0:2])

# Create a grid to discretize the state space
state_grid = discretization.create_uniform_grid(env.observation_space.low, env.observation_space.high, bins)
state_samples = np.array([env.observation_space.sample() for i in range(10)])

print('state_grid:', state_grid)

q_agent = QLearningAgent(env, state_grid)
discretized_state_samples = np.array([discretization.discretize(sample, state_grid) for sample in state_samples])
visualizer.visualize_samples(discretization.transform_cart_simples_to_array(state_samples), discretization.transform_cart_discretized_to_array(discretized_state_samples), state_grid[0:2])
plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space

runner = Runner()
scores = runner.run(q_agent, env, discretization)

# Plot scores obtained per episode
plt.plot(scores); plt.title("Scores");
plt.show()
rolling_mean = visualizer.plot_scores(scores)

visualizer.visualize_agent(env, q_agent, discretization)

# Run in test mode and analyze scores obtained
test_scores = runner.run(q_agent, env, discretization, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = visualizer.plot_scores(test_scores, rolling_window=10)