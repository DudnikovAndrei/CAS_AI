
import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment
from Discretization import Discretization
from QLearningAgent import QLearningAgent

# Create an environment and set random seed
environment = Environment('MountainCar-v0')
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

print(np.linspace(-1.0, 1.0, 10))
print(np.linspace(-1.0, 1.0, 10)[1:-1])

discreditation = Discretization()

low = [-1.0, -5.0]
high = [1.0, 5.0]
bins = (10, 10)
grids = discreditation.create_uniform_grid(low, high, bins)  # [test]
print(grids)

samples = np.array(
    [[-1.0, -5.0],
     [-0.81, -4.1],
     [-0.8, -4.0],
     [-0.5, 0.0],
     [0.2, -1.9],
     [0.8, 4.0],
     [0.81, 4.1],
     [1.0, 5.0]])

# Create a grid to discretize the state space
state_grid = discreditation.create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
q_agent = QLearningAgent(env, state_grid)

discretized_samples = np.array([q_agent.discretize(sample, grids) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")
print("\nGrid:", repr(grids), sep="\n")

samples.shape, type(grids)
discreditation.visualize_samples(samples, discretized_samples, grids, low, high)

# Obtain some samples from the space, discretize them, and then visualize them
state_samples = np.array([env.observation_space.sample() for i in range(10)])
discretized_state_samples = np.array([q_agent.discretize(sample, state_grid) for sample in state_samples])
discreditation.visualize_samples(state_samples, discretized_state_samples, state_grid, env.observation_space.low, env.observation_space.high)
plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space

scores = discreditation.run(q_agent, env)

# Plot scores obtained per episode
plt.plot(scores); plt.title("Scores");
rolling_mean = discreditation.plot_scores(scores)

# Run in test mode and analyze scores obtained
test_scores = discreditation.run(q_agent, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = discreditation.plot_scores(test_scores)

discreditation.plot_q_table(q_agent.q_table)

# TODO: Create a new agent with a different state space grid
state_grid_new = discreditation.create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
q_agent_new = QLearningAgent(env, state_grid_new)
q_agent_new.scores = []  # initialize a list to store scores for this agent

# Train it over a desired number of episodes and analyze scores
# Note: This cell can be run multiple times, and scores will get accumulated
q_agent_new.scores += discreditation.run(q_agent_new, env, num_episodes=50000)  # accumulate scores
rolling_mean_new = discreditation.plot_scores(q_agent_new.scores)

# Run in test mode and analyze scores obtained
test_scores = discreditation.run(q_agent_new, env, num_episodes=100, mode='test')
print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
_ = discreditation.plot_scores(test_scores)

# Visualize the learned Q-table
discreditation.plot_q_table(q_agent_new.q_table)

environment.show_environment_with_qagent(q_agent_new)
