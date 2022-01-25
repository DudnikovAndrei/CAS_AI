
import sys
import numpy as np
import pandas as pd
import matplotlib.collections as mc
import matplotlib.pyplot as plt

class Discretization():
    def create_uniform_grid(self, low, high, bins=(10, 10)):
        """Define a uniformly-spaced grid that can be used to discretize a space.

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        bins : tuple
            Number of bins along each corresponding dimension.

        Returns
        -------
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        """
        grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
        print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
        return grid

    def visualize_samples(self, samples, discretized_samples, grid, low=None, high=None):
        """Visualize original and discretized samples on a given 2-dimensional grid."""

        fig, ax = plt.subplots(figsize=(10, 10))

        # Show grid
        ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
        ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
        ax.grid(True)

        # If bounds (low, high) are specified, use them to set axis limits
        if low is not None and high is not None:
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
        else:
            # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
            low = [splits[0] for splits in grid]
            high = [splits[-1] for splits in grid]

        array_low = np.array([low]).T
        array_high = np.array([high]).T
        grid_1 = grid

        # Map each discretized sample (which is really an index) to the center of corresponding grid cell
        grid_extended = np.hstack((array_low, grid_1, array_high))  # add low and high ends
        grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
        locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples

        ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
        ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
        ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  # add a line connecting each original-discretized sample
        ax.legend(['original', 'discretized'])

    def run(self, agent, env, num_episodes=20000, mode='train'):
        """Run agent in given reinforcement learning environment and return scores."""
        scores = []
        max_avg_score = -np.inf
        for i_episode in range(1, num_episodes + 1):
            # Initialize episode
            state = env.reset()
            action = agent.reset_episode(state)
            total_reward = 0
            done = False

            # Roll out steps until done
            while not done:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = agent.act(state, reward, done, mode)

            # Save final score
            scores.append(total_reward)

            # Print episode stats
            if mode == 'train':
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score
                if i_episode % 100 == 0:
                    print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                    sys.stdout.flush()

        return scores

    def plot_scores(self, scores, rolling_window=100):
        """Plot scores and optional rolling mean using specified window."""
        plt.plot(scores);
        plt.title("Scores");
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean);
        return rolling_mean

    def plot_q_table(self, q_table):
        """Visualize max Q-value for each state and corresponding action."""
        q_image = np.max(q_table, axis=2)  # max Q-value for each state
        q_actions = np.argmax(q_table, axis=2)  # best action for each state

        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(q_image, cmap='jet');
        cbar = fig.colorbar(cax)
        for x in range(q_image.shape[0]):
            for y in range(q_image.shape[1]):
                ax.text(x, y, q_actions[x, y], color='white',
                        horizontalalignment='center', verticalalignment='center')
        ax.grid(False)
        ax.set_title("Q-table, size: {}".format(q_table.shape))
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')