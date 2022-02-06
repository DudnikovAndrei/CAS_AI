
import pandas as pd
import matplotlib.collections as mc
import numpy as np
import matplotlib.pyplot as plt
import time

from IPython import display

# Set plotting options
try:
    get_ipython().magic("matplotlib inline")
except:
    plt.ion()

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
plt.interactive(True)

class Visualizer():
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

        plt.show()

    def plot_scores(self, scores, rolling_window=100):
        """Plot scores and optional rolling mean using specified window."""
        plt.plot(scores);
        plt.title("Scores");
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean);
        plt.show()
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

    def visualize_agent(self, env, q_agent, discretization):
        state = env.reset()
        score = 0
        img = plt.imshow(env.render(mode='rgb_array'))
        for t in range(1000):
            action = q_agent.act(state, discretization, mode='test')
            img.set_data(env.render(mode='rgb_array'))
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            state, reward, done, _ = env.step(action)
            score += reward
            time.sleep(0.2)
            if done:
                print('Score: ', score)
                break

        env.close()