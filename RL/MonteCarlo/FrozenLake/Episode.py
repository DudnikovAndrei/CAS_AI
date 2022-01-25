
import random
import numpy as np

class Episode():
    # our starting policy
    def generate_episode(self, env, policy):
        env.reset()
        episode = []
        done = False

        while not done:
            timestep = []
            state = env.env.s
            timestep.append(state)
            n = random.uniform(0, sum(policy[state].values()))
            top_range = 0
            for prob in policy[state].items():
                top_range += prob[1]
                if n < top_range:
                    action = prob[0]
                    break

            next_state, reward, done, info = env.step(action)
            episode.append((next_state, action, reward))
        return episode

    def generate_episode_from_Q(self, env, Q, epsilon, nA, control):
        """ generates an episode from following the epsilon-greedy policy """
        episode = []
        state = env.reset()
        while True:
            action = np.random.choice(np.arange(nA), p=control.get_probs(Q[state], epsilon, nA)) \
                if state in Q else env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def generate_episode_from_limit_stochastic(self, env):
        episode = []
        state = env.reset()
        while True:
            probs = [[0.5, 0.25, 0.0, 0.25], [0.25, 0.5, 0.25, 0.0], [0.0, 0.25, 0.5, 0.25], [0.25, 0.0, 0.25, 0.5]]
            action_choice = np.random.choice(np.arange(4))
            action = np.random.choice(np.arange(4), p=probs[action_choice])
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                if reward == 1:
                    print('Game Over! Reward: 1')
                    print('')
                break
        return episode

