
import sys
import numpy as np

from collections import defaultdict

class Control():
    # control
    def get_probs(self, Q_s, epsilon, nA):  # epsilon greedy
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nA  # wenn epsilon 1 ist zu Beginn und ich habe 5 Aktionen, dann ist meine Policy [1/5, 1/5,..., 1/5]
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    # evaluation
    def update_Q(self, episode, Q, alpha, gamma):  # policy improvement oder control
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            old_Q = Q[state][actions[i]]
            # Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
            Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1+i)]) - old_Q)
            # print('state:', state)
            # print('action:', actions[i])
            # print('q-value:', Q[state][actions[i]])
        return Q

    def mc_control(self, env, episode, control, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
        nA = env.action_space.n
        # initialize empty dictionary of arrays
        Q = defaultdict(lambda: np.zeros(nA))
        epsilon = eps_start
        # loop over episodes
        for i_episode in range(1, num_episodes + 1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
            # set the value of epsilon
            epsilon = max(epsilon * eps_decay, eps_min)
            # generate an episode by following epsilon-greedy policy
            ### start punkt
            get_episode = episode.generate_episode_from_Q(env, Q, epsilon, nA, control)
            # update the action-value function estimate using the episode
            Q = self.update_Q(get_episode, Q, alpha, gamma)  #
        # determine the policy corresponding to the final action-value function estimate
        policy = dict((k, np.argmax(v)) for k, v in Q.items())
        return policy, Q
