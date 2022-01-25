import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy


env = gym.make('Blackjack-v1')

# Each state is a 3-tuple of:
#
# the player's current sum  ∈{0,1,…,31} ,
# the dealer's face up card  ∈{1,…,11} , and
# whether or not the player has a usable ace (no  =0 , yes  =1 ).


# for i_episode in range(3):
#     state = env.reset()
#     while True:
#         print(state)
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         if done:
#             print('End game! Reward: ', reward)
#             print('You won :)\n') if reward > 0 else print('You lost :(\n')
#             break


# our starting policy
def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

for i in range(3):
    print(generate_episode_from_limit_stochastic(env))


def mc_prediction_q(env, num_episodes, generate_episode, gamma=.8):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # generate an episode
        episode = generate_episode(env)

        # obtain the states, actions, and rewards
        states, actions, rewards = zip(*episode)

        # prepare for discounting
        discounts = np.array([gamma ** i for i in range(len(rewards))])

        print("discounts:", discounts)
        print("rewards:", rewards)
        print("states:", states)

        # update the sum of the returns, number of visits, and action-value
        # function estimates for each state-action pair in the episode
        for i, (state, action) in enumerate(zip(states, actions)):
            n_steps_after_state = len(rewards[i:])
            print("rewards[i:]:", rewards[i:])
            print("discounts[:n_steps_after_state]:", discounts[:n_steps_after_state])
            print("----------------------")

            returns_sum[state][action] += sum(rewards[i:] * discounts[:n_steps_after_state])
            N[state][action] += 1.0
            Q[state][action] = returns_sum[state][action] / N[state][action]
            print("Q-Value:", Q[state][action])
            print("State:", state)
            print("Action:", action)
    return Q


Q = mc_prediction_q(env, 50, generate_episode_from_limit_stochastic) # nur prediction

# obtain the corresponding state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())


def generate_episode_from_Q(env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    while True:
        pb = get_probs(Q[state], epsilon, nA)
        action = np.random.choice(np.arange(nA), p=pb) if state in Q else env.action_space.sample()
        print('Action:', action)
        print('State:', state)
        print("")

        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


# control
def get_probs(Q_s, epsilon, nA):  # epsilon greedy
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA  # wenn epsilon 1 ist zu Beginn und ich habe 5 Aktionen, dann ist meine Policy [1/5, 1/5,..., 1/5]
    print('epsilon:', epsilon)
    print('policy_s:', policy_s)
    best_a = np.argmax(Q_s)
    print('best action:', best_a)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    print('policy_s[best_a]:', policy_s[best_a])
    return policy_s


# evaluation
def update_Q(env, episode, Q, alpha, gamma):  # policy improvement oder control
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    for i, state in enumerate(states):
        n_steps_after_state = len(rewards[i:])
        old_Q = Q[state][actions[i]]
        # Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:n_steps_after_state]) - old_Q)
        print('state:', state)
        print('action:', actions[i])
        print('q-value:', Q[state][actions[i]])
    return Q


def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        ### start punkt
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma) #
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q


policy, Q = mc_control(env, 500000, 0.02)


# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)


