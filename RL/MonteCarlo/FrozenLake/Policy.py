
class Policy():
    def create_random_policy(self, env):
        policy = {}
        for key in range(0, env.observation_space.n):
            p = {}
            for action in range(0, env.action_space.n):
                p[action] = 1 / env.action_space.n
            policy[key] = p
        return policy

    def create_state_action_dictionary(env, policy):
        Q = {}
        for key in policy.kes():
            Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
        return Q

    def test_policy(self, env, policy, episode):
        wins = 0
        r = 100
        for i in range(r):
            w = episode.generate_episode(env, policy) [-1][-1]
            if w == 1:
                wins += 1
        return wins / r