
from tqdm import tqdm
from BranchingDQN import BranchingDQN

from utils import AgentConfig
from utils import BranchingTensorEnv
import utils

def dqn_train():
    args = utils.arguments()
    env = BranchingTensorEnv(args.env)
    config = AgentConfig()

    agent = BranchingDQN(env.observation_space.shape[0], env.action_space.n, config)
    state = env.reset()
    ep_reward = 0.
    recap = []

    p_bar = tqdm(total=config.max_frames)
    epsilon = config.epsilon_start  # initialize epsilon
    for frame in range(config.max_frames):
        action = agent.get_action(state, env.action_space.n, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        if done:
            epsilon = max(config.epsilon_final, config.epsilon_decay * epsilon)  # decrease epsilon
            state = env.reset()
            recap.append(ep_reward)
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            ep_reward = 0.

        p_bar.update(1)

        if frame % 1000 == 0:
            utils.save(agent, recap, args)

if __name__ == "__main__":
    scores = dqn_train()