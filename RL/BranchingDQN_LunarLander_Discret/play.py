from tqdm import tqdm
import torch

from model import DuelingNetwork
from utils import BranchingTensorEnv
import utils

def run():
    args = utils.arguments()
    env = BranchingTensorEnv(args.env)

    agent = DuelingNetwork(env.observation_space.shape[0], env.action_space.n)
    agent.load_state_dict(torch.load('./runs/{}/model_state_dict'.format(args.env)))

    print(agent)

    for ep in tqdm(range(10)):
        s = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad(): # kein training gerade, nur abruf vom ergebnis
                out = agent(s).squeeze(0)
            action = torch.argmax(out, dim=0).numpy()
            s, r, done, _ = env.step(action)

            env.render()
            ep_reward += r

        print('Ep reward: {:.3f}'.format(ep_reward))

    env.close()

if __name__ == "__main__":
    run()
