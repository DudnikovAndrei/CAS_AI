
import torch.nn as nn

class DuelingNetwork(nn.Module):
    def __init__(self, obs, ac):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(obs, 128),
                                   nn.ReLU(), 
                                   nn.Linear(128,128),
                                   nn.ReLU())

        # self.value_head = nn.Linear(128, 1)
        # self.adv_head = nn.Linear(128, ac)
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.adv_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ac)
        )

    def forward(self, x):
        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)
        q_val = value + (adv - adv.mean())

        return q_val
