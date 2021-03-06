import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F

# VAE
class VAE(pl.LightningModule):
    def __init__(self, n_obs):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_obs, 400)
        self.fc21 = nn.Linear(400, 20) # mu 30
        self.fc22 = nn.Linear(400, 20) # log 30 variance
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, n_obs)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3) # torch.sigmoid(self.fc4(h3))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar