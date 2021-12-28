import numpy as np
import pandas as pd
import torch
from torch import optim
import matplotlib.pyplot as plt

from SP500_VAE.SP500 import SP500DataSet
from SP500_VAE.VAE_Lightning import VAE
from VAE_Trainer import Trainer

# DataSet
sp500_dataset = SP500DataSet()
train_loader = sp500_dataset.get_train_loader()
test_loader = sp500_dataset.get_test_loader()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = train_loader.dataset.tensors[0].data.shape[1]  # + 1
model = VAE(size).to(device)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

trainer = Trainer(model, device, size)
for epoch in range(1, 10):
    trainer.train(epoch, train_loader, optimizer)
    trainer.test(test_loader)
    #with torch.no_grad():
    #    sample = torch.randn(64, 20).to(trainer.device)
    #    sample = trainer.model.decode(sample).cpu()

trainer.save()


model = VAE(trainer.size)
model.load_state_dict(torch.load('./runs/vae/vae_state_dict'))


train_loader, val_loader, test_loader = sp500_dataset.get_data_loaders()
samples = []
for i in range(100):
  with torch.no_grad():
    sample_feat, _, _ = model(train_loader.dataset.tensors[0])
    sample_labels = train_loader.dataset.tensors[1]
    new_data_sample = torch.cat([sample_feat, sample_labels], dim=1)
    samples.append(new_data_sample.numpy())


generated_np = np.vstack(samples)
generated_df = pd.DataFrame(generated_np)
orig = train_loader.dataset.tensors[0].detach().numpy()[0, :]
gen = generated_np[0, :-1]

plt.plot(orig[:100])
plt.plot(gen[:100])