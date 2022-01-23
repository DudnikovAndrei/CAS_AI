import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch import optim
from torch.utils.data import DataLoader

from LoadData import LoadData
from Model import Net
from Trainer import Trainer
from Attack import Attack

# Datas
train_data = LoadData().get_train_data()
test_data = LoadData().get_test_data()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.NLLLoss() # NLLLoss mit log_softmax ist wie Categorical Crossentropy mit Softmax
batch_size = 32

trainer = Trainer(device, model, loss_fn)

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

plt.figure(figsize=(10,5))
for i in range(25):
  plt.subplot(5,5,i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(train_data.data[i])

plt.show()

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainer.train(train_dataloader, optimizer)
    trainer.test(test_dataloader)
print("Done!")

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "cifar10_model.pth"
torch.save(model.state_dict(), pretrained_model)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

attack = Attack(device)

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = attack.test(model, test_dataloader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        ex = np.transpose(ex, (1, 2, 0))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
