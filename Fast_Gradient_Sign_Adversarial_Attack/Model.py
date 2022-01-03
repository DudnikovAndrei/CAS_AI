import torch.nn as nn
import torch.nn.functional as F

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 32 x 32 x 3
        x = self.conv1(x)
        # 28 x 28 x 128
        x = F.relu(F.max_pool2d(x, 2))
        # 14 x 14 x 128
        x = self.conv2(x)
        # 12 x 12 x 32
        x = F.relu(F.max_pool2d(x, 2))
        # 6 x 6 x 32
        x = x.view(-1, 1152) # wie flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def output_shape(SH, K, P, S, D):
      return ((SH[0] + 2*P[0] - D[0]*(K[0]-1) - 1)/S[0])+1, ((SH[1] + 2*P[1] - D[1]*(K[1]-1) - 1)/S[1])+1


