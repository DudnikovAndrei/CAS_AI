import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15.0, 7.0)

x_train = torch.rand(1000)
x_train = x_train * 10.0 - 5.0

y_sub_train = torch.cos(x_train)**2
plt.plot(x_train.numpy(), y_sub_train.numpy(), 'o')
plt.title('$ y = cos^2(x) $')
plt.show()

noisy = torch.randn(y_sub_train.shape) / 3.
plt.plot(x_train.numpy(), noisy.numpy(), 'o')
plt.show()

y_train = y_sub_train + noisy
plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('noisy $ y = cos^2(x) $')
plt.show()

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_val = torch.linspace(-5, 5, 100)
y_val = torch.cos(x_val.data)**2

plt.plot(x_val.numpy(), y_val.numpy(), 'o')
plt.show()

x_val.unsqueeze_(1)
y_val.unsqueeze_(1)

class OurNet(torch.nn.Module):
    def __init__(self, n_hid_n):
        super(OurNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hid_n)
        self.act1 = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(n_hid_n, n_hid_n)
        # self.act2 = torch.nn.Sigmoid()
        # self.fc3 = torch.nn.Linear(n_hid_n, n_hid_n)
        # self.act3 = torch.nn.Sigmoid()
        self.fc4 = torch.nn.Linear(n_hid_n, 1)
        # 1 вход -> n нейронов -> n нейронов -> n нейронов -> n нейронов -> выход 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        # x = self.fc2(x)
        # x = self.act2(x)
        # x = self.fc3(x)
        # x = self.act3(x)
        x = self.fc4(x)
        return x

def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', c = 'g', label='то что должно быть')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c = 'r', label = 'Предсказание сети')
    plt.legend(loc='upper left')
    plt.show()

def loss(pred, true):
    sq = (pred-true)**2
    return sq.mean()

our_net = OurNet(100)
predict(our_net, x_val, y_val)
optimiser = torch.optim.Adam(our_net.parameters(), lr=0.001)

for e in range(10000):
    optimiser.zero_grad()
    y_pred = our_net.forward(x_train)
    loss_val = loss(y_pred, y_train)
    if not e % 2000:
        print(loss_val)

    loss_val.backward()
    optimiser.step()

predict(our_net, x_val, y_val)


