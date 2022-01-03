from __future__ import print_function
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from six.moves import urllib

class LoadData():
    def __init__(self, batch_size=32, train_val_test_split=[80, 10, 10]):
        self.Ntest = 1000
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.transform = self.loadData()

    def loadData(self):
        # NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
        #       see, https://github.com/pytorch/vision/issues/3497 for more information
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform

    def get_data(self, isTrainData):
        # Download data from open datasets.
        data = datasets.CIFAR10(
            root="data",
            train=isTrainData,
            download=True,
            transform=self.transform
        )

        return data

    def get_train_data(self):
        return self.get_data(isTrainData=True)

    def get_test_data(self):
        return self.get_data(isTrainData=False)
