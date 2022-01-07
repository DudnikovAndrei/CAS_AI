import torch.nn as nn
import torch.optim as optim

from Discriminator import Discriminator
from Genarator import Generator

class CreateGenAndDisc():
    def __init__(self, config):
        self.config = config

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def create_discriminator(self):
        # Create the Discriminator
        netD = Discriminator(self.config.nc, self.config.ndf).to(self.config.device)

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(self.weights_init)

        # Print the model
        print(netD)

        return netD

    def create_generator(self):
        # Create the generator
        netG = Generator(self.config.nz, self.config.ngf, self.config.nc).to(self.config.device)

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netG.apply(self.weights_init)

        # Print the model
        print(netG)

        return netG

    def get_optimisers(self, netD, netG):
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))

        return optimizerD, optimizerG