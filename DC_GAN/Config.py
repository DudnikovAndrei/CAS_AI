import torch
import torch.nn as nn

class Config():
    def __init__(self):
        self.device = torch.device("cuda")

        # Set random seem for reproducibility
        self.manualSeed = 999

        self.dataroot = "imageA"

        # Number of workers for dataloader
        self.workers = 4

        # Batch size during training
        self.batch_size = 128

        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = 64

        # Number of channels in the training images. For color images this is 3
        self.nc = 3

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100

        # Size of feature maps in generator
        self.ngf = 64

        # Size of feature maps in discriminator
        self.ndf = 64

        # Number of training epochs
        self.num_epochs = 5

        # Learning rate for optimizers
        self.lr = 0.0002

        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0