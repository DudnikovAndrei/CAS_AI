from __future__ import print_function
import random
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Decide which device we want to run on
from torchvision.datasets.utils import download_and_extract_archive

from Config import Config
from CreateGenAndDisc import CreateGenAndDisc
from Plot import Plot
from Training import Training

config = Config()
manualSeed = config.manualSeed

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

download_and_extract_archive('https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip', '.', './imageA')

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=config.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(config.image_size),
                               transforms.CenterCrop(config.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

plot = Plot(dataloader, config.device)
plot.plot_training_images()

createGenAndDisc = CreateGenAndDisc(config)
netD = createGenAndDisc.create_discriminator()
netG = createGenAndDisc.create_generator()

optimizerD, optimizerG = createGenAndDisc.get_optimisers(netD, netG)

# Training Loop
training_loop = Training(config.device)
img_list, G_losses, D_losses = training_loop.train(config.num_epochs, dataloader, netD, netG, config.real_label, config.fake_label, config.criterion,
                    config.nz, optimizerD, optimizerG, config.fixed_noise, vutils)

plot.plot_results(img_list, G_losses, D_losses)