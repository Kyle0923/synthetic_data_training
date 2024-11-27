# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
import json


LAMBDA = 10 # Gradient penalty lambda hyperparameter

LATENT_FEATURES = 64
BATCH_SIZE = 64*3

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, channels=3):
        super(Generator, self).__init__()
        self.ngf = LATENT_FEATURES
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, self.ngf * 8, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. ``(self.ngf*8) x 3 x 3``
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. ``(self.ngf*4) x 12 x 12``
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. ``(self.ngf*2) x 48 x 48``
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. ``(self.ngf) x 192 x 192``
            nn.ConvTranspose2d( self.ngf, channels, kernel_size=8, stride=4, padding=2, bias=False),
            nn.Tanh()
            # state size. ``(channels) x 768 x 768``
        )

    def forward(self, input):
        return self.model(input)

# Discriminator (Critic)
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.ndf = LATENT_FEATURES
        self.model = nn.Sequential(
            # input is ``(channels) x 768 x 768``
            nn.Conv2d(in_channels=channels, out_channels=self.ndf, kernel_size=8, stride=3, padding=3, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(self.ndf) x 256 x 256``

            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(self.ndf*2) x 64 x 64``

            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(self.ndf*4) x 16 x 16``

            nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(self.ndf*8) x 4 x 4``

            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.model(x)


# Generate images
def img_gen(generator_path, noise_dim, device, num_images=1, save_path="generated_image.png"):
    generator = Generator(noise_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    z = torch.randn(num_images**2, noise_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(z)

    fake_images = (fake_images + 1) / 2
    vutils.save_image(fake_images, save_path, nrow=num_images)
    print(f"Generated images saved to {save_path}.")

    # plt.imshow(fake_images[0].permute(1, 2, 0).cpu().numpy())
    # plt.axis("off")
    # plt.show()

# Main
def main():
    noise_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate images for each group
    for group in ['colon_aca', 'colon_n']:
        # group = "colon_aca"
        ep=100
        img_gen(f"generator_{group}_4000_ep{ep}.pth", noise_dim, device, num_images=3, save_path=f"img_gen_{group}_{ep}.png")

if __name__ == "__main__":
    # model = Generator(100)
    # summary(model, input_size=(1, 100, 1, 1))
    # model = Discriminator(3)
    # summary(model, input_size=(1, 3, 768, 768))
    main()
