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
from tqdm import tqdm  # Import tqdm for the progress bar
import json

LAMBDA = 10 # Gradient penalty lambda hyperparameter

LATENT_FEATURES = 128
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
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. ``(self.ngf) x 96 x 96``
            nn.ConvTranspose2d( self.ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. ``(channels) x 192 x 192``
        )

    def forward(self, input):
        return self.model(input)

# Discriminator (Critic)
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.ndf = LATENT_FEATURES
        self.model = nn.Sequential(
            # input is ``(channels) x 192 x 192``
            nn.Conv2d(in_channels=channels, out_channels=self.ndf, kernel_size=8, stride=3, padding=3, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(self.ndf) x 64 x 64``

            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(self.ndf*2) x 32 x 32``

            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
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

# custom weights initialization called on ``generator`` and ``discriminator``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Gradient Penalty
# adopted from https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py#L129
# credit: Marvin Cao
def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # flatten the gradients to it calculates norm batchwise
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# DataLoader for a specific group
def get_dataloader(data_dir, group, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_idx = dataset.class_to_idx[group]
    indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
    subset = Subset(dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=True)

# Training GAN with progress bar
def train_gan(generator, discriminator, dataloader, device, noise_dim, save_path, epochs=2000):
    G_losses = []
    D_losses = []
    lr = .1

    # criterion = nn.BCELoss()

    # # Create batch of latent vectors that we will use to visualize
    # #  the progression of the generator
    # fixed_noise = torch.randn(4, noise_dim, 1, 1, device=device)

    # real_label = 1.
    # fake_label = 0.

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=5, min_lr=1e-4
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5, min_lr=1e-4
    )

    for epoch in range(1, epochs+1):
        # Create a progress bar using tqdm
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")

        for i, (real_data, _) in enumerate(progress_bar):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_data = generator(z).detach()

            # Goal:
            # discriminator(real_data) => 1
            # discriminator(fake_data) => 0

            loss_D = -torch.mean(discriminator(real_data)) + torch.mean(discriminator(fake_data))
            # gp = gradient_penalty(discriminator, real_data, fake_data, device)
            # loss_D += gp
            loss_D.backward()
            optimizer_D.step()
            scheduler_D.step(loss_D)

            # # Clip weights of discriminator
            # for p in discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            # Train Generator every 5 steps
            if i % 5 == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, noise_dim, 1, 1, device=device)
                fake_data = generator(z)

                loss_G = -torch.mean(discriminator(fake_data))
                loss_G.backward()
                optimizer_G.step()
                scheduler_G.step(loss_G)

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Update progress bar description with loss values
            # progress_bar.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item(), gp=gp.item())
            progress_bar.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

        # Optionally, save intermediate models and generate images during training
        if epoch % 100 == 0:
            ep_save_path = f"{save_path}_ep{epoch}.pth"
            print(f"Epoch [{epoch}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
            torch.save(generator.state_dict(), f"generator_{ep_save_path}")
            print(f"Generator is saved as 'generator_{ep_save_path}'.")
            torch.save(discriminator.state_dict(), f"discriminator_{ep_save_path}")
            print(f"Discriminator is saved as 'discriminator_{ep_save_path}'.")

    return {"G_losses":G_losses, "D_losses": D_losses}


# Train GAN for a group
def train_group_gan(group, dataloader, noise_dim, image_size, epochs, device):
    print(f"Training GAN for group: {group}")

    generator = Generator(noise_dim, channels=3).to(device)
    discriminator = Discriminator(channels=3).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    save_path = f"{group}_{len(dataloader.dataset)}"
    losses = train_gan(generator, discriminator, dataloader, device, noise_dim, save_path, epochs)

    with open(f'training_record_{group}_{len(dataloader.dataset)}.json', 'w') as file:
        json.dump(losses, file)

# Generate images
def img_gen(generator_path, noise_dim, device, num_images=1, save_path="generated_image.png"):
    generator = Generator(noise_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    z = torch.randn(num_images, noise_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(z)

    fake_images = (fake_images + 1) / 2
    vutils.save_image(fake_images, save_path, nrow=num_images)
    print(f"Generated images saved to {save_path}.")

    plt.imshow(fake_images[0].permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.show()

# Main
def main():
    train_data_dir = './lung_colon_image_set/colon_image_sets/train'
    noise_dim = 100
    image_size = 192
    batch_size = BATCH_SIZE
    epochs = 20000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train GANs for each group
    # for group in ['colon_aca', 'colon_n']:
    for group in ['colon_aca']:
        dataloader = get_dataloader(train_data_dir, group, image_size, batch_size)
        train_group_gan(group, dataloader, noise_dim, image_size, epochs, device)

    # Generate images for each group
    # for group in ['colon_aca', 'colon_n']:
    #     img_gen(f"generator_{group}.pth", noise_dim, device, num_images=5, save_path=f"generated_{group}.png")

if __name__ == "__main__":
    # model = Generator(100)
    # summary(model, input_size=(1, 100, 1, 1))
    # model = Discriminator(3)
    # summary(model, input_size=(1, 3, 192, 192))
    # exit()
    main()
