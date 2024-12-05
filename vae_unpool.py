# https://avandekleut.github.io/vae/

import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import time
import sys
from datetime import timedelta
import numpy as np
from collections import defaultdict
import shutil # for __file__

from pytorch_msssim import ssim

# for signal handling
import os
import signal

test_training = False
graceful_exit = False

LATENT_DIM = 32
BATCH_SIZE = 24

IMAGE_SIZE = 768

BETA = 1e-8 # KL-Div factor

LOG_PATH = "VAE_colon_n_first8"

LEAKY_RELU_SLOPE = 0.01

class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        self.latent_dim = LATENT_DIM
        self.model = nn.Sequential(
            # input is ``(channels) x 768 x 768``
            nn.Conv2d(in_channels=channels, out_channels=self.latent_dim, kernel_size=8, stride=3, padding=3),
            # state size. ``(self.latent_dim) x 256 x 256``
            nn.MaxPool2d(kernel_size=2, stride=2),
            # state size. ``(self.latent_dim) x 128 x 128``

            nn.Conv2d(in_channels=self.latent_dim, out_channels=self.latent_dim * 4, kernel_size=4, stride=2, padding=1),
            # state size. ``(self.latent_dim*4) x 64 x 64``
            nn.MaxPool2d(kernel_size=2, stride=2),
            # state size. ``(self.latent_dim*4) x 32 x 32``

            nn.Conv2d(in_channels=self.latent_dim * 4, out_channels=self.latent_dim * 8, kernel_size=4, stride=2, padding=1),
            # state size. ``(self.latent_dim*8) x 16 x 16``
            nn.MaxPool2d(kernel_size=2, stride=2),
            # state size. ``(self.latent_dim*8) x 8 x 8``

            nn.Conv2d(in_channels=self.latent_dim * 8, out_channels=self.latent_dim * 16, kernel_size=4, stride=2, padding=1),
            # state size. ``(self.latent_dim*16) x 4 x 4``
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.latent_dim * 16, out_channels=self.latent_dim, kernel_size=4, stride=2, padding=1),
            # state size. ``(self.latent_dim) x 2 x 2``
            nn.ReLU(True),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(self.latent_dim * (2 ** 2), self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim * (2 ** 2), self.latent_dim)

    def forward(self, x):
        z = self.model(x)
        mu = self.fc_mu(z)
        # mu = torch.clamp(mu, min=-10, max=10)
        # mu = nn.Tanh()(self.fc_mu(z))*10
        logvar = self.fc_logvar(z)
        # logvar = torch.clamp(logvar, max=np.log(10))
        # logvar = nn.Tanh()(self.fc_logvar(z)) * np.log(10)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, channels=3):
        super(Decoder, self).__init__()
        self.latent_dim = LATENT_DIM
        upsample_mode = "nearest"
        self.model = nn.Sequential(
            nn.Flatten(),
            # input is Z, going into a convolution
            nn.Linear(self.latent_dim, self.latent_dim * 3 * 3),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(self.latent_dim, 3, 3)),
            # state size. ``(self.latent_dim) x 3 x 3``
            nn.ConvTranspose2d(self.latent_dim, self.latent_dim * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.latent_dim * 16),
            nn.ReLU(True),
            # state size. ``(self.latent_dim*16) x 6 x 6``
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            # state size. ``(self.latent_dim*16) x 12 x 12``
            nn.ConvTranspose2d(self.latent_dim * 16, self.latent_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.latent_dim * 8),
            nn.ReLU(True),
            # state size. ``(self.latent_dim*8) x 24 x 24``
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            # state size. ``(self.latent_dim*8) x 48 x 48``
            nn.ConvTranspose2d(self.latent_dim * 8, self.latent_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.latent_dim * 4),
            nn.ReLU(True),
            # state size. ``(self.latent_dim*4) x 96 x 96``
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            # state size. ``(self.latent_dim*4) x 192 x 192``
            nn.ConvTranspose2d(self.latent_dim * 4, self.latent_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.latent_dim * 2),
            nn.ReLU(True),
            # state size. ``(self.latent_dim*4) x 384 x 384``
            nn.ConvTranspose2d(self.latent_dim * 2, channels, kernel_size=4, stride=2, padding=1),
            # state size. ``(channels) x 768 x 768``
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_channels)
        self.decoder = Decoder(img_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

# DataLoader for a specific group
def get_dataloader_for_group(data_dir, group, image_size, batch_size):
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

# DataLoader for a specific group
def get_dataloader(data_dir, group, image_size, batch_size, first_n=-1):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_idx = dataset.class_to_idx[group]
    indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
    subset = Subset(dataset, indices)
    if first_n > 0:
        subset = Subset(subset, range(first_n))

    return DataLoader(subset, batch_size=batch_size, shuffle=True)

def calc_ssim_loss(recon_x, x):
    return 1 - ssim(recon_x, x, data_range=1.0, size_average=True)

# Training GAN with progress bar
def train_vae(group, dataloader, device, epochs=2000):
    global test_training, graceful_exit, BETA
    print(f"Start training, latent_dim: {LATENT_DIM}, BETA: {BETA}")
    vae = VariationalAutoencoder(LATENT_DIM).to(device)
    vae.apply(weights_init)
    save_name = f"{group}_{len(dataloader.dataset)}"

    stats = []

    lr = 1e-4
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    gaussian = {}

    start_time = time.time()
    for epoch in range(1, epochs+1):
        def save_model():
            ep_save_path = f"{save_name}_ep{epoch}"
            print(f"Epoch [{epoch}/{epochs}] stat: {stat}")
            torch.save(vae.state_dict(), f"{LOG_PATH}/vae_{ep_save_path}.pth")
            print(f"Model is saved as '{LOG_PATH}/vae_{ep_save_path}.pth'.")
            with open(f'{LOG_PATH}/vae_{ep_save_path}_gaussian.json', 'w') as file:
                gaussian_to_file = {}
                gaussian_to_file["mean"] = gaussian["mean"].tolist()
                gaussian_to_file["scale_tril"] = gaussian["scale_tril"].tolist()

                json.dump(gaussian_to_file, file)

        def finish_training():
            end_time = time.time()
            save_model()
            elapsed_time = end_time - start_time
            duration = str(timedelta(seconds=elapsed_time))
            start_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
            with open(f'{LOG_PATH}/VAE_training_record_{save_name}.json', 'w') as file:
                json.dump({
                            "date": start_date,
                            "duration": duration,
                            "epoch": epoch,
                            "hyperparam": {"LATENT_DIM": LATENT_DIM, "BETA": BETA},
                            "stats": stats
                           },
                           file)

        latent_vec = []

        # Create a progress bar using tqdm
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")

        if test_training:
            save_model()
            img_gen_group(vae, gaussian, group, device, num_images=9, save_name=f"img_gen_vae_{group}_ep{epoch-1}", save_as_grid=True)
            vae.train()
            test_training = False

        for x, y in progress_bar:
            if graceful_exit:
                finish_training()
                sys.exit(0)

            optimizer.zero_grad()

            x = x.to(device)
            recon_x, mu, logvar, z = vae(x)

            latent_vec.append({"mu": mu.cpu().detach(), "logvar": logvar.cpu().detach()}) # put to CPU to avoid cuda OOM

            recon_loss = nn.MSELoss()(recon_x, x)
            ssim_loss = calc_ssim_loss(recon_x, x)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            if (ssim_loss.item() < 0.35):
                if BETA < 1e-2:
                    BETA = min(1e-2, BETA * 10)
                    print(f"updating BETA to {BETA:.1e}")
            # kl_div = torch.clamp(kl_div, max=1)
            loss = 10*recon_loss + ssim_loss + BETA * kl_div
            loss.backward()
            optimizer.step()

            stat = {"recon_loss": recon_loss.item(), "ssim_loss": ssim_loss.item(), "kl-div": kl_div.item(), "BETA": BETA}
            stats.append({"_ep": epoch, **stat})

            progress_bar.set_postfix(**stat)

        # epoch done

        latents_mean = torch.cat([item["mu"] for item in latent_vec], dim=0)
        latents_logvar = torch.cat([item["logvar"] for item in latent_vec], dim=0)
        mean = latents_mean.mean(dim=0)
        logvar_avg = latents_logvar.mean(dim=0) / 2

        scale_tril = logvar_avg.exp()
        gaussian = {"mean": mean, "scale_tril": scale_tril}

        if epoch % 500 == 0:
            save_model()
            img_gen_group(vae, gaussian, group, device, num_images=16, save_name=f"img_gen_vae_{group}_ep{epoch}", save_as_grid=True)
            vutils.save_image(recon_x, f"img_gen_vae_{group}_RECON_ep{epoch}.jpg", nrow=5, format="JPEG")
            vae.train()

    finish_training()

    return vae

# def fit_latent_space(vae, train_data_path, group, device):
#     print("Loading training data")
#     training_loader = get_dataloader_for_group(train_data_path, group, IMAGE_SIZE, BATCH_SIZE*2)
#     latent_vectors = []

#     for x, y in tqdm(training_loader, desc=f"encoding training data {group}"):
#         x = x.to(device)
#         mu, log_var = vae.encoder(x)
#         z = vae.reparameterize(mu, log_var)
#         latent_vectors.append(z.detach())

#     # Convert to tensor for numerical operations
#     latents = torch.cat(latent_vectors, dim=0)
#     mean = latents.mean(dim=0)
#     cov = torch.cov(latents.T)
#     return mean, cov

# Generate images
def img_gen(vae_path, gaussian_path, group, device, data_dir=None, num_images=1, save_path=".", save_name="gen", save_as_grid=False):
    vae = VariationalAutoencoder(LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vae_path))
    vae.eval()

    gaussian = None
    if gaussian_path:
        with open(gaussian_path, 'r') as file:
            gaussian = json.load(file)
            for key in gaussian:
                gaussian[key] = torch.tensor(gaussian[key])

    # for group in groups:
        # if not gaussians:
        #     # for legacy models without gaussian json
        #     mean, cov = fit_latent_space(vae, data_dir, group, device)
        #     gaussians = { group: {"mean": mean.tolist(), "cov": cov.tolist()} }
    img_gen_group(vae, gaussian, group, device, num_images, save_path=f"{save_path}/{group}", save_name=f"{save_name}_{group}", save_as_grid=save_as_grid)

def img_gen_group(vae, gaussian, group, device, num_images=1, save_path=".", save_name="gen", save_as_grid=False):
    IMG_FORMAT = "jpeg"
    os.makedirs(save_path, exist_ok=True)

    mean, scale_tril = gaussian.values()
    # mean, cov = torch.tensor(mean), torch.tensor(cov)
    z = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale_tril)).sample((num_images,))

    # z = torch.randn(num_images, vae.latent_dim)
    z = z.to(device)

    fake_images = vae.decoder(z)

    fake_images = (fake_images + 1) / 2
    if save_as_grid:
        image_path = os.path.join(save_path, f"{save_name}.{IMG_FORMAT}")
        vutils.save_image(fake_images, image_path, nrow=int(np.sqrt(num_images)), format=IMG_FORMAT)
        print(f"Generated images saved to {image_path}.")
    else:
        for idx, img in enumerate(fake_images):
            image_path = os.path.join(save_path, f"{save_name}_{idx+1}.{IMG_FORMAT}")
            vutils.save_image(img, image_path, format=IMG_FORMAT)
        print(f"Images saved to {save_path}.")


def signal_handler_test_onfly(sig, frame):
    global test_training
    print(f"Signal {sig} received. Preparing to test on-the-fly ...")
    test_training = True

exit_count = 0
def signal_handler_on_exit(sig, frame):
    global graceful_exit, exit_count
    exit_count += 1
    if exit_count > 3:
        sys.exit(0)
    print(f"Signal {sig} received. Preparing graceful exit ...")
    graceful_exit = True

# Main
def main(train=0):
    if train:
        PID = os.getpid()
        # use sigbus to trigger training on-the-fly
        signal.signal(signal.SIGBUS, signal_handler_test_onfly)
        signal.signal(signal.SIGINT, signal_handler_on_exit)
        signal.signal(signal.SIGTERM, signal_handler_on_exit)
        print(f"PID: {PID}, use ``kill -s SIGBUS {PID}`` to trigger test")

        os.makedirs(LOG_PATH, exist_ok=True)
        print(f"LOG path: {LOG_PATH}")
        current_script = os.path.abspath(__file__)
        destination_path = os.path.join(LOG_PATH, os.path.basename(current_script))
        shutil.copy(current_script, destination_path)

    train_data_dir = './lung_colon_image_set/colon_image_sets/train'
    image_size = 768
    first_n = 8
    batch_size = BATCH_SIZE
    epochs = 20000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if train:
        # for group in ['colon_aca', 'colon_n']:
        for group in ['colon_n']:
            dataloader = get_dataloader(train_data_dir, group, image_size, batch_size, first_n)
            train_vae(group, dataloader, device, epochs)

    if not train:
        # for group in ['colon_aca', 'colon_n']:
        # groups = ['colon_aca', 'colon_n']
        ep = 5000
        group = 'colon_n'
        dataset_size = 8
        # groups = ['colon_aca']
        img_gen(f"{LOG_PATH}/vae_{group}_{dataset_size}_ep{ep}.pth", f"{LOG_PATH}/vae_{group}_{dataset_size}_ep{ep}_gaussian.json", group, device, num_images=20, save_path=f"gen_img/", save_name="gen", save_as_grid=False)
        # img_gen(f"{LOG_PATH}/vae_8000_ep{ep}.pth", None, groups, device, data_dir=train_data_dir, num_images=9, save_path=f"gen_img/", save_name="gen", save_as_grid=True)

if __name__ == "__main__":
    # model = Encoder()
    # summary(model, input_size=(1, 3, 768, 768))
    # model = Decoder()
    # summary(model, input_size=(1, LATENT_DIM, 1, 1))
    # exit()
    main()
