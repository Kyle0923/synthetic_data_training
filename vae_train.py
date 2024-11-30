import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import time
import json
from tqdm import tqdm
from datetime import timedelta

# Global Parameters
LATENT_FEATURES = 32  # Reduced latent features to prevent OOM
BATCH_SIZE = 18  # Reduced batch size
EPOCHS = 5000
IMAGE_SIZE = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(
                img_channels, LATENT_FEATURES, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                LATENT_FEATURES, LATENT_FEATURES * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                LATENT_FEATURES * 2,
                LATENT_FEATURES * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(LATENT_FEATURES * 4 * (IMAGE_SIZE // 8) ** 2, latent_dim)
        self.fc_logvar = nn.Linear(
            LATENT_FEATURES * 4 * (IMAGE_SIZE // 8) ** 2, latent_dim
        )
        self.fc_decoder_input = nn.Linear(
            latent_dim, LATENT_FEATURES * 4 * (IMAGE_SIZE // 8) ** 2
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (LATENT_FEATURES * 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8)),
            nn.ConvTranspose2d(
                LATENT_FEATURES * 4,
                LATENT_FEATURES * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                LATENT_FEATURES * 2, LATENT_FEATURES, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                LATENT_FEATURES, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decoder_input(z)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def train_vae(group, dataloader, latent_dim, save_path, epochs, device):
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision training
    losses = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        # Create a tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")

        for x, _ in progress_bar:
            x = x.to(device)
            optimizer.zero_grad()

            # Mixed precision training
            with torch.cuda.amp.autocast():
                recon_x, mu, logvar = model(x)
                loss = loss_function(recon_x, x, mu, logvar)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Update tqdm progress bar with loss
            progress_bar.set_postfix(Loss=loss.item())

        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"{save_path}_ep{epoch}.pth")
            print(f"Model saved at epoch {epoch}")

    elapsed_time = str(timedelta(seconds=time.time() - start_time))
    with open(f"training_record_{save_path}.json", "w") as f:
        json.dump({"duration": elapsed_time, "losses": losses}, f)

    print("Training Complete.")
    return


def get_dataloader(data_dir, group, image_size, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_idx = dataset.class_to_idx[group]
    indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    train_data_dir = "./lung_colon_image_set/colon_image_sets/train"
    group = "colon_n"
    dataloader = get_dataloader(train_data_dir, group, IMAGE_SIZE, BATCH_SIZE)
    save_path = f"vae_{group}_{len(dataloader.dataset)}"
    train_vae(group, dataloader, LATENT_FEATURES, save_path, EPOCHS, DEVICE)
