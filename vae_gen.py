import torch
import torchvision.utils as vutils
from torchvision import transforms
import os
from vae_train import VAE  # Import VAE model from the trained file
import torch

LATENT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_images(model_path, num_images=10, save_path="generated_images.png"):
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    z = torch.randn(num_images, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        generated_images = model.decoder(z)

    generated_images = (generated_images + 1) / 2  # Denormalize to [0, 1]
    vutils.save_image(
        generated_images, save_path, nrow=5
    )  # Adjust nrow as per your requirement
    print(f"Generated images saved to {save_path}.")


def main():
    model_path = "vae_model_final.pth"  # Path to the trained VAE model
    generate_images(model_path, num_images=9, save_path="generated_images.png")


if __name__ == "__main__":
    main()
