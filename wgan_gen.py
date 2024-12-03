import torch
from torchinfo import summary
import torch.nn as nn
import torch.utils.data as data
from torchvision import utils as vutils
import os

LAMBDA = 10  # Gradient penalty lambda hyperparameter
LATENT_FEATURES = 64
BATCH_SIZE = 1  # Reduced batch size for memory efficiency


# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, channels=3):
        super(Generator, self).__init__()
        self.ngf = LATENT_FEATURES
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                noise_dim, self.ngf * 8, kernel_size=3, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.ngf * 8,
                self.ngf * 4,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.ngf * 4,
                self.ngf * 2,
                kernel_size=8,
                stride=4,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.ngf * 2, self.ngf, kernel_size=8, stride=4, padding=2, bias=False
            ),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                self.ngf, channels, kernel_size=8, stride=4, padding=2, bias=False
            ),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.model(input)


# Generate images
def img_gen(
    generator_path, noise_dim, device, num_images=1, save_path="gen_img", batch_size=1
):
    os.makedirs(save_path, exist_ok=True)

    generator = Generator(noise_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    for i in range(0, num_images, batch_size):
        z = torch.randn(min(batch_size, num_images - i), noise_dim, 1, 1, device=device)
        with torch.no_grad():
            fake_images = generator(z)

        fake_images = (fake_images + 1) / 2  # Normalize to [0, 1]
        for idx, img in enumerate(fake_images):
            image_path = os.path.join(save_path, f"image_{i + idx + 1}.png")
            vutils.save_image(img, image_path)

        # Free memory
        del z, fake_images
        torch.cuda.empty_cache()

    print(f"Images saved to {save_path}.")


# Main
def main():
    noise_dim = 100
    num_images = 3000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate images for each group
    for group in ["colon_aca", "colon_n"]:
        # for group in ["colon_aca"]:
        ep = 3300
        img_gen(
            f"generator_{group}_4000_ep{ep}.pth",
            noise_dim,
            device,
            num_images=num_images,
            save_path=f"synthetic_data/{group}/",
            batch_size=BATCH_SIZE,
        )


if __name__ == "__main__":
    main()
