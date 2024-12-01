from wgan_768x768 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
first_n = 50
noise_dim = 100

# for group in ['colon_aca', 'colon_n']:
for group in ['colon_n']:
    # group = "colon_aca"
    ep=2000
    generator_path = f"generator_{group}_{first_n}_ep{ep}.pth"
    img_gen(generator_path, noise_dim, device, num_images=50, save_path=f"gen_img/{group}/", save_name=f"gen_{group}")
