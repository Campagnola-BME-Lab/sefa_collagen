"""Generate real and reconstructed image pairs using the StyleGAN2 generator and encoder.

This script samples latents, synthesizes real images, reconstructs them via the encoder,
and saves paired images for SSIM or other reconstruction evaluations.
"""

import os
import torch
import random
from PIL import Image
from tqdm import tqdm
import torchvision.utils as vutils
import legacy
from model import Encoder

# Settings (use repo-relative paths so script works for cloned copies)
g_pkl_path = "checkpoints/stylegan2_ada_collagen512.pkl"
encoder_ckpt_path = "checkpoints/StyleGan2_Encoder/best_encoder.pth"

output_base = "output/SSIM_EncoderImages"
num_runs = 10
num_images = 100
image_size = (512, 512)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Generator
print(f"Loading generator from {g_pkl_path}")
with open(g_pkl_path, "rb") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

# Load Encoder
print(f"Loading encoder from {encoder_ckpt_path}")
encoder = Encoder(size=512, w_dim=512).to(device).eval()
encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))

for run in range(1, num_runs + 1):
    run_dir = os.path.join(output_base, f"run_{run:02d}")
    real_dir = os.path.join(run_dir, "real")
    recon_dir = os.path.join(run_dir, "recon")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    print(f"\n=== Run {run}/{num_runs} ===")

    with torch.no_grad():
        for i in tqdm(range(num_images)):
            z = torch.randn(1, G.z_dim, device=device)
            c = torch.zeros([1, G.c_dim], device=device)

            w = G.mapping(z, c)
            img_real = G.synthesis(w)

            img_gray = img_real.mean(dim=1, keepdim=True)
            pred_w = encoder(img_gray)
            img_recon = G.synthesis(pred_w)

            img_real_out = ((img_real + 1) / 2).clamp(0, 1)
            img_recon_out = ((img_recon + 1) / 2).clamp(0, 1)

            vutils.save_image(img_real_out, os.path.join(real_dir, f"{i:05d}.png"))
            vutils.save_image(img_recon_out, os.path.join(recon_dir, f"{i:05d}.png"))

print(f"\nSaved all encoder image pairs under {output_base}")

