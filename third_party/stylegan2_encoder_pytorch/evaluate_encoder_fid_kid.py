import os
import torch
import torchvision.utils as vutils
from torch_fidelity import calculate_metrics
from tqdm import tqdm
import csv
from PIL import Image
from glob import glob
import random
import shutil

from model import Encoder
from legacy import load_network_pkl

def normalize_tensor(tensor):
    """Normalize tensor from [0, 255] to [-1, 1]"""
    return tensor / 127.5 - 1.0

def main():
    # --- Settings ---
    # Use repo-relative paths to the checkpoints so the script works for cloned copies.
    g_pkl_path = "checkpoints/stylegan2_ada_collagen512.pkl"
    encoder_ckpt_path = "checkpoints/StyleGan2_Encoder/best_encoder.pth"

    output_fake_dir = "KID-eval/encoder_eval_fake"
    output_real_dir = "KID-eval/encoder_eval_real"
    csv_path = "KID-eval/encoder_fid_kid.csv"

    num_images = 100
    num_repeats = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Generator ---
    print(f"Loading generator from {g_pkl_path}")
    with open(g_pkl_path, "rb") as f:
        G = load_network_pkl(f)['G_ema'].to(device).eval()

    # --- Load Encoder ---
    print(f"Loading encoder from {encoder_ckpt_path}")
    encoder = Encoder(size=512, w_dim=512).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt_path))
    encoder.eval()

    # --- Prepare output folders ---
    os.makedirs(output_fake_dir, exist_ok=True)
    os.makedirs(output_real_dir, exist_ok=True)

    # --- Prepare CSV output ---
    csv_header = ['Run', 'FID', 'KID']
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

        for run in range(num_repeats):
            print(f"\n=== Run {run+1}/{num_repeats} ===")

            # Clear output folders
            for f in os.listdir(output_real_dir):
                os.remove(os.path.join(output_real_dir, f))
            for f in os.listdir(output_fake_dir):
                os.remove(os.path.join(output_fake_dir, f))

            # --- Generate real and reconstructed images ---
            with torch.no_grad():
                for i in tqdm(range(num_images)):
                    z = torch.randn(1, G.z_dim).to(device)
                    c = torch.zeros([1, G.c_dim], device=device)  # dummy conditioning label

                    w = G.mapping(z, c)
                    img_real = G.synthesis(w)

                    # Prepare input for encoder
                    img_gray = img_real.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                    pred_w = encoder(img_gray)
                    img_recon = G.synthesis(pred_w)

                    # Postprocess
                    img_real_out = ((img_real + 1) / 2).clamp(0, 1)
                    img_recon_out = ((img_recon + 1) / 2).clamp(0, 1)

                    # Save for metric evaluation (expand grayscale to 3 channels for torch-fidelity)
                    vutils.save_image(img_real_out.expand(-1, 3, -1, -1), os.path.join(output_real_dir, f"{i:05d}.png"))
                    vutils.save_image(img_recon_out.expand(-1, 3, -1, -1), os.path.join(output_fake_dir, f"{i:05d}.png"))

            # --- Calculate metrics ---
            print("Calculating FID/KID...")
            metrics = calculate_metrics(
                input1=output_fake_dir,
                input2=output_real_dir,
                cuda=torch.cuda.is_available(),
                isc=False,
                fid=True,
                kid=True,
                kid_subset_size=min(100, num_images),
                verbose=False
            )

            fid = metrics['frechet_inception_distance']
            kid = metrics['kernel_inception_distance_mean']

            print(f"Run {run+1} FID: {fid:.3f}  KID: {kid:.5f}")

            # Write to CSV
            writer.writerow([run+1, fid, kid])

    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()

