import os
import torch
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from PIL import Image
from model import Encoder
from legacy import load_network_pkl  # from StyleGAN2-ADA / SeFA compatibility
from utils import postprocess
from torch_fidelity import calculate_metrics
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

def normalize_tensor(tensor):
    """Normalize tensor from [0, 255] to [-1, 1]"""
    return tensor / 127.5 - 1.0

def train_encoder(args):
    os.makedirs(args.out_dir, exist_ok=True)
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load StyleGAN2-ADA generator
    print(f"Loading generator from {args.g_pkl_path}")
    with open(args.g_pkl_path, "rb") as f:
        G = load_network_pkl(f)['G_ema'].to(device).eval()

    encoder = Encoder(size=args.img_size, w_dim=args.w_dim).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(log_dir=os.path.join('runs', f'encoder_{args.run_name}'))

    best_kid = float('inf')
    step = 0
    for epoch in range(args.num_epochs):
        encoder.train()
        losses = []

        for _ in tqdm(range(10000 // args.batch_size), desc=f"Epoch {epoch:02d}"):
            # Sample latent
            z = torch.randn(args.batch_size, G.z_dim, device=device)
            c = torch.zeros([args.batch_size, G.c_dim], device=device)  #dummy conditioning label
            w = G.mapping(z, c)  # Shape: [B, 1, 512] or [B, n_latents, 512]
            if w.ndim == 2:
                w = w.unsqueeze(1).repeat(1, args.n_latents, 1)

            with torch.no_grad():
                img = G.synthesis(w)
            img_gray = img.mean(dim=1, keepdim=True)  # Grayscale

            pred_w = encoder(img_gray)
            loss = loss_fn(pred_w, w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), step)
            step += 1
            
            #memory issues block
            torch.cuda.empty_cache()
            #del img, img_gray, w, z, pred_w, loss

        print(f"[Epoch {epoch}] Loss: {np.mean(losses):.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"encoder_epoch{epoch+1:03d}.pt")
            torch.save(encoder.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Save a few reconstructions
        # Folders to store images for evaluation
        eval_dir_real = os.path.join(args.out_dir, 'eval_real')
        eval_dir_fake = os.path.join(args.out_dir, 'eval_fake')
        os.makedirs(eval_dir_real, exist_ok=True)
        os.makedirs(eval_dir_fake, exist_ok=True)

        print(f"[Eval] Generating images for FID/KID at epoch {epoch}")

        G.eval()
        encoder.eval()

        with torch.no_grad():
            for i in range(100):  # You can change this count
                z = torch.randn(1, G.z_dim).to(device)
                c = torch.zeros([1, G.c_dim], device=device)  #dummy conditioning label

                w = G.mapping(z, c)
                img_real = G.synthesis(w)

                img_gray = img_real.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                pred_w = encoder(img_gray)
                img_recon = G.synthesis(pred_w)

                # Postprocess
                img_real_out = ((img_real + 1) / 2).clamp(0, 1)
                img_recon_out = ((img_recon + 1) / 2).clamp(0, 1)

                # Save for metric evaluation (expand grayscale to 3 channels because of torch fidelity)
                vutils.save_image(img_real_out.expand(-1, 3, -1, -1), os.path.join(eval_dir_real, f"{i:03d}.png"))
                vutils.save_image(img_recon_out.expand(-1, 3, -1, -1), os.path.join(eval_dir_fake, f"{i:03d}.png"))


        # Compute metrics
        metrics = calculate_metrics(
            input1=eval_dir_fake,
            input2=eval_dir_real,
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=True,
            kid_subset_size=50,
            verbose=False
        )
        fid = metrics['frechet_inception_distance']
        kid = metrics['kernel_inception_distance_mean']
        print(f"[Eval] FID: {metrics['frechet_inception_distance']:.2f}, "
              f"KID: {metrics['kernel_inception_distance_mean']:.4f}")
        writer.add_scalar('Validation/KID', kid, epoch)
        writer.add_scalar('Validation/FID', fid, epoch)
        if kid < best_kid:
            best_kid = kid
            best_path = os.path.join(args.out_dir, 'best_encoder.pth')
            torch.save(encoder.state_dict(), best_path)
            print(f"Saved BEST model with KID={kid:.5f}")
    torch.save(encoder.state_dict(), os.path.join(args.out_dir, 'final_encoder.pth'))
    
    writer.close()
    print("Saved final encoder model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--g_pkl_path",
        type=str,
        default='checkpoints/stylegan2_ada_collagen512.pkl',
        help="Path to StyleGAN2-ADA .pkl file"  # Path to the pretrained StyleGAN2-ADA generator pickle
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="third_party/stylegan2_encoder_pytorch/checkpoints/encoder_run0",
        help="Directory to save encoder checkpoints and evaluation outputs"  # Output directory for checkpoints and eval images
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Generator image resolution (height/width)"  # Image resolution used by generator and encoder
    )
    parser.add_argument(
        "--w_dim",
        type=int,
        default=512,
        help="Dimensionality of w (W space)"  # Size of the intermediate latent vector w
    )
    parser.add_argument(
        "--n_latents",
        type=int,
        default=14,
        help="Number of latent tokens/ws the encoder should output"  # Number of ws (num_ws) per sample
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"  # Number of samples per training iteration
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=300,
        help="Total number of training epochs"  # How many epochs to train for
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer"  # Adam learning rate
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"  # Frequency (in epochs) to save intermediate checkpoints
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='default_run',
        help="Name used for TensorBoard/run labeling"  # Identifier for logging/run name
    )
    args = parser.parse_args()

    train_encoder(args)
