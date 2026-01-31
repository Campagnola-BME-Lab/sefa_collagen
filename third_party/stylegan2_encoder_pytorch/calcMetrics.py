import torch
from torch_fidelity import calculate_metrics
import argparse
import os

def main(args):
    eval_real_dir = f"{args.output_dir}/eval_real"
    eval_fake_dir = f"{args.output_dir}/eval_fake"

    # Sanity check
    n_real = len(os.listdir(eval_real_dir))
    n_fake = len(os.listdir(eval_fake_dir))
    print(f"Found {n_real} real images and {n_fake} fake images.")

    metrics = calculate_metrics(
        input1=eval_fake_dir,
        input2=eval_real_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=True,
        kid_subset_size=50,
        return_dict=True
    )

    
    #print(metrics)
    
    print("\n=== Final Validation Results ===")
    if 'fid' in metrics:
        print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    else:
        print("FID not computed (possible missing/corrupt images?)")

    if 'kid' in metrics:
        print(f"KID: {metrics['kernel_inception_distance_mean']:.6f}")
        print(f"KID std: {metrics['kernel_inception_distance_std']:.6f}")
    else:
        print("KID not computed (possible missing/corrupt images?)")

    print("===============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output_dir where eval/{reals,fakes} are saved')
    args = parser.parse_args()

    main(args)
