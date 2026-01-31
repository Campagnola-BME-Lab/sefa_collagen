StyleGAN2 Encoder (modified for ADA and SeFa)
============================================

This fork adapts the StyleGAN2 encoder code for compatibility with
StyleGAN2-ADA and the SeFa (Closed-Form Factorization) workflow.

Highlights
- Encoder and training code were modified from the original
	bryandlee/stylegan2-encoder-pytorch to work with StyleGAN2-ADA
	checkpoints and SeFa data flows (grayscale input, multi-w latent outputs).
- Training loop and loss weighting differ from upstream; the script
	saves the best model based on KID and supports evaluation (FID/KID).
- Model shape/IO was adapted so the encoder outputs `n_latents` ws
	tokens (default 14) which are passed directly to the StyleGAN2-ADA
	synthesis network.

Where to put checkpoints
- Place the generator pickle and encoder checkpoints under the repo `checkpoints/` folder:
	- Generator (StyleGAN2-ADA): `checkpoints/stylegan2_ada_collagen512.pkl`
	- Encoder checkpoints: `checkpoints/StyleGan2_Encoder/*.pth`
- The scripts use repo-relative defaults so they work for cloned copies.

Quick start (recommended)
1. Create environment (conda or pip). An example environment file is available at the repo root:
	 - conda: `conda env create -f environment_droplet.yml`

2. Place checkpoints under `checkpoints/` (see above).

3. Train encoder (example):
	 ```
	 python third_party/stylegan2_encoder_pytorch/train_encoder_forsefa.py \
			 --g_pkl_path checkpoints/stylegan2_ada_collagen512.pkl \
			 --out_dir checkpoints/encoder_run \
			 --img_size 512 \
			 --n_latents 14 \
			 --batch_size 4 \
			 --num_epochs 300
	 ```

4. Generate reconstruction pairs for SSIM evaluation:
	 ```
	 python third_party/stylegan2_encoder_pytorch/SSIM_image_generator.py
	 ```
	 Output defaults to `output/SSIM_EncoderImages` (repo-relative).

5. Evaluate encoder via repeated FID/KID runs:
	 ```
	 python third_party/stylegan2_encoder_pytorch/evaluate_encoder_fid_kid.py
	 ```

Streamlit / SeFa integration
- The main SeFa UI (`interface.py`) expects generator and encoder checkpoints under `checkpoints/`.
- If you want to run the Streamlit UI:
	```
	streamlit run interface.py
	```
Citation / references
- Based on: https://github.com/bryandlee/stylegan2-encoder-pytorch
- StyleGAN2 papers and implementatons: see links in the original upstream project.




