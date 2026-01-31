# Introduction

In this repository, we have modified SeFa with support for StyleGAN2-ADA after training on a repository of collagen images from SHG microscopy. The interface has been extended to facilitate external analysis tools. Semantic behavior is described in the detail in the paper.

## Setup

The environment droplet (`.yml`) file has been provided.

- **CUDA 11.1** is required for StyleGAN2-ADA functionality.
- Developed and tested on **Ubuntu 24.04.2 LTS** with an **NVIDIA RTX 3080**.
- **GCC 8** and **GCC 10.5** were both verified to work.

## Interface

The interface has been expanded with a variety of analysis tool, and is based on [StreamLit](https://www.streamlit.io/). This interface can be locally launched with

```bash
CUDA_VISIBLE_DEVICES=0
streamlit run interface.py
```

The interface will launch in the browser.

## Third-party encoder

See `third_party/stylegan2_encoder_pytorch/README.md` for details about the modified StyleGAN2 encoder, training scripts, and evaluation tools used by this project.

## BibTeX

Manuscript has been submitted.

## References

- SeFa (Closed-Form Factorization of Latent Semantics) — paper and project page:
	- Paper: https://arxiv.org/pdf/2007.06600.pdf
	- Project: https://genforce.github.io/sefa/
- StyleGAN2-ADA PyTorch (official NVLabs implementation): https://github.com/NVlabs/stylegan2-ada-pytorch
- stylegan2-encoder-pytorch (encoder repo adapted here): https://github.com/bryandlee/stylegan2-encoder-pytorch