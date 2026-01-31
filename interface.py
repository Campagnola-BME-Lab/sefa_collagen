# python 3.7
"""
Streamlit interface for SeFa: model loading, sampling, synthesis, projection UI, and traversal export.

Provides utilities to load and cache generators, sample and synthesize latent codes,
project images, and generate semantic traversals as image sequences or GIFs. It wires these
capabilities into a Streamlit UI for interactive exploration of GAN latent semantics.
"""

import numpy as np
import torch
import streamlit as st
import SessionState
import io
import shutil
import pandas as pd
from models import parse_gan_type
from utils import to_tensor, postprocess, load_generator, factorize_weight
from projector import ImageProjector
#from lightencoder import LightEncoder
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from third_party.stylegan2_encoder.model import Encoder


@st.cache(allow_output_mutation=True)
def get_model(model_name):
    # Load and cache a generator instance by name.
    return load_generator(model_name)

@st.cache(allow_output_mutation=True)
def factorize_model(model, layer_idx):
    # Compute and cache factorization (eigenvectors/values) for a model and layer selection.
    return factorize_weight(model, layer_idx)

def sample(model, gan_type, num=1):
    # Sample random latent codes and apply model-specific preprocessing to obtain usable codes/ws.
    codes = torch.randn(num, model.z_space_dim).cuda()
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes, trunc_psi=0.7, trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes, trunc_psi=0.5, trunc_layers=18)
    elif gan_type == 'stylegan2_ada':
        codes = model.mapping(codes)
        codes = model.truncation(codes, trunc_psi=0.7, trunc_layers=8)
    return codes.detach().cpu().numpy()

def synthesize(model, gan_type, code, noise_mode='const'):
    with torch.no_grad():
        # Synthesize images from code tensors, adjusting shapes to match the model's expected ws format.
        code_tensor = torch.as_tensor(code).to(dtype=torch.float32, device='cuda')

        if code_tensor.ndim == 2:
            code_tensor = code_tensor.unsqueeze(1).repeat(1, model.num_ws, 1)
        elif code_tensor.ndim == 3 and code_tensor.shape[1] != model.num_ws:
            code_tensor = code_tensor.repeat(1, model.num_ws, 1)
        elif code_tensor.ndim == 4:
            code_tensor = code_tensor.squeeze(0)
            if code_tensor.ndim == 2:
                code_tensor = code_tensor.unsqueeze(0).repeat(1, model.num_ws, 1)

        images = model.synthesis(code_tensor, noise_mode=noise_mode)
        images = postprocess(images)

        if images.shape[0] == 1:
            return images[0]
        return images


def create_gif(image_folder, output_path, duration=100):
    # Create an animated GIF from PNG files in image_folder.
    import imageio
    images = []
    files = sorted(os.listdir(image_folder))
    for file in files:
        if file.endswith('.png'):
            img_path = os.path.join(image_folder, file)
            images.append(imageio.imread(img_path))
    if images:
        imageio.mimsave(output_path, images, duration=duration/1000.0)

def generate_multi_sample_traversals(model, gan_type, boundaries, semantic_indices, output_dir='output', step_size=1.0, step_range=(-10, 10), noise_mode='random', n_samples=5, original_code=None, layer_selections=None):
    os.makedirs(output_dir, exist_ok=True)
    z_dim = model.z_space_dim if hasattr(model, 'z_space_dim') else model.z_dim
    num_ws = model.num_ws
    records = []
    # Generate traversal images across samples, semantics, layer groups and steps, saving images and metadata.

    if layer_selections is None:
        layer_selections = [['all']]

    # Iterate over each random sample (or use provided original_code when applicable).
    for sample_idx in range(n_samples):
        if original_code is not None and n_samples == 1:
            w = torch.tensor(np.copy(original_code)).cuda()
            if w.ndim == 2:
                w = w.unsqueeze(0)
        else:
            z = torch.randn(1, z_dim).cuda()
            w = model.mapping(z)
            if isinstance(w, dict):
                w = w['w']
            w = model.truncation(w, trunc_psi=0.7, trunc_layers=8)
            if w.ndim == 2:
                w = w.unsqueeze(1)
            if w.shape[1] == 1:
                w = w.repeat(1, num_ws, 1)

    # For each semantic index to traverse.
    for sem_idx in semantic_indices:
            # For each selected layer group (or 'all' to apply to every ws layer).
            for group in layer_selections:
                if group == ['all']:
                    layer_idx = list(range(num_ws))
                    layer_label = 'all'
                else:
                    layer_idx = group
                    layer_label = f"{layer_idx[0]}-{layer_idx[-1]}"

                layer_dir = os.path.join(output_dir, f'semantic_{sem_idx:03d}', f'layers_{layer_label}')
                os.makedirs(layer_dir, exist_ok=True)
                boundary = boundaries[sem_idx][np.newaxis, np.newaxis, :]

                # For each traversal step, edit the code by adding boundary * step and synthesize an image.
                for step in np.arange(step_range[0], step_range[1] + step_size, step_size):
                    boundary_np = boundary.squeeze()
                    edited_code = w.cpu().numpy() + step * boundary_np
                    img = synthesize(model, gan_type, edited_code, noise_mode=noise_mode)
                    img_pil = Image.fromarray(np.squeeze(img))

                    filename = f'sample_{sample_idx:03d}_layers_{layer_label}_weight_{step:+.2f}.png'
                    full_path = os.path.join(layer_dir, filename)
                    img_pil.save(full_path)

                    records.append({
                        'sample_idx': sample_idx,
                        'semantic_idx': sem_idx,
                        'step': step,
                        'layer_group': layer_label,
                        'filename': full_path
                    })

    # Save metadata to CSV
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'traversal_metadata.csv'), index=False)

def main():
    st.title('Closed-Form Factorization of Latent Semantics in GANs')
    st.sidebar.title('Options')
    reset = st.sidebar.button('Reset')

    model_name = st.sidebar.selectbox('Model to Interpret', ['stylegan2_ada_collagen512'])
    model = get_model(model_name)
    gan_type = parse_gan_type(model)

    available_layers = ['all', '0-2', '3-4', '5-7'] if model.num_layers <= 8 else ['all', '0-1', '2-5', '6-13']
    layer_idx = st.sidebar.selectbox('Layers to Interpret', available_layers)
    layers, boundaries, eigen_values = factorize_model(model, layer_idx)

    max_semantics = len(eigen_values)
    noise_const = st.sidebar.checkbox("Use constant noise (deterministic)", value=False)
    noise_mode = 'const' if noise_const else 'random'

    use_encoder = st.sidebar.checkbox("Use Encoder for Projection", value=False)
    encoder_checkpoint = st.sidebar.text_input("Path to Encoder Checkpoint", value='/home/melchomps/Documents/GradSchool/sefa/checkpoints/StyleGan2_Encoder/best_encoder.pth')


    uploaded_file = st.sidebar.file_uploader("Upload an image to project", type=["png", "jpg", "jpeg"])
    proj_steps = st.sidebar.number_input("Projection steps", value=500, min_value=100, max_value=2000, step=100)
    proj_lr = st.sidebar.number_input("Projection learning rate", value=0.01, format="%.5f")
    use_perceptual = st.sidebar.checkbox("Use LPIPS Perceptual Loss", value=True)

    step_size_for_sliders = st.sidebar.number_input("Fine Step Size (+/- Buttons)", value=0.5, step=0.1, format="%.2f")
    semantics_to_show = st.sidebar.number_input("How many semantics to show?", value=min(10, max_semantics), min_value=1, max_value=max_semantics, step=1)

    comparison_placeholder = st.empty()
    button_placeholder = st.empty()

    state = SessionState.get(
        model_name=model_name,
        code_idx=0,
        codes=None,
        original_code=None,
        original_img=None,
        steps={}
    )

    writer = SummaryWriter(log_dir='runs/Style_encoder')  # TensorBoard Logger

    if use_encoder and os.path.exists(encoder_checkpoint):
        
        encoder = Encoder(
            size=model.img_resolution,
            w_dim=model.w_dim,
        ).cuda()
        encoder.load_state_dict(torch.load(encoder_checkpoint))
        encoder.eval()
        print(f"Loaded Encoder checkpoint from {encoder_checkpoint}")
    else:
        encoder = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Ensure grayscale

        if use_encoder:
            # Convert PIL image to tensor in [-1, 1]
            image_array = np.array(image).astype(np.float32)

            # Normalize to [-1, 1]
            image_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0   # shape: [1, 1, H, W]

            # Resize to generator resolution (e.g. 512x512)
            image_tensor = F.interpolate(image_tensor, size=(model.img_resolution, model.img_resolution), mode='bilinear', align_corners=False)

            # Move to GPU
            image_tensor = image_tensor.cuda()

            # Project into latent space
            with torch.no_grad():
                code = encoder(image_tensor)
            if code.shape[1] == 1:
                code = code.repeat(1, model.num_ws, 1)

            # Store code + image state
            state.codes = code
            state.original_code = np.copy(code.detach().cpu().numpy())
            state.original_img = synthesize(model, gan_type, state.original_code, noise_mode=noise_mode)
            state.steps = {}

        else:
            projector = ImageProjector(model)
            with st.spinner("Projecting image to latent space..."):
                code = projector.project(image, steps=proj_steps, lr=proj_lr, perceptual_loss=use_perceptual)
            state.codes = code[np.newaxis]
            state.original_code = np.copy(state.codes)
            state.original_img = synthesize(model, gan_type, state.original_code, noise_mode=noise_mode)
            state.steps = {}
    else:
        try:
            base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
        except FileNotFoundError:
            base_codes = sample(model, gan_type)

        if state.codes is None:
            state.codes = base_codes[0:1]
            state.original_code = np.copy(state.codes)
            state.original_img = synthesize(model, gan_type, state.original_code, noise_mode=noise_mode)

        if button_placeholder.button('Random', key="random_button"):
            state.code_idx += 1
            if state.code_idx < base_codes.shape[0]:
                state.codes = base_codes[state.code_idx][np.newaxis]
            else:
                state.codes = sample(model, gan_type)
            state.original_code = np.copy(state.codes)
            state.original_img = synthesize(model, gan_type, state.original_code, noise_mode=noise_mode)
            state.steps = {}

    max_step = {'pggan': 5.0, 'stylegan': 2.0, 'stylegan2': 15.0, 'stylegan2_ada': 10.0}.get(gan_type, 10.0)

    st.sidebar.markdown("## Semantic Sliders")
    rerun_needed = False
    for sem_idx in range(semantics_to_show):
        col1, col2, col3 = st.sidebar.beta_columns([6, 1, 1])
        with col1:
            slider_val = st.slider(
                f"{sem_idx:03d} (eig {eigen_values[sem_idx]:.2f})",
                min_value=-max_step,
                max_value=max_step,
                value=float(state.steps.get(sem_idx, 0.0)),
                step=0.04 * max_step,
                key=f"slider_{sem_idx}"
            )
        with col2:
            if st.button('+', key=f"plus_{sem_idx}"):
                state.steps[sem_idx] = min(state.steps.get(sem_idx, 0.0) + step_size_for_sliders, max_step)
                rerun_needed = True
        with col3:
            if st.button('-', key=f"minus_{sem_idx}"):
                state.steps[sem_idx] = max(state.steps.get(sem_idx, 0.0) - step_size_for_sliders, -max_step)
                rerun_needed = True
        if not rerun_needed:
            state.steps[sem_idx] = slider_val

    if rerun_needed:
        st.experimental_rerun()

    edited_code = np.copy(state.original_code)
    for sem_idx, step in state.steps.items():
        boundary = boundaries[sem_idx][np.newaxis, np.newaxis, :]
        edited_code += boundary * step

    edited_img = synthesize(model, gan_type, edited_code, noise_mode=noise_mode)

    st.write("### Comparison")
    comparison_placeholder.image([state.original_img / 255.0, edited_img / 255.0], caption=["Original", "Edited"], use_column_width=True)

    st.sidebar.markdown("---")
    st.sidebar.title("Generate Semantic Traversals")

    selected_semantics = st.sidebar.multiselect(
        'Select semantics to traverse',
        options=[f'{i:03d}' for i in range(len(eigen_values))],
        default=[f'{i:03d}' for i in range(min(5, len(eigen_values)))]
    )
    selected_semantics = [int(s) for s in selected_semantics]
    
    selected_layers = st.sidebar.multiselect(
        'Select layer groups to traverse',
        options=['all', '0-2', '3-4', '5-7']
    )
    layer_map = {
        '0-2': list(range(0, 3)),
        '3-4': list(range(3, 5)),
        '5-7': list(range(5, 8)),
        'all': ['all']
    }
    layer_selections = [layer_map[label] for label in selected_layers]

    traverse_step_size = st.sidebar.number_input('Step size between frames', value=1.0, min_value=0.01, step=0.01, format="%.2f")
    traverse_start = st.sidebar.number_input('Start value', value=-10.0, step=0.5, format="%.1f")
    traverse_end = st.sidebar.number_input('End value', value=10.0, step=0.5, format="%.1f")
    n_samples = st.sidebar.number_input("Number of random samples", value=10, step=1)

    output_folder = st.sidebar.text_input('Output folder', value='output_traversals')
    make_gif = st.sidebar.checkbox('Also create GIF animation?', value=True)
    use_current_image = st.sidebar.checkbox("Use current image", value=False)

    if st.sidebar.button('Generate Traversals'):
        generate_multi_sample_traversals(
            model=model,
            gan_type=gan_type,
            original_code=state.original_code,
            boundaries=boundaries,
            semantic_indices=selected_semantics,
            output_dir=output_folder,
            step_size=traverse_step_size,
            step_range=(traverse_start, traverse_end),
            noise_mode=noise_mode,
            n_samples=n_samples,
            layer_selections=layer_selections
        )

        if make_gif:
            for sem_idx in selected_semantics:
                semantic_dir = os.path.join(output_folder, f'semantic_{sem_idx:03d}')
                gif_path = os.path.join(output_folder, f'semantic_{sem_idx:03d}.gif')
                create_gif(semantic_dir, gif_path, duration=100)

        st.sidebar.success(f"Traversals saved to `{output_folder}` folder.")

if __name__ == '__main__':
    main()

