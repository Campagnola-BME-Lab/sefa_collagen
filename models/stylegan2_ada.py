"""StyleGAN2-ADA Generator Wrapper for SeFa."""

import torch
import torch.nn as nn
import legacy


class Generator(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.device = device
        self.z_space_dim = 512  # Latent space (Z)
        self.w_space_dim = 512  # Intermediate latent space (W)
        self.num_ws = None      # Will be set after loading
        self.c_dim = 0          # No conditioning in most cases

        # Load pre-trained network.
        with open(model_path, 'rb') as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        self.num_ws = self.G.num_ws
        self.w_avg = self.G.mapping.w_avg.clone().detach()
        # Latent dimension for StyleGAN2-ADA
        self.latent_dim = self.G.z_dim
        
        # Number of layers in the synthesis network
        self.num_layers = len(list(self.G.synthesis.children()))
        self.z_dim = self.G.z_dim
        self.w_dim = self.G.w_dim
        self.c_dim = self.G.c_dim
        self.img_resolution = self.G.img_resolution
        self.img_channels = self.G.img_channels
        self.mapping = self.G.mapping
        self.synthesis = self.G.synthesis  # ✅Must be the full module
        self.w_avg = self.G.mapping.w_avg.clone().detach()

    def mapping(self, z, c=None):
        if c is None and self.c_dim > 0:
            c = torch.zeros([z.shape[0], self.c_dim], device=self.device)
        elif c is None:
            c = torch.zeros([z.shape[0], 0], device=self.device)
        ws = self.G.mapping(z, c)
        return ws

    def truncation(self, ws, trunc_psi=0.7, trunc_layers=8):
        ws = ws.clone()
        ws[:, :trunc_layers] = self.w_avg + (ws[:, :trunc_layers] - self.w_avg) * trunc_psi
        return ws

    def synthesis(self, ws, noise_mode='random'):
        images = self.G.synthesis(ws, noise_mode=noise_mode)
        return images

    def forward(self, z, c=None, trunc_psi=0.7, trunc_layers=8):
        ws = self.mapping(z, c)
        ws = self.truncation(ws, trunc_psi, trunc_layers)
        images = self.synthesis(ws)
        return {'image': images, 'ws': ws}

    def get_layer_names(self):
        layer_names = []
        for name, module in self.G.synthesis.named_modules():
            if hasattr(module, 'weight'):
                layer_names.append(name)
        return layer_names

    def get_layer(self, layer_name):
        module = dict(self.G.synthesis.named_modules()).get(layer_name, None)
        if module is None:
            raise ValueError(f'Layer {layer_name} not found.')
        return module

    def set_layer(self, layer_name, new_layer):
        modules = dict(self.G.synthesis.named_modules())
        if layer_name in modules:
            parent_module = self.G.synthesis
            *path, name = layer_name.split('.')
            for p in path:
                parent_module = getattr(parent_module, p)
            setattr(parent_module, name, new_layer)
        else:
            raise ValueError(f'Layer {layer_name} not found.')
            
            
            
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initialized dummy Discriminator.")

    def forward(self, x):
        print(f"Discriminator received input of shape: {x.shape}")
        return torch.zeros(x.size(0), 1, device=x.device)