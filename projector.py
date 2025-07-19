import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from lpips import LPIPS

class ImageProjector:
    def __init__(self, generator, device='cuda', lpips_net='vgg'):
        """Initializes the projector with a generator and configures loss.

        Args:
            generator: The GAN generator model.
            device: Device to run computation on (e.g., 'cuda').
            lpips_net: Backbone network for LPIPS ('vgg', 'alex', or 'squeeze').
        """
        self.device = device
        self.G = generator.to(self.device).eval()
        self.z_dim = generator.z_dim
        self.w_dim = generator.w_dim
        self.num_ws = generator.num_ws
        self.w_avg = generator.mapping.w_avg if hasattr(generator.mapping, 'w_avg') else torch.zeros([self.w_dim], device=self.device)
        self.perceptual_loss_fn = LPIPS(net=lpips_net).to(self.device)

    def preprocess_image(self, image, resolution):
        """Prepares and normalizes the target image to match the generator output.

        Args:
            image: Input image (PIL or ndarray).
            resolution: Target resolution to resize to.

        Returns:
            Tensor suitable for GAN input.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert('L') if image.mode != 'L' else image
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    @torch.no_grad()
    def find_best_initial_w(self, target_image, num_trials=10):
        """Finds a good starting latent code by sampling z and minimizing MSE.

        Args:
            target_image: The image to match.
            num_trials: Number of z samples to test.

        Returns:
            Best initial w latent code.
        """
        best_loss = float('inf')
        best_w = None
        for _ in range(num_trials):
            z = torch.randn([1, self.z_dim], device=self.device)
            w = self.G.mapping(z)
            if w.ndim == 2:
                w = w.unsqueeze(1).repeat(1, self.num_ws, 1)
            synth = self.G.synthesis(w)
            loss = F.mse_loss(synth, target_image)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_w = w
        return best_w.detach().clone()

    def project(self, image, steps=1000, lr=0.01, perceptual_loss=True, verbose=False, init_noise_scale=0.05, noise_reg_weight=1e5):
        """Projects an input image into the GAN's latent space.

        Args:
            image: Target image to project.
            steps: Number of optimization steps.
            lr: Learning rate.
            perceptual_loss: Whether to use LPIPS perceptual loss.
            verbose: If True, prints progress.
            init_noise_scale: Noise scale for w initialization.
            noise_reg_weight: Regularization weight for noise.

        Returns:
            Optimized latent code (w space).
        """
        target_resolution = self.G.img_resolution
        target_image = self.preprocess_image(image, target_resolution)

        w_init = self.find_best_initial_w(target_image, num_trials=10)
        w_opt = w_init + init_noise_scale * torch.randn_like(w_init)
        w_opt = w_opt.detach().clone().requires_grad_(True)

        noises = []
        if hasattr(self.G.synthesis, 'named_buffers') and callable(getattr(self.G.synthesis, 'named_buffers')):
            for name, buf in self.G.synthesis.named_buffers():
                if "noise_const" in name:
                    buf.requires_grad = True
                    noises.append(buf)
        else:
            noise = torch.randn([1, 1, target_resolution, target_resolution], device=self.device, requires_grad=True)
            noises.append(noise)

        optimizer = torch.optim.Adam([w_opt] + noises, lr=lr)

        def noise_regularization(noises):
            """Applies spatial correlation regularization to noise maps."""
            reg = 0.0
            for noise in noises:
                size = noise.shape[2]
                while size >= 8:
                    reg += (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    reg += (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                    size //= 2
                    noise = F.avg_pool2d(noise, kernel_size=2)
            return reg

        for step in range(int(steps)):
            optimizer.zero_grad()
            synth_image = self.G.synthesis(w_opt)

            synth_image_norm = (synth_image + 1.0) / 2.0
            target_image_norm = (target_image + 1.0) / 2.0

            mse = F.mse_loss(synth_image, target_image)
            loss = mse
            if perceptual_loss:
                perceptual = self.perceptual_loss_fn(synth_image_norm.expand(-1, 3, -1, -1),
                                                     target_image_norm.expand(-1, 3, -1, -1))
                loss += perceptual.mean()

            if noises:
                loss += noise_reg_weight * noise_regularization(noises)

            loss.backward()
            optimizer.step()

            if verbose and step % 100 == 0:
                print(f"Step {step:04d}, MSE: {mse.item():.4f}, Total Loss: {loss.item():.4f}")

        return w_opt.detach().cpu().numpy()