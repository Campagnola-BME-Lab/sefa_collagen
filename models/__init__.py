# python3.7
"""Collects all available models together."""

from .model_zoo import MODEL_ZOO
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .stylegan2_ada import Generator
from .stylegan2_ada import Discriminator

__all__ = [
    'MODEL_ZOO', 'PGGANGenerator', 'PGGANDiscriminator', 'StyleGANGenerator',
    'StyleGANDiscriminator', 'StyleGAN2Generator', 'StyleGAN2Discriminator',
    'build_generator', 'build_discriminator', 'build_model'
]

_GAN_TYPES_ALLOWED = ['pggan', 'stylegan', 'stylegan2','stylegan2_ada']
_MODULES_ALLOWED = ['generator', 'discriminator']


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    if gan_type == 'stylegan2_ada':
        return Generator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_discriminator(gan_type, resolution, **kwargs):
    """Builds discriminator by GAN type.

    Args:
        gan_type: GAN type to which the discriminator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """
    if gan_type not in _GAN_TYPES_ALLOWED:
        raise ValueError(f'Invalid GAN type: `{gan_type}`!\n'
                         f'Types allowed: {_GAN_TYPES_ALLOWED}.')

    if gan_type == 'pggan':
        return PGGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANDiscriminator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Discriminator(resolution, **kwargs)
    if gan_type == 'stylegan2_ada':
        return Discriminator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def build_model(gan_type, module, resolution, **kwargs):
    """Builds a GAN module (generator/discriminator/etc).

    Args:
        gan_type: GAN type to which the model belong.
        module: GAN module to build, such as generator or discrimiantor.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the discriminator.

    Raises:
        ValueError: If the `module` is not supported.
        NotImplementedError: If the `module` is not implemented.
    """
    if module not in _MODULES_ALLOWED:
        raise ValueError(f'Invalid module: `{module}`!\n'
                         f'Modules allowed: {_MODULES_ALLOWED}.')

    if module == 'generator':
        return build_generator(gan_type, resolution, **kwargs)
    if module == 'discriminator':
        return build_discriminator(gan_type, resolution, **kwargs)
    raise NotImplementedError(f'Unsupported module `{module}`!')


def parse_gan_type(module):
    """Parses GAN type of a given module.

    Args:
        module: The module to parse GAN type from.

    Returns:
        A string, indicating the GAN type.

    Raises:
        ValueError: If the GAN type is unknown.
    """
    if hasattr(module, 'gan_type'):
        #return module.gan_type  # Use the explicitly set GAN type
        return 'stylegan2_ada'
    elif isinstance(module, (PGGANGenerator, PGGANDiscriminator)):
        return 'pggan'
    elif isinstance(module, (StyleGANGenerator, StyleGANDiscriminator)):
        return 'stylegan'
    elif isinstance(module, (StyleGAN2Generator, StyleGAN2Discriminator)):
        return 'stylegan2'
    elif isinstance(module, (Generator, Discriminator)):
        return 'stylegan2_ada'
    raise ValueError(f'Unable to parse GAN type from type `{type(module)}`!')
