"""Utility functions."""

import base64
import os
import subprocess
import cv2
import numpy as np

import torch


__all__ = ['postprocess', 'load_generator', 'factorize_weight',
           'HtmlPageVisualizer']

CHECKPOINT_DIR = 'checkpoints'


def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`.

    Args:
      array: The input array to convert.

    Returns:
      A `torch.Tensor` with dtype `torch.FloatTensor` on cuda device.
    """
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()


def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.

    Args:
        images: A `torch.Tensor` with shape `NCHW` to process.
        min_val: The minimum value of the input tensor. (default: -1.0)
        max_val: The maximum value of the input tensor. (default: 1.0)

    Returns:
        A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images
