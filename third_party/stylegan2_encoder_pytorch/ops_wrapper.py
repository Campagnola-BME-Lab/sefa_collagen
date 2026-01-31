# ops_wrapper.py

import math
import torch
import torch.nn as nn
from torch_utils.ops.bias_act import bias_act

def fused_leaky_relu(x, bias, negative_slope=0.2, scale=math.sqrt(2)):
    # `bias` should be a 1D tensor with shape [channels]
    return bias_act(x, bias, act='lrelu', alpha=negative_slope, gain=scale)

class FusedLeakyReLU(nn.Module):
    def __init__(self, channels, negative_slope=0.2, scale=math.sqrt(2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return bias_act(x, self.bias, act='lrelu', alpha=self.negative_slope, gain=self.scale)

