import torch
from torch.utils.cpp_extension import load
from torch.utils.cpp_extension import load

upfirdn2d_plugin = load(
    name="upfirdn2d_plugin",
    sources=[
        "/home/melchomps/Documents/GradSchool/sefa/torch_utils/ops/upfirdn2d.cpp",
        "/home/melchomps/Documents/GradSchool/sefa/torch_utils/ops/upfirdn2d.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math"],
    verbose=True
)

print("Successfully compiled upfirdn2d_plugin!")

bias_act_plugin = load(
    name="bias_act_plugin",
    sources=[
        "/home/melchomps/Documents/GradSchool/sefa/torch_utils/ops/bias_act.cpp",
        "/home/melchomps/Documents/GradSchool/sefa/torch_utils/ops/bias_act.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math"],
    verbose=True
)

print("Successfully compiled bias_act_plugin!")
