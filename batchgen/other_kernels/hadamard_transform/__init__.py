import torch
from torch.utils.cpp_extension import load
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_csrc_dir = os.path.join(_current_dir, "csrc")

_hadamard_cuda = load(
    name="fast_hadamard_transform_cuda",
    sources=[
        os.path.join(_csrc_dir, "hadamard_binding.cpp"),
        os.path.join(_csrc_dir, "fast_hadamard_transform_cuda.cu"),
    ],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=[
        '-O3',
        '-std=c++17',
        '--use_fast_math',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
    ],
    extra_include_paths=[_csrc_dir],
    verbose=False,
)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_cuda.fast_hadamard_transform(x, scale)
