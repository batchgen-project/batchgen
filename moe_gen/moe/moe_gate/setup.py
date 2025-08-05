from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_moe_gate',
    ext_modules=[
        CUDAExtension('fused_moe_gate', [
            'moe_gate_cuda.cpp',
            'moe_gate_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })