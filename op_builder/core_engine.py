import os

from .builder import CUDAOpBuilder

BATCHGEN_CORE_ROOT = "core/"


class CoreEngineBuilder(CUDAOpBuilder):
    BUILD_VAR = "MOE_BUILD_CORE_ENGINE"
    NAME = "core_engine"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"batchgen.core_engine"

    def sources(self):
        return [
            f"{BATCHGEN_CORE_ROOT}/utils.cpp",
            f"{BATCHGEN_CORE_ROOT}/batchgen_Binding.cpp",
            f"{BATCHGEN_CORE_ROOT}/batchgen.cpp",
            f"{BATCHGEN_CORE_ROOT}/DtoH_Engine/DtoH_Engine.cpp",
            f"{BATCHGEN_CORE_ROOT}/Hetero_Attn/Hetero_Attn.cpp",
            f"{BATCHGEN_CORE_ROOT}/HtoD_Engine/HtoD_Engine.cu",
            f"{BATCHGEN_CORE_ROOT}/HtoD_Engine/HtoD_Engine_Kernels.cu",  # Your CUDA kernel
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/KV_Storage.cpp",
            f"{BATCHGEN_CORE_ROOT}/Weights_Storage/Weights_Storage.cpp",
            f"{BATCHGEN_CORE_ROOT}/Parameter_Server/Parameter_Server.cpp",
            f"{BATCHGEN_CORE_ROOT}/Parameter_Server/posix_shm.cpp",
            f"{BATCHGEN_CORE_ROOT}/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp",
            f"{BATCHGEN_CORE_ROOT}/GPU_KV_Buffer/GPU_KV_Buffer.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_manager.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_backend.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_worker_view.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_kv_page_table.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/uva_copy_kernel.cu",
            f"{BATCHGEN_CORE_ROOT}/Hetero_Attn/CPU_Kernels/grouped_query_attention_cpu_avx2_omp.cpp",
            f"{BATCHGEN_CORE_ROOT}/allocator.cpp",
        ]

    def include_paths(self):
        return ["core/", "external"]

    def cxx_args(self):
        """C++ compiler flags - DON'T call super() to avoid conflicts"""
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        
        args = [
            "-O2",
            "-std=c++17",  # Must be C++17 for PyTorch
            "-fPIC",
            "-Wall",
            "-fipa-pta",
            "-ffast-math",
            "-fno-unsafe-math-optimizations",
            "-fprefetch-loop-arrays",
            "-fopenmp",
            "-Wno-reorder",
            CPU_ARCH,
            SIMD_WIDTH,
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ]
        return args

    def nvcc_args(self):
        """CUDA compiler flags for .cu files"""
        args = super().nvcc_args()  # Get compute capabilities from base class
        args += [
            "-O3",
            "--use_fast_math",
            "-std=c++17",  # Match C++ standard
        ]
        return args

    def extra_ldflags(self):
        import torch

        flags = []

        # CUDA_HOME lib paths (system installs, e.g. /usr/local/cuda/lib64)
        cuda_home = torch.utils.cpp_extension.CUDA_HOME
        if cuda_home:
            for subdir in ("lib64", "lib"):
                lib_dir = os.path.join(cuda_home, subdir)
                if os.path.isdir(lib_dir):
                    flags.append(f"-L{lib_dir}")

        # Conda stubs dir (libcuda.so stub for linking in conda envs)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            stubs_dir = os.path.join(conda_prefix, "lib", "stubs")
            if os.path.isdir(stubs_dir):
                flags.append(f"-L{stubs_dir}")

        flags += [
            '-lnuma',
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lpthread',
            # '-ltcmalloc',
            '-lcufile',
        ]
        return flags

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)