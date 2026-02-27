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
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_prefix_cache.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_worker_view.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_kv_page_table.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/uva_copy_kernel.cu",
            (
                f"{BATCHGEN_CORE_ROOT}/Hetero_Attn/CPU_Kernels/"
                "grouped_query_attention_cpu_avx2_omp.cpp"
            ),
            f"{BATCHGEN_CORE_ROOT}/allocator.cpp",
        ]

    def include_paths(self):
        return ["core/", "external"]

    def cxx_args(self):
        """C++ compiler flags - DON'T call super() to avoid conflicts"""
        cpu_arch = self.cpu_arch()
        simd_width = self.simd_width()

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
            cpu_arch,
            simd_width,
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
        return [
            '-lnuma',
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lpthread',
            # '-ltcmalloc',
            '-lcufile',
        ]

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
