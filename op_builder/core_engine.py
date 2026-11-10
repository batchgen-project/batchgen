import os

from .builder import CUDAOpBuilder

BATCHGEN_CORE_ROOT = "core/"

# The CPU attention kernel of attention modes 0 and 2 is written in AVX2
# intrinsics (<immintrin.h>), so it exists only in x86_64 builds. On any other
# host the source is left out and Hetero_Attn.cpp, guarded by __x86_64__,
# refuses those modes with CPUAttentionUnavailableError.
X86_CPU_ATTENTION_SOURCE = (
    f"{BATCHGEN_CORE_ROOT}/Hetero_Attn/CPU_Kernels/grouped_query_attention_cpu_avx2_omp.cpp"
)


def _torch_cxx_std() -> str:
    """C++ standard required by the active Torch (>=2.15 headers need C++20)."""
    import torch

    major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    return "-std=c++20" if (major, minor) >= (2, 15) else "-std=c++17"


def _libucx_paths():
    """Locate the pip-installed ``libucx-cu12`` wheel's headers/libs.

    The core_engine's distributed-weight daemon uses the UCP memory-handle API
    (``UCP_OP_ATTR_FIELD_MEMH``), which needs UCX >= 1.14. The ``libucx-cu12``
    wheel (declared in requirements.txt) ships a modern UCX (headers + libs) as
    a portable manylinux package, and the runtime loads UCX from that wheel.
    Building against any other UCX is refused. Returns ``(include_dir, lib_dir)``.
    """
    try:
        import libucx
    except ImportError as exc:
        raise RuntimeError(
            "core_engine needs the libucx-cu12 wheel; run "
            "`pip install -r requirements.txt` before building"
        ) from exc
    base = os.path.dirname(os.path.abspath(libucx.__file__))
    inc = os.path.join(base, "include")
    lib = os.path.join(base, "lib")
    if not (os.path.isdir(inc) and os.path.isdir(lib)):
        raise RuntimeError(f"libucx wheel at {base} has no include/ and lib/ directories")
    return inc, lib


class CoreEngineBuilder(CUDAOpBuilder):
    BUILD_VAR = "MOE_BUILD_CORE_ENGINE"
    NAME = "core_engine"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"batchgen.core_engine"

    def sources(self):
        x86_cpu_attention = (
            [X86_CPU_ATTENTION_SOURCE] if self._cpu_info()["arch"] == "X86_64" else []
        )
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
            f"{BATCHGEN_CORE_ROOT}/Weights_Storage/distributed_weight_daemon.cpp",
            f"{BATCHGEN_CORE_ROOT}/Parameter_Server/Parameter_Server.cpp",
            f"{BATCHGEN_CORE_ROOT}/Parameter_Server/posix_shm.cpp",
            f"{BATCHGEN_CORE_ROOT}/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp",
            f"{BATCHGEN_CORE_ROOT}/GPU_KV_Buffer/GPU_KV_Buffer.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_manager.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_backend.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_prefix_cache_coordinator.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_paged_kv_worker_view.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/host_kv_page_table.cpp",
            f"{BATCHGEN_CORE_ROOT}/KV_Storage/uva_copy_kernel.cu",
            *x86_cpu_attention,
            f"{BATCHGEN_CORE_ROOT}/allocator.cpp",
        ]

    def include_paths(self):
        paths = ["core/", "external"]
        ucx_inc, _ = _libucx_paths()
        paths.append(ucx_inc)
        return paths

    def cxx_args(self):
        """C++ compiler flags - DON'T call super() to avoid conflicts"""
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        
        args = [
            "-O2",
            _torch_cxx_std(),  # C++ standard the active Torch headers require
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
            _torch_cxx_std(),  # Match C++ standard
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

        # UCX from the pip libucx-cu12 wheel: link against it and bake an rpath so
        # the .so resolves libucp/libuct/libucs/libucm at runtime with no
        # LD_LIBRARY_PATH.
        _, ucx_lib = _libucx_paths()
        flags.append(f"-L{ucx_lib}")
        flags.append(f"-Wl,-rpath,{ucx_lib}")

        flags += [
            '-lnuma',
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lpthread',
            # '-ltcmalloc',
            '-lcufile',
            '-lucp',
            '-luct',
            '-lucs',
            '-lucm',
        ]
        return flags

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
