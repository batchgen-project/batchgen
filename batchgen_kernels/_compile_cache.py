"""Compile-cache configuration for the batchgen_kernels build (setup.py only).

torch >= 2.15 treats ``CXX`` as one program (``shlex.join``) and prepends
ccache/sccache itself unless ``TORCH_NO_COMPILER_WRAPPER`` is set; torch 2.9
writes ``CXX`` into the ninja file verbatim and adds no launcher. A multi-word
``CXX="ccache g++"`` therefore works only on 2.9 and becomes
``ccache 'ccache g++'`` on 2.15. The one form both accept is a single
executable: ``scripts/cache_cxx.sh`` runs the launcher, and
``TORCH_NO_COMPILER_WRAPPER`` keeps newer torch from adding a second one.

nvcc goes through ``PYTORCH_NVCC`` + ``scripts/cache_nvcc.sh``; torch uses
``PYTORCH_NVCC`` verbatim on every version. ``CC`` stays a plain compiler
because torch passes it to nvcc as ``-ccbin``.
"""

import os
import shlex
import shutil

CACHE_TOOLS = ("sccache", "ccache")


class MultiWordCompilerError(RuntimeError):
    """CXX names more than one program, which torch >= 2.15 cannot run."""


def configure_compile_cache(environ, scripts_dir, cuda_home, which=shutil.which):
    """Point CXX/PYTORCH_NVCC at the cache wrappers; return the tool or None.

    ``environ`` is updated in place. A caller-set single-program ``CXX`` is
    kept (an explicit compiler choice). Without the repo's ``scripts/``
    wrappers (a build outside the checkout) nothing is configured and torch's
    own compiler handling applies.
    """
    cxx = environ.get("CXX")
    if cxx is not None and len(shlex.split(cxx)) != 1:
        raise MultiWordCompilerError(
            f"CXX={cxx!r} names more than one program; torch >= 2.15 runs CXX "
            "as a single executable. Unset CXX (the build then uses "
            "scripts/cache_cxx.sh when ccache/sccache is installed) or set it "
            "to one compiler executable."
        )
    cxx_wrapper = os.path.realpath(os.path.join(scripts_dir, "cache_cxx.sh"))
    nvcc_wrapper = os.path.realpath(os.path.join(scripts_dir, "cache_nvcc.sh"))
    if not (os.path.isfile(cxx_wrapper) and os.path.isfile(nvcc_wrapper)):
        return None
    for tool in CACHE_TOOLS:
        if which(tool):
            if cxx is None:
                environ["CXX"] = cxx_wrapper
                environ["CACHE_CXX_LAUNCHER"] = tool
                environ["TORCH_NO_COMPILER_WRAPPER"] = "1"
            nvcc = os.path.join(cuda_home, "bin", "nvcc")
            environ.setdefault("PYTORCH_NVCC", f"{nvcc_wrapper} {tool} {nvcc}")
            return tool
    return None
