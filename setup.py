# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025-2026                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

import io
import os
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )

ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, ROOT_DIR)
# sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from torch.utils import cpp_extension

from op_builder.all_ops import ALL_OPS

RED_START = "\033[31m"
RED_END = "\033[0m"
ERROR = f"{RED_START} [ERROR] {RED_END}"


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def abort(msg):
    print(f"{ERROR} {msg}")
    assert False, msg


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


class CustomBuildPy(build_py):
    """Custom build that copies source directories into batchgen/ for packaging."""

    def run(self):
        self._prepare_package_data()
        super().run()

    def _prepare_package_data(self):
        root = Path(ROOT_DIR)
        batchgen_dir = root / "batchgen"

        # Directories to copy into batchgen/
        for src_name in ["core", "external", "op_builder"]:
            src = root / src_name
            dst = batchgen_dir / src_name

            if not src.exists():
                print(f"[WARNING] Source directory {src} does not exist, skipping")
                continue

            # Remove symlink if exists
            if dst.is_symlink():
                print(f"Removing symlink {dst}")
                dst.unlink()

            # Remove existing directory to get fresh copy
            if dst.exists() and dst.is_dir():
                shutil.rmtree(dst)

            # Copy the directory
            print(f"Copying {src} -> {dst}")
            shutil.copytree(src, dst)


install_requires = fetch_requirements("requirements.txt")

ext_modules = []

BUILD_OP_DEFAULT = int(os.environ.get("BUILD_OPS", 0))

if BUILD_OP_DEFAULT:
    assert torch_available, "Unable to pre-compile ops without torch installed. Please install torch before attempting to pre-compile ops."
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
    install_ops = dict.fromkeys(ALL_OPS.keys(), False)
    for op_name, builder in ALL_OPS.items():
        if builder is not None:
            op_compatible = builder.is_compatible()
            compatible_ops[op_name] = op_compatible
            if not op_compatible:
                abort(f"Unable to pre-compile {op_name}")
            ext_modules.append(builder.builder())

cmdclass = {
    "build_py": CustomBuildPy,
    "build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True),
}

print(f"find_packages: {find_packages()}")

# install all files in the package, rather than just the egg
setup(
    name="batchgen",
    version=os.getenv("BATCHGEN_VERSION", "1.0.10.post5"),
    packages=find_packages(
        include=[
            "batchgen",
            "batchgen.*",
        ],
    ),
    package_data={
        "batchgen": [
            # C++ source files for JIT compilation
            "core/**/*.cpp",
            "core/**/*.cu",
            "core/**/*.h",
            "core/**/*.hpp",
            "core/**/*.cc",
            "external/**/*.h",
            "external/**/*.hpp",
            "external/**/*.cpp",
            "external/**/*.cc",
            "external/**/*.rst",
            # JIT-compiled kernel sources under batchgen/other_kernels/*/csrc/
            # (e.g. hadamard_transform/csrc/* — required for runtime
            # torch.utils.cpp_extension.load on the wheel-only install path).
            "other_kernels/**/*.cpp",
            "other_kernels/**/*.cu",
            "other_kernels/**/*.cuh",
            "other_kernels/**/*.h",
            "other_kernels/**/*.hpp",
            "other_kernels/**/*.cc",
            # Op builder Python files
            "op_builder/**/*.py",
            # Data files (tokenizers, configs, etc.)
            "**/*.json",
            "**/*.parquet",
            "**/*.jinja",
            "**/*.model",
            # Compiled binaries
            "**/*.so",
        ],
    },
    exclude_package_data={
        "batchgen": [
            "storage/batches/*",
            "storage/files/*",
            "storage/files_meta/*",
            "storage/outputs/*",
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    author="EfficientMoE Team",
    description="High-throughput offline batch inference engine for MoE models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache License 2.0",
    python_requires=">=3.11",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
