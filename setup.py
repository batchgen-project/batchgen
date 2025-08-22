# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
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
import sys

from setuptools import find_packages, setup

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
    "build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)
}

print(f"find_packages: {find_packages()}")

# install all files in the package, rather than just the egg
setup(
    name="batchgen",
    version=os.getenv("MOEGEN_VERSION", "0.1"),
    packages=find_packages(
        exclude=["op_builder", "op_builder.*", "external/*"],
        include=[
            "batchgen",
            "batchgen.core_engine",
            "batchgen.*",
            "batchgen.models",
            "batchgen.models.**",
        ],
    ),
    package_data={
        "batchgen": [
            "**/*.cpp",
            "**/*.h",
            "**/*.cc",
            "**/*.hpp",
            "**/*.py",
            "**/*.so",
        ],
        "batchgen.core_engine": ["**/*.so"],
    },
    include_package_data=True,
    install_requires=install_requires,
    author="EfficientMoE Team",
    description="BatchGen is a library for High-throughput offline inference for MoE models with limited GPUs",
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
