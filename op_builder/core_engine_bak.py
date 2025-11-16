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

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# op_builder/async_io.py
#
# Part of the DeepSpeed Project, under the Apache-2.0 License.
# See https://github.com/microsoft/DeepSpeed/blob/master/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0

# MoE-Infinity: replaced AsyncIOBuilder with CoreEngineBuilder

import glob
import os

from .builder import OpBuilder

batchgen_CORE_ROOT = "core/"


class CoreEngineBuilder(OpBuilder):
    BUILD_VAR = "MOE_BUILD_CORE_ENGINE"
    NAME = "core_engine"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f"batchgen.core_engine"

    def sources(self):
        return [
            f"{batchgen_CORE_ROOT}/utils.cpp",
            f"{batchgen_CORE_ROOT}/batchgen_Binding.cpp",
            f"{batchgen_CORE_ROOT}/batchgen.cpp",
            f"{batchgen_CORE_ROOT}/DtoH_Engine/DtoH_Engine.cpp",
            f"{batchgen_CORE_ROOT}/Hetero_Attn/Hetero_Attn.cpp",
            f"{batchgen_CORE_ROOT}/HtoD_Engine/HtoD_Engine.cpp",
            f"{batchgen_CORE_ROOT}/HtoD_Engine/HtoD_Engine_Kernels.cu", 
            f"{batchgen_CORE_ROOT}/KV_Storage/KV_Storage.cpp",
            f"{batchgen_CORE_ROOT}/Weights_Storage/Weights_Storage.cpp",
            f"{batchgen_CORE_ROOT}/Parameter_Server/Parameter_Server.cpp",
            f"{batchgen_CORE_ROOT}/Parameter_Server/posix_shm.cpp",
            f"{batchgen_CORE_ROOT}/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp",
            f"{batchgen_CORE_ROOT}/GPU_KV_Buffer/GPU_KV_Buffer.cpp",
            f"{batchgen_CORE_ROOT}/Hetero_Attn/CPU_Kernels/grouped_query_attention_cpu_avx2_omp.cpp",
            f"{batchgen_CORE_ROOT}/allocator.cpp",
        ]

    def include_paths(self):
        return ["core/", "external"]

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            "-Wall",
            "-O2",
            "-std=c++17",
            "-fipa-pta",
            "-ffast-math",
            "-fno-unsafe-math-optimizations",
            "-fprefetch-loop-arrays",
            "-fopenmp",
            "-shared",
            "-fPIC",
            "-Wno-reorder",
            # '-DSPDLOG_HEADER_ONLY',
            CPU_ARCH,
            "-fopenmp",
            SIMD_WIDTH,
            "-I/usr/local/cuda/include",
            "-L/usr/local/cuda/lib64",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-lcuda",
            "-lcudart",
            "-lcublas",
            "-lpthread",
            "-ltcmalloc",
            "-lcufile",
            "-lnuma",
        ]

    def extra_ldflags(self):
        return ['-lnuma']

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
