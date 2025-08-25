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

# op_builder/__init__.py
#
# Part of the DeepSpeed Project, under the Apache-2.0 License.
# See https://github.com/microsoft/DeepSpeed/blob/master/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0

# MoE-Infinity: replced builder_closure with PrefetchBuilder

import importlib
import os
import pkgutil
import sys

from .builder import OpBuilder, get_default_compute_capabilities
from .core_engine import CoreEngineBuilder

# Do not remove, required for abstract accelerator to detect if we have a deepspeed or 3p op_builder
__deepspeed__ = True

# List of all available op builders from deepspeed op_builder
try:
    import batchgen.op_builder  # noqa: F401

    op_builder_dir = "batchgen.op_builder"
except ImportError:
    op_builder_dir = "op_builder"

this_module = sys.modules[__name__]


def builder_closure(member_name):
    return CoreEngineBuilder


# reflect builder names and add builder closure, such as 'TransformerBuilder()' creates op builder wrt current accelerator
for _, module_name, _ in pkgutil.iter_modules(
    [os.path.dirname(this_module.__file__)]
):
    if module_name != "all_ops" and module_name != "builder":
        module = importlib.import_module(
            f".{module_name}", package=op_builder_dir
        )
        for member_name in module.__dir__():
            if (
                member_name.endswith("Builder")
                and member_name != "OpBuilder"
                and member_name != "CUDAOpBuilder"
            ):
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = builder_closure(member_name)
