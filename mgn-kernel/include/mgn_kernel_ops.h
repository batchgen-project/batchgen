/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

#include "scalar_type.hpp"

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

using fptr_t = int64_t;

/*
 * From csrc/moe
 */

std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor);

std::vector<torch::Tensor> expert_bincount_cuda(
    torch::Tensor eids,
    int64_t routed_expert_start_idx,
    int64_t experts_per_rank,
    torch::Device device);

std::vector<torch::Tensor> fused_moe_token_dispatch_cuda(
    torch::Tensor global_x,
    torch::Tensor topk_idx,
    torch::Tensor token_idx,
    torch::Tensor topk_pos,
    int64_t routed_expert_start_idx,
    int64_t routed_expert_end_idx);

/*
 * From csrc/elementwise
 */
torch::Tensor simple_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);