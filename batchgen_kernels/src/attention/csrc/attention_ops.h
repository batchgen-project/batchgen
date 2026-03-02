#pragma once
#include <torch/extension.h>

// RMSNorm: standalone
// num_valid_tokens: optional 1-element int32 device tensor for CUDA graph padding skip
torch::Tensor rmsnorm_forward(
    torch::Tensor input,    // [*, hidden_size] BF16/FP16
    torch::Tensor weight,   // [hidden_size]
    float eps,
    c10::optional<torch::Tensor> num_valid_tokens = c10::nullopt);

// Fused Add + RMSNorm: residual += hidden, normed = rmsnorm(residual)
// Returns (normed, residual_updated)
std::vector<torch::Tensor> add_rmsnorm_forward(
    torch::Tensor residual, // [*, hidden_size] modified in-place
    torch::Tensor hidden,   // [*, hidden_size]
    torch::Tensor weight,   // [hidden_size]
    float eps,
    c10::optional<torch::Tensor> num_valid_tokens = c10::nullopt);

// Fused RoPE for Q and K (YaRN half-dim rotation)
// Returns (q_rot, k_rot)
std::vector<torch::Tensor> rope_forward(
    torch::Tensor query,    // [B, S, num_heads, head_dim]
    torch::Tensor key,      // [B, S, num_kv_heads, head_dim]
    torch::Tensor cos,      // [B, S, head_dim]
    torch::Tensor sin,      // [B, S, head_dim]
    int half_dim,
    c10::optional<torch::Tensor> num_valid_tokens = c10::nullopt);

// QKV Split (allocating): returns new (q, k, v) tensors
std::vector<torch::Tensor> qkv_split_forward(
    torch::Tensor qkv,
    int q_size,
    int kv_size,
    c10::optional<torch::Tensor> num_valid_tokens = c10::nullopt);

// QKV Split (in-place): writes to pre-allocated output tensors
void qkv_split_inplace(
    torch::Tensor qkv,
    torch::Tensor q_out,
    torch::Tensor k_out,
    torch::Tensor v_out,
    int q_size,
    int kv_size,
    c10::optional<torch::Tensor> num_valid_tokens = c10::nullopt);
