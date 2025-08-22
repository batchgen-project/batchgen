// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

/*
        CPU kernel for grouped query attention with AVX2 optimizations.
        Tailored for BFloat16 data type and decoding phase.

        @param
                query_states: torch::Tensor, [bsz, num_attn_head, 1, head_dim]
                key_states: std::vector<c10::BFloat16*>, bsz x [cache_length,
   num_kv_head, head_dim]. Address of host kv storage of sequences.
                value_states: std::vector<c10::BFloat16*>, bsz x [cache_length,
   num_kv_head, head_dim]. Address of host kv storage of sequences.
                cache_length: int64_t, length of the cache.
                attention_mask: torch::Tensor, [bsz, cache_length]
                group_size: int64_t, number of attention heads per kv head.
                num_threads: int64_t, number of threads to use.
                num_attention_heads: int64_t, number of attention heads.
                num_kv_head: int64_t, number of kv heads.
                head_dim: int64_t, dimension of each attention head.

        @return
                torch::Tensor, [bsz, num_attn_head, 1, head_dim]. Attention
   output.
*/
torch::Tensor grouped_query_attention_cpu_avx2(
    const torch::Tensor& query_states,
    const std::vector<c10::BFloat16*>& key_states,
    const std::vector<c10::BFloat16*>& value_states,
    const torch::Tensor& attention_mask, int64_t cache_length,
    int64_t group_size, int64_t num_attention_heads, int64_t num_kv_head,
    int64_t head_dim, int64_t num_threads);

// torch::Tensor grouped_query_attention_cpu_avx2_deepseekv2(
//     const torch::Tensor& query_states,
//     const std::vector<c10::BFloat16*>& key_states,
//     const std::vector<c10::BFloat16*>& value_states,
//     const torch::Tensor& attention_mask, int64_t cache_length,
//     int64_t group_size, int64_t num_attention_heads, int64_t num_kv_head,
//     int64_t qk_head_dim, int64_t v_head_dim, int64_t num_threads);
