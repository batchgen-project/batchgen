import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

MODULE_UNDER_TEST = "batchgen.attention.mla.torch_backend"

mla_decode_fn = None
try:
    mod = __import__(
        MODULE_UNDER_TEST, fromlist=["mla_decoding_torch_with_fp8_kv"]
    )
    mla_decode_fn = getattr(mod, "mla_decoding_torch_with_fp8_kv")
except Exception as e:
    raise RuntimeError(
        f"Failed to import mla_decoding_torch_with_fp8_kv from {MODULE_UNDER_TEST}. Please check the import path. Original error: {e}"
    )


class FakeMLA(nn.Module):
    """
    Minimal implementation required for the tested function:
    - Linear layers for q_*, kv_*, o_proj, etc.
    - The shape of kv_b_proj's weights must satisfy view(num_heads, -1, kv_lora_rank)
    - rotary_emb: returns (cos, sin); in this unit test, rotary_pos_emb will be patched as identity
    """

    def __init__(
        self,
        num_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        q_lora_rank=12,
        v_head_dim=8,
        kv_lora_rank=12,
        hidden_dim=32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        # q
        self.q_a_proj = nn.Linear(hidden_dim, q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(
            q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # kv
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank)

        # kv_b_proj.weight.view(num_heads, -1, kv_lora_rank)
        # let out_features = (qk_nope_head_dim + v_head_dim) * num_heads
        out_feats = (self.qk_nope_head_dim + self.v_head_dim) * self.num_heads
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, out_feats, bias=False)

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden_dim, bias=False
        )

        self.softmax_scale = self.q_head_dim**-0.5

        self.out_embed_dim = hidden_dim

    def rotary_emb(self, k_pe, seq_len: int):
        """
        Returns (cos, sin). Shape is not important because rotary_pos_emb will be patched as identity.
        """
        # Return placeholder tensors for cos/sin with the same shape as k_pe
        cos = torch.ones_like(k_pe)
        sin = torch.zeros_like(k_pe)
        return cos, sin

    # Bind the function under test to this class for easier testing
    mla_decoding_torch_with_fp8_kv = mla_decode_fn


class TestMLADecodingFP8KV(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # DeepSeek-R1 Configuration
        self.bsz = 8
        self.num_heads = 128
        self.qk_nope = 128
        self.qk_rope = 64
        self.q_lora_rank = 1536
        self.v_dim = 128
        self.kv_r = 512
        self.hidden_dim = 7168

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FakeMLA(
            num_heads=self.num_heads,
            qk_nope_head_dim=self.qk_nope,
            qk_rope_head_dim=self.qk_rope,
            q_lora_rank=self.q_lora_rank,
            v_head_dim=self.v_dim,
            kv_lora_rank=self.kv_r,
            hidden_dim=self.hidden_dim,
        ).eval().to(self.device).to(torch.bfloat16)


        # cache length
        self.seq_len = 8193
        self.max_seqlen = 1151  # max length that does not include padding
        self.max_seqlen_pad = (
            1152  # actual allocated length (>= max_seqlen, aligned to 64)
        )
        self.cur_pos = 5  # the first valid token position (0-indexed)

        # input hidden states
        self.hidden_states = torch.randn(
            self.bsz,
            1,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.attention_mask = torch.zeros(
            self.bsz, self.seq_len, device=self.device, dtype=torch.int64
        )
        self.attention_mask[:, self.cur_pos:] = 1  # valid tokens

        self.q_position_ids = torch.full(
            (self.bsz, 1), self.cur_pos, device=self.device, dtype=torch.int64
        )

        # FP8 KV cache (the underlying byte content does not matter, as the dequantization function will be mocked)
        kv_dim = self.kv_r + self.qk_rope
        self.past_key_states = torch.zeros(
            self.bsz,
            self.max_seqlen_pad,
            kv_dim,
            device=self.device,
            dtype=torch.float8_e4m3fn,
        )
        self.past_value_states = torch.empty(
            1
        )  # Not used but kept for signature compatibility
        # scale: scaling parameter for each token (corresponds to kv_dim)
        self.scale = torch.ones(
            self.bsz,
            self.max_seqlen_pad,
            kv_dim,
            device=self.device,
            dtype=torch.float32,
        )

        # Other arguments (not used in the function or affecting shapes)
        self.cache_seqlens = torch.full(
            (self.bsz,), self.cur_pos + 1, device=self.device, dtype=torch.int32
        )
        self.weight_scale = None

    def _patch_helpers(self):
        """
        Returns contextmanager to patch three external functions at once:
        - dequant_compressed_kv_per_token
        - per_token_blocked_quantize_bf16_to_fp8
        - rotary_pos_emb
        """
        # Dequantization: return all-zero compressed KV (shape [b, max_seqlen_pad, kv_dim])
        def fake_dequant(past_key_states, scale, max_seqlen):
            bsz, T, kv_dim = self.past_key_states.shape
            assert T == self.max_seqlen_pad
            return torch.zeros(bsz, T, kv_dim, device=self.device, dtype=torch.bfloat16)

        # Quantization: return (fp8=original input clamped to uint8, scale=all 1s), making it easy to assert write-back
        def fake_quant_bf16_to_fp8(offload_kv):
            fp8 = (offload_kv.float() * 10).to(torch.float8_e4m3fn)  # Construct a distinguishable byte result
            scale = torch.ones_like(offload_kv, dtype=torch.float32)
            return fp8, scale

        # RoPE position transformation: for simplicity, return identity
        def fake_rotary_pos_emb(x, cos, sin, pos_ids):
            return x

        patches = [
            patch(f"{MODULE_UNDER_TEST}.dequant_compressed_kv_per_token", side_effect=fake_dequant),
            patch(f"{MODULE_UNDER_TEST}.per_token_blocked_quantize_bf16_to_fp8", side_effect=fake_quant_bf16_to_fp8),
            patch(f"{MODULE_UNDER_TEST}.rotary_pos_emb", side_effect=fake_rotary_pos_emb),
        ]
        return patches

    def test_forward_basic_shapes_and_updates(self):
        """
        1) Forward pass runs without exceptions.
        2) Output shape == (bsz, 1, num_heads * v_head_dim) (after o_proj).
        3) past_key_states and scale are updated at the q_position_ids position.
        """
        for p in self._patch_helpers():
            p.start()
        self.addCleanup(patch.stopall)
    
        out, new_past, new_scale = self.model.mla_decoding_torch_with_fp8_kv(
            self.hidden_states,
            self.past_key_states.clone(),
            self.past_value_states,
            self.attention_mask.clone(),
            self.q_position_ids.clone(),
            self.scale.clone(),
            self.cache_seqlens.clone(),
            self.max_seqlen,
            self.weight_scale,
        )

        # 1) Shape assertion
        self.assertEqual(out.shape, (self.bsz, 1, self.model.out_embed_dim))

        # 2) KV/scale write-back assertion (should be written at cur_pos)
        bidx = torch.arange(self.bsz, device=self.device)
        # new_past/new_scale have the same shape as the input
        self.assertEqual(new_past.shape, self.past_key_states.shape)
        self.assertEqual(new_scale.shape, self.scale.shape)

        # There are updates (different from initial all-zero/all-one)
        self.assertFalse(
            torch.equal(
            new_past[bidx, self.cur_pos, :],
            self.past_key_states[bidx, self.cur_pos, :],
            )
        )

    def test_mask_left_padding_path(self):
        """
        test MLA decoding with a short attention mask that triggers left padding via F.pad.
        """
        for p in self._patch_helpers():
            p.start()
        self.addCleanup(patch.stopall)

        # Construct a shorter mask (length only 4) to ensure F.pad left padding is triggered
        short_mask = torch.zeros(
            self.bsz, 4, device=self.device, dtype=torch.int64
        )
        short_mask[:, -1] = 1

        out, *_ = self.model.mla_decoding_torch_with_fp8_kv(
            self.hidden_states,
            self.past_key_states.clone(),
            self.past_value_states,
            short_mask,
            self.q_position_ids.clone(),
            self.scale.clone(),
            self.cache_seqlens.clone(),
            self.max_seqlen,
            self.weight_scale,
        )
        self.assertEqual(out.shape, (self.bsz, 1, self.model.out_embed_dim))


if __name__ == "__main__":
    unittest.main()
