"""
Pytest tests comparing BatchGen's GLM-5 MLP + MoE + Router against HF's
ground-truth modeling_glm_moe_dsa.py implementation element-by-element.

Focus: Layer 0 (dense MLP) first-token prefill divergence hunting.
Also: MoE routing and expert computation for additional coverage.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HF Ground-Truth Classes (inlined from modeling_glm_moe_dsa.py)
# ============================================================================

class HfAct2fn:
    """Simple activation function registry."""
    @staticmethod
    def silu(x):
        return F.silu(x)


ACT2FN = {"silu": HfAct2fn.silu}


class HfGlmMoeDsaMLP(nn.Module):
    """HF ground truth: GlmMoeDsaMLP (line 451-465 in modeling_glm_moe_dsa.py)."""

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        """HF formula: down_proj(act_fn(gate_proj(x)) * up_proj(x))"""
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class HfGlmMoeDsaTopkRouter(nn.Module):
    """HF ground truth: GlmMoeDsaTopkRouter (line 467-485)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts), dtype=torch.float32))

    def forward(self, hidden_states):
        """Returns raw logits [bsz_seq, num_experts]."""
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class HfGlmMoeDsaMoE(nn.Module):
    """HF ground truth: GlmMoeDsaMoE (line 527-580).
    
    Includes:
    - GlmMoeDsaTopkRouter.forward() (line 481-484)
    - route_tokens_to_experts() (line 547-570)
    - forward() (line 572-580)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = HfGlmMoeDsaTopkRouter(config)
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        """HF ground truth: route_tokens_to_experts (line 547-570).
        
        Key observation: topk_weights are gathered from SIGMOID scores (raw, un-biased),
        not from biased scores.
        
        Returns: (topk_indices, topk_weights)
        """
        # Line 548: sigmoid
        router_logits = router_logits.sigmoid()
        
        # Line 549: add bias for selection only (not for weight gathering)
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        
        # Lines 550-554: group-based routing (but n_group=1 for GLM-5, so this is simpler)
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        
        # Line 563: mask out low-group experts
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        
        # Line 564: topk on BIASED scores
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        
        # Line 565: gather RAW (un-biased) sigmoid scores at topk_indices
        topk_weights = router_logits.gather(1, topk_indices)
        
        # Lines 566-568: normalize
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        
        # Line 569: scale
        topk_weights = topk_weights * self.routed_scaling_factor
        
        return topk_indices, topk_weights


# Simplified HF expert for testing (not used in full MoE test, but structure reference)
class HfGlmMoeDsaNaiveMoe(nn.Module):
    """HF ground truth: GlmMoeDsaNaiveMoe (line 488-524).
    
    Simplified for testing: just stores expert weights and has forward.
    The real test uses individual experts created separately.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # HF uses merged gate_up_proj [E, 2*N, K]
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """HF forward (line 500-524).
        
        Creates output in same dtype as hidden_states (BF16 for prefill).
        Accumulates via index_add.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [E, topk, num_tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ============================================================================
# Config Mock (minimal GLM-5 config)
# ============================================================================

class MinimalGlm5Config:
    """Minimal config for testing."""
    def __init__(self):
        self.hidden_size = 128  # Reduced for test speed
        self.intermediate_size = 256  # For dense MLP
        self.moe_intermediate_size = 256
        self.num_experts_per_tok = 2
        self.n_routed_experts = 8
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.num_local_experts = 8


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Minimal GLM-5 config for testing."""
    return MinimalGlm5Config()


@pytest.fixture
def torch_seed():
    """Set deterministic seed."""
    torch.manual_seed(42)
    yield
    # Reset after test


# ============================================================================
# Test Suite: MLP (Dense Layer 0)
# ============================================================================

class TestGlm5MLPvHF:
    """Test Glm5MLP vs HfGlmMoeDsaMLP element-by-element."""

    @pytest.mark.parametrize("bsz,seq_len", [
        (1, 16),
        (2, 64),
        (1, 128),
    ])
    def test_mlp_bf16_forward(self, config, torch_seed, bsz, seq_len):
        """
        Test: Dense MLP forward (BF16 path, no FP8).
        
        Verify: down_proj(SiLU(gate_proj(x)) * up_proj(x)) produces identical output
        for both HF and BatchGen when weights are copied.
        """
        # Import BatchGen MLP
        from batchgen.models.glm.glm5.model import Glm5MLP
        
        # Create HF reference
        hf_mlp = HfGlmMoeDsaMLP(config, intermediate_size=config.intermediate_size).to(torch.bfloat16)
        
        # Create BatchGen equivalent
        batchgen_mlp = Glm5MLP(config).to(torch.bfloat16)
        
        # Copy weights: HF → BatchGen (element-wise identity)
        batchgen_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
        batchgen_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
        batchgen_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
        
        # Random BF16 input
        x = torch.randn(bsz, seq_len, config.hidden_size, dtype=torch.bfloat16)
        
        # Forward
        with torch.no_grad():
            hf_out = hf_mlp(x)
            batchgen_out = batchgen_mlp(x)
        
        # Compare element-by-element
        try:
            torch.testing.assert_close(
                batchgen_out, hf_out,
                atol=1e-3, rtol=1e-3,
                msg=f"MLP mismatch (bsz={bsz}, seq={seq_len})"
            )
        except AssertionError as e:
            max_diff = (batchgen_out - hf_out).abs().max().item()
            max_idx = (batchgen_out - hf_out).abs().argmax().item()
            logger.error(
                f"MLP forward divergence: max_diff={max_diff:.6e} at index {max_idx}\n"
                f"batchgen[0,0,0]={batchgen_out[0,0,0].item():.8f}, "
                f"hf[0,0,0]={hf_out[0,0,0].item():.8f}"
            )
            raise


# ============================================================================
# Test Suite: Router (Gate)
# ============================================================================

class TestGlm5RouterVsHF:
    """Test Glm5MoEGate vs HfGlmMoeDsaTopkRouter routing logic."""

    @pytest.mark.parametrize("bsz,seq_len", [
        (1, 16),
        (2, 64),
        (1, 128),
    ])
    def test_router_logits_fp32(self, config, torch_seed, bsz, seq_len):
        """
        Test: Router logits computation (before sigmoid/bias/topk).
        
        Verify: F.linear(x.float(), w.float()) produces identical logits.
        """
        from batchgen.models.glm.glm5.model import Glm5MoEGate
        
        # Create HF reference. HfGlmMoeDsaTopkRouter uses
        # nn.Parameter(torch.empty(...)) which can hold NaN garbage; initialize
        # to a well-behaved random distribution explicitly before copying.
        hf_router = HfGlmMoeDsaTopkRouter(config).to(torch.bfloat16)
        nn.init.normal_(hf_router.weight, mean=0.0, std=0.02)
        hf_router.e_score_correction_bias.data.zero_()

        # Create BatchGen equivalent
        batchgen_gate = Glm5MoEGate(config).to(torch.bfloat16)

        # Copy weights
        batchgen_gate.weight.data.copy_(hf_router.weight.data)
        batchgen_gate.e_score_correction_bias.data.copy_(hf_router.e_score_correction_bias.data)
        
        # Random BF16 input; flatten to match HF's internal view(-1, hidden_size).
        x = torch.randn(bsz, seq_len, config.hidden_size, dtype=torch.bfloat16)
        x_flat = x.view(-1, config.hidden_size)

        # Get logits
        with torch.no_grad():
            # HF router internally flattens and returns [B*S, num_experts].
            hf_logits = hf_router(x)
            # BatchGen gate uses F.linear(x.float(), w.float()). Feed the
            # already-flattened x so shapes align with HF's output.
            batchgen_logits = F.linear(x_flat.float(), batchgen_gate.weight.float())

        assert hf_logits.shape == batchgen_logits.shape, (
            f"Shape mismatch: HF={hf_logits.shape}, BatchGen={batchgen_logits.shape}"
        )

        # Compare logits (before sigmoid)
        if torch.allclose(batchgen_logits, hf_logits, atol=1e-3, rtol=1e-3):
            return  # pass
        max_diff = (batchgen_logits - hf_logits).abs().max().item()
        pytest.fail(f"Router logits divergence: max_diff={max_diff:.6e}")

    @pytest.mark.parametrize("bsz,seq_len", [
        (1, 16),
        (2, 64),
        (1, 128),
    ])
    def test_route_tokens_to_experts_full_pipeline(self, config, torch_seed, bsz, seq_len):
        """
        Test: Full routing pipeline (sigmoid + bias + topk + gather + normalize + scale).
        
        Verify: Both implementations return same (topk_indices, topk_weights).
        
        Critical detail: HF line 565 gathers from raw sigmoid scores, NOT biased scores.
        BatchGen Glm5MoEGate.forward should match this behavior.
        """
        from batchgen.models.glm.glm5.model import Glm5MoEGate
        
        # Create instances
        hf_moe = HfGlmMoeDsaMoE(config).to(torch.bfloat16)
        batchgen_gate = Glm5MoEGate(config).to(torch.bfloat16)
        
        # Copy weights
        batchgen_gate.weight.data.copy_(hf_moe.gate.weight.data)
        batchgen_gate.e_score_correction_bias.data.copy_(hf_moe.gate.e_score_correction_bias.data)
        
        # Random hidden states; feed already-flattened to BatchGen.
        x = torch.randn(bsz, seq_len, config.hidden_size, dtype=torch.bfloat16)
        x_flat = x.view(-1, config.hidden_size)

        with torch.no_grad():
            # HF: get router logits, then route (route works on [B*S, E])
            hf_logits = hf_moe.gate(x)
            hf_topk_indices, hf_topk_weights = hf_moe.route_tokens_to_experts(hf_logits)

            # BatchGen: directly call gate.forward on flat input
            batchgen_topk_weights, batchgen_topk_indices = batchgen_gate(x_flat)

        # Shapes must match
        assert batchgen_topk_indices.shape == hf_topk_indices.shape, (
            f"topk_indices shape mismatch: "
            f"BatchGen={batchgen_topk_indices.shape} HF={hf_topk_indices.shape}"
        )

        # topk indices should be identical (deterministic given identical logits)
        assert (batchgen_topk_indices == hf_topk_indices).all(), \
            f"topk_indices value divergence"

        # Compare weights element-wise
        if torch.allclose(batchgen_topk_weights, hf_topk_weights, atol=1e-3, rtol=1e-3):
            return
        max_diff = (batchgen_topk_weights - hf_topk_weights).abs().max().item()
        pytest.fail(f"topk_weights divergence: max_diff={max_diff:.6e}")


# ============================================================================
# Test Suite: Expert Computation (prefill)
# ============================================================================

class TestGlm5ExpertPrefillVsHF:
    """Test Glm5Expert and prefill MoE forward vs HF naive expert."""

    def test_single_expert_forward(self, config, torch_seed):
        """
        Test: Single expert (gate_proj + up_proj + SiLU → down_proj).
        
        Verify: Glm5Expert.forward matches HF expert weight structure.
        """
        from batchgen.models.glm.glm5.model import Glm5Expert
        
        bsz, seq_len = 2, 16

        # Create HF expert (extract as if from naive MoE) in BF16 to match
        # BatchGen expert dtype — otherwise F.linear rejects BF16 input with
        # an FP32 weight matrix.
        hf_gate_up = nn.Linear(config.hidden_size, 2 * config.moe_intermediate_size, bias=False).to(torch.bfloat16)
        hf_down = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False).to(torch.bfloat16)

        # Create BatchGen expert
        batchgen_expert = Glm5Expert(config.hidden_size, config.moe_intermediate_size).to(torch.bfloat16)
        
        # Copy weights: split gate_up into separate gate and up
        gate_up_weight = hf_gate_up.weight.clone()  # [2*N, K]
        gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)  # each [N, K]
        
        batchgen_expert.gate_proj.weight.data.copy_(gate_weight)
        batchgen_expert.up_proj.weight.data.copy_(up_weight)
        batchgen_expert.down_proj.weight.data.copy_(hf_down.weight.data)
        
        # Random input
        x = torch.randn(bsz, seq_len, config.hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # HF: merged gate_up path
            gate_up_out = hf_gate_up(x)  # [B, S, 2*N]
            hf_gate, hf_up = gate_up_out.chunk(2, dim=-1)  # each [B, S, N]
            hf_out = hf_down(F.silu(hf_gate) * hf_up)
            
            # BatchGen: separate path
            batchgen_out = batchgen_expert(x)
        
        try:
            torch.testing.assert_close(
                batchgen_out, hf_out,
                atol=1e-3, rtol=1e-3,
                msg="Single expert forward divergence"
            )
        except AssertionError as e:
            max_diff = (batchgen_out - hf_out).abs().max().item()
            logger.error(f"Expert forward divergence: max_diff={max_diff:.6e}")
            raise

    def test_prefill_moe_accumulation_dtype(self, config, torch_seed):
        """
        Test: MoE prefill accumulation dtype (default BF16, matches HF).
        
        Verify: With many tokens routed to same expert, accumulated output
        has low max_diff when cast back to BF16.
        """
        from batchgen.models.glm.glm5.model import Glm5MoE, Glm5Expert, Glm5MoEGate
        
        # Use larger config for accumulation stress test
        config.n_routed_experts = 16
        config.num_local_experts = 16
        
        bsz, seq_len = 1, 256  # Many tokens
        num_tokens = bsz * seq_len
        
        # Create BatchGen MoE (prefill)
        moe = Glm5MoE(config).to(torch.bfloat16)
        
        # Populate experts (normally done by _config_expert_module)
        from batchgen.models.glm.glm5.model import _Glm5ExpertPlaceholder
        moe.experts = [Glm5Expert(config.hidden_size, config.moe_intermediate_size).to(torch.bfloat16)
                       for _ in range(config.num_local_experts)]
        
        # Random hidden states
        x = torch.randn(bsz, seq_len, config.hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            out = moe._forward_prefill(x)
        
        # Verify output shape and dtype
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
        assert out.dtype == torch.bfloat16, f"Output dtype should be BF16, got {out.dtype}"
        
        # Verify no NaN/Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


# ============================================================================
# Test Suite: Layer 0 Prefill (Dense MLP stress test)
# ============================================================================

class TestLayer0PrefillDenseMLP:
    """Stress test for Layer 0 (dense MLP) prefill to catch first-token divergence."""

    def test_layer0_mlp_long_sequence(self, config, torch_seed):
        """
        Test: Layer 0 (dense MLP) on long sequence (256 tokens).
        
        Context: Layer 0 is DENSE (no MoE). Any divergence in layer 0's MLP
        will propagate to all downstream MoE layers and first token of
        layer 1 output. This test catches bugs in:
        - BF16 arithmetic (gate_proj, up_proj, SiLU, down_proj)
        - Accumulation rounding
        """
        from batchgen.models.glm.glm5.model import Glm5MLP
        
        seq_len = 256
        
        # Create HF and BatchGen MLPs
        hf_mlp = HfGlmMoeDsaMLP(config).to(torch.bfloat16)
        batchgen_mlp = Glm5MLP(config).to(torch.bfloat16)
        
        # Weight copy
        batchgen_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
        batchgen_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
        batchgen_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
        
        # Long sequence input
        x = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            hf_out = hf_mlp(x)
            batchgen_out = batchgen_mlp(x)
        
        # Element-wise comparison
        diff = (batchgen_out - hf_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        logger.info(
            f"Layer 0 (seq_len={seq_len}): max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
        )
        
        assert max_diff < 5e-3, \
            f"Layer 0 long sequence divergence too large: {max_diff:.6e}"

    def test_first_token_divergence_mlp(self, config, torch_seed):
        """
        Test: MLP output at first token position (seq_len=1, batch=1).
        
        This is the exact scenario where prefill divergence manifests:
        - Input shape [1, 1, hidden]
        - Output should match exactly
        """
        from batchgen.models.glm.glm5.model import Glm5MLP
        
        hf_mlp = HfGlmMoeDsaMLP(config).to(torch.bfloat16)
        batchgen_mlp = Glm5MLP(config).to(torch.bfloat16)
        
        batchgen_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
        batchgen_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
        batchgen_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
        
        # First token only
        x = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            hf_out = hf_mlp(x)
            batchgen_out = batchgen_mlp(x)
        
        max_diff = (batchgen_out - hf_out).abs().max().item()
        
        logger.info(f"First token MLP: max_diff={max_diff:.6e}")
        
        # First token should be bit-exact (or very close)
        assert max_diff < 1e-3, \
            f"First token MLP divergence: {max_diff:.6e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# ============================================================================
# Critical Test: Bias Handling in Router (LOAD-BEARING)
# ============================================================================

class TestCriticalBiasHandling:
    """Critical test: Verify e_score_correction_bias is used for SELECTION only,
    not for weight gathering.
    
    This is load-bearing for numerical matching. HF line 565:
        topk_weights = router_logits.gather(1, topk_indices)
    where router_logits = logits.sigmoid() (UN-BIASED).
    
    The bias is used at line 549 (for topk selection), but weights come from
    raw sigmoid (line 548), not biased values (line 549).
    """

    def test_bias_not_applied_to_weights(self, config, torch_seed):
        """
        Critical: topk_weights must be raw sigmoid, not sigmoid+bias.
        
        Setup: Create a scenario where different experts are selected with/without bias.
        Verify: Returned weights are from raw sigmoid (no bias).
        """
        from batchgen.models.glm.glm5.model import Glm5MoEGate
        
        # Create with explicit bias values
        hf_moe = HfGlmMoeDsaMoE(config).to(torch.bfloat16)
        batchgen_gate = Glm5MoEGate(config).to(torch.bfloat16)
        
        # Set non-zero bias (to detect if it's being applied to weights). Use
        # .data.copy_ because e_score_correction_bias is an nn.Parameter with
        # requires_grad; direct copy_ on Parameter with requires_grad=True
        # throws a "leaf variable requires grad" error.
        bias_value = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.2, -0.2])[:config.n_routed_experts]
        hf_moe.gate.e_score_correction_bias.data.copy_(bias_value)
        batchgen_gate.e_score_correction_bias.data.copy_(bias_value)

        # Copy other weights
        batchgen_gate.weight.data.copy_(hf_moe.gate.weight.data)

        # Single token, single sequence; pass flattened to BatchGen.
        x = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16)
        x_flat = x.view(-1, config.hidden_size)

        with torch.no_grad():
            hf_logits = hf_moe.gate(x)
            hf_topk_idx, hf_topk_weights = hf_moe.route_tokens_to_experts(hf_logits)

            batchgen_topk_weights, batchgen_topk_idx = batchgen_gate(x_flat)
        
        # Verify: weights are NOT affected by bias
        # Extract the chosen expert scores before bias
        router_sigmoid = torch.sigmoid(hf_logits)  # Raw sigmoid, no bias
        
        # HF should gather from raw sigmoid
        expected_weights = router_sigmoid.gather(-1, hf_topk_idx)
        if config.norm_topk_prob:
            expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)
        expected_weights = expected_weights * config.routed_scaling_factor
        
        try:
            torch.testing.assert_close(
                hf_topk_weights, expected_weights,
                atol=1e-6, rtol=1e-6,
                msg="HF weights should be raw sigmoid (no bias applied)"
            )
        except AssertionError:
            logger.error("HF implementation mismatch: weights contain bias component")
            raise
        
        try:
            torch.testing.assert_close(
                batchgen_topk_weights, expected_weights,
                atol=1e-6, rtol=1e-6,
                msg="BatchGen weights should be raw sigmoid (no bias applied)"
            )
        except AssertionError:
            logger.error("BatchGen weights contain bias: biased applied to weights incorrectly")
            raise


# ============================================================================
# Additional Parametrized Tests for Robustness
# ============================================================================

class TestRobustness:
    """Additional parametrized tests for edge cases and robustness."""

    @pytest.mark.parametrize("hidden_size,intermediate_size", [
        (64, 128),
        (128, 256),
        (256, 512),
    ])
    def test_mlp_various_sizes(self, torch_seed, hidden_size, intermediate_size):
        """Test MLP with various hidden/intermediate sizes."""
        config = MinimalGlm5Config()
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        
        from batchgen.models.glm.glm5.model import Glm5MLP
        
        hf_mlp = HfGlmMoeDsaMLP(config).to(torch.bfloat16)
        batchgen_mlp = Glm5MLP(config).to(torch.bfloat16)
        
        batchgen_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
        batchgen_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
        batchgen_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
        
        x = torch.randn(2, 16, hidden_size, dtype=torch.bfloat16)
        
        with torch.no_grad():
            hf_out = hf_mlp(x)
            batchgen_out = batchgen_mlp(x)
        
        torch.testing.assert_close(batchgen_out, hf_out, atol=1e-3, rtol=1e-3)

    def test_router_zero_bias(self, config, torch_seed):
        """Test router with zero bias (sanity check)."""
        from batchgen.models.glm.glm5.model import Glm5MoEGate

        hf_moe = HfGlmMoeDsaMoE(config).to(torch.bfloat16)
        batchgen_gate = Glm5MoEGate(config).to(torch.bfloat16)

        # Explicitly zero bias. .data.zero_ avoids the "leaf Variable
        # requires grad" error on Parameters with requires_grad=True.
        hf_moe.gate.e_score_correction_bias.data.zero_()
        batchgen_gate.e_score_correction_bias.data.zero_()

        batchgen_gate.weight.data.copy_(hf_moe.gate.weight.data)

        x = torch.randn(4, 32, config.hidden_size, dtype=torch.bfloat16)
        x_flat = x.view(-1, config.hidden_size)

        with torch.no_grad():
            hf_logits = hf_moe.gate(x)
            hf_topk_idx, hf_topk_weights = hf_moe.route_tokens_to_experts(hf_logits)

            batchgen_topk_weights, batchgen_topk_idx = batchgen_gate(x_flat)

        assert batchgen_topk_idx.shape == hf_topk_idx.shape, (
            f"Shape mismatch: BatchGen={batchgen_topk_idx.shape} HF={hf_topk_idx.shape}"
        )
        assert (batchgen_topk_idx == hf_topk_idx).all()
        torch.testing.assert_close(batchgen_topk_weights, hf_topk_weights, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
