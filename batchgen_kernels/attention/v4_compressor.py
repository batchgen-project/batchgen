# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn.functional as F
from torch import nn


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.ones(hidden_size, dtype=torch.float32),
            persistent=False,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(dtype)


def _runtime_linear(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    scale: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    if weight is None:
        raise RuntimeError(
            f"DeepSeek-V4 compressor {name} weight is not loaded."
        )
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        _linear_from_weight,
    )

    return _linear_from_weight(x, weight, scale)


class DeepSeekV4Compressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        eps: float,
        overlap: bool = False,
        rotate: bool = False,
        init_weights: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        self.rotate = rotate
        self.coeff = 2 if overlap else 1
        self.register_buffer(
            "ape",
            torch.empty(
                compress_ratio, self.coeff * head_dim, dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "wkv_weight",
            torch.empty(
                self.coeff * head_dim, hidden_size, dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "wgate_weight",
            torch.empty(
                self.coeff * head_dim, hidden_size, dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer("wkv_scale", None, persistent=False)
        self.register_buffer("wgate_scale", None, persistent=False)
        self.norm = _RMSNorm(head_dim, eps)
        if init_weights:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ape, std=0.02)
        nn.init.xavier_uniform_(self.wkv_weight)
        nn.init.xavier_uniform_(self.wgate_weight)

    def _reshape_projected(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.coeff, self.head_dim)

    def _chunk_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.view(-1, self.compress_ratio)[:, 0].to(torch.int64)

    def _compress_chunks(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        num_chunks = kv.shape[0]
        ape = self.ape.view(self.compress_ratio, self.coeff, self.head_dim)
        kv = kv.float().reshape(
            num_chunks, self.compress_ratio * self.coeff, self.head_dim
        )
        gate = gate.float().reshape(
            num_chunks, self.compress_ratio * self.coeff, self.head_dim
        )
        ape = ape.float().reshape(
            self.compress_ratio * self.coeff, self.head_dim
        )
        score = gate + ape.unsqueeze(0)
        weights = F.softmax(score, dim=1)
        pooled = (kv * weights).sum(dim=1)
        pooled = self.norm(pooled)
        pooled = self._apply_rope(pooled, positions, cos_sin_cache)
        return self._maybe_rotate(pooled)

    def _maybe_rotate(self, x: torch.Tensor) -> torch.Tensor:
        if not self.rotate or x.numel() == 0:
            return x
        from batchgen_kernels.attention.dsa.fused_indexer_score import (
            get_hadamard_matrix,
        )

        H = get_hadamard_matrix(x.shape[-1], x.device, torch.float32)
        return (x.float() @ H).to(x.dtype)

    def _apply_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0 or self.rope_head_dim == 0:
            return x
        out = x.clone()
        half = self.rope_head_dim // 2
        cache = cos_sin_cache.index_select(0, positions.to(torch.long))
        cos = cache[:, :half]
        sin = cache[:, half:]
        rope = out[:, -self.rope_head_dim :].float().view(out.shape[0], half, 2)
        even = rope[..., 0]
        odd = rope[..., 1]
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        out[:, -self.rope_head_dim :] = (
            torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(out.dtype)
        )
        return out

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states.new_empty(0, self.head_dim)
        num_chunks = hidden_states.shape[0] // self.compress_ratio
        if num_chunks == 0:
            return hidden_states.new_empty(0, self.head_dim)
        tokens = num_chunks * self.compress_ratio
        hidden_states = hidden_states[:tokens]
        positions = positions[:tokens]
        kv = self._reshape_projected(
            _runtime_linear(
                hidden_states, self.wkv_weight, self.wkv_scale, "wkv"
            )
        ).view(
            num_chunks,
            self.compress_ratio,
            self.coeff,
            self.head_dim,
        )
        gate = self._reshape_projected(
            _runtime_linear(
                hidden_states, self.wgate_weight, self.wgate_scale, "wgate"
            )
        ).view(
            num_chunks,
            self.compress_ratio,
            self.coeff,
            self.head_dim,
        )
        return self._compress_chunks(
            kv,
            gate,
            self._chunk_positions(positions),
            cos_sin_cache,
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = []
        for hidden_state, position in zip(hidden_states, positions):
            hidden = hidden_state.unsqueeze(0)
            kv = _runtime_linear(
                hidden, self.wkv_weight, self.wkv_scale, "wkv"
            ).squeeze(0)
            gate = _runtime_linear(
                hidden, self.wgate_weight, self.wgate_scale, "wgate"
            ).squeeze(0)
            slot = int(position.item()) % self.compress_ratio
            kv_state[slot].copy_(kv)
            if self.overlap:
                score_state[slot].copy_(gate)
            else:
                score_state[slot].copy_(gate + self.ape[slot])
            if slot == self.compress_ratio - 1:
                chunk_pos = (
                    torch.div(
                        position.view(1),
                        self.compress_ratio,
                        rounding_mode="floor",
                    )
                    * self.compress_ratio
                )
                if self.overlap:
                    chunk_kv = kv_state.view(
                        1,
                        self.compress_ratio,
                        self.coeff,
                        self.head_dim,
                    )
                    chunk_gate = score_state.view(
                        1,
                        self.compress_ratio,
                        self.coeff,
                        self.head_dim,
                    )
                    outputs.append(
                        self._compress_chunks(
                            chunk_kv,
                            chunk_gate,
                            chunk_pos,
                            cos_sin_cache,
                        )
                    )
                else:
                    pooled = (
                        kv_state.float()
                        * torch.softmax(score_state.float(), dim=0)
                    ).sum(dim=0, keepdim=True)
                    pooled = self.norm(pooled)
                    pooled = self._apply_rope(pooled, chunk_pos, cos_sin_cache)
                    outputs.append(self._maybe_rotate(pooled))
        if outputs:
            output = torch.cat(outputs, dim=0)
        else:
            output = hidden_states.new_empty(0, self.head_dim)
        return output, kv_state, score_state

    def seed_decode_state(
        self,
        hidden_states: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for hidden_state, position in zip(hidden_states, positions):
            hidden = hidden_state.unsqueeze(0)
            kv = _runtime_linear(
                hidden, self.wkv_weight, self.wkv_scale, "wkv"
            ).squeeze(0)
            gate = _runtime_linear(
                hidden, self.wgate_weight, self.wgate_scale, "wgate"
            ).squeeze(0)
            slot = int(position.item()) % self.compress_ratio
            kv_state[slot].copy_(kv)
            if self.overlap:
                score_state[slot].copy_(gate)
            else:
                score_state[slot].copy_(gate + self.ape[slot])
        return kv_state, score_state
