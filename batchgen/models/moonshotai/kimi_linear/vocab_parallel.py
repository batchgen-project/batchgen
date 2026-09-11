"""Vocab-parallel embedding and lm_head for the head-parallel (TP-G) K3 decode.

Slice 2 of DECODE_CONCURRENCY_PLAN.md: ``embed_tokens`` and ``lm_head`` are
2.19 GiB each per rank on the world16 K3 and were replicated on every rank of
the TP8 attention group. Each rank now holds the vocab rows
``[c * V/G, (c + 1) * V/G)`` (c = tp rank):

* ``VocabParallelEmbedding``: masked local lookup + one all_reduce over the
  TP group (the group's rows are replicated, so every rank needs the full
  embedding of every row). Once per forward, graph-capturable.
* ``VocabParallelLMHead``: local ``[rows, V/G]`` logits + one all_gather over
  the TP group, returned as the full ``[..., V]`` tensor the worker's
  sampling consumes (52 MB at 160 rows, NVLink). The greedy fast path
  (all_gather of per-rank argmax only) can replace the gather later.

Both keep the nn.Embedding / nn.Linear call contracts, so the prefill and
decode callers do not change.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):
    def __init__(self, weight_shard: torch.Tensor, vocab_start: int, tp_group,
                 padding_idx=None):
        super().__init__()
        self.weight = nn.Parameter(weight_shard.detach(), requires_grad=False)
        self.vocab_start = int(vocab_start)
        self.shard_size = int(weight_shard.shape[0])
        self.tp_group = tp_group
        self.padding_idx = padding_idx
        self.embedding_dim = int(weight_shard.shape[1])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        local = input_ids - self.vocab_start
        mask = (local >= 0) & (local < self.shard_size)
        local = torch.where(mask, local, torch.zeros_like(local))
        out = F.embedding(local, self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        dist.all_reduce(out, group=self.tp_group)
        return out


class VocabParallelLMHead(nn.Module):
    def __init__(self, weight_shard: torch.Tensor, vocab_size: int, tp_size: int,
                 tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight_shard.detach(), requires_grad=False)
        self.vocab_size = int(vocab_size)
        self.tp_size = int(tp_size)
        self.tp_group = tp_group
        self.in_features = int(weight_shard.shape[1])
        self.out_features = self.vocab_size
        self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        local = F.linear(hidden_states, self.weight).contiguous()   # [..., V/G]
        gathered = local.new_empty((self.tp_size,) + tuple(local.shape))
        dist.all_gather_into_tensor(gathered, local, group=self.tp_group)
        return assemble_gathered_logits(gathered, self.vocab_size)


def assemble_gathered_logits(gathered: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """[G, ..., V/G] (rank-major all_gather) -> [..., V] with rank c's slice at
    columns [c*V/G, (c+1)*V/G)."""
    lead = tuple(gathered.shape[1:-1])
    perm = list(range(1, gathered.dim() - 1)) + [0, gathered.dim() - 1]
    return gathered.permute(*perm).reshape(*lead, vocab_size)


def vocab_shard_bounds(vocab_size: int, tp_size: int, tp_rank: int):
    if vocab_size % tp_size:
        raise RuntimeError(
            f"vocab size {vocab_size} is not divisible by the TP size {tp_size}")
    rows = vocab_size // tp_size
    return tp_rank * rows, (tp_rank + 1) * rows
