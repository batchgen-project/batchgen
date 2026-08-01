# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear Parallel Strategy Manager (PSM).

BF16-only design (simpler than K2.5):
  - One model build serves both phases; wrappers route prefill/decode.
  - MoE prefill: pure DP — all 256 BF16 experts host-offloaded and streamed
    per rank by the copy engine (no collectives, no INT4/marlin).
  - MoE decode (decode_moe_mode="resident_ep", M4 P0.3): each rank holds its
    EP-8 shard resident (32 experts x 26 MoE layers, stacked BF16, ~11.8 GB)
    and runs all_gather -> fused_moe_bf16 -> comm.all_reduce(SUM) per layer
    (batchgen.moe.fused_moe_bf16_resident seam). "streamed" falls back to
    the prefill-style streaming path.
  - Attention: NoPE-MLA layers use paged KV + FlashMLA decode; KDA layers
    use pooled conv/recurrent state (KimiLinearKDAWrapper pools).
  - Decode CUDA graphs (decode_graph_mode="graph"|"compare", M5.2 Phase A):
    per-layer attention spans captured/replayed by KimiLinearDecodeGraph
    (cuda_graph_segments.py); the MoE stays eager between replays because its
    collectives cannot be captured. "eager" (default) installs nothing.

PSM <-> worker contract (methods, AttnWrapperBase class attrs the worker writes
each step, injected module attributes, comm handoff, weight-storage sharing,
KDA state pools) is documented in:
    batchgen-context/architecture/PSM_WORKER_CONTRACT.md
Read that first when modifying this file or batchgen_worker.py.
"""

import logging
import time
import types

import torch

from batchgen.models.wrappers import AttnWrapperBase

from .serving_modules import (
    kda_decode_serving,
    kda_prefill_serving,
    mla_decoding_nope_with_pagekv,
    mla_prefill_nope,
    mla_prefill_nope_prepacked,
    moe_forward_serving,
)
from .wrappers import (
    KimiLinearAttnWrapper,
    KimiLinearExpertWrapper,
    KimiLinearKDAWrapper,
)


def _replace_param(root_module, dotted_name, tensor):
    """Replace a leaf nn.Parameter by object (materializes meta params).

    `param.data = tensor` (set_data) cannot convert a meta parameter to a
    concrete tensor, so we swap the Parameter object instead.
    """
    *parent, leaf = dotted_name.split(".")
    mod = root_module.get_submodule(".".join(parent)) if parent else root_module
    mod._parameters[leaf] = torch.nn.Parameter(tensor.detach(), requires_grad=False)


class KimiLinearParallelStrategyManager:
    """Parallel Strategy Manager for Kimi-Linear (KDA + NoPE-MLA hybrid)."""

    def __init__(
        self,
        loaded_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        rank,
        global_rank,
        world_size,
    ):
        self.loaded_model_config = loaded_model_config
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.skeleton_state_dict = skeleton_state_dict
        self.rank = rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.model = None
        self.weight_copy_task = {}
        self.state_dict_name_map = {}

        self.num_experts = int(getattr(model_config, "n_routed_experts", 256) or 256)
        assert self.num_experts % world_size == 0, (
            f"num_experts {self.num_experts} not divisible by world_size {world_size}"
        )
        self.experts_per_rank = self.num_experts // world_size
        self.local_expert_start = self.global_rank * self.experts_per_rank

        self._kda_pool_slots = int(getattr(
            engine_config.GPU_Buffer_Config, "kda_state_slots", 256
        ))
        self._comm = None
        self._resident_ep_built = False
        self._decode_graph = None

    # ------------------------------------------------------------------ #
    #  Phase configuration                                                #
    # ------------------------------------------------------------------ #

    def set_comm(self, comm):
        """Receive the BatchGen NCCL communicator from the worker. Consumed by
        the resident-EP decode MoE (all_gather + all_reduce per MoE layer);
        the streamed prefill path uses no collectives."""
        self._comm = comm

    def _build_weight_copy_task(self):
        """Modules the copy engine must stream (host-offloaded), in layer-major
        ascending order. Only routed experts are streamed; attn/kda/shared are
        resident (empty lists). The MoE forward MUST consume routed experts in
        this exact order per layer so the producer never stalls."""
        cfg = self.loaded_model_config
        task = {"attn": [], "kda_attn": [], "shared_expert": [],
                "routed_expert": []}
        for layer_idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None or moe.experts is None:
                continue
            for e_idx in range(len(moe.experts)):
                task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{e_idx}"
                )
        return task

    def configure_prefill(self):
        """Build the model (if needed) and switch to prefill phase (pure DP)."""
        self.loaded_model_config.phase = "prefill"
        self.loaded_model_config.ep_size = 1  # pure DP; routed experts streamed
        if self.model is None:
            self._build_model()
        AttnWrapperBase.phase = "prefill"
        KimiLinearExpertWrapper.phase = "prefill"
        self.weight_copy_task = self._build_weight_copy_task()
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Switch to decode phase (model already built).

        decode_moe_mode="resident_ep" (default, set by the planner):
        materialize the stacked EP shard once and return an EMPTY
        routed_expert copy task — the worker then never starts the decode
        H2D streamer (batchgen_worker._load_decode_model gates on a
        non-empty routed_expert list). "streamed": legacy pure-DP streaming,
        identical to prefill.
        """
        self.loaded_model_config.phase = "decode"
        if comm is not None:
            self._comm = comm
        if self.model is None:
            self._build_model()
        AttnWrapperBase.phase = "decode"
        KimiLinearExpertWrapper.phase = "decode"
        self.weight_copy_task = self._build_weight_copy_task()
        if self._decode_moe_mode() == "resident_ep":
            self._init_resident_ep_decode()
            # Resident shards serve every routed expert: decode streams
            # nothing, so the copy engine gets no decode expert tasks.
            self.weight_copy_task["routed_expert"] = []
        self._init_decode_graph()
        return self.model, self.weight_copy_task

    def _decode_moe_mode(self):
        """Planner-set decode MoE execution mode (config-driven, no env vars):
        "resident_ep" (M4 P0.3 default) or "streamed" (fallback)."""
        return getattr(self.engine_config.EP_Config, "decode_moe_mode",
                       "resident_ep")

    def _decode_graph_mode(self):
        """Planner-set decode CUDA-graph mode (config-driven, no env vars):
        "eager" (default until M5.5), "graph" or "compare"."""
        return getattr(self.engine_config.Basic_Config, "decode_graph_mode",
                       "eager")

    def _init_decode_graph(self):
        """Install the Phase-A decode CUDA-graph adapter (M5.2), if asked for.

        Per-layer attention spans are captured lazily (first use of a bucket)
        and replayed with the MoE running eagerly between them — its resident-EP
        forward does all_gather/all_reduce, which must not be captured in
        Phase A. See cuda_graph_segments.py for the capture-structure rationale.
        """
        mode = self._decode_graph_mode()
        if mode not in ("graph", "compare"):
            return
        if self._decode_graph is not None:
            self._decode_graph.set_mode(mode)
            return
        from .cuda_graph_segments import KimiLinearDecodeGraph

        basic = self.engine_config.Basic_Config
        self._decode_graph = KimiLinearDecodeGraph(
            self.model,
            self.loaded_model_config,
            device=basic.device_torch,
            buckets=getattr(basic, "decode_graph_buckets", None),
            mode=mode,
            compare_every=getattr(basic, "decode_graph_compare_every", None),
            rank=self.rank,
        )
        self._decode_graph.install()

    def _init_resident_ep_decode(self):
        """Materialize the stacked EP-8 BF16 shards ONCE (idempotent) and
        attach a ResidentEPMoELayer to every MoE block.

        Source is the host copy-engine weight storage (core_engine.get_tensor,
        the exact tensors the streamed path consumes) — after this one-time
        H2D there is no per-step expert traffic in decode. HBM arithmetic for
        the ~11.8 GB/rank shards lives at the allocation site
        (batchgen.moe.fused_moe_bf16_resident.build_layer_shard).
        """
        if self._resident_ep_built:
            return
        from batchgen.moe.fused_moe_bf16_resident import (
            build_resident_ep_layers,
        )

        assert self._comm is not None, (
            "resident-EP decode needs the NCCL communicator (worker passes "
            "it via configure_decoding(comm=...) or set_comm)"
        )
        cfg = self.loaded_model_config
        build_resident_ep_layers(
            self.model.model.layers,
            self.core_engine.get_tensor,
            self._comm,
            self.world_size,
            self.global_rank,
            self.local_expert_start,
            self.experts_per_rank,
            cfg.moe_intermediate_size,
            self.engine_config.Basic_Config.device_torch,
        )
        self._resident_ep_built = True

    def set_num_tokens_per_rank(self, num_tokens_per_rank):
        """Worker hook (duck-typed by _sync_decode_moe_rank_counts): per-step
        MAX decode rows across ranks. Defines the padded all_gather /
        all_reduce layout of the resident-EP decode MoE — every rank
        (including empty ones) sizes the global buffer from this scalar."""
        if not self._resident_ep_built:
            return
        from batchgen.moe.fused_moe_bf16_resident import ResidentEPMoELayer

        ResidentEPMoELayer.set_num_tokens_per_rank(num_tokens_per_rank)

    # ------------------------------------------------------------------ #
    #  Model build                                                        #
    # ------------------------------------------------------------------ #

    def _build_model(self):
        from .model import KimiLinearForCausalLM

        start = time.perf_counter()
        cfg = self.loaded_model_config
        device = self.engine_config.Basic_Config.device_torch

        # Construct on META device: this allocates NO weight storage, so every
        # rank shares the single weight copy held in the parameter server's
        # shm (skeleton_state_dict) and its own GPU module tensors — we never
        # materialize a per-rank CPU copy of the 48B model (in particular the
        # 256 experts/layer are never allocated on the host).
        with torch.device("meta"):
            self.model = KimiLinearForCausalLM(cfg)
        logging.info(f"Rank {self.rank}: model skeleton constructed on meta "
                     f"({time.perf_counter() - start:.1f}s)")

        # 1. skeleton weights (embeddings, norms, gates, lm_head) — assigned
        #    from the shared shm storage (zero-copy across ranks).
        self._load_model_skeleton()

        # 2. RESIDENT module weights from the core engine (attn/kda/shared are
        #    small; routed experts are host-offloaded & streamed, see step 3).
        self._load_attn_modules()
        self._load_kda_modules()
        self._load_shared_expert_modules()

        # 3. serving method injection + wrappers
        self._config_attn_modules()
        self._config_kda_modules()
        self._config_expert_modules()   # wrap routed experts as streamed
        self._config_lm_head_hook()
        self._wrap_logits_output()

        # 4. KDA state pools
        kda_indices = [
            i for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)
        ]
        KimiLinearKDAWrapper.init_state_pools(
            kda_indices,
            num_slots=self._kda_pool_slots,
            num_heads=cfg.kda_num_heads,
            head_dim=cfg.kda_head_dim,
            conv_width=cfg.kda_conv_size,
            proj_size=cfg.kda_num_heads * cfg.kda_head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        self.model.eval()
        self.model.to(device)
        if self.rank == 0:
            logging.info(
                f"Kimi-Linear model configured in {time.perf_counter() - start:.1f}s "
                f"(EP {self.world_size}, {self.experts_per_rank} experts/rank, "
                f"KDA slots {self._kda_pool_slots})"
            )

    def _load_model_skeleton(self):
        """Load non-module weights from the checkpoint state dict."""
        device = self.engine_config.Basic_Config.device_torch
        # Loaded via core_engine instead: self_attn (attn/kda), experts (stacked),
        # shared experts. Everything else (embed, norms, router gate, attn_res,
        # lm_head) comes from the checkpoint dict here.
        skip_fragments = (".self_attn.", ".block_sparse_moe.experts.",
                          ".block_sparse_moe.shared_experts.")
        n_loaded = 0
        missing = []
        for key, param in self.model.named_parameters():
            if any(f in key for f in skip_fragments):
                continue
            if key in self.skeleton_state_dict:
                # Materialize on GPU from the shared shm tensor (native dtype).
                _replace_param(self.model, key,
                               self.skeleton_state_dict[key].to(device))
                n_loaded += 1
            elif param.is_meta:
                missing.append(key)
        if missing:
            raise RuntimeError(
                f"Kimi-Linear skeleton: {len(missing)} params not found in "
                f"weight storage (still on meta): {missing[:8]}"
            )
        if self.rank == 0:
            logging.info(f"Skeleton: {n_loaded} tensors loaded from shared storage")

    def _load_attn_modules(self):
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if cfg.is_kda_layer(layer_idx):
                continue
            attn = self.model.model.layers[layer_idx].self_attn
            tensors = self.core_engine.get_tensor(f"attn_{layer_idx}")
            for name, p in list(attn.named_parameters()):
                if name in tensors:
                    _replace_param(attn, name, tensors[name].to(device=device))
                elif self.rank == 0:
                    logging.warning(f"attn_{layer_idx}: missing tensor {name}")

    def _load_kda_modules(self):
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if not cfg.is_kda_layer(layer_idx):
                continue
            kda = self.model.model.layers[layer_idx].self_attn
            tensors = self.core_engine.get_tensor(f"kda_attn_{layer_idx}")
            for name, p in list(kda.named_parameters()):
                if name in tensors:
                    _replace_param(kda, name, tensors[name].to(device=device))
                elif self.rank == 0:
                    logging.warning(f"kda_attn_{layer_idx}: missing tensor {name}")

    def _load_shared_expert_modules(self):
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            if not hasattr(layer, "block_sparse_moe"):
                continue
            shared = layer.block_sparse_moe.shared_experts
            if shared is None:
                continue
            tensors = self.core_engine.get_tensor(f"shared_expert_{layer_idx}")
            for name, p in list(shared.named_parameters()):
                if name in tensors:
                    _replace_param(shared, name, tensors[name].to(device=device))
                elif self.rank == 0:
                    logging.warning(f"shared_expert_{layer_idx}: missing {name}")

    # ------------------------------------------------------------------ #
    #  Serving method injection                                           #
    # ------------------------------------------------------------------ #

    def _config_attn_modules(self):
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if cfg.is_kda_layer(layer_idx):
                continue
            attn = self.model.model.layers[layer_idx].self_attn
            attn.mla_prefill_nope = types.MethodType(mla_prefill_nope, attn)
            attn.mla_prefill_nope_prepacked = types.MethodType(
                mla_prefill_nope_prepacked, attn
            )
            attn.mla_decoding_nope_with_pagekv = types.MethodType(
                mla_decoding_nope_with_pagekv, attn
            )
            self.model.model.layers[layer_idx].self_attn = KimiLinearAttnWrapper(
                attn, layer_idx, self.core_engine, self.engine_config,
                self.model_config, persistent=True,
            )

    def _config_kda_modules(self):
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            if not cfg.is_kda_layer(layer_idx):
                continue
            kda = self.model.model.layers[layer_idx].self_attn
            kda.kda_prefill_serving = types.MethodType(kda_prefill_serving, kda)
            kda.kda_decode_serving = types.MethodType(kda_decode_serving, kda)
            self.model.model.layers[layer_idx].self_attn = KimiLinearKDAWrapper(
                kda, layer_idx, self.core_engine, self.engine_config,
                self.model_config, persistent=True,
            )

    def _config_expert_modules(self):
        """Wrap routed experts as streamed (host-offloaded) BF16 experts and
        inject the pure-DP MoE forward. Shared experts stay resident.

        Routed-expert weights are NOT resident: KimiLinearExpertWrapper
        (persistent=False) streams each expert from the copy-engine buffer per
        forward. Expert module params are emptied here so meta is cleared and
        `model.to(device)` is safe; the copy engine fills them per step.
        """
        device = self.engine_config.Basic_Config.device_torch
        cfg = self.loaded_model_config
        for layer_idx in range(cfg.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
            if not hasattr(layer, "block_sparse_moe"):
                continue
            moe = layer.block_sparse_moe
            if moe.experts is not None:
                for e_idx in range(len(moe.experts)):
                    expert = moe.experts[e_idx]
                    if expert is None:
                        continue
                    # Clear meta/params -> empty GPU tensors (streamed later).
                    # (`p.data =` cannot materialize meta params — swap the
                    # Parameter object via _replace_param instead.)
                    for name, _ in list(expert.named_parameters()):
                        _replace_param(expert, name,
                                       torch.empty(0, device=device))
                    moe.experts[e_idx] = KimiLinearExpertWrapper(
                        expert, layer_idx, e_idx, self.core_engine,
                        self.engine_config, self.model_config,
                        persistent=False,
                    )
            moe.forward = types.MethodType(moe_forward_serving, moe)

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(
            self._lm_head_forward_pre_hook
        )

    def _wrap_logits_output(self):
        """Worker expects HF-style outputs with .logits; our forward returns a
        raw tensor — wrap it."""
        model = self.model
        orig_forward = model.forward

        def forward_with_logits(self, *args, **kwargs):
            # The worker passes attention_mask=None; it would ride **kwargs
            # into the decoder-layer call and collide with the model's own
            # explicit attention_mask= argument (TypeError: multiple values).
            # Drop it here to keep model.py parity-clean.
            kwargs.pop("attention_mask", None)
            logits = orig_forward(*args, **kwargs)
            return types.SimpleNamespace(logits=logits)

        model.forward = types.MethodType(forward_with_logits, model)
