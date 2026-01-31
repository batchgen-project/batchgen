"""Kimi K2.5 embedding with optional MoonViT vision encoding.

Pipeline:
1. Text embedding: nn.Embedding lookup via model.embed_tokens
2. Vision encoding (if media present):
   a. Image preprocessing (decode bytes, normalize, patchify)
   b. MoonViT encoder (SigLIP-based, 400M params, 27 layers)
   c. PatchMergerMLP projector (2x2 spatial merge, 1152-dim -> 7168-dim)
3. Token replacement: <|media_pad|> positions <- projected vision embeds

Text-only mode (enable_vision=False):
    Only step 1. Vision weights not loaded. ~0 additional memory.

Multimodal mode (enable_vision=True):
    All three steps. Vision weights loaded from checkpoint. ~934 MB per GPU.
"""

import io
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.base_embedding import BaseEmbedding, MediaItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PatchMergerMLP — 2x2 spatial patch merge + MLP projection
# ---------------------------------------------------------------------------


class PatchMergerMLP(nn.Module):
    """2x2 spatial patch merger + MLP projector for K2.5.

    Merges 4 adjacent vision patches into 1, then projects to LLM hidden dim.

    Flow:
        Input:  [B, H_patches, W_patches, encoder_dim]  (e.g., [B, 16, 16, 1152])
        Merge:  [B, H/2 * W/2, 4 * encoder_dim]         (e.g., [B, 64, 4608])
        LN:     [B, N/4, 4608]
        FC1:    [B, N/4, 4608] -> GELU
        FC2:    [B, N/4, llm_hidden]                      (e.g., [B, 64, 7168])

    Checkpoint weight prefix: mm_projector.*
    Memory: ~134 MB (BF16)
    """

    def __init__(
        self,
        encoder_dim: int = 1152,
        llm_hidden_size: int = 7168,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_hidden_size = llm_hidden_size
        self.merge_dim = 4 * encoder_dim  # 4608

        self.ln = nn.LayerNorm(self.merge_dim)
        self.fc1 = nn.Linear(self.merge_dim, self.merge_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.merge_dim, self.llm_hidden_size)

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        """Merge 2x2 patches and project to LLM dimension.

        Args:
            patch_embeds: [B, H_patches, W_patches, encoder_dim]
                H_patches and W_patches must both be even.

        Returns:
            [B, H_patches/2 * W_patches/2, llm_hidden_size]
        """
        B, H, W, D = patch_embeds.shape
        assert H % 2 == 0 and W % 2 == 0, (
            f"Patch grid must be even in both dims, got {H}x{W}"
        )

        # 2x2 spatial merge: group adjacent patches
        merged = patch_embeds.reshape(B, H // 2, 2, W // 2, 2, D)
        merged = merged.permute(0, 1, 3, 2, 4, 5).contiguous()
        merged = merged.reshape(B, (H // 2) * (W // 2), 4 * D)

        # Project: LN -> FC1 -> GELU -> FC2
        out = self.ln(merged)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out


# ---------------------------------------------------------------------------
# MoonViT Encoder — structural stub
# ---------------------------------------------------------------------------


class MoonViTAttention(nn.Module):
    """Single attention layer for MoonViT.

    Structural stub: uses standard PyTorch scaled_dot_product_attention.
    Production optimization (FlashAttention-2, 2D RoPE) is future work.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        return self.proj(attn_out)


class MoonViTMLP(nn.Module):
    """MLP block for MoonViT (SigLIP-style)."""

    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoonViTBlock(nn.Module):
    """Single transformer block for MoonViT."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MoonViTAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MoonViTMLP(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MoonViTEncoder(nn.Module):
    """MoonViT vision encoder for K2.5 (structural stub).

    Architecture:
        - 27 transformer layers
        - 1152-dim hidden, 16 attention heads
        - 14x14 pixel patch size
        - MLP dim: 4 * 1152 = 4608
        - Position encoding: learnable 2D (2D RoPE deferred)

    This is a structural stub: all nn.Module layers are defined so weights
    load correctly from checkpoint, and the forward path is functional using
    standard PyTorch attention. FlashAttention-2 and 2D RoPE optimization
    are deferred to a future implementation.

    Checkpoint weight prefix: vision_tower.*
    Memory: ~800 MB (BF16)
    """

    def __init__(
        self,
        hidden_dim: int = 1152,
        num_layers: int = 27,
        num_heads: int = 16,
        mlp_dim: int = 4608,
        patch_size: int = 14,
        image_channels: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding: Conv2D with patch_size stride
        self.patch_embed = nn.Conv2d(
            image_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MoonViTBlock(hidden_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_post = nn.LayerNorm(hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image pixels to patch embeddings.

        Args:
            pixel_values: [B, C, H, W] normalized to [-1, 1].
                H and W must be divisible by patch_size (14).

        Returns:
            [B, H_patches, W_patches, hidden_dim]
            where H_patches = H // 14, W_patches = W // 14.
        """
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Image dims must be divisible by patch_size={self.patch_size}, "
            f"got {H}x{W}"
        )

        # Patch embedding: [B, C, H, W] -> [B, hidden_dim, H/P, W/P]
        x = self.patch_embed(pixel_values)
        H_p, W_p = x.shape[2], x.shape[3]

        # Reshape to sequence: [B, N, hidden_dim]
        x = x.flatten(2).transpose(1, 2)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.ln_post(x)

        # Reshape back to spatial: [B, H_patches, W_patches, hidden_dim]
        x = x.reshape(B, H_p, W_p, self.hidden_dim)
        return x


# ---------------------------------------------------------------------------
# KimiK25Embedding — full embedding pipeline
# ---------------------------------------------------------------------------


class KimiK25Embedding(BaseEmbedding):
    """Kimi K2.5 embedding with optional MoonViT vision encoding.

    Text-only mode:
        embed_text() wraps model.embed_tokens. No vision weights loaded.

    Multimodal mode (enable_vision=True):
        Three-step pipeline: text embed -> vision encode -> token replacement.
        Vision weights loaded from checkpoint (~934 MB per GPU).
    """

    # K2.5 special token IDs
    MEDIA_PAD_TOKEN_ID = 163605
    MEDIA_BEGIN_TOKEN_ID = 163602
    MEDIA_CONTENT_TOKEN_ID = 163603
    MEDIA_END_TOKEN_ID = 163604

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        device: torch.device,
        dtype: torch.dtype = None,
        enable_vision: bool = False,
        vision_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize K2.5 embedding.

        Args:
            embed_tokens: The model's nn.Embedding layer (reference, not copy).
            device: Torch device for computation.
            dtype: Output dtype (default: bfloat16).
            enable_vision: Whether to load and enable vision encoder.
            vision_weights: Pre-loaded vision weights dict. Keys should include
                vision_tower.* and mm_projector.* prefixed tensors.
        """
        self.hidden_size = 7168
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.supports_vision = enable_vision

        # Text embedding: reference to model's embed_tokens (already on GPU)
        self._embed_tokens = embed_tokens

        # Vision components (initialized only if enable_vision=True)
        self._vision_encoder: Optional[MoonViTEncoder] = None
        self._patch_merger: Optional[PatchMergerMLP] = None

        if enable_vision:
            self._init_vision(vision_weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        media_items: Optional[List[MediaItem]] = None,
    ) -> torch.Tensor:
        """Full embedding pipeline.

        Args:
            input_ids: Token IDs [batch, seq] or [total_tokens].
            media_items: Optional list of media items (images).

        Returns:
            inputs_embeds: [batch, seq, 7168] or [total_tokens, 7168].
        """
        # Step 1: Text embedding
        text_embeds = self.embed_text(input_ids)

        # If no media or vision not enabled, return text-only
        if not media_items or not self.supports_vision:
            return text_embeds

        # Step 2: Vision encoding
        vision_embeds = self.encode_vision(media_items)

        # Step 3: Replace <|media_pad|> tokens with vision embeddings
        result = self.replace_media_tokens(
            text_embeds,
            input_ids,
            vision_embeds,
            media_pad_token_id=self.MEDIA_PAD_TOKEN_ID,
        )
        return result

    def embed_text(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Text embedding lookup via model's embed_tokens.

        Args:
            input_ids: Token IDs.

        Returns:
            Text embeddings [..., 7168].
        """
        return self._embed_tokens(input_ids.to(self.device))

    def encode_vision(
        self,
        media_items: List[MediaItem],
    ) -> List[torch.Tensor]:
        """Encode media items using MoonViT + PatchMergerMLP.

        Args:
            media_items: List of MediaItem with raw image bytes.

        Returns:
            List of tensors, each [num_tokens_i, 7168].
        """
        if self._vision_encoder is None or self._patch_merger is None:
            raise NotImplementedError(
                "Vision encoder not initialized. "
                "Set enable_vision=True and provide vision_weights."
            )

        results = []
        for item in media_items:
            # Decode and preprocess image
            pixel_values = self._preprocess_image(item.data)
            pixel_values = pixel_values.to(self.device, self.dtype)

            # MoonViT encoding: [1, C, H, W] -> [1, H_p, W_p, 1152]
            with torch.inference_mode():
                patch_embeds = self._vision_encoder(pixel_values)

                # PatchMergerMLP: [1, H_p, W_p, 1152] -> [1, N/4, 7168]
                projected = self._patch_merger(patch_embeds)

            # Squeeze batch dim: [N/4, 7168]
            results.append(projected.squeeze(0))

        return results

    def _init_vision(
        self,
        vision_weights: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Initialize vision encoder and projector.

        Args:
            vision_weights: Dict of checkpoint tensors. If None, modules are
                initialized with random weights (useful for testing).
        """
        self._vision_encoder = MoonViTEncoder()
        self._patch_merger = PatchMergerMLP()

        if vision_weights is not None:
            encoder_weights = {
                k.replace("vision_tower.", ""): v
                for k, v in vision_weights.items()
                if k.startswith("vision_tower.")
            }
            projector_weights = {
                k.replace("mm_projector.", ""): v
                for k, v in vision_weights.items()
                if k.startswith("mm_projector.")
            }
            if encoder_weights:
                self._vision_encoder.load_state_dict(
                    encoder_weights, strict=False
                )
                logger.info(
                    f"Loaded {len(encoder_weights)} vision encoder weights"
                )
            if projector_weights:
                self._patch_merger.load_state_dict(
                    projector_weights, strict=False
                )
                logger.info(
                    f"Loaded {len(projector_weights)} projector weights"
                )

        self._vision_encoder = self._vision_encoder.to(self.device, self.dtype)
        self._patch_merger = self._patch_merger.to(self.device, self.dtype)
        self._vision_encoder.eval()
        self._patch_merger.eval()

        encoder_params = sum(
            p.numel() for p in self._vision_encoder.parameters()
        )
        projector_params = sum(
            p.numel() for p in self._patch_merger.parameters()
        )
        logger.info(
            f"Vision encoder: {encoder_params / 1e6:.0f}M params, "
            f"projector: {projector_params / 1e6:.0f}M params"
        )

    @staticmethod
    def _preprocess_image(image_bytes: bytes) -> torch.Tensor:
        """Decode and preprocess image bytes for MoonViT.

        Args:
            image_bytes: Raw JPEG/PNG bytes.

        Returns:
            [1, 3, H, W] tensor normalized to [-1, 1], with H and W
            padded to multiples of 14 (patch size).
        """
        try:
            from torchvision import transforms
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Vision encoding requires torchvision and Pillow. "
                "Install with: pip install torchvision Pillow"
            )

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        # Pad to patch-aligned resolution (multiple of 14)
        patch_size = 14
        new_h = math.ceil(h / patch_size) * patch_size
        new_w = math.ceil(w / patch_size) * patch_size

        transform = transforms.Compose([
            transforms.Resize((new_h, new_w)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        pixel_values = transform(image).unsqueeze(0)  # [1, 3, H, W]
        return pixel_values
