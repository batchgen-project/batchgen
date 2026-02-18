"""Base embedding abstraction for BatchGen.

This module defines the abstract base class for all BatchGen embedders,
providing a unified API for converting token IDs (and optional media)
into input embeddings for the LLM.

Each model's embedding class inherits BaseEmbedding and implements
model-specific logic. No registry — import and use directly.

Usage:
    from batchgen.models.moonshotai.kimi_k25.embedding import KimiK25Embedding

    embedder = KimiK25Embedding(model=model, device=device)
    inputs_embeds = embedder(input_ids)                        # text-only
    inputs_embeds = embedder(input_ids, media_items=[item])    # multimodal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MediaItem:
    """A single media item (image or video frame) to be encoded.

    Uses raw bytes to avoid PIL dependency at the interface level.
    The model-specific embedding implementation handles decoding.

    Attributes:
        data: Raw bytes of the media (JPEG/PNG/etc)
        media_type: "image" or "video_frame"
        metadata: Optional dict with resolution hints, frame index, etc.
    """

    data: bytes
    media_type: str = "image"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEmbedding(ABC):
    """Abstract base class for all BatchGen embedding implementations.

    The embedding class encapsulates the full pipeline from token IDs to
    the inputs_embeds tensor that the LLM's forward() accepts.

    For text-only models, this wraps nn.Embedding.
    For multimodal models, this also handles vision encoding and token
    replacement (e.g., replacing <|media_pad|> tokens with vision embeddings).

    Attributes:
        hidden_size: LLM hidden dimension (output embedding dimension).
        device: Torch device for computation.
        dtype: Torch dtype for output embeddings.
        supports_vision: Whether this embedder can process images/video.
    """

    hidden_size: int
    device: torch.device
    dtype: torch.dtype
    supports_vision: bool = False

    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor,
        media_items: Optional[List[MediaItem]] = None,
    ) -> torch.Tensor:
        """Convert token IDs (and optional media) to input embeddings.

        This is the main entry point. For text-only, it performs nn.Embedding
        lookup. For multimodal, it additionally runs vision encoding and
        replaces placeholder tokens with vision embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len] or [total_tokens] (packed).
            media_items: Optional list of media items to encode.
                Length must match the number of media placeholders in input_ids.

        Returns:
            inputs_embeds: [batch, seq_len, hidden_size] or
                [1, total_tokens, hidden_size] (packed).
        """

    @abstractmethod
    def embed_text(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Text-only embedding lookup.

        Args:
            input_ids: Token IDs.

        Returns:
            Text embeddings with trailing dimension == hidden_size.
        """

    def encode_vision(
        self,
        media_items: List[MediaItem],
    ) -> List[torch.Tensor]:
        """Encode media items into vision embeddings.

        Default implementation raises NotImplementedError.
        Override in multimodal subclasses.

        Args:
            media_items: List of media items to encode.

        Returns:
            List of tensors, each [num_tokens_i, hidden_size].

        Raises:
            NotImplementedError: If vision is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support vision encoding. "
            f"supports_vision={self.supports_vision}"
        )

    def replace_media_tokens(
        self,
        text_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        vision_embeds: List[torch.Tensor],
        media_pad_token_id: int,
    ) -> torch.Tensor:
        """Replace media placeholder tokens with vision embeddings.

        Default implementation for Encoder-Projector-Concat models:
        finds all positions where input_ids == media_pad_token_id and
        replaces the corresponding text embeddings with vision embeddings.

        Args:
            text_embeds: Text embeddings [batch, seq, hidden] or [total, hidden].
            input_ids: Original token IDs (same leading dims as text_embeds).
            vision_embeds: List of vision embedding tensors, one per media item.
                Each tensor is [num_tokens_i, hidden_size].
            media_pad_token_id: Token ID of the media placeholder.

        Returns:
            Modified embeddings with placeholders replaced.

        Raises:
            ValueError: If vision token count doesn't match pad positions.
        """
        result = text_embeds.clone()
        flat_ids = input_ids.reshape(-1)
        flat_result = result.reshape(-1, result.shape[-1])

        media_positions = (flat_ids == media_pad_token_id).nonzero(as_tuple=True)[0]

        if len(media_positions) == 0:
            return result

        all_vision = torch.cat(vision_embeds, dim=0)

        if all_vision.shape[0] != media_positions.shape[0]:
            raise ValueError(
                f"Vision token count mismatch: "
                f"{all_vision.shape[0]} vision tokens vs "
                f"{media_positions.shape[0]} media_pad positions in input_ids"
            )

        flat_result[media_positions] = all_vision.to(flat_result.dtype)
        return result

    def __call__(
        self,
        input_ids: torch.LongTensor,
        media_items: Optional[List[MediaItem]] = None,
    ) -> torch.Tensor:
        """Convenience: delegates to forward()."""
        return self.forward(input_ids, media_items)
