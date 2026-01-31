# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Test Kimi K2.5 embedding pipeline.

This test verifies:
1. Text embedding (embed_text) output shapes
2. PatchMergerMLP spatial merge + projection shapes
3. Media pad token replacement correctness
4. Multi-image pad replacement
5. Forward pipeline (text-only and multimodal)
6. Vision flag behavior
7. Base class encode_vision raises NotImplementedError

All tests use random/mock weights. No checkpoint or GPU required.

Usage:
    python -m pytest tests/test_kimi_k25_embedding.py -v
    or
    python tests/test_kimi_k25_embedding.py
"""

import os
import sys
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batchgen.models.base_embedding import BaseEmbedding, MediaItem
from batchgen.models.moonshotai.kimi_k25.embedding import (
    KimiK25Embedding,
    MoonViTEncoder,
    PatchMergerMLP,
)


# ============================================================================ #
# Test Constants (K2.5 dimensions)
# ============================================================================ #

VOCAB_SIZE = 164000
HIDDEN_SIZE = 7168
ENCODER_DIM = 1152
PATCH_SIZE = 14
MEDIA_PAD_TOKEN_ID = 163605


# ============================================================================ #
# Test Utilities
# ============================================================================ #


def make_mock_embed_tokens(vocab_size: int = VOCAB_SIZE, hidden_size: int = HIDDEN_SIZE):
    """Create a mock nn.Embedding for testing."""
    return nn.Embedding(vocab_size, hidden_size)


def make_mock_embedder(
    enable_vision: bool = False,
    vocab_size: int = VOCAB_SIZE,
    hidden_size: int = HIDDEN_SIZE,
):
    """Create a KimiK25Embedding with mock weights for testing."""
    embed_tokens = make_mock_embed_tokens(vocab_size, hidden_size)
    device = torch.device("cpu")
    dtype = torch.float32  # Use float32 on CPU for stability

    return KimiK25Embedding(
        embed_tokens=embed_tokens,
        device=device,
        dtype=dtype,
        enable_vision=enable_vision,
        vision_weights=None,  # Random init for testing
    )


# ============================================================================ #
# Test Functions
# ============================================================================ #


def test_text_embedding_basic():
    """Verify embed_text() output shape matches nn.Embedding."""
    embedder = make_mock_embedder(enable_vision=False)

    # Single sequence
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 32))
    output = embedder.embed_text(input_ids)
    assert output.shape == (1, 32, HIDDEN_SIZE), (
        f"Expected (1, 32, {HIDDEN_SIZE}), got {output.shape}"
    )

    # Batch of sequences
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 64))
    output = embedder.embed_text(input_ids)
    assert output.shape == (4, 64, HIDDEN_SIZE), (
        f"Expected (4, 64, {HIDDEN_SIZE}), got {output.shape}"
    )

    # 1D input (packed tokens)
    input_ids = torch.randint(0, VOCAB_SIZE, (128,))
    output = embedder.embed_text(input_ids)
    assert output.shape == (128, HIDDEN_SIZE), (
        f"Expected (128, {HIDDEN_SIZE}), got {output.shape}"
    )

    print("PASS: embed_text() output shapes are correct")


def test_patch_merger_mlp_shapes():
    """Verify PatchMergerMLP produces correct output shapes at multiple resolutions."""
    merger = PatchMergerMLP(encoder_dim=ENCODER_DIM, llm_hidden_size=HIDDEN_SIZE)

    # Test grid sizes: (H_patches, W_patches) -> (H/2 * W/2, hidden_size)
    test_cases = [
        # (H_patches, W_patches, expected_output_tokens)
        (8, 8, 16),       # 224x224 image
        (16, 16, 64),     # 448x448 image (K2.5 default)
        (32, 32, 256),    # 896x896 image
        (64, 64, 1024),   # 1792x1792 image (K2.5 max)
    ]

    for H_p, W_p, expected_tokens in test_cases:
        patch_embeds = torch.randn(1, H_p, W_p, ENCODER_DIM)
        output = merger(patch_embeds)

        assert output.shape == (1, expected_tokens, HIDDEN_SIZE), (
            f"Grid {H_p}x{W_p}: expected (1, {expected_tokens}, {HIDDEN_SIZE}), "
            f"got {output.shape}"
        )

    # Batch dimension
    patch_embeds = torch.randn(3, 16, 16, ENCODER_DIM)
    output = merger(patch_embeds)
    assert output.shape == (3, 64, HIDDEN_SIZE), (
        f"Batch=3: expected (3, 64, {HIDDEN_SIZE}), got {output.shape}"
    )

    print("PASS: PatchMergerMLP shapes are correct for all resolutions")


def test_media_pad_replacement():
    """Verify replace_media_tokens() replaces correct positions."""
    embedder = make_mock_embedder(enable_vision=False)

    seq_len = 20
    num_vision_tokens = 4

    # Create input_ids with some media_pad tokens in the middle
    input_ids = torch.randint(0, 1000, (1, seq_len))
    pad_start = 5
    input_ids[0, pad_start:pad_start + num_vision_tokens] = MEDIA_PAD_TOKEN_ID

    # Create text embeddings
    text_embeds = torch.randn(1, seq_len, HIDDEN_SIZE)
    original_text = text_embeds.clone()

    # Create vision embeddings
    vision_embeds = [torch.randn(num_vision_tokens, HIDDEN_SIZE) * 100]  # Large scale to distinguish

    # Replace
    result = embedder.replace_media_tokens(
        text_embeds, input_ids, vision_embeds, MEDIA_PAD_TOKEN_ID
    )

    # Check: non-pad positions should be unchanged
    for i in range(seq_len):
        if i < pad_start or i >= pad_start + num_vision_tokens:
            assert torch.equal(result[0, i], original_text[0, i]), (
                f"Position {i} was modified but should be unchanged"
            )

    # Check: pad positions should be replaced with vision embeds
    for i in range(num_vision_tokens):
        assert torch.allclose(
            result[0, pad_start + i],
            vision_embeds[0][i],
            atol=1e-6,
        ), f"Pad position {pad_start + i} was not replaced correctly"

    print("PASS: replace_media_tokens() replaces correct positions")


def test_media_pad_replacement_multiple_images():
    """Verify replace_media_tokens() with multiple contiguous pad runs."""
    embedder = make_mock_embedder(enable_vision=False)

    seq_len = 30
    # Two images: 4 tokens each, at different positions
    input_ids = torch.randint(0, 1000, (1, seq_len))
    # Image 1: positions 3-6
    input_ids[0, 3:7] = MEDIA_PAD_TOKEN_ID
    # Image 2: positions 15-18
    input_ids[0, 15:19] = MEDIA_PAD_TOKEN_ID

    text_embeds = torch.randn(1, seq_len, HIDDEN_SIZE)
    original_text = text_embeds.clone()

    # Two sets of vision embeddings
    vision_1 = torch.ones(4, HIDDEN_SIZE) * 10.0   # Image 1: all 10s
    vision_2 = torch.ones(4, HIDDEN_SIZE) * -10.0   # Image 2: all -10s
    vision_embeds = [vision_1, vision_2]

    result = embedder.replace_media_tokens(
        text_embeds, input_ids, vision_embeds, MEDIA_PAD_TOKEN_ID
    )

    # Image 1 positions should have value ~10
    for i in range(4):
        assert torch.allclose(result[0, 3 + i], vision_1[i], atol=1e-6), (
            f"Image 1, position {3 + i}: expected ~10, got {result[0, 3 + i, 0].item():.2f}"
        )

    # Image 2 positions should have value ~-10
    for i in range(4):
        assert torch.allclose(result[0, 15 + i], vision_2[i], atol=1e-6), (
            f"Image 2, position {15 + i}: expected ~-10, got {result[0, 15 + i, 0].item():.2f}"
        )

    # Non-pad positions unchanged
    for i in [0, 1, 2, 7, 8, 14, 19, 20, 29]:
        assert torch.equal(result[0, i], original_text[0, i]), (
            f"Position {i} was modified but should be unchanged"
        )

    print("PASS: Multiple image pad replacement is correct")


def test_media_pad_replacement_count_mismatch():
    """Verify replace_media_tokens() raises ValueError on token count mismatch."""
    embedder = make_mock_embedder(enable_vision=False)

    input_ids = torch.randint(0, 1000, (1, 20))
    input_ids[0, 5:9] = MEDIA_PAD_TOKEN_ID  # 4 pad positions

    text_embeds = torch.randn(1, 20, HIDDEN_SIZE)

    # Provide wrong number of vision tokens (3 instead of 4)
    vision_embeds = [torch.randn(3, HIDDEN_SIZE)]

    try:
        embedder.replace_media_tokens(
            text_embeds, input_ids, vision_embeds, MEDIA_PAD_TOKEN_ID
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "mismatch" in str(e).lower(), f"Unexpected error message: {e}"

    print("PASS: Token count mismatch raises ValueError")


def test_forward_text_only():
    """Verify forward(input_ids, media_items=None) equals embed_text()."""
    embedder = make_mock_embedder(enable_vision=False)

    input_ids = torch.randint(0, VOCAB_SIZE, (2, 48))

    # forward() with no media should equal embed_text()
    forward_output = embedder.forward(input_ids, media_items=None)
    text_output = embedder.embed_text(input_ids)

    assert torch.equal(forward_output, text_output), (
        "forward() without media should equal embed_text()"
    )

    # Also test via __call__
    call_output = embedder(input_ids)
    assert torch.equal(call_output, text_output), (
        "__call__() without media should equal embed_text()"
    )

    print("PASS: forward() text-only equals embed_text()")


def test_forward_with_mock_vision():
    """End-to-end forward with mock vision encoder."""
    embedder = make_mock_embedder(enable_vision=True)

    seq_len = 32
    num_vision_tokens = 64  # 16x16 patches after 2x2 merge

    # Build input_ids with media pad tokens
    input_ids = torch.randint(0, 1000, (1, seq_len))
    input_ids[0, 10:10 + num_vision_tokens] = MEDIA_PAD_TOKEN_ID

    # We need to mock encode_vision since we don't have real images
    # Create fake vision embeddings that encode_vision would produce
    mock_vision = torch.randn(num_vision_tokens, HIDDEN_SIZE)

    # Patch encode_vision to return our mock embeddings
    original_encode = embedder.encode_vision
    embedder.encode_vision = lambda items: [mock_vision.to(embedder.dtype)]

    media_items = [MediaItem(data=b"fake_image_bytes")]
    result = embedder.forward(input_ids, media_items=media_items)

    # Restore
    embedder.encode_vision = original_encode

    # Output shape should match input
    assert result.shape == (1, seq_len, HIDDEN_SIZE), (
        f"Expected (1, {seq_len}, {HIDDEN_SIZE}), got {result.shape}"
    )

    # Vision positions should contain the mock vision embeddings
    for i in range(num_vision_tokens):
        assert torch.allclose(
            result[0, 10 + i],
            mock_vision[i].to(result.dtype),
            atol=1e-5,
        ), f"Vision position {10 + i} not correctly replaced"

    print("PASS: forward() with mock vision produces correct output")


def test_supports_vision_flag():
    """Verify supports_vision=False ignores media_items."""
    embedder = make_mock_embedder(enable_vision=False)

    input_ids = torch.randint(0, VOCAB_SIZE, (1, 20))
    media_items = [MediaItem(data=b"fake_image_bytes")]

    # With vision disabled, media_items should be ignored
    result_with_media = embedder.forward(input_ids, media_items=media_items)
    result_without_media = embedder.forward(input_ids, media_items=None)

    assert torch.equal(result_with_media, result_without_media), (
        "With supports_vision=False, media_items should be ignored"
    )

    print("PASS: supports_vision=False correctly ignores media_items")


def test_encode_vision_not_implemented():
    """Verify BaseEmbedding.encode_vision() raises NotImplementedError."""
    # Create a minimal concrete subclass that doesn't override encode_vision
    class MinimalEmbedding(BaseEmbedding):
        def __init__(self):
            self.hidden_size = 128
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.supports_vision = False

        def forward(self, input_ids, media_items=None):
            return torch.zeros(1)

        def embed_text(self, input_ids):
            return torch.zeros(1)

    embedder = MinimalEmbedding()

    try:
        embedder.encode_vision([MediaItem(data=b"test")])
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "MinimalEmbedding" in str(e), (
            f"Error should mention class name, got: {e}"
        )

    print("PASS: Base class encode_vision() raises NotImplementedError")


def test_moonvit_encoder_shapes():
    """Verify MoonViTEncoder forward produces correct output shapes."""
    encoder = MoonViTEncoder(
        hidden_dim=ENCODER_DIM,
        num_layers=2,  # Use fewer layers for test speed
        num_heads=16,
        mlp_dim=4608,
        patch_size=PATCH_SIZE,
    )

    # 224x224 image -> 16x16 patches
    pixel_values = torch.randn(1, 3, 224, 224)
    output = encoder(pixel_values)
    assert output.shape == (1, 16, 16, ENCODER_DIM), (
        f"224x224: expected (1, 16, 16, {ENCODER_DIM}), got {output.shape}"
    )

    # 448x448 image -> 32x32 patches
    pixel_values = torch.randn(1, 3, 448, 448)
    output = encoder(pixel_values)
    assert output.shape == (1, 32, 32, ENCODER_DIM), (
        f"448x448: expected (1, 32, 32, {ENCODER_DIM}), got {output.shape}"
    )

    print("PASS: MoonViTEncoder output shapes are correct")


# ============================================================================ #
# Main
# ============================================================================ #


def run_all_tests():
    """Run all K2.5 embedding tests."""
    print("=" * 60)
    print("Kimi K2.5 Embedding Test Suite")
    print("=" * 60)

    tests = [
        ("Text Embedding Basic", test_text_embedding_basic),
        ("PatchMergerMLP Shapes", test_patch_merger_mlp_shapes),
        ("Media Pad Replacement", test_media_pad_replacement),
        ("Multi-Image Pad Replacement", test_media_pad_replacement_multiple_images),
        ("Pad Replacement Count Mismatch", test_media_pad_replacement_count_mismatch),
        ("Forward Text-Only", test_forward_text_only),
        ("Forward With Mock Vision", test_forward_with_mock_vision),
        ("Supports Vision Flag", test_supports_vision_flag),
        ("Base encode_vision NotImplemented", test_encode_vision_not_implemented),
        ("MoonViT Encoder Shapes", test_moonvit_encoder_shapes),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"Test: {name}")
        print("-" * 60)
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
