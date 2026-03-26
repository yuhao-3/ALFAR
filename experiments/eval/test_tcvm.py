"""
Test script for TCVM implementation

This script performs basic sanity checks on the TCVM utilities:
1. Test attention extraction and top-K selection
2. Test KV cache masking strategies
3. Test contrastive logit computation
"""

import torch
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcvm_utils import (
    get_topk_visual_indices,
    mask_visual_kv_cache,
    get_topk_context_indices,
    mask_context_kv_cache,
    compute_tcvm_contrastive_logits
)


def test_topk_visual_indices():
    """Test top-K visual token extraction"""
    print("=" * 50)
    print("Test 1: Top-K Visual Token Extraction")
    print("=" * 50)

    # Simulate attention weights: [batch=2, num_visual_tokens=576]
    batch_size = 2
    num_visual_tokens = 576
    img_start_idx = 35
    top_k = 20

    # Create random attention weights
    visual_attn = torch.randn(batch_size, num_visual_tokens)

    # Get top-K indices
    topk_indices = get_topk_visual_indices(
        visual_attn,
        img_start_idx=img_start_idx,
        top_k=top_k
    )

    print(f"Input shape: {visual_attn.shape}")
    print(f"Output shape: {topk_indices.shape}")
    print(f"Top-K indices (first batch): {topk_indices[0]}")
    print(f"Min index: {topk_indices.min().item()} (should be >= {img_start_idx})")
    print(f"Max index: {topk_indices.max().item()} (should be < {img_start_idx + num_visual_tokens})")

    assert topk_indices.shape == (batch_size, top_k), f"Shape mismatch: {topk_indices.shape}"
    assert topk_indices.min() >= img_start_idx, "Indices below img_start_idx!"
    assert topk_indices.max() < img_start_idx + num_visual_tokens, "Indices exceed visual range!"

    print("✓ Test passed!\n")


def test_topk_context_indices():
    """Test top-K context token extraction"""
    print("=" * 50)
    print("Test 1b: Top-K Context Token Extraction")
    print("=" * 50)

    # Simulate attention weights: [batch=2, num_context_tokens=100]
    batch_size = 2
    num_context_tokens = 100
    context_start_idx = 650  # After image + question + prompt
    top_k = 20

    # Create random attention weights
    context_attn = torch.randn(batch_size, num_context_tokens)

    # Get top-K indices
    topk_indices = get_topk_context_indices(
        context_attn,
        context_start_idx=context_start_idx,
        top_k=top_k
    )

    print(f"Input shape: {context_attn.shape}")
    print(f"Output shape: {topk_indices.shape}")
    print(f"Top-K indices (first batch): {topk_indices[0]}")
    print(f"Min index: {topk_indices.min().item()} (should be >= {context_start_idx})")
    print(f"Max index: {topk_indices.max().item()} (should be < {context_start_idx + num_context_tokens})")

    assert topk_indices.shape == (batch_size, top_k), f"Shape mismatch: {topk_indices.shape}"
    assert topk_indices.min() >= context_start_idx, "Indices below context_start_idx!"
    assert topk_indices.max() < context_start_idx + num_context_tokens, "Indices exceed context range!"

    print("✓ Test passed!\n")


def test_kv_cache_masking():
    """Test KV cache masking strategies"""
    print("=" * 50)
    print("Test 2: KV Cache Masking")
    print("=" * 50)

    # Simulate KV cache structure for 2 layers
    batch_size = 1
    num_heads = 32
    seq_len = 700  # Visual tokens (576) + text tokens
    head_dim = 128
    num_layers = 2
    top_k = 20

    # Create dummy KV cache
    past_kv = []
    for _ in range(num_layers):
        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        past_kv.append((key_states, value_states))

    past_kv = tuple(past_kv)

    # Indices to mask (simulating top-K visual tokens)
    topk_indices = torch.randint(35, 611, (batch_size, top_k))

    # Test each masking strategy
    for strategy in ['zero', 'mean', 'noise']:
        print(f"\nTesting strategy: {strategy}")
        masked_kv = mask_visual_kv_cache(
            past_kv,
            topk_indices,
            strategy=strategy,
            detach=True
        )

        print(f"  Original KV cache layers: {len(past_kv)}")
        print(f"  Masked KV cache layers: {len(masked_kv)}")

        # Check structure is preserved
        assert len(masked_kv) == len(past_kv), "Layer count mismatch!"

        # Check each layer
        for layer_idx, ((orig_k, orig_v), (masked_k, masked_v)) in enumerate(zip(past_kv, masked_kv)):
            assert masked_k.shape == orig_k.shape, f"Key shape mismatch at layer {layer_idx}"
            assert masked_v.shape == orig_v.shape, f"Value shape mismatch at layer {layer_idx}"

            # Verify masking happened at the right positions
            for batch_idx in range(batch_size):
                mask_pos = topk_indices[batch_idx]

                if strategy == 'zero':
                    # Check that masked positions are zeros
                    masked_region_k = masked_k[batch_idx, :, mask_pos, :]
                    assert torch.allclose(masked_region_k, torch.zeros_like(masked_region_k)), \
                        f"Zero masking failed for keys at layer {layer_idx}"

                # Check that non-masked positions are unchanged
                unmasked_mask = torch.ones(seq_len, dtype=torch.bool)
                unmasked_mask[mask_pos] = False
                unmasked_pos = torch.where(unmasked_mask)[0]

                orig_unmasked_k = orig_k[batch_idx, :, unmasked_pos, :]
                masked_unmasked_k = masked_k[batch_idx, :, unmasked_pos, :]
                assert torch.allclose(orig_unmasked_k, masked_unmasked_k), \
                    f"Unmasked positions were modified at layer {layer_idx}"

        print(f"  ✓ Strategy '{strategy}' passed!")

    print("\n✓ All masking tests passed!\n")


def test_context_kv_cache_masking():
    """Test context KV cache masking strategies"""
    print("=" * 50)
    print("Test 2b: Context KV Cache Masking")
    print("=" * 50)

    # Simulate KV cache structure for 2 layers
    batch_size = 1
    num_heads = 32
    seq_len = 800  # Visual tokens + question + prompt + context
    head_dim = 128
    num_layers = 2
    top_k = 20
    context_start_idx = 650

    # Create dummy KV cache
    past_kv = []
    for _ in range(num_layers):
        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        past_kv.append((key_states, value_states))

    past_kv = tuple(past_kv)

    # Indices to mask (simulating top-K context tokens)
    topk_indices = torch.randint(context_start_idx, context_start_idx + 100, (batch_size, top_k))

    # Test each masking strategy
    for strategy in ['zero', 'mean', 'noise']:
        print(f"\nTesting strategy: {strategy}")
        masked_kv = mask_context_kv_cache(
            past_kv,
            topk_indices,
            strategy=strategy,
            detach=True
        )

        print(f"  Original KV cache layers: {len(past_kv)}")
        print(f"  Masked KV cache layers: {len(masked_kv)}")

        # Check structure is preserved
        assert len(masked_kv) == len(past_kv), "Layer count mismatch!"

        # Check each layer
        for layer_idx, ((orig_k, orig_v), (masked_k, masked_v)) in enumerate(zip(past_kv, masked_kv)):
            assert masked_k.shape == orig_k.shape, f"Key shape mismatch at layer {layer_idx}"
            assert masked_v.shape == orig_v.shape, f"Value shape mismatch at layer {layer_idx}"

            # Verify masking happened at the right positions
            for batch_idx in range(batch_size):
                mask_pos = topk_indices[batch_idx]

                if strategy == 'zero':
                    # Check that masked positions are zeros
                    masked_region_k = masked_k[batch_idx, :, mask_pos, :]
                    assert torch.allclose(masked_region_k, torch.zeros_like(masked_region_k)), \
                        f"Zero masking failed for keys at layer {layer_idx}"

                # Check that non-masked positions are unchanged
                unmasked_mask = torch.ones(seq_len, dtype=torch.bool)
                unmasked_mask[mask_pos] = False
                unmasked_pos = torch.where(unmasked_mask)[0]

                orig_unmasked_k = orig_k[batch_idx, :, unmasked_pos, :]
                masked_unmasked_k = masked_k[batch_idx, :, unmasked_pos, :]
                assert torch.allclose(orig_unmasked_k, masked_unmasked_k), \
                    f"Unmasked positions were modified at layer {layer_idx}"

        print(f"  ✓ Strategy '{strategy}' passed!")

    print("\n✓ All context masking tests passed!\n")


def test_contrastive_logits():
    """Test contrastive logit computation"""
    print("=" * 50)
    print("Test 3: Contrastive Logit Computation")
    print("=" * 50)

    batch_size = 2
    vocab_size = 32000

    # Simulate base and masked logits
    logits_base = torch.randn(batch_size, vocab_size)
    logits_masked = torch.randn(batch_size, vocab_size)

    # Test with APC
    print("Testing with Adaptive Plausibility Constraint (APC)...")
    cd_logits_apc = compute_tcvm_contrastive_logits(
        logits_base,
        logits_masked,
        alpha=1.0,
        beta=0.7,
        apply_apc=True
    )

    print(f"  Input shape: {logits_base.shape}")
    print(f"  Output shape: {cd_logits_apc.shape}")

    # Count how many tokens are masked to -inf
    num_masked = (cd_logits_apc == -float("inf")).sum(dim=-1)
    print(f"  Tokens masked by APC: {num_masked.tolist()}")

    assert cd_logits_apc.shape == logits_base.shape, "Shape mismatch!"
    assert (num_masked > 0).all(), "APC should mask some tokens!"

    # Test without APC
    print("\nTesting without APC...")
    cd_logits_no_apc = compute_tcvm_contrastive_logits(
        logits_base,
        logits_masked,
        alpha=1.0,
        beta=0.7,
        apply_apc=False
    )

    num_masked_no_apc = (cd_logits_no_apc == -float("inf")).sum(dim=-1)
    print(f"  Tokens masked: {num_masked_no_apc.tolist()} (should be 0)")

    assert num_masked_no_apc.sum() == 0, "No masking should occur without APC!"

    print("\n✓ Contrastive logit tests passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("TCVM-KAR Unit Tests")
    print("=" * 50 + "\n")

    try:
        test_topk_visual_indices()
        test_topk_context_indices()
        test_kv_cache_masking()
        test_context_kv_cache_masking()
        test_contrastive_logits()

        print("=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nTCVM-KAR implementation is ready for integration.")
        print("The Knowledge-Aware Router can now dynamically choose between")
        print("masking visual tokens OR context tokens based on attention patterns.")
        print("\nNext step: Test with actual LLaVA model on multimodal RAG tasks.\n")

    except Exception as e:
        print("\n" + "=" * 50)
        print("✗ TEST FAILED")
        print("=" * 50)
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
