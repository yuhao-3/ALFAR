"""
TCVM (Token-Level Causal Visual Masking) Utilities

This module provides helper functions for implementing TCVM:
- Extract top-K attended visual tokens
- Mask KV cache at specific positions
- Run counterfactual forward pass with masked visual context
"""

import torch
from typing import Tuple, Optional, List


def get_topk_visual_indices(
    visual_attn_weights: torch.Tensor,
    img_start_idx: int,
    top_k: int
) -> torch.Tensor:
    """
    Extract indices of top-K attended visual tokens.

    Args:
        visual_attn_weights: [batch, num_visual_tokens] or [num_visual_tokens]
                            attention weights from current token to visual tokens
        img_start_idx: starting index of visual tokens in sequence (typically 35)
        top_k: number of top attended tokens to extract

    Returns:
        topk_indices: [batch, top_k] or [top_k] absolute sequence indices

    Example:
        >>> visual_attn = torch.randn(1, 576)  # 576 visual tokens
        >>> indices = get_topk_visual_indices(visual_attn, img_start_idx=35, top_k=20)
        >>> indices.shape
        torch.Size([1, 20])
        >>> assert indices.min() >= 35 and indices.max() < 35 + 576
    """
    # Handle both batched and unbatched inputs
    if visual_attn_weights.dim() == 1:
        visual_attn_weights = visual_attn_weights.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Get top-K values and their relative indices within visual token range
    # BUGFIX: Ensure k doesn't exceed available visual tokens
    actual_k = min(top_k, visual_attn_weights.size(-1))
    _, topk_relative_idx = torch.topk(visual_attn_weights, k=actual_k, dim=-1)

    # Convert to absolute sequence indices
    topk_absolute_idx = topk_relative_idx + img_start_idx

    if squeeze_output:
        topk_absolute_idx = topk_absolute_idx.squeeze(0)

    return topk_absolute_idx


def mask_visual_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    topk_indices: torch.Tensor,
    strategy: str = 'zero',
    detach: bool = True
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Clone and mask KV cache at specific visual token positions.

    This creates a counterfactual KV cache where the top-K attended visual tokens
    are masked out, simulating the model's behavior without those visual cues.

    Args:
        past_key_values: tuple of (key_states, value_states) pairs for each layer
                        - key_states: [batch, num_heads, seq_len, head_dim]
                        - value_states: [batch, num_heads, seq_len, head_dim]
        topk_indices: [batch, top_k] or [top_k] indices of tokens to mask
        strategy: masking strategy
            - 'zero': set key/value to zeros
            - 'mean': replace with mean of all visual tokens
            - 'noise': replace with Gaussian noise (mean=0, std=0.01)
        detach: if True, detach cloned tensors to save memory

    Returns:
        masked_past_kv: cloned and masked KV cache (same structure as input)

    Note:
        The function clones the entire KV cache to avoid modifying the original.
        For efficiency, only the specified indices are modified.
    """
    if topk_indices.dim() == 1:
        topk_indices = topk_indices.unsqueeze(0)

    batch_size = topk_indices.shape[0]
    masked_kv = []

    for layer_idx, (key_states, value_states) in enumerate(past_key_values):
        # Clone to avoid modifying original
        if detach:
            key_masked = key_states.clone().detach()
            value_masked = value_states.clone().detach()
        else:
            key_masked = key_states.clone()
            value_masked = value_states.clone()

        # Apply masking strategy for each batch
        for batch_idx in range(batch_size):
            mask_positions = topk_indices[batch_idx]  # [top_k]

            if strategy == 'zero':
                # Zero out the key and value at masked positions
                key_masked[batch_idx, :, mask_positions, :] = 0.0
                value_masked[batch_idx, :, mask_positions, :] = 0.0

            elif strategy == 'mean':
                # Replace with mean of all tokens (global average pooling)
                mean_key = key_states[batch_idx].mean(dim=1, keepdim=True)  # [num_heads, 1, head_dim]
                mean_value = value_states[batch_idx].mean(dim=1, keepdim=True)

                # Broadcast mean to masked positions
                key_masked[batch_idx, :, mask_positions, :] = mean_key.expand(-1, len(mask_positions), -1)
                value_masked[batch_idx, :, mask_positions, :] = mean_value.expand(-1, len(mask_positions), -1)

            elif strategy == 'noise':
                # Replace with Gaussian noise
                key_shape = key_states[batch_idx, :, mask_positions, :].shape
                value_shape = value_states[batch_idx, :, mask_positions, :].shape

                key_noise = torch.randn(key_shape, device=key_states.device, dtype=key_states.dtype) * 0.01
                value_noise = torch.randn(value_shape, device=value_states.device, dtype=value_states.dtype) * 0.01

                key_masked[batch_idx, :, mask_positions, :] = key_noise
                value_masked[batch_idx, :, mask_positions, :] = value_noise

            else:
                raise ValueError(f"Unknown masking strategy: {strategy}. Choose from ['zero', 'mean', 'noise']")

        masked_kv.append((key_masked, value_masked))

    return tuple(masked_kv)


def get_topk_context_indices(
    context_attn_weights: torch.Tensor,
    context_start_idx: int,
    top_k: int
) -> torch.Tensor:
    """
    Extract indices of top-K attended context tokens.

    Args:
        context_attn_weights: [batch, num_context_tokens] or [num_context_tokens]
                             attention weights from current token to context tokens
        context_start_idx: starting index of context tokens in sequence
        top_k: number of top attended tokens to extract

    Returns:
        topk_indices: [batch, top_k] or [top_k] absolute sequence indices

    Example:
        >>> context_attn = torch.randn(1, 100)  # 100 context tokens
        >>> indices = get_topk_context_indices(context_attn, context_start_idx=650, top_k=20)
        >>> indices.shape
        torch.Size([1, 20])
        >>> assert indices.min() >= 650 and indices.max() < 650 + 100
    """
    # Handle both batched and unbatched inputs
    if context_attn_weights.dim() == 1:
        context_attn_weights = context_attn_weights.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Get top-K values and their relative indices within context token range
    # BUGFIX: Ensure k doesn't exceed available context tokens
    actual_k = min(top_k, context_attn_weights.size(-1))
    _, topk_relative_idx = torch.topk(context_attn_weights, k=actual_k, dim=-1)

    # Convert to absolute sequence indices
    topk_absolute_idx = topk_relative_idx + context_start_idx

    if squeeze_output:
        topk_absolute_idx = topk_absolute_idx.squeeze(0)

    return topk_absolute_idx


def mask_context_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    topk_indices: torch.Tensor,
    strategy: str = 'zero',
    detach: bool = True
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Clone and mask KV cache at specific context token positions.

    This creates a counterfactual KV cache where the top-K attended context tokens
    are masked out, simulating the model's behavior without those context cues.

    Args:
        past_key_values: tuple of (key_states, value_states) pairs for each layer
                        - key_states: [batch, num_heads, seq_len, head_dim]
                        - value_states: [batch, num_heads, seq_len, head_dim]
        topk_indices: [batch, top_k] or [top_k] indices of tokens to mask
        strategy: masking strategy
            - 'zero': set key/value to zeros
            - 'mean': replace with mean of all context tokens
            - 'noise': replace with Gaussian noise (mean=0, std=0.01)
        detach: if True, detach cloned tensors to save memory

    Returns:
        masked_past_kv: cloned and masked KV cache (same structure as input)

    Note:
        The function clones the entire KV cache to avoid modifying the original.
        For efficiency, only the specified indices are modified.
    """
    if topk_indices.dim() == 1:
        topk_indices = topk_indices.unsqueeze(0)

    batch_size = topk_indices.shape[0]
    masked_kv = []

    for layer_idx, (key_states, value_states) in enumerate(past_key_values):
        # Clone to avoid modifying original
        if detach:
            key_masked = key_states.clone().detach()
            value_masked = value_states.clone().detach()
        else:
            key_masked = key_states.clone()
            value_masked = value_states.clone()

        # Apply masking strategy for each batch
        for batch_idx in range(batch_size):
            mask_positions = topk_indices[batch_idx]  # [top_k]

            if strategy == 'zero':
                # Zero out the key and value at masked positions
                key_masked[batch_idx, :, mask_positions, :] = 0.0
                value_masked[batch_idx, :, mask_positions, :] = 0.0

            elif strategy == 'mean':
                # Replace with mean of all tokens (global average pooling)
                mean_key = key_states[batch_idx].mean(dim=1, keepdim=True)  # [num_heads, 1, head_dim]
                mean_value = value_states[batch_idx].mean(dim=1, keepdim=True)

                # Broadcast mean to masked positions
                key_masked[batch_idx, :, mask_positions, :] = mean_key.expand(-1, len(mask_positions), -1)
                value_masked[batch_idx, :, mask_positions, :] = mean_value.expand(-1, len(mask_positions), -1)

            elif strategy == 'noise':
                # Replace with Gaussian noise
                key_shape = key_states[batch_idx, :, mask_positions, :].shape
                value_shape = value_states[batch_idx, :, mask_positions, :].shape

                key_noise = torch.randn(key_shape, device=key_states.device, dtype=key_states.dtype) * 0.01
                value_noise = torch.randn(value_shape, device=value_states.device, dtype=value_states.dtype) * 0.01

                key_masked[batch_idx, :, mask_positions, :] = key_noise
                value_masked[batch_idx, :, mask_positions, :] = value_noise

            else:
                raise ValueError(f"Unknown masking strategy: {strategy}. Choose from ['zero', 'mean', 'noise']")

        masked_kv.append((key_masked, value_masked))

    return tuple(masked_kv)


def tcvm_counterfactual_forward(
    model,
    input_ids: torch.LongTensor,
    masked_past_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    attention_mask: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run forward pass with masked KV cache (lightweight, only current token).

    This performs a single-step forward pass using the counterfactual KV cache
    where top-K attended visual tokens have been masked. This is efficient because
    the KV cache already contains all previous token representations.

    Args:
        model: LLaVA model instance
        input_ids: current input tokens [batch, seq_len]
                  (typically just last token: [batch, 1])
        masked_past_kv: masked past_key_values from mask_visual_kv_cache()
        attention_mask: attention mask [batch, seq_len] (optional)
        images: image tensor (not used in counterfactual, included for API consistency)

    Returns:
        next_token_logits: [batch, vocab_size] logits for next token

    Note:
        We only compute logits for the last token since KV cache handles the prefix.
        This is much faster than recomputing the entire sequence.
    """
    # Ensure we only process the last token
    if input_ids.shape[1] > 1:
        input_ids = input_ids[:, -1:]

    # Forward pass with masked KV cache
    # Set output_attentions=False to save memory (we don't need them for counterfactual)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            past_key_values=masked_past_kv,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            images=None,  # Visual tokens already in KV cache
        )

    # Extract logits for the last (and only) position
    next_token_logits = outputs.logits[:, -1, :]

    return next_token_logits


def compute_tcvm_contrastive_logits(
    logits_base: torch.Tensor,
    logits_masked: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.7,
    apply_apc: bool = True
) -> torch.Tensor:
    """
    Compute TCVM contrastive logits with Adaptive Plausibility Constraint (APC).

    Formula:
        P_final = P_base - alpha * P_masked

    Where tokens that don't drop in probability when visual evidence is removed
    are likely hallucinations (relying on LM priors, not visual grounding).

    Args:
        logits_base: [batch, vocab_size] logits from full visual context
        logits_masked: [batch, vocab_size] logits from masked visual context
        alpha: contrastive penalty weight (higher = stronger penalization)
        beta: plausibility threshold (typically 0.1-0.7)
              only tokens with P_base >= beta * max(P_base) are considered
        apply_apc: if True, apply Adaptive Plausibility Constraint

    Returns:
        contrastive_logits: [batch, vocab_size] final logits after TCVM

    Example:
        >>> base = torch.randn(1, 32000)
        >>> masked = torch.randn(1, 32000)
        >>> final = compute_tcvm_contrastive_logits(base, masked, alpha=1.0, beta=0.7)
        >>> assert final.shape == base.shape
    """
    if apply_apc:
        # Adaptive Plausibility Constraint: only modify plausible tokens
        # Tokens with low base probability are masked out
        cutoff = torch.log(torch.tensor(beta, device=logits_base.device)) + \
                 logits_base.max(dim=-1, keepdim=True).values

        # Compute contrastive difference
        contrastive_logits = logits_base - alpha * logits_masked

        # Mask implausible tokens (set to -inf)
        contrastive_logits = contrastive_logits.masked_fill(
            logits_base < cutoff,
            -float("inf")
        )
    else:
        # Simple contrastive decoding without plausibility constraint
        contrastive_logits = logits_base - alpha * logits_masked

    return contrastive_logits
