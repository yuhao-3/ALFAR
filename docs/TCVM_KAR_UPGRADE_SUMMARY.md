# TCVM-KAR (Knowledge-Aware Router) Upgrade Summary

## Overview
Successfully upgraded TCVM (Token-Level Causal Visual Masking) to **TCVM-KAR** (Knowledge-Aware Router) to support Multimodal RAG settings. The upgrade enables dynamic routing between visual and context token masking based on attention patterns.

## Problem Statement
The original TCVM implementation only masked visual patches, which incorrectly penalized tokens generated faithfully from retrieved textual context. In multimodal RAG scenarios where both visual and textual evidence are present, this approach was suboptimal.

## Solution: Knowledge-Aware Router (KAR)
The KAR dynamically chooses whether to mask **visual tokens** OR **context tokens** based on the current token's attention distribution, using a routing metric λ_t.

### Routing Metric
```
λ_t = Σ(visual_attention) / (Σ(visual_attention) + Σ(context_attention))

Decision:
- if λ_t > 0.5: Mask visual tokens (vision-dominant)
- if λ_t ≤ 0.5: Mask context tokens (context-dominant)
```

## Implementation Details

### Task 1: Context Masking Utilities (tcvm_utils.py)
Added two new functions mirroring the visual masking utilities:

1. **`get_topk_context_indices(context_attn_weights, context_start_idx, top_k)`**
   - Extracts indices of top-K attended context tokens
   - Location: `experiments/eval/tcvm_utils.py:139-179`

2. **`mask_context_kv_cache(past_key_values, topk_indices, strategy, detach)`**
   - Masks KV cache at specific context token positions
   - Supports same strategies as visual masking: 'zero', 'mean', 'noise'
   - Location: `experiments/eval/tcvm_utils.py:182-261`

### Task 2: KAR Router Logic (vcd_sample.py)
Upgraded the TCVM block with intelligent routing:

1. **Dual Attention Extraction** (lines 157-168)
   ```python
   visual_attn_weights = outputs.attentions[-1][:, :, -1, img_start_idx:img_end_idx].mean(dim=1)
   context_attn_weights = outputs.attentions[-1][:, :, -1, context_start_idx:context_end_idx].mean(dim=1)
   ```

2. **Lambda_t Computation** (lines 170-173)
   ```python
   attn_vis_sum = visual_attn_weights.sum(dim=-1)
   attn_ctx_sum = context_attn_weights.sum(dim=-1)
   lambda_t = attn_vis_sum / (attn_vis_sum + attn_ctx_sum + 1e-9)
   ```

3. **Dynamic Branching** (lines 175-247)
   - Determines dominant modality using `is_visual_dominant = lambda_t > 0.5`
   - Routes to appropriate masking function
   - Handles both single-sample and batched inputs

4. **Debug Logging** (lines 179-183)
   - Optional per-step logging controlled by `tcvm_debug` parameter
   - Format: `[KAR Router] Step {step}: lambda_t={value:.4f} -> Masking {Vision|Context}`

5. **Automatic Context Index Calculation** (lines 155-173)
   - Auto-computes context indices from `question_len`, `prompt_len`, `context_len`
   - Fallback to visual-only TCVM if context indices unavailable
   - Location: `experiments/eval/vcd_sample.py:160-173`

### Task 3: Counterfactual Forward Pass
The existing `tcvm_counterfactual_forward()` function works seamlessly with both visual and context masking. The mathematical formulation:

```
P_final = P_base - α · P_masked
```

applies beautifully to both cases, requiring **no code changes**.

### Task 4: Kwargs Propagation
Context indices are automatically propagated through the generation pipeline:

1. **Evaluation Script** → passes `question_len`, `prompt_len`, `context_len`
2. **LLaVA Forward** (llava_llama.py:58-84) → includes parameters in signature
3. **Generation Pipeline** → adds to `model_kwargs`
4. **Sample Function** → extracts via `model_kwargs.get()`

All parameters are already in place in `llava_llama.py:73-82`.

## Testing
All unit tests pass successfully (experiments/eval/test_tcvm.py):

```
✓ Test 1: Top-K Visual Token Extraction
✓ Test 1b: Top-K Context Token Extraction
✓ Test 2: KV Cache Masking (visual)
✓ Test 2b: Context KV Cache Masking
✓ Test 3: Contrastive Logit Computation
```

## Usage

### Enable TCVM-KAR in Evaluation Scripts
The existing ALFAR evaluation scripts already pass the necessary parameters:

```python
model.generate(
    ...,
    use_tcvm=True,              # Enable TCVM-KAR
    tcvm_topk=20,               # Number of tokens to mask
    tcvm_alpha=1.0,             # Contrastive penalty weight
    tcvm_beta=0.7,              # Plausibility threshold
    tcvm_mask_strategy='zero',  # Masking strategy
    tcvm_debug=False,           # Enable debug logging
    question_len=question_len,  # Auto-calculated
    prompt_len=prompt_len,      # Auto-calculated
    context_len=context_len,    # Auto-calculated
    img_start_idx=35,
    img_end_idx=611
)
```

### Enable Debug Mode
To see routing decisions:
```python
use_tcvm=True,
tcvm_debug=True
```

Output:
```
[KAR Router] Auto-calculated context indices: start=650, end=750
[KAR Router] Step 0: lambda_t=0.6234 -> Masking Vision
[KAR Router] Step 1: lambda_t=0.3891 -> Masking Context
...
```

## Key Features

1. **Backward Compatible**: Falls back to visual-only TCVM if context indices unavailable
2. **Automatic Index Calculation**: No manual calculation needed in evaluation scripts
3. **Flexible Batching**: Handles both single and batched inputs with majority voting
4. **Debug Support**: Optional per-step logging for analysis
5. **Unified Contrastive Logic**: Same mathematical formulation for both modalities

## Files Modified

1. `experiments/eval/tcvm_utils.py` - Added context masking utilities
2. `experiments/eval/vcd_sample.py` - Implemented KAR router logic
3. `experiments/eval/test_tcvm.py` - Extended tests for context masking

## Next Steps

1. **Run full evaluation** on multimodal RAG benchmarks (A-OKVQA, ViQuAE, etc.)
2. **Compare performance** between:
   - Original TCVM (visual-only)
   - TCVM-KAR (adaptive routing)
   - ALFAR baseline
3. **Analyze routing patterns**: Track λ_t distributions to understand when each modality dominates
4. **Hyperparameter tuning**: Experiment with routing threshold (currently 0.5)

## Mathematical Foundation

The KAR preserves the core TCVM intuition: tokens that remain high-probability when critical evidence is removed are likely hallucinations. By adaptively selecting which evidence to mask (visual vs. context), the router maximizes the discriminative power of the contrastive objective.

**Intuition**:
- If the model is heavily attending to visual tokens → mask visual evidence
- If the model is heavily attending to context tokens → mask context evidence
- This ensures we're always testing the most relevant evidence source for each token

## Citation
If you use TCVM-KAR, please cite both the original TCVM work and acknowledge the KAR extension for multimodal RAG settings.
