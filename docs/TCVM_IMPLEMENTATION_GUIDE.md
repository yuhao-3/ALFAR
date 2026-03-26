# TCVM Implementation Guide

## Overview

**TCVM (Token-Level Causal Visual Masking)** is a training-free, inference-time decoding strategy designed to mitigate object hallucination in Large Vision-Language Models (LVLMs).

Unlike global visual contrastive methods (e.g., VCD), TCVM applies **fine-grained, dynamic counterfactual masking** at the token level. It identifies the specific visual patches that the model attends to for each generated token, masks them, and uses contrastive decoding to penalize hallucinations.

---

## Core Mechanism

### Problem with VCD
- **Global noise**: VCD applies noise to the entire image
- **Coarse-grained**: Destroys background context and inter-object relationships
- **Not token-specific**: Cannot identify which visual patches caused hallucination

### TCVM Solution
- **Token-level masking**: Dynamically masks only the top-K attended visual patches for each token
- **Fine-grained**: Preserves visual context except for suspected hallucination sources
- **Causal inference**: Measures causal dependence by observing probability changes

---

## Algorithm Pipeline

For each decoding step $t$:

### 1. Base Forward Pass
- Input: Full image $V$ and prefix $x_{<t}$
- Output: Base logits $P_{base}(x_t | V, x_{<t})$
- Extract: Cross-attention map $A_t$ for the current token

### 2. Dynamic Masking Identification
- Extract top-$K$ visual patches with highest attention weights from $A_t$
- Typically $K = 20$ tokens (~3.5% of 576 patches)

### 3. Counterfactual Construction
- Clone the KV cache from the base forward pass
- Mask the top-$K$ visual token representations using one of:
  - **Zero-masking**: Set to 0
  - **Mean-masking**: Replace with average of all visual tokens
  - **Noise-masking**: Replace with Gaussian noise

### 4. Counterfactual Forward Pass
- Run forward pass with masked KV cache
- Only compute for current token (efficient via KV cache reuse)
- Output: Masked logits $P_{mask}(x_t | V_{mask}, x_{<t})$

### 5. Contrastive Decoding
- If $P_{base} - P_{mask} < \epsilon$: Token probability barely drops despite visual evidence removal → hallucination
- Adjust logits: $P_{final} = P_{base} - \alpha \cdot P_{mask}$
- Apply Adaptive Plausibility Constraint (APC) to prevent grammatical collapse

---

## Implementation Details

### File Structure

```
experiments/eval/
├── tcvm_utils.py          # Core TCVM utilities
├── vcd_sample.py          # Modified sampling loop with TCVM branch
└── test_tcvm.py           # Unit tests
```

### Core Functions (tcvm_utils.py)

#### 1. `get_topk_visual_indices()`
Extracts indices of top-K attended visual tokens.

**Input**:
- `visual_attn_weights`: [batch, num_visual_tokens] attention weights
- `img_start_idx`: Starting index of visual tokens (typically 35)
- `top_k`: Number of top tokens to extract (default: 20)

**Output**:
- `topk_indices`: [batch, top_k] absolute sequence indices

#### 2. `mask_visual_kv_cache()`
Clones and masks KV cache at specific positions.

**Input**:
- `past_key_values`: Tuple of (key, value) pairs for each layer
- `topk_indices`: [batch, top_k] indices to mask
- `strategy`: 'zero' | 'mean' | 'noise'
- `detach`: Whether to detach tensors (saves memory)

**Output**:
- `masked_past_kv`: Cloned and masked KV cache

**Strategies**:
- **Zero**: `key[indices] = 0`, `value[indices] = 0`
- **Mean**: `key[indices] = mean(all_keys)`, `value[indices] = mean(all_values)`
- **Noise**: `key[indices] = N(0, 0.01)`, `value[indices] = N(0, 0.01)`

#### 3. `tcvm_counterfactual_forward()`
Runs forward pass with masked KV cache (lightweight, only current token).

**Input**:
- `model`: LLaVA model instance
- `input_ids`: Current token [batch, 1]
- `masked_past_kv`: Masked KV cache
- `attention_mask`: Attention mask (optional)

**Output**:
- `next_token_logits`: [batch, vocab_size] logits from counterfactual

#### 4. `compute_tcvm_contrastive_logits()`
Computes contrastive logits with Adaptive Plausibility Constraint.

**Input**:
- `logits_base`: [batch, vocab_size] from full visual context
- `logits_masked`: [batch, vocab_size] from masked context
- `alpha`: Contrastive penalty weight (default: 1.0)
- `beta`: Plausibility threshold (default: 0.7)
- `apply_apc`: Whether to apply APC (default: True)

**Output**:
- `contrastive_logits`: [batch, vocab_size] final logits

**Formula**:
```
cutoff = log(beta) + max(P_base)
P_final = P_base - alpha * P_masked
P_final[P_base < cutoff] = -inf  # APC: mask implausible tokens
```

---

## Integration into Generation Loop

### Modified vcd_sample.py

The TCVM branch is added **parallel** to the existing VCD branch:

```python
# Line 133: Check for TCVM mode
use_tcvm = model_kwargs.get("use_tcvm", False)

if use_tcvm:
    # TCVM branch (lines 142-197)
    # 1. Extract visual attention
    visual_attn_weights = outputs.attentions[-1][:, :, -1, img_start_idx:img_end_idx]
    visual_attn_weights = visual_attn_weights.mean(dim=1)  # Average across heads

    # 2. Get top-K indices
    topk_indices = get_topk_visual_indices(visual_attn_weights, img_start_idx, top_k)

    # 3. Mask KV cache
    masked_past_kv = mask_visual_kv_cache(outputs.past_key_values, topk_indices, strategy)

    # 4. Counterfactual forward
    logits_masked = tcvm_counterfactual_forward(model, input_ids, masked_past_kv, ...)

    # 5. Contrastive decoding
    cd_logits = compute_tcvm_contrastive_logits(logits_base, logits_masked, alpha, beta)

    # 6. Sample
    next_tokens = sample_from_logits(cd_logits)

elif use_cd:
    # VCD branch (existing)
    ...

else:
    # Standard generation (existing)
    ...
```

### Backward Compatibility

- **VCD mode**: Works unchanged (triggered by `images_cd != None`)
- **TCVM mode**: Triggered by `use_tcvm=True`
- **Standard mode**: Neither VCD nor TCVM active (default)

---

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_tcvm` | bool | False | Enable TCVM mode |
| `tcvm_topk` | int | 20 | Number of visual tokens to mask (~3.5% of 576) |
| `tcvm_alpha` | float | 1.0 | Contrastive penalty weight (higher = stronger) |
| `tcvm_beta` | float | 0.7 | Plausibility threshold for APC |
| `tcvm_mask_strategy` | str | 'zero' | Masking strategy: 'zero', 'mean', or 'noise' |
| `img_start_idx` | int | 35 | Starting index of visual tokens |
| `img_end_idx` | int | 611 | Ending index of visual tokens (576 patches) |

---

## Usage Example

### Running TCVM Evaluation

All evaluation scripts now support TCVM via CLI arguments. Here are examples for each dataset:

#### InfoSeek / ViQuAE (Multiple-Choice)

```bash
cd experiments/eval

# Standard VCD baseline (existing)
python alfar_mc_llava.py \
    --model-path /path/to/llava-1.5-7b \
    --image-folder /path/to/infoseek/images \
    --dataset infoseek \
    --answers-file ../result/infoseek_vcd.jsonl \
    --cd_beta 0.7 \
    --att_alpha 0.2

# TCVM with zero-masking
python alfar_mc_llava.py \
    --model-path /path/to/llava-1.5-7b \
    --image-folder /path/to/infoseek/images \
    --dataset infoseek \
    --answers-file ../result/infoseek_tcvm.jsonl \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7 \
    --tcvm_mask_strategy zero

# TCVM with mean-masking (ablation)
python alfar_mc_llava.py \
    --model-path /path/to/llava-1.5-7b \
    --image-folder /path/to/infoseek/images \
    --dataset infoseek \
    --answers-file ../result/infoseek_tcvm_mean.jsonl \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_mask_strategy mean
```

#### A-OKVQA / OK-VQA (Open-Ended)

```bash
cd experiments/eval

# TCVM on A-OKVQA
python alfar_okvqa_llava.py \
    --model-path /path/to/llava-1.5-7b \
    --image-folder /path/to/coco/val2017 \
    --dataset aokvqa \
    --answers-file ../result/aokvqa_tcvm.csv \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7

# TCVM on OK-VQA
python alfar_okvqa_llava.py \
    --model-path /path/to/llava-1.5-7b \
    --image-folder /path/to/coco/val2017 \
    --dataset okvqa \
    --answers-file ../result/okvqa_tcvm.csv \
    --use_tcvm \
    --tcvm_topk 30 \
    --tcvm_alpha 1.5
```

#### E-VQA (Evidence-Based)

```bash
cd experiments/eval

# TCVM on E-VQA
python alfar_evqa_llava.py \
    --model-path /path/to/llava-1.5-7b \
    --image-folder /path/to/inaturalist2021 \
    --answers-file ../result/evqa_tcvm.json \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7
```

### CLI Arguments Reference

All evaluation scripts now support the following TCVM arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_tcvm` | flag | False | Enable TCVM decoding mode |
| `--tcvm_topk` | int | 20 | Number of visual tokens to mask (~3.5% of 576) |
| `--tcvm_alpha` | float | 1.0 | Contrastive penalty weight (higher = stronger) |
| `--tcvm_beta` | float | 0.7 | Plausibility threshold for APC (0.1-0.9) |
| `--tcvm_mask_strategy` | str | 'zero' | Masking strategy: 'zero', 'mean', or 'noise' |

**Modified Files**:
- `experiments/eval/alfar_mc_llava.py` - Lines 131-135, 174-178
- `experiments/eval/alfar_okvqa_llava.py` - Lines 115-119, 165-169
- `experiments/eval/alfar_evqa_llava.py` - Lines 99-103, 139-143

---

## SLURM Job Submission

For large-scale evaluation on GPU clusters, use the provided SLURM scripts:

### Submit Individual Jobs

```bash
cd slurm_jobs

# Submit InfoSeek evaluation
sbatch run_infoseek_tcvm.slurm

# Submit ViQuAE evaluation
sbatch run_viquae_tcvm.slurm

# Submit A-OKVQA evaluation
sbatch run_aokvqa_tcvm.slurm

# Submit OK-VQA evaluation
sbatch run_okvqa_tcvm.slurm

# Submit E-VQA evaluation
sbatch run_evqa_tcvm.slurm
```

### Submit All Jobs at Once

```bash
cd slurm_jobs

# Submit all datasets
bash run_all_tcvm.sh

# Submit specific datasets only
bash run_all_tcvm.sh infoseek viquae
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View live output (replace JOB_ID with your job ID)
tail -f ../logs/tcvm_infoseek_JOB_ID.out

# Check completed job logs
ls -lht ../logs/tcvm_*.out
```

### SLURM Configuration

All TCVM jobs use:
- **Partition**: gpu-a100
- **GPUs**: 1x A100
- **Memory**: 64GB
- **CPUs**: 8 cores
- **Time limit**: 24 hours

**Output files**:
- Predictions: `experiments/result/{dataset}_tcvm_results.{jsonl|csv|json}`
- Logs: `logs/tcvm_{dataset}_{job_id}.{out|err}`

---

## Testing

Run unit tests to verify implementation:

```bash
cd experiments/eval
python test_tcvm.py
```

**Expected output**:
```
✓ Test 1: Top-K Visual Token Extraction - PASSED
✓ Test 2: KV Cache Masking (zero/mean/noise) - PASSED
✓ Test 3: Contrastive Logit Computation - PASSED
✓ ALL TESTS PASSED!
```

---

## Performance Considerations

### Memory Usage
- **KV Cache Cloning**: Creates a full copy of past_key_values
- **Mitigation**: Use `detach=True` to free gradients
- **Impact**: +50% memory during decoding (temporary)

### Inference Latency
- **Extra Forward Pass**: 1 additional forward pass per decoding step
- **Optimization**: Only computes last token (KV cache reuse)
- **Impact**: ~2x slower than standard generation, but faster than VCD (no prompt re-encoding)

### Ablation Recommendations

1. **Top-K values**: Test 10, 20, 30, 50 (3.5%, 5%, 9% of patches)
2. **Masking strategies**: Compare zero vs. mean vs. noise
3. **Alpha values**: Sweep 0.5, 1.0, 1.5, 2.0
4. **Beta values**: Test 0.5, 0.7, 0.9 (APC strictness)

---

## Differentiation from Related Methods

| Method | Level | Mechanism | TCVM Difference |
|--------|-------|-----------|-----------------|
| **VCD** | Image-level | Global image noise | TCVM masks **specific patches** dynamically |
| **OPERA** | Token-level | Attention penalty (beam search) | TCVM uses **contrastive decoding**, not beam search |
| **ICD** | Instruction-level | Weak vs. strong instructions | TCVM operates on **visual modality**, not text |

**Key Novelty**: TCVM is the first method to apply **token-specific, patch-level visual masking** with causal inference via contrastive decoding.

---

## Next Steps

1. ✅ **Integration**: TCVM arguments added to all evaluation scripts
2. **Evaluation**: Run POPE benchmark to measure hallucination reduction
   - Baseline: Standard LLaVA-1.5 generation
   - VCD: Existing visual contrastive decoding
   - TCVM: Token-level causal visual masking
3. **Ablation**: Systematic hyperparameter sweep
   - Top-K values: 10, 20, 30, 50 tokens
   - Masking strategies: zero, mean, noise
   - Alpha values: 0.5, 1.0, 1.5, 2.0
   - Beta values: 0.5, 0.7, 0.9
4. **Analysis**: Visualize which patches are masked for hallucinated vs. correct tokens
   - Extract attention maps during generation
   - Compare masked regions for correct vs. hallucinated predictions
   - Validate causal relationship between visual patches and token probabilities

---

## References

### Key Files
- **Implementation**: `experiments/eval/tcvm_utils.py`
- **Integration**: `experiments/eval/vcd_sample.py` (lines 133-197)
- **Testing**: `experiments/eval/test_tcvm.py`

### Relevant Code Sections
- **Attention extraction**: `vcd_sample.py:155-158`
- **KV masking**: `tcvm_utils.py:52-106`
- **Contrastive decoding**: `tcvm_utils.py:140-178`

---

## Author Notes

**Implementation Status**: ✅ Phase 1-4 Complete

- [x] Core utilities implemented (`tcvm_utils.py`)
- [x] Sampling loop modified (`vcd_sample.py`)
- [x] Unit tests passed (`test_tcvm.py`)
- [x] CLI arguments added to all evaluation scripts
- [x] SLURM job scripts created for all datasets
- [x] Documentation complete with usage examples
- [ ] Run evaluation on InfoSeek, ViQuAE, A-OKVQA, OK-VQA, E-VQA (Next: Phase 5)
- [ ] Hyperparameter ablation and analysis (Next: Phase 6)

**Ready for**: Large-scale GPU evaluation on knowledge-grounded VQA benchmarks.
