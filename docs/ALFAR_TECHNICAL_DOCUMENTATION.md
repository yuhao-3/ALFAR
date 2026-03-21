# ALFAR Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Adaptive Attention Allocation](#adaptive-attention-allocation)
4. [Adaptive Logit Fusion](#adaptive-logit-fusion)
5. [Evaluation Scripts](#evaluation-scripts)
6. [Key Parameters](#key-parameters)
7. [Code Flow](#code-flow)

---

## Overview

ALFAR (Adaptive Logit Fusion and Attention Re-weighting) is a framework for knowledge-grounded Visual Question Answering (VQA). It enhances Vision-Language Models (VLMs) by:

1. **Adaptive Attention Allocation**: Dynamically adjusts attention weights to balance image features and retrieved textual context
2. **Adaptive Logit Fusion**: Uses contrastive decoding to combine predictions from context-aware and context-free branches

The system is designed to work with LLaVA models and supports multiple VQA benchmarks.

---

## System Architecture

### Core Components

```
ALFAR/
├── experiments/
│   ├── eval/
│   │   ├── alfar_mc_llava.py          # Multiple-choice VQA (InfoSeek, ViQuAE)
│   │   ├── alfar_okvqa_llava.py       # Open-ended VQA (A-OKVQA, OK-VQA)
│   │   ├── alfar_evqa_llava.py        # Evidence-based VQA (E-VQA)
│   │   ├── vcd_sample.py              # Contrastive decoding sampling
│   │   └── attention.py               # Adaptive attention mechanism
│   └── llava/
│       └── model/
│           └── language_model/
│               └── llava_llama.py     # Modified LLaVA model
├── evaluation/
│   ├── eval_mc.py                     # Multiple-choice evaluation
│   ├── eval_okvqa.py                  # OK-VQA evaluation
│   └── eval_evqa.py                   # E-VQA evaluation
└── data/
    ├── eval_data/                     # Question datasets
    ├── retrieval_result/              # Retrieved knowledge indices
    └── wiki/                          # Wikipedia knowledge base
```

---

## Adaptive Attention Allocation

### Location
**File**: `experiments/eval/attention.py`

### Purpose
Dynamically re-weights attention to balance:
- Visual information from the image
- Textual context from retrieved knowledge

### Mechanism

The adaptive attention operates at the **attention weight** level before softmax normalization:

#### 1. Image Attention Suppression
```python
# Line 96-99 in attention.py
attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
    - attn_weights[:, :, -1, img_start_idx:img_end_idx] * self.ret_sim * self.alpha
    + attn_weights[:, :, -1, img_start_idx:img_end_idx]
)
```

**Formula**:
```
A_img' = A_img - (ret_sim × α × A_img)
       = A_img × (1 - ret_sim × α)
```

Where:
- `A_img`: Original attention weights for image tokens
- `ret_sim`: Retrieval similarity score (0-1, how relevant the retrieved context is)
- `α` (alpha): Attention allocation parameter (`att_alpha`)
- Higher `ret_sim` → More suppression of image attention
- Higher `α` → More aggressive re-weighting

#### 2. Context Attention Boosting
```python
# Line 101-105 in attention.py
# Calculate weight based on context-to-question attention
weight = softmax(A[context, question].sum(question_dim))

# Boost context attention
attn_weights[:, :, -1, context_start_idx:context_end_idx] = (
    attn_weights[:, :, -1, context_start_idx:context_end_idx] * self.alpha * self.weight
    + attn_weights[:, :, -1, context_start_idx:context_end_idx]
)
```

**Formula**:
```
weight = softmax(Σ_q A[context, question])
A_ctx' = A_ctx + (α × weight × A_ctx)
       = A_ctx × (1 + α × weight)
```

Where:
- `A_ctx`: Original attention weights for context tokens
- `weight`: Importance of each context token based on its attention to the question
- More relevant context tokens get larger boosts

#### 3. Token Position Indices

The mechanism operates on specific token ranges:
- **Image tokens**: `[img_start_idx : img_end_idx]` (typically 35-611 for 576 image patches)
- **Question tokens**: `[img_end_idx : img_end_idx + question_len]`
- **Prompt tokens**: Next `prompt_len` tokens (e.g., "Answer the question using...")
- **Context tokens**: `[context_start : context_start + context_len]`

### Integration

The attention modification is applied to specific layers via monkey patching:

```python
# experiments/eval/attention.py lines 131-144
def llama_modify(model, start_layer, end_layer, use_attn, alpha, use_cfg,
                 img_start_idx, img_end_idx, question_len, prompt_len, context_len, ret_sim):
    modify_layers = list(range(start_layer, end_layer))
    for i in modify_layers:
        model.layers[i].self_attn.use_attn = use_attn
        model.layers[i].self_attn.alpha = alpha
        # ... set other parameters
        model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.layers[i].self_attn)
```

### Key Insights

1. **Retrieval-Aware**: The suppression strength adapts to retrieval quality (`ret_sim`)
2. **Token-Selective**: Only modifies attention for the last predicted token (autoregressive generation)
3. **Question-Guided**: Context boost is weighted by relevance to the question
4. **Layer-Specific**: Applied only to specified transformer layers

---

## Adaptive Logit Fusion

### Location
**File**: `experiments/eval/vcd_sample.py`

### Purpose
Combines predictions from two forward passes using contrastive decoding:
1. **Context-aware branch**: Model with retrieved context + image
2. **Context-free branch**: Model with only image (no textual context)

This amplifies context-dependent predictions while suppressing context-independent ones.

### Mechanism

#### 1. Dual Forward Passes
```python
# vcd_sample.py lines 111-142
# Forward pass WITH context
outputs = self(**model_inputs, ...)
next_token_logits = outputs.logits[:, -1, :]

# Forward pass WITHOUT context (contrastive)
outputs_cd = self(**model_inputs_cd, ...)
next_token_logits_cd = outputs_cd.logits[:, -1, :]
```

#### 2. Dynamic Fusion Weight Calculation
```python
# vcd_sample.py lines 146-152
cd_beta = model_kwargs.get("cd_beta")  # Base fusion strength
context_start_idx = img_end_idx + question_len + prompt_len

# Extract attention weights from last layer, last head, last token
image_attn = outputs.attentions[-1][-1, :, -1, img_start_idx:img_end_idx].mean(dim=0).sum()
context_attn = outputs.attentions[-1][-1, :, -1, context_start_idx:context_start_idx+context_len].mean(dim=0).sum()

# Calculate adaptive fusion coefficient
cd_alpha = image_attn / context_attn
```

**Interpretation**:
- `cd_alpha = image_attn / context_attn`
- If `image_attn > context_attn` → `cd_alpha > 1` → Model relies more on image
- If `context_attn > image_attn` → `cd_alpha < 1` → Model relies more on context

#### 3. Contrastive Logit Combination
```python
# vcd_sample.py line 154
diffs = (1 + 1/cd_alpha) * next_token_logits - (1 - cd_alpha) * next_token_logits_cd
```

**Formula**:
```
L_contrast = (1 + 1/cd_alpha) × L_ctx - (1 - cd_alpha) × L_no_ctx

When cd_alpha >> 1 (image-focused):
  L_contrast ≈ L_ctx - (1 - cd_alpha) × L_no_ctx
  (Less contrastive emphasis)

When cd_alpha << 1 (context-focused):
  L_contrast ≈ (1 + 1/cd_alpha) × L_ctx - 2 × L_no_ctx
  (Strong contrastive emphasis)
```

Where:
- `L_ctx`: Logits from context-aware forward pass
- `L_no_ctx`: Logits from context-free forward pass
- `cd_alpha`: Attention-based fusion coefficient

#### 4. Plausibility Filtering
```python
# vcd_sample.py lines 156-157
cutoff = log(cd_beta) + max(next_token_logits)
cd_logits = diffs.masked_fill(next_token_logits < cutoff, -inf)
```

**Purpose**: Filter out implausible tokens
- `cd_beta` (typically 0.7): Plausibility threshold
- Only keeps tokens where `L_ctx >= log(cd_beta) + max(L_ctx)`
- Prevents low-confidence tokens from being selected

#### 5. Sampling
```python
# vcd_sample.py lines 160-163
cd_logits = logits_processor(input_ids, cd_logits)  # Apply temperature
cd_logits = logits_warper(input_ids, cd_logits)     # Apply top-p/top-k
cd_probs = nn.functional.softmax(cd_logits, dim=-1)
next_tokens = torch.multinomial(cd_probs, num_samples=1)
```

### Key Insights

1. **Attention-Guided Fusion**: Uses attention distribution to determine fusion weights dynamically
2. **Contrastive Amplification**: Tokens that benefit from context get higher scores
3. **Context-Independence Detection**: Suppresses tokens that would be predicted even without context
4. **Plausibility Constraint**: Ensures generated tokens remain reasonable

---

## Evaluation Scripts

### 1. Multiple-Choice VQA (`alfar_mc_llava.py`)

**Datasets**: InfoSeek, ViQuAE

**Key Features**:
- Retrieves top-1 Wikipedia context using pre-computed indices
- Uses retrieval similarity score (`ret_sim`) for adaptive attention
- Single-token generation (answer is A/B/C/D)

**Prompt Structure**:
```
Context-aware:
  <image> + Question + "Answer based on context. Context: [wiki_summary] Options: A:... B:... C:... D:..."

Context-free:
  <image> + Question + "Answer based on your knowledge. Options: A:... B:... C:... D:..."
```

**Key Parameters**:
```python
model.generate(
    input_ids,
    images=raw_image_tensor,
    images_cd=input_ids1,       # Context-free prompt
    cd_beta=0.7,                # Logit fusion threshold
    att_alpha=0.2,              # Attention allocation (InfoSeek/ViQuAE)
    ret_sim=retrieval_sim[i],   # Retrieval quality score
    max_new_tokens=1            # Single answer token
)
```

### 2. Open-Ended VQA (`alfar_okvqa_llava.py`)

**Datasets**: A-OKVQA, OK-VQA

**Key Features**:
- Uses dense captions as context from pre-generated knowledge file
- Fixed `ret_sim=0.9` (assumes high-quality captions)
- Multi-token generation (up to 10 tokens)

**Prompt Structure**:
```
Context-aware:
  <image> + Question + "Answer using a single word or phrase based on the given context. Context: [caption]"

Context-free:
  <image> + Question + "Answer using a single word or phrase based on your knowledge."
```

**Key Parameters**:
```python
model.generate(
    ...
    cd_beta=0.7,
    att_alpha=0.4,              # Higher than MC tasks
    ret_sim=0.9,                # Fixed high quality
    max_new_tokens=10
)
```

### 3. Evidence-Based VQA (`alfar_evqa_llava.py`)

**Dataset**: E-VQA (iNaturalist subset)

**Key Features**:
- Uses evidence field from dataset as context
- Fixed `ret_sim=1.0` (gold evidence)
- Filters for templated questions only

**Key Parameters**:
```python
model.generate(
    ...
    cd_beta=0.7,
    att_alpha=0.1,              # Lower (evidence is very reliable)
    ret_sim=1.0,                # Perfect evidence
    max_new_tokens=10
)
```

---

## Key Parameters

### `cd_beta` (Logit Fusion Threshold)
- **Type**: Float (0-1)
- **Default**: 0.7
- **Purpose**: Controls plausibility filtering in contrastive decoding
- **Effect**:
  - Lower → More conservative (only high-confidence context-dependent tokens)
  - Higher → More permissive (allows more tokens through)

### `att_alpha` (Attention Allocation Strength)
- **Type**: Float (0-1)
- **Dataset-Specific Values**:
  - **E-VQA**: 0.1 (gold evidence, minimal adjustment needed)
  - **InfoSeek/ViQuAE**: 0.2 (balanced image-text tasks)
  - **A-OKVQA/OK-VQA**: 0.4 (requires more context emphasis)
- **Purpose**: Controls how aggressively to re-weight attention
- **Effect**:
  - Lower → Smaller attention shifts
  - Higher → Larger attention reallocation from image to context

### `ret_sim` (Retrieval Similarity)
- **Type**: Float (0-1)
- **Values**:
  - **InfoSeek/ViQuAE**: Retrieved from `retrieval_sim` array (variable quality)
  - **A-OKVQA/OK-VQA**: 0.9 (high-quality dense captions)
  - **E-VQA**: 1.0 (gold evidence)
- **Purpose**: Indicates quality/relevance of retrieved context
- **Effect**: Modulates attention suppression strength
  - Higher → More image attention suppression
  - Lower → Less modification

### Image Token Indices
- **`img_start_idx`**: 35 (start of image patch tokens)
- **`img_end_idx`**: 611 (end of image patch tokens, 576 patches total)
- **Note**: Depends on LLaVA prompt template and image encoder configuration

---

## Code Flow

### Complete Inference Pipeline

```
1. DATA LOADING
   ├─ Load questions from JSON/CSV
   ├─ Load retrieval indices (InfoSeek/ViQuAE only)
   ├─ Load knowledge base (Wikipedia/captions/evidence)
   └─ Load LLaVA model + image processor

2. PER-SAMPLE PROCESSING
   ├─ Prepare dual prompts:
   │  ├─ Context-aware: <img> + Q + "...based on context: [knowledge]"
   │  └─ Context-free: <img> + Q + "...based on your knowledge"
   ├─ Tokenize prompts
   ├─ Compute token lengths (question_len, prompt_len, context_len)
   └─ Load and preprocess image

3. ADAPTIVE ATTENTION SETUP (via llama_modify)
   ├─ Inject attention modification into model layers
   ├─ Set parameters: alpha, ret_sim, token indices
   └─ Monkey-patch forward methods

4. GENERATION (via vcd_sample.py)
   ├─ For each token:
   │  ├─ Forward pass #1: Context-aware
   │  │  ├─ Apply adaptive attention (attention.py)
   │  │  │  ├─ Suppress image attention: A_img × (1 - ret_sim × α)
   │  │  │  └─ Boost context attention: A_ctx × (1 + α × weight)
   │  │  └─ Get logits L_ctx
   │  │
   │  ├─ Forward pass #2: Context-free
   │  │  └─ Get logits L_no_ctx
   │  │
   │  ├─ Extract attention weights from last layer
   │  ├─ Compute cd_alpha = image_attn / context_attn
   │  │
   │  ├─ Adaptive logit fusion:
   │  │  ├─ diffs = (1 + 1/cd_alpha) × L_ctx - (1 - cd_alpha) × L_no_ctx
   │  │  └─ Filter: keep only if L_ctx >= log(cd_beta) + max(L_ctx)
   │  │
   │  ├─ Apply temperature + top-p sampling
   │  └─ Sample next token
   │
   └─ Repeat until max_new_tokens or EOS

5. POST-PROCESSING
   ├─ Decode generated tokens
   ├─ Save predictions to file
   └─ Run evaluation script
```

### Key Function Calls

```python
# Main evaluation script (e.g., alfar_mc_llava.py)
model.generate(
    input_ids,                    # Context-aware prompt
    images=image_tensor,
    images_cd=input_ids_no_ctx,   # Triggers contrastive decoding
    cd_beta=0.7,
    att_alpha=0.2,
    ret_sim=0.85,
    img_start_idx=35,
    img_end_idx=611,
    question_len=15,
    prompt_len=20,
    context_len=100,
    ...
)
↓
# LlamaForCausalLM.generate() → sample()
# vcd_sample.py: evolve_vcd_sampling() patches GenerationMixin.sample
↓
# For each generation step:
model.forward()  # Context-aware
  ↓
  # llava_llama.py: forward()
  ↓
  # LlamaModel layers with modified attention
  # attention.py: llama_new_forward()
  #   - Modify attn_weights for image/context tokens
  #   - Apply softmax → matmul with values

model.forward()  # Context-free (cd branch)

# Combine logits with adaptive fusion
# Sample from fused distribution
```

---

## Implementation Details

### Modified Components

1. **`vcd_sample.py`**:
   - Patches `transformers.generation.utils.GenerationMixin.sample`
   - Adds contrastive decoding logic

2. **`attention.py`**:
   - Replaces forward pass of specified LlamaAttention layers
   - Modifies attention weights before softmax

3. **`llava_llama.py`**:
   - Accepts additional parameters: `images_cd`, `cd_beta`, `att_alpha`, etc.
   - Passes through to generation

### Parameter Passing

```python
# Parameters flow:
eval_model()  # alfar_*_llava.py
  ↓
model.generate(**params)
  ↓
sample() # vcd_sample.py
  ↓
model.forward(**model_kwargs)
  ↓
LlamaModel.forward()
  ↓
LlamaAttention.forward() # Modified by attention.py
```

---

## Dataset-Specific Configurations

| Dataset   | Type | att_alpha | ret_sim | max_tokens | Context Source     |
|-----------|------|-----------|---------|------------|--------------------|
| InfoSeek  | MC   | 0.2       | Variable| 1          | Wikipedia (top-1)  |
| ViQuAE    | MC   | 0.2       | Variable| 1          | Wikipedia (top-1)  |
| A-OKVQA   | Open | 0.4       | 0.9     | 10         | Dense captions     |
| OK-VQA    | Open | 0.4       | 0.9     | 10         | Dense captions     |
| E-VQA     | Open | 0.1       | 1.0     | 10         | Gold evidence      |

**Rationale**:
- E-VQA: Low α (0.1) because evidence is gold-standard
- InfoSeek/ViQuAE: Medium α (0.2) for balanced image-text retrieval
- OK-VQA family: High α (0.4) because questions heavily rely on external knowledge

---

## References

### Key Files
- **Evaluation**: `experiments/eval/alfar_{mc,okvqa,evqa}_llava.py`
- **Attention**: `experiments/eval/attention.py`
- **Logit Fusion**: `experiments/eval/vcd_sample.py`
- **Model**: `experiments/llava/model/language_model/llava_llama.py`

### Data Requirements
- **Questions**: `data/eval_data/{mc,okvqa,evqa}/`
- **Retrieval**: `data/retrieval_result/` (indices and similarity scores)
- **Knowledge**: `data/wiki/wiki_with_image.npy` + mapping files
- **Images**: Dataset-specific directories

---

## Summary

ALFAR achieves adaptive knowledge integration through:

1. **Adaptive Attention Allocation**:
   - Dynamically suppresses image attention based on retrieval quality
   - Boosts relevant context tokens based on question attention
   - Applied at attention weight level in transformer layers

2. **Adaptive Logit Fusion**:
   - Uses contrastive decoding with context-aware and context-free branches
   - Dynamically adjusts fusion weights based on attention distribution
   - Filters implausible predictions with `cd_beta` threshold

3. **Dataset-Specific Tuning**:
   - Varies `att_alpha` based on context reliability
   - Adapts `ret_sim` to retrieval quality
   - Optimized for different question types (MC vs open-ended)

This combination allows the model to automatically balance visual and textual information sources for accurate knowledge-grounded visual question answering.
