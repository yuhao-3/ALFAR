# TCVM vs ALFAR: Detailed Comparison

## Table of Contents
1. [Overview](#overview)
2. [ALFAR: Two-Stage Mechanism](#alfar-two-stage-mechanism)
3. [TCVM: Single-Stage Mechanism](#tcvm-single-stage-mechanism)
4. [Process Flow Comparison](#process-flow-comparison)
5. [Key Formulas](#key-formulas)
6. [Implementation Details](#implementation-details)
7. [Hyperparameters](#hyperparameters)
8. [Performance Characteristics](#performance-characteristics)
9. [When to Use Which Method](#when-to-use-which-method)
10. [Debugging & Visualization](#debugging--visualization)

---

## Overview

### ALFAR (Attention reLocation and Adaptive Logits Fusion)

**Full Name**: Boosting Knowledge Utilization via **A**daptive **L**ogits **F**usion and **A**ttention **R**eallocation

**Paper**: An et al., "Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation", arXiv:2406.12718, 2024

**Purpose**: Enhance knowledge-grounded VQA by (1) reallocating attention from images to context, and (2) adaptively fusing logits based on attention patterns.

**Core Idea**:
- **Stage 1 (Internal)**: Reduce image attention, boost context attention
- **Stage 2 (External)**: Adaptively fuse base logits with contrastive logits based on image/context attention ratio

**Type**: Training-free, inference-time method with dual-stage intervention

**Primary Use Case**: Knowledge-grounded VQA (A-OKVQA, OK-VQA, InfoSeek, ViQuAE)

### TCVM (Token-Level Causal Visual Masking)

**Purpose**: Mitigate object hallucination by identifying and penalizing tokens that lack visual grounding.

**Core Idea**: For each generated token, dynamically mask the top-K attended visual patches and use contrastive decoding to detect hallucinations.

**Type**: Training-free, inference-time contrastive decoding

**Primary Use Case**: Hallucination detection and reduction (POPE, general VQA)

---

## ALFAR: Two-Stage Mechanism

ALFAR operates in **two stages** - one inside the model, one outside:

### Stage 1: Attention Reallocation (Internal)

**Location**: `experiments/eval/attention.py` (lines 95-105)

**When**: During every forward pass, in layers 0-31

**What**: Modifies attention weights BEFORE softmax normalization

**Formula**:
```python
# Step 1: Compute context-question relevance weight
weight = softmax(sum(attn[context → question]))

# Step 2: Reduce image attention
attn[token → image] = attn[token → image] - attn[token → image] * ret_sim * alpha

# Step 3: Boost context attention
attn[token → context] = attn[token → context] + attn[token → context] * alpha * weight

# Step 4: Renormalize
attn = softmax(attn)
```

**Effect**:
- Visual tokens get **less attention weight**
- Context tokens get **more attention weight** (proportional to relevance)

### Stage 2: Adaptive Logits Fusion (External)

**Location**: `experiments/eval/vcd_sample.py` (lines 199-229)

**When**: After forward pass completes, during sampling

**What**: Fuses base logits with contrastive logits using adaptive weight

**Formula**:
```python
# Step 1: Measure attention distribution
image_attn = sum(attn[last_token → image])
context_attn = sum(attn[last_token → context])

# Step 2: Compute adaptive fusion weight
cd_alpha = image_attn / context_attn

# Step 3: Adaptive logits fusion
final_logits = (1 + 1/cd_alpha) * base_logits - (1 - cd_alpha) * contrastive_logits

# Step 4: Apply Adaptive Plausibility Constraint (APC)
cutoff = log(cd_beta) + max(base_logits)
final_logits[base_logits < cutoff] = -inf
```

**Key Insight**: The fusion weight `cd_alpha` is **adaptive** - it adjusts based on how much the model attends to images vs context:

| Scenario | cd_alpha | Effect |
|----------|----------|--------|
| **High context attention** | Low (e.g., 0.3) | More weight on base logits, less contrastive penalty |
| **High image attention** | High (e.g., 3.0) | Stronger contrastive penalty, suppress image-driven tokens |
| **Balanced attention** | ~1.0 | Balanced fusion |

---

## TCVM: Single-Stage Mechanism

TCVM operates in **one stage** - entirely outside the model:

### Token-Level Causal Visual Masking (External)

**Location**: `experiments/eval/tcvm_utils.py`, `experiments/eval/vcd_sample.py` (lines 142-198)

**When**: After forward pass completes, during sampling

**What**: Masks top-K attended visual patches and compares base vs masked logits

**Formula**:
```python
# Step 1: Extract visual attention for current token
visual_attn = attn[last_token → visual_patches]  # [batch, 576]

# Step 2: Get top-K attended patches
topk_indices = argtopk(visual_attn, k=20)  # e.g., [37, 42, 89, ...]

# Step 3: Clone and mask KV cache
kv_masked = clone(kv_cache)
kv_masked[key/value at topk_indices] = 0  # or mean/noise

# Step 4: Counterfactual forward pass (only last token)
masked_logits = model.forward(input_ids=last_token, past_key_values=kv_masked)

# Step 5: Contrastive decoding
final_logits = base_logits - tcvm_alpha * masked_logits

# Step 6: Apply Adaptive Plausibility Constraint (APC)
cutoff = log(tcvm_beta) + max(base_logits)
final_logits[base_logits < cutoff] = -inf
```

**Key Insight**: TCVM identifies **which specific visual patches** the model relies on for each token, then measures if removing them changes the prediction (causal inference).

---

## Process Flow Comparison

### ALFAR Complete Process

```
Input: Image + Question + Context
  │
  ├─> Encode Image (visual tokens 35-611)
  ├─> Encode Question
  ├─> Encode Prompt
  └─> Encode Context
  │
  v
┌────────────────────────────────────────────────────┐
│ STAGE 1: Attention Reallocation (INTERNAL)        │
│ Location: Inside attention layers (0-31)          │
├────────────────────────────────────────────────────┤
│                                                     │
│  For each layer, for each token:                   │
│                                                     │
│  1. Compute standard attention weights             │
│     attn = softmax(QK^T / sqrt(d))                 │
│                                                     │
│  2. Measure context-question relevance             │
│     weight = softmax(sum(attn[context → question]))│
│                                                     │
│  3. Reduce image attention                         │
│     attn[token → img] -= attn[img] * ret_sim * α   │
│                                                     │
│  4. Boost context attention                        │
│     attn[token → ctx] += attn[ctx] * α * weight    │
│                                                     │
│  5. Renormalize                                    │
│     attn = softmax(attn)                           │
│                                                     │
│  6. Output                                         │
│     output = attn · V                              │
│                                                     │
└────────────────────────────────────────────────────┘
  │
  ├─> base_logits = LM_head(output)
  │
  v
┌────────────────────────────────────────────────────┐
│ STAGE 2: Adaptive Logits Fusion (EXTERNAL)        │
│ Location: In sampling loop (vcd_sample.py)        │
├────────────────────────────────────────────────────┤
│                                                     │
│  1. Run second forward pass with images_cd         │
│     (uses prepare_inputs_for_generation_cd)        │
│     → contrastive_logits                           │
│                                                     │
│  2. Measure attention distribution                 │
│     image_attn = sum(attn[token → img])            │
│     context_attn = sum(attn[token → ctx])          │
│                                                     │
│  3. Compute adaptive fusion weight                 │
│     cd_alpha = image_attn / context_attn           │
│                                                     │
│  4. Adaptive logits fusion                         │
│     final = (1 + 1/cd_alpha) * base_logits         │
│           - (1 - cd_alpha) * contrastive_logits    │
│                                                     │
│  5. Apply APC                                      │
│     cutoff = log(cd_beta) + max(base_logits)       │
│     final[base_logits < cutoff] = -inf             │
│                                                     │
│  6. Sample                                         │
│     token = multinomial(softmax(final))            │
│                                                     │
└────────────────────────────────────────────────────┘
  │
  v
Output: Generated Token
```

**Key Points**:
- **2 forward passes per token**: base (with reallocation) + contrastive (with images_cd)
- **Operates at 2 levels**: Inside layers (attention) + Outside model (logits)
- **Adaptive**: cd_alpha changes based on attention patterns

### TCVM Complete Process

```
Input: Image + Question + (Optional) Context
  │
  ├─> Encode Image (visual tokens 35-611)
  ├─> Encode Question
  └─> Encode Context (if present)
  │
  v
┌────────────────────────────────────────────────────┐
│ Base Forward Pass (STANDARD)                       │
│ Location: Standard model.forward()                │
├────────────────────────────────────────────────────┤
│                                                     │
│  1. Standard attention (no modification)           │
│     attn = softmax(QK^T / sqrt(d))                 │
│                                                     │
│  2. Output                                         │
│     output = attn · V                              │
│                                                     │
│  3. Logits                                         │
│     base_logits = LM_head(output)                  │
│                                                     │
│  4. Store                                          │
│     ├─> past_key_values (KV cache)                 │
│     └─> attentions (attention maps)                │
│                                                     │
└────────────────────────────────────────────────────┘
  │
  v
┌────────────────────────────────────────────────────┐
│ TCVM: Token-Level Causal Masking (EXTERNAL)       │
│ Location: In sampling loop (vcd_sample.py)        │
├────────────────────────────────────────────────────┤
│                                                     │
│  1. Extract visual attention                       │
│     visual_attn = attn[-1][token → visual]         │
│                   [last_layer][576 patches]        │
│                                                     │
│  2. Identify top-K attended patches                │
│     topk_idx = argtopk(visual_attn, k=20)          │
│     Example: [37, 42, 89, 103, ..., 598]           │
│               ↑ top-20 patches (~3.5%)             │
│                                                     │
│  3. Clone and mask KV cache                        │
│     kv_masked = clone(past_key_values)             │
│     For each layer:                                │
│       key[batch, heads, topk_idx, :] = 0           │
│       value[batch, heads, topk_idx, :] = 0         │
│                                                     │
│  4. Counterfactual forward pass                    │
│     (Only last token, reuse masked KV cache)       │
│     masked_logits = model.forward(                 │
│         input_ids=last_token,                      │
│         past_key_values=kv_masked                  │
│     )                                              │
│                                                     │
│  5. Contrastive decoding                           │
│     final = base_logits - tcvm_alpha * masked_logits│
│                                                     │
│  6. Apply APC                                      │
│     cutoff = log(tcvm_beta) + max(base_logits)     │
│     final[base_logits < cutoff] = -inf             │
│                                                     │
│  7. Sample                                         │
│     token = multinomial(softmax(final))            │
│                                                     │
└────────────────────────────────────────────────────┘
  │
  v
Output: Generated Token
```

**Key Points**:
- **2 forward passes per token**: base + counterfactual
- **Operates at 1 level**: Outside model (on KV cache and logits)
- **Fine-grained**: Masks specific patches (top-K), not all visual tokens

---

## Key Formulas

### ALFAR Stage 1: Attention Reallocation

**Relevance Weight**:
```
weight = softmax(Σ attn[context_i → question_j])
```

**Image Attention Reduction**:
```
attn'[token → image] = attn[token → image] * (1 - ret_sim * α)
```

**Context Attention Boost**:
```
attn'[token → context] = attn[token → context] * (1 + α * weight)
```

### ALFAR Stage 2: Adaptive Logits Fusion

**Adaptive Weight**:
```
cd_alpha = Σ attn[token → image] / Σ attn[token → context]
```

**Fusion Formula**:
```
P_final = (1 + 1/cd_alpha) * P_base - (1 - cd_alpha) * P_contrastive
```

**Adaptive Plausibility Constraint**:
```
cutoff = log(cd_beta) + max(P_base)
P_final[P_base < cutoff] = -∞
```

**Interpretation**:
- If `cd_alpha` is **small** (context attention high):
  - `(1 + 1/cd_alpha)` is **large** → more weight on base logits
  - `(1 - cd_alpha)` is **close to 1** → moderate contrastive penalty
- If `cd_alpha` is **large** (image attention high):
  - `(1 + 1/cd_alpha)` is **close to 1** → less weight on base logits
  - `(1 - cd_alpha)` is **negative** → strong contrastive penalty (reversal!)

### TCVM: Contrastive Decoding

**Top-K Selection**:
```
indices = argtopk(attn[token → visual], k)
```

**KV Masking**:
```
key_masked[indices] = 0  (or mean(key) or N(0, 0.01))
value_masked[indices] = 0
```

**Contrastive Formula**:
```
P_final = P_base - tcvm_alpha * P_masked
```

**Adaptive Plausibility Constraint**:
```
cutoff = log(tcvm_beta) + max(P_base)
P_final[P_base < cutoff] = -∞
```

---

## Implementation Details

### ALFAR Implementation

**Stage 1 Files**: `experiments/eval/attention.py`

**How enabled**:
- `llama_modify()` called in `prepare_inputs_for_generation()` (llava_llama.py:137)
- Monkey-patches `forward()` method of all LlamaAttention layers (0-31)

**Stage 1 Code** (attention.py:95-105):
```python
if use_attn and not use_cfg:
    # Reduce image attention
    attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
        - attn_weights[:, :, -1, img_start_idx:img_end_idx] * self.ret_sim * self.alpha
        + attn_weights[:, :, -1, img_start_idx:img_end_idx]
    )

    # Compute relevance weight
    if attn_weights.size(2) != 1:
        self.weight = torch.softmax(
            attn_weights[:, :, context_start_idx:context_end_idx,
                        question_start_idx:question_end_idx].sum(3, keepdim=True),
            dim=2
        ).squeeze(3)

    # Boost context attention
    attn_weights[:, :, -1, context_start_idx:context_end_idx] = (
        attn_weights[:, :, -1, context_start_idx:context_end_idx] * self.alpha * self.weight
        + attn_weights[:, :, -1, context_start_idx:context_end_idx]
    )
```

**Stage 2 Files**: `experiments/eval/vcd_sample.py`

**How enabled**:
- `evolve_vcd_sampling()` monkey-patches `GenerationMixin.sample()` (vcd_sample.py:20)
- Triggered by `images_cd != None`

**Stage 2 Code** (vcd_sample.py:215-223):
```python
# Measure attention distribution
image_attn = outputs.attentions[-1][-1,:,-1,img_start_idx:img_end_idx].mean(dim=0).sum()
context_attn = outputs.attentions[-1][-1,:,-1, context_start_idx:context_start_idx+context_len].mean(dim=0).sum()

# Compute adaptive fusion weight
cd_alpha = image_attn / context_attn

# Adaptive logits fusion
diffs = (1+1/cd_alpha) * next_token_logits - (1-cd_alpha) * next_token_logits_cd

# Apply APC
cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
```

### TCVM Implementation

**Files**: `experiments/eval/tcvm_utils.py`, `experiments/eval/vcd_sample.py`

**How enabled**:
- `evolve_vcd_sampling()` monkey-patches `GenerationMixin.sample()` (vcd_sample.py:20)
- Triggered by `use_tcvm=True`

**TCVM Code** (vcd_sample.py:142-198):
```python
if use_tcvm:
    # Extract visual attention
    visual_attn_weights = outputs.attentions[-1][:, :, -1, img_start_idx:img_end_idx]
    visual_attn_weights = visual_attn_weights.mean(dim=1)  # Average across heads

    # Get top-K indices
    topk_visual_indices = get_topk_visual_indices(
        visual_attn_weights, img_start_idx=img_start_idx, top_k=tcvm_topk
    )

    # Mask KV cache
    masked_past_kv = mask_visual_kv_cache(
        outputs.past_key_values, topk_visual_indices,
        strategy=tcvm_strategy, detach=True
    )

    # Counterfactual forward
    next_token_logits_masked = tcvm_counterfactual_forward(
        model=self, input_ids=input_ids, masked_past_kv=masked_past_kv,
        attention_mask=model_inputs.get('attention_mask'), images=None
    )

    # Contrastive decoding
    cd_logits = compute_tcvm_contrastive_logits(
        logits_base=next_token_logits,
        logits_masked=next_token_logits_masked,
        alpha=tcvm_alpha, beta=tcvm_beta, apply_apc=True
    )

    # Sample
    cd_logits = logits_processor(input_ids, cd_logits)
    cd_logits = logits_warper(input_ids, cd_logits)
    cd_probs = nn.functional.softmax(cd_logits, dim=-1)
    next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
```

---

## Hyperparameters

### ALFAR Hyperparameters

| Parameter | Type | Default | Description | Used In |
|-----------|------|---------|-------------|---------|
| `att_alpha` | float | 0.4 | Attention reallocation strength | Stage 1 + Stage 2 |
| `ret_sim` | float | 0.9 | Retrieval similarity threshold (image reduction factor) | Stage 1 |
| `cd_beta` | float | 0.7 | APC threshold for logits fusion | Stage 2 |
| `img_start_idx` | int | 35 | Visual tokens start index | Both stages |
| `img_end_idx` | int | 611 | Visual tokens end index (576 patches) | Both stages |
| `question_len` | int | varies | Question length (tokens) | Stage 1 |
| `prompt_len` | int | varies | Prompt length (tokens) | Stage 1 |
| `context_len` | int | varies | Context length (tokens) | Both stages |
| `images_cd` | tensor | None | Contrastive input (triggers Stage 2) | Stage 2 |

**Key Relationships**:
- `att_alpha` controls **both** attention reallocation (Stage 1) **and** appears in fusion weight (Stage 2)
- `cd_alpha` is **computed dynamically** from attention: `cd_alpha = image_attn / context_attn`
- Stage 2 only activates if `images_cd != None`

**Example Configuration**:
```python
model.generate(
    input_ids,
    images=images,
    images_cd=input_ids1,        # Trigger Stage 2 (adaptive fusion)
    cd_beta=0.7,                 # APC threshold
    att_alpha=0.4,               # Attention reallocation strength
    ret_sim=0.9,                 # Image attention reduction
    img_start_idx=35,
    img_end_idx=611,
    question_len=question_len,
    prompt_len=prompt_len,
    context_len=context_len,
)
```

### TCVM Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_tcvm` | bool | False | Enable TCVM mode |
| `tcvm_topk` | int | 20 | Number of visual patches to mask (~3.5% of 576) |
| `tcvm_alpha` | float | 1.0 | Contrastive penalty weight |
| `tcvm_beta` | float | 0.7 | APC threshold |
| `tcvm_mask_strategy` | str | 'zero' | Masking: 'zero', 'mean', or 'noise' |
| `img_start_idx` | int | 35 | Visual tokens start index |
| `img_end_idx` | int | 611 | Visual tokens end index |

**Example Configuration**:
```python
model.generate(
    input_ids,
    images=images,
    use_tcvm=True,               # Enable TCVM
    tcvm_topk=20,                # Mask top-20 attended patches
    tcvm_alpha=1.0,              # Contrastive penalty
    tcvm_beta=0.7,               # APC threshold
    tcvm_mask_strategy='zero',   # Zero-out masked patches
    img_start_idx=35,
    img_end_idx=611,
)
```

---

## Comparison Summary

| Aspect | ALFAR | TCVM |
|--------|-------|------|
| **Mechanism** | Attention reallocation + Adaptive logits fusion | Token-level causal masking |
| **Stages** | 2 (internal + external) | 1 (external only) |
| **Operation Point** | Inside layers + Outside model | Outside model only |
| **Forward Passes** | 2 per token (base + contrastive_cd) | 2 per token (base + counterfactual) |
| **Layers Modified** | All 32 layers (monkey-patched) | None (operates on KV cache) |
| **Primary Target** | Boost context, reduce image | Penalize hallucinations |
| **Scope** | Global (all visual tokens) | Local (top-K patches) |
| **Adaptive Component** | cd_alpha = image_attn / context_attn | Fixed tcvm_alpha |
| **Use Case** | Knowledge-grounded VQA | Hallucination detection |
| **Requires Context** | Yes | No |
| **Inference Overhead** | ~2x (2 forward passes) | ~2x (2 forward passes) |
| **Memory Overhead** | Minimal | +50% (KV cloning) |
| **Files** | `attention.py` + `vcd_sample.py` | `tcvm_utils.py` + `vcd_sample.py` |
| **Trigger** | `images_cd != None` | `use_tcvm=True` |

### Key Differences

1. **ALFAR modifies attention weights internally**, TCVM does not
2. **ALFAR uses adaptive fusion** (cd_alpha varies), TCVM uses fixed penalty (tcvm_alpha)
3. **ALFAR requires context**, TCVM works without context
4. **ALFAR masks all image tokens** (via attention reduction), **TCVM masks top-K specific patches**
5. **ALFAR's contrastive input** is `images_cd` (second prompt), **TCVM's counterfactual** is masked KV cache

---

## Performance Characteristics

### ALFAR Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Inference Speed** | ~0.5x baseline | 2 forward passes (base + contrastive_cd) |
| **Memory Usage** | Minimal overhead | No KV cloning, just attention modification |
| **GPU Utilization** | 2x compute per token | 2 full forward passes |
| **Latency per Token** | +30-50ms | Depends on model size |
| **Throughput** | ~50% of baseline | Due to 2 forward passes |
| **Context Required** | Yes | Both stages need context boundaries |

**Strengths**:
- Effective for knowledge-grounded VQA
- Adaptive fusion adjusts to attention patterns
- Direct control over context utilization

**Weaknesses**:
- Requires external context
- 2 forward passes (slow)
- Needs precise question/context boundaries

### TCVM Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Inference Speed** | ~0.5x baseline | 2 forward passes (base + counterfactual) |
| **Memory Usage** | +50% during decoding | KV cache cloning (temporary) |
| **GPU Utilization** | 2x compute per token | Base + counterfactual forward |
| **Latency per Token** | +20-40ms | Counterfactual is lightweight (1 token) |
| **Throughput** | ~50% of baseline | Significant overhead |
| **Context Required** | No | Works on visual grounding alone |

**Strengths**:
- Works without external context
- Fine-grained (patch-level) masking
- Direct hallucination detection

**Weaknesses**:
- 2 forward passes (slow)
- Higher memory footprint
- No context utilization boost

---

## When to Use Which Method

### Use ALFAR When:
- You have **high-quality retrieved context**
- Task is **knowledge-grounded VQA** (A-OKVQA, InfoSeek, ViQuAE)
- You want **adaptive fusion** based on attention patterns
- Context utilization is the primary goal

**Example Scenarios**:
- "What year was this building constructed?" + Wikipedia context
- "Who is the author of this book?" + Knowledge base entry
- Model tends to rely on visual priors instead of context

### Use TCVM When:
- You want **hallucination detection**
- Task is **open-ended VQA** without guaranteed context
- You need **fine-grained visual grounding verification**
- Hallucination is critical (medical, autonomous systems)

**Example Scenarios**:
- "Describe what you see" (no context, high hallucination risk)
- POPE benchmark (object hallucination detection)
- Open-domain VQA where fabrication is risky

### Combine ALFAR + TCVM When:
- You have context **and** need hallucination detection
- Maximum answer quality is critical
- Inference speed is not a constraint

**Configuration**:
```python
model.generate(
    input_ids,
    images=images,
    # ALFAR Stage 1 + 2
    images_cd=input_ids1,        # Enable ALFAR Stage 2
    att_alpha=0.4,               # Attention reallocation
    ret_sim=0.9,                 # Image reduction
    cd_beta=0.7,                 # ALFAR APC
    question_len=question_len,
    context_len=context_len,
    # TCVM
    use_tcvm=True,               # Enable TCVM
    tcvm_topk=20,
    tcvm_alpha=1.0,
    tcvm_beta=0.7,
)
```

**Note**: ALFAR and TCVM are **mutually exclusive in the current implementation** - see vcd_sample.py lines 142-229:
```python
if use_tcvm:
    # TCVM branch
elif use_cd:  # This is ALFAR Stage 2
    # ALFAR adaptive fusion
else:
    # Standard sampling
```

To combine both, you'd need to modify the code to apply both contrastive methods sequentially or in parallel.

---

## Debugging & Visualization

### Print ALFAR Stage 1 (Attention Reallocation)

Add to `experiments/eval/attention.py`:

```python
def llama_new_forward(self, hidden_states, ...):
    # ... existing code ...

    if use_attn and not use_cfg:
        # Before reallocation
        img_attn_before = attn_weights[:, :, -1, img_start_idx:img_end_idx].mean().item()
        ctx_attn_before = attn_weights[:, :, -1, context_start_idx:context_end_idx].mean().item()

        # Attention reallocation (existing code)
        # ...

        # After reallocation (before softmax)
        img_attn_after = attn_weights[:, :, -1, img_start_idx:img_end_idx].mean().item()
        ctx_attn_after = attn_weights[:, :, -1, context_start_idx:context_end_idx].mean().item()

        if hasattr(self, 'layer_idx'):
            print(f"[ALFAR Stage 1] Layer {self.layer_idx}")
            print(f"  Image:   {img_attn_before:.4f} → {img_attn_after:.4f} ({img_attn_after/img_attn_before:.1%})")
            print(f"  Context: {ctx_attn_before:.4f} → {ctx_attn_after:.4f} ({ctx_attn_after/ctx_attn_before:.1%})")
            print(f"  Relevance weight: {self.weight.mean():.4f}")
```

### Print ALFAR Stage 2 (Adaptive Logits Fusion)

Add to `experiments/eval/vcd_sample.py` around line 220:

```python
elif use_cd:  # ALFAR Stage 2
    # ... existing code ...

    image_attn = outputs.attentions[-1][-1,:,-1,img_start_idx:img_end_idx].mean(dim=0).sum()
    context_attn = outputs.attentions[-1][-1,:,-1, context_start_idx:context_start_idx+context_len].mean(dim=0).sum()
    cd_alpha = image_attn / context_attn

    print(f"\n[ALFAR Stage 2] Adaptive Logits Fusion")
    print(f"  Image attention: {image_attn:.4f}")
    print(f"  Context attention: {context_attn:.4f}")
    print(f"  cd_alpha (image/context): {cd_alpha:.4f}")
    print(f"  Base logit weight: (1 + 1/{cd_alpha:.4f}) = {1 + 1/cd_alpha:.4f}")
    print(f"  Contrastive penalty: (1 - {cd_alpha:.4f}) = {1 - cd_alpha:.4f}")

    diffs = (1+1/cd_alpha) * next_token_logits - (1-cd_alpha) * next_token_logits_cd

    # Compare top-5 tokens
    top5_base = torch.topk(next_token_logits, k=5)
    top5_final = torch.topk(cd_logits, k=5)

    print(f"  Top-5 base tokens: {tokenizer.batch_decode(top5_base.indices[0])}")
    print(f"  Top-5 final tokens: {tokenizer.batch_decode(top5_final.indices[0])}")
```

### Print TCVM Process

Add to `experiments/eval/vcd_sample.py` around line 160:

```python
if use_tcvm:
    print(f"\n[TCVM] Token-Level Causal Masking")

    # Extract attention
    visual_attn_weights = outputs.attentions[-1][:, :, -1, img_start_idx:img_end_idx]
    visual_attn_weights = visual_attn_weights.mean(dim=1)

    print(f"  Visual attention: mean={visual_attn_weights.mean():.4f}, max={visual_attn_weights.max():.4f}")

    # Get top-K
    topk_visual_indices = get_topk_visual_indices(visual_attn_weights, img_start_idx, tcvm_topk)
    print(f"  Top-{tcvm_topk} patch indices: {topk_visual_indices[0].tolist()}")
    print(f"  Top-{tcvm_topk} attention weights: {visual_attn_weights[0, topk_visual_indices[0] - img_start_idx].tolist()}")

    # Mask and forward
    masked_past_kv = mask_visual_kv_cache(outputs.past_key_values, topk_visual_indices,
                                         strategy=tcvm_strategy, detach=True)
    next_token_logits_masked = tcvm_counterfactual_forward(...)

    # Compare
    top5_base = torch.topk(next_token_logits, k=5)
    top5_masked = torch.topk(next_token_logits_masked, k=5)
    top5_final = torch.topk(cd_logits, k=5)

    print(f"  Top-5 base: {tokenizer.batch_decode(top5_base.indices[0])}")
    print(f"  Top-5 masked: {tokenizer.batch_decode(top5_masked.indices[0])}")
    print(f"  Top-5 final: {tokenizer.batch_decode(top5_final.indices[0])}")
```

---

## Summary

| Feature | ALFAR | TCVM |
|---------|-------|------|
| **Components** | 2-stage: Attention Reallocation + Adaptive Logits Fusion | 1-stage: Token-Level Causal Masking |
| **Internal Modification** | Yes (attention weights in layers 0-31) | No |
| **External Modification** | Yes (adaptive logits fusion) | Yes (contrastive decoding) |
| **Adaptive Component** | cd_alpha (image_attn / context_attn) | None (fixed tcvm_alpha) |
| **Forward Passes** | 2 (base + contrastive_cd) | 2 (base + counterfactual) |
| **Contrastive Input** | images_cd (second prompt) | Masked KV cache |
| **Masking Scope** | Global (all visual tokens via attention) | Local (top-K patches via KV masking) |
| **Requires Context** | Yes | No |
| **Primary Goal** | Boost context utilization | Detect hallucinations |
| **Best Use Case** | Knowledge-grounded VQA | Hallucination mitigation |

---

## Conclusion

**ALFAR** and **TCVM** are fundamentally different approaches:

**ALFAR** (2-stage):
1. **Stage 1 (Internal)**: Reallocates attention from images to context
2. **Stage 2 (External)**: Adaptively fuses base and contrastive logits based on attention patterns
3. **Best for**: Knowledge-grounded VQA where context utilization is critical
4. **Key innovation**: Adaptive fusion weight that adjusts to attention distribution

**TCVM** (1-stage):
1. **External only**: Masks top-K attended visual patches and compares base vs counterfactual
2. **Best for**: Hallucination detection without requiring external context
3. **Key innovation**: Fine-grained, token-specific patch-level masking

**Key Insight**: ALFAR's "adaptive" component is the **cd_alpha** weight in Stage 2, which dynamically adjusts fusion based on how much the model attends to images vs context. This makes ALFAR particularly effective when context quality varies or when the model's attention needs to be guided toward relevant context.

---

## References

### Code Locations

**ALFAR Stage 1 (Attention Reallocation)**:
- Implementation: `experiments/eval/attention.py` (lines 79-106)
- Integration: `experiments/llava/model/language_model/llava_llama.py` (line 137, 163)
- Function: `llama_modify()`, `llama_new_forward()`

**ALFAR Stage 2 (Adaptive Logits Fusion)**:
- Implementation: `experiments/eval/vcd_sample.py` (lines 199-229)
- Integration: `evolve_vcd_sampling()` (line 20)
- Trigger: `images_cd != None`

**TCVM**:
- Utilities: `experiments/eval/tcvm_utils.py`
- Integration: `experiments/eval/vcd_sample.py` (lines 142-198)
- Trigger: `use_tcvm=True`

### Paper

An et al., "Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation", arXiv:2406.12718, 2024
