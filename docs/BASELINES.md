# ALFAR Baseline Methods Documentation

**Last Updated**: 2026-04-07
**Status**: Implemented

---

## Table of Contents

1. [Overview](#overview)
2. [Baseline Methods](#baseline-methods)
3. [Implementation Details](#implementation-details)
4. [Running Baselines](#running-baselines)
5. [Evaluation](#evaluation)
6. [Expected Results](#expected-results)
7. [References](#references)

---

## Overview

This document describes all baseline methods implemented for comparison with ALFAR. The ALFAR paper compares against several state-of-the-art training-free decoding methods.

### Baseline Categories

1. **Knowledge Conflict Methods**: CD, CAD, AdaCAD, Entropy, COIECD
2. **Hallucination Mitigation Methods**: VCD, AGLA

### Implementation Status

| Method | Status | File | Description |
|--------|--------|------|-------------|
| **No-Context** | ✅ Implemented | `no_context_llava_okvqa.py` | Parametric knowledge only |
| **Regular MRAG** | ✅ Implemented | `regular_mrag_llava_okvqa.py` | Standard RAG |
| **VCD** | ✅ Implemented | `baseline_all_okvqa_llava.py` | Visual Contrastive Decoding |
| **CD** | ✅ Implemented | `baseline_all_okvqa_llava.py` | Contrastive Decoding |
| **CAD** | ✅ Implemented | `baseline_all_okvqa_llava.py` | Context-Aware Decoding |
| **AdaCAD** | ✅ Implemented | `baseline_all_okvqa_llava.py` | Adaptive CAD |
| **Entropy** | ✅ Implemented | `baseline_all_okvqa_llava.py` | Entropy-based Decoding |
| **COIECD** | ✅ Implemented | `baseline_all_okvqa_llava.py` | Contextual Info-Entropy Decoding |
| **AGLA** | ⏳ Planned | TBD | Assembly of Global and Local Attention |
| **ALFAR** | ✅ Implemented | `alfar_okvqa_llava.py` | Full ALFAR method |

---

## Baseline Methods

### 1. Visual Contrastive Decoding (VCD)

**Purpose**: Mitigate object hallucinations by contrasting outputs from original vs distorted visual inputs.

**Method**:
```
VCD(x) = log P_original(x) - alpha * log P_distorted(x)
```

**Key Parameters**:
- `vcd_alpha`: Weight for distorted logits (default: 0.5)
- `vcd_blur_radius`: Gaussian blur radius for image distortion (default: 10.0)

**Reference**:
- Paper: "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding" (CVPR 2024 Highlight)
- GitHub: [DAMO-NLP-SG/VCD](https://github.com/DAMO-NLP-SG/VCD)

**When to Use**: When visual hallucinations are a concern, especially for object recognition tasks.

---

### 2. Contrastive Decoding (CD)

**Purpose**: Generate better text by contrasting expert (large) and amateur (small or no-context) models.

**Method**:
```
CD(x) = log P_expert(x) - alpha * log P_amateur(x)
```

**Key Parameters**:
- `cd_alpha`: Weight for amateur model contrast (default: 0.5)

**Reference**:
- Paper: "Contrastive Decoding: Open-ended Text Generation as Optimization" (ACL 2023)
- arXiv: [2210.15097](https://arxiv.org/abs/2210.15097)

**When to Use**: General text generation improvement, particularly for open-ended generation.

---

### 3. Context-Aware Decoding (CAD)

**Purpose**: Make model trust provided evidence more by amplifying difference between with/without context.

**Method**:
```
CAD(x) = (1 + alpha) * log P_context(x) - alpha * log P_no_context(x)
```

**Key Parameters**:
- `cad_alpha`: Context amplification weight (default: 0.5)

**Reference**:
- Paper: "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding" (NAACL 2024)
- Authors: Shi et al.

**When to Use**: When you want to override model's parametric knowledge with retrieved context.

**Strengths**:
- Significantly improves faithfulness to context
- Works well with high-quality retrieved evidence
- Training-free

**Limitations**:
- Can overcorrect when conflict degree varies
- May degrade performance on low-conflict examples

---

### 4. Adaptive Context-Aware Decoding (AdaCAD)

**Purpose**: Dynamically adjust context amplification based on detected knowledge conflict degree.

**Method**:
```
conflict_degree = KL(P_context || P_no_context)
alpha_adaptive = alpha_max * f(conflict_degree)
AdaCAD(x) = (1 + alpha_adaptive) * log P_context(x) - alpha_adaptive * log P_no_context(x)
```

**Key Parameters**:
- `adacad_alpha_max`: Maximum amplification weight (default: 1.0)
- `adacad_threshold`: Conflict detection threshold (default: 0.5)

**Reference**:
- Paper: "AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge" (2024)
- arXiv: [2409.07394](https://arxiv.org/abs/2409.07394)

**When to Use**: When conflict degree varies across samples (some high-conflict, some low-conflict).

**Advantages over CAD**:
- Reduces overcorrection on low-conflict examples
- Automatically adapts to conflict degree
- Better overall performance across varying scenarios

---

### 5. Entropy-Based Decoding

**Purpose**: Guide decoding using entropy, preferring confident (low-entropy) predictions.

**Method**:
```
H(P) = -Σ P(x) log P(x)
If H(P) > threshold:
    P_adjusted(x) = P(x) / (1 + beta * (H(P) - threshold))
```

**Key Parameters**:
- `entropy_threshold`: Threshold for high entropy (default: 1.0)
- `entropy_temperature`: Temperature for controlling entropy (default: 0.5)

**When to Use**: When model uncertainty needs to be reduced or managed.

**Effect**: Sharpens probability distribution when model is uncertain.

---

### 6. COIECD (Contextual Information-Entropy Constraint Decoding)

**Purpose**: Combine CAD with entropy constraints for better decoding.

**Method**:
```
CAD_scores = (1 + alpha) * log P_context(x) - alpha * log P_no_context(x)
H = Entropy(CAD_scores)
If H > threshold:
    COIECD_scores = CAD_scores / (1 + beta * (H - threshold))
```

**Key Parameters**:
- `coiecd_alpha`: CAD alpha (default: 0.5)
- `coiecd_temperature`: Temperature for entropy constraint (default: 0.7)

**When to Use**: When you need both context amplification and uncertainty reduction.

**Advantages**:
- Combines benefits of CAD and entropy-based decoding
- Reduces uncertain predictions while amplifying context

---

### 7. AGLA (Assembly of Global and Local Attention)

**Purpose**: Reduce hallucinations by combining global and local visual attention.

**Status**: **Planned** (requires significant modifications to attention mechanism)

**Method**:
- Global features: For response generation
- Local features: For visual discrimination (prompt-relevant)
- Logit fusion: Calibrate using both feature types

**Reference**:
- Paper: "Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention" (CVPR 2025)
- GitHub: [Lackel/AGLA](https://github.com/Lackel/AGLA)
- Author: Same as ALFAR (Lackel)

**When to Use**: For tasks requiring fine-grained visual discrimination.

---

## Implementation Details

### Unified Implementation

All baselines (except AGLA) are implemented in a single unified script:

**File**: `experiments/eval/baseline_all_okvqa_llava.py`

This script supports all baseline methods via the `--method` parameter.

### Method Selection

```python
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cad \              # Choose: vcd, cd, cad, adacad, entropy, coiecd
    --dataset aokvqa \
    --cad-alpha 0.5
```

### Architecture

```
baseline_all_okvqa_llava.py
├── Common setup (model, data loading)
├── Method selection
│   ├── VCD: Distorted image generation
│   ├── CD: No-context as amateur
│   ├── CAD: Context amplification
│   ├── AdaCAD: Adaptive amplification
│   ├── Entropy: Temperature-based
│   └── COIECD: CAD + entropy
└── Unified evaluation
```

### Logits Processing Module

**File**: `experiments/eval/baseline_logits_processors.py`

Contains logits processor classes for advanced implementations (future use):
- `ContrastiveDecodingLogitsProcessor`
- `ContextAwareDecodingLogitsProcessor`
- `AdaptiveCADLogitsProcessor`
- `EntropyBasedDecodingLogitsProcessor`
- `COIECDLogitsProcessor`
- `VCDLogitsProcessor`

---

## Running Baselines

### Quick Start

**Single Baseline**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cad \
    --dataset aokvqa \
    --model-path /path/to/llava_1.5_7b \
    --image-folder /path/to/coco/val2014 \
    --cad-alpha 0.5
```

**All Baselines via SLURM**:
```bash
bash slurm_jobs/run_all_baselines_aokvqa.sh
```

### Method-Specific Examples

**VCD (Visual Contrastive Decoding)**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method vcd \
    --dataset aokvqa \
    --vcd-alpha 0.5 \
    --vcd-blur-radius 10.0
```

**CD (Contrastive Decoding)**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cd \
    --dataset aokvqa \
    --cd-alpha 0.5
```

**CAD (Context-Aware Decoding)**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method cad \
    --dataset aokvqa \
    --cad-alpha 0.5
```

**AdaCAD (Adaptive CAD)**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method adacad \
    --dataset aokvqa \
    --adacad-alpha-max 1.0
```

**Entropy-Based**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method entropy \
    --dataset aokvqa \
    --entropy-temperature 0.5
```

**COIECD**:
```bash
python experiments/eval/baseline_all_okvqa_llava.py \
    --method coiecd \
    --dataset aokvqa \
    --coiecd-alpha 0.5 \
    --coiecd-temperature 0.7
```

### SLURM Execution

**Submit CAD Baseline**:
```bash
sbatch slurm_jobs/run_baseline_cad_aokvqa.slurm
```

**Monitor Jobs**:
```bash
squeue -u $USER
tail -f logs/baseline_cad_aokvqa_*.out
```

---

## Evaluation

### Running Evaluation

**For OKVQA/AOKVQA**:
```bash
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/cad_aokvqa_results.csv
```

### Result Files

Results are saved in CSV format:
```
experiments/result/
├── cad_aokvqa_results.csv
├── cd_aokvqa_results.csv
├── vcd_aokvqa_results.csv
├── adacad_aokvqa_results.csv
├── entropy_aokvqa_results.csv
└── coiecd_aokvqa_results.csv
```

### Comparing Baselines

Create comparison table:
```python
import pandas as pd

methods = ['no_context', 'regular_mrag', 'cad', 'cd', 'vcd', 'adacad', 'entropy', 'coiecd', 'alfar']
results = {}

for method in methods:
    df = pd.read_csv(f'experiments/result/{method}_aokvqa_results.csv')
    # Evaluate...
    results[method] = accuracy

print(pd.DataFrame(results))
```

---

## Expected Results

### A-OKVQA Validation Set (n=1145)

Based on motivation experiments and paper results:

| Method | Expected Accuracy | vs No-Context | vs Regular MRAG | Notes |
|--------|------------------|---------------|-----------------|-------|
| **No-Context** | ~46% | - | - | Parametric only |
| **Regular MRAG** | ~46% | 0% | - | Context without amplification |
| **CD** | ~48-50% | +2-4% | +2-4% | Text contrastive |
| **VCD** | ~48-52% | +2-6% | +2-6% | Visual contrastive |
| **CAD** | ~50-55% | +4-9% | +4-9% | Context amplification |
| **AdaCAD** | ~52-57% | +6-11% | +6-11% | Adaptive amplification |
| **Entropy** | ~47-50% | +1-4% | +1-4% | Uncertainty reduction |
| **COIECD** | ~51-56% | +5-10% | +5-10% | CAD + entropy |
| **ALFAR** | **~60%** | **+14%** | **+14%** | Full method |

**Note**: These are estimates based on the three-way analysis. Actual results will vary based on hyperparameter tuning.

### Hyperparameter Sensitivity

**CAD Alpha**:
- 0.1-0.3: Weak amplification, safe but limited improvement
- 0.4-0.6: Moderate amplification, balanced performance
- 0.7-1.0: Strong amplification, high performance but may overcorrect

**VCD Blur Radius**:
- 5-10: Moderate distortion, balanced
- 10-15: Strong distortion, stronger effect
- 15+: Very strong distortion, may harm performance

---

## References

### Papers

1. **VCD**: Li et al., "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding", CVPR 2024
2. **CD**: Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization", ACL 2023
3. **CAD**: Shi et al., "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding", NAACL 2024
4. **AdaCAD**: "AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge", arXiv 2024
5. **AGLA**: An et al., "Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention", CVPR 2025
6. **ALFAR**: An et al., "Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation", 2024

### Code References

- VCD: https://github.com/DAMO-NLP-SG/VCD
- AGLA: https://github.com/Lackel/AGLA
- ALFAR: https://github.com/Lackel/ALFAR

---

## Implementation Notes

### Simplifications

Due to infrastructure constraints, some baselines are implemented in simplified form:

1. **VCD**: Uses blur distortion instead of masking (original VCD uses various distortions)
2. **CD**: Uses no-context as "amateur" model instead of separate small model
3. **AdaCAD**: Uses fixed adaptive rule instead of learned conflict detector
4. **Entropy/COIECD**: Uses temperature control instead of full logits modification

These simplifications allow running all baselines with the existing codebase while maintaining the core principles of each method.

### Future Improvements

1. **Full VCD**: Implement multiple distortion types (masking, noise, etc.)
2. **Full CD**: Add separate amateur model support
3. **Full AdaCAD**: Implement learned conflict detector
4. **AGLA**: Implement full attention assembly mechanism
5. **Logits Processors**: Integrate custom logits processors into generation loop

---

## Troubleshooting

### Common Issues

**Empty Predictions**:
- Ensure `evolve_vcd_sampling()` is called
- Check that `vcd_sample.py` is properly imported

**CUDA OOM**:
- Reduce batch size
- Use smaller blur radius for VCD
- Increase SLURM memory allocation

**Poor Baseline Performance**:
- Tune hyperparameters (alpha, temperature, etc.)
- Verify correct context retrieval
- Check image loading

**Method Not Working**:
- Verify method name is correct
- Check that all required parameters are provided
- Review logs for error messages

---

## Experiment Checklist

### Before Running

- [ ] Environment activated
- [ ] Data files verified (questions, images, knowledge)
- [ ] Model checkpoint available
- [ ] Output directory exists
- [ ] Method and parameters selected

### After Running

- [ ] Results file generated
- [ ] Evaluation completed
- [ ] Accuracy compared to baselines
- [ ] Results documented

---

**Version**: 1.0
**Last Updated**: 2026-04-07
**Maintained by**: ALFAR Project Team
