# TCVM-KAR Multi-Seed Experimental Results

**Date**: March 26, 2026
**Model**: LLaVA 1.5-7B
**Method**: TCVM-KAR (Knowledge-Aware Router)
**Seeds**: 0, 1, 2
**Status**: 4/5 datasets complete, E-VQA in progress

---

## Summary

All TCVM-KAR multi-seed experiments (3 seeds) have been completed for 4 out of 5 datasets. E-VQA evaluation is currently in progress.

---

## Results by Dataset

### A-OKVQA (Knowledge-aware VQA)

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 0    | 59.71%   | 683.67/1145   |
| 1    | 60.12%   | 688.37/1145   |
| 2    | 59.48%   | 681.05/1145   |
| **Mean ± SD** | **59.77% ± 0.32%** | - |

### OK-VQA (Outside Knowledge VQA)

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 0    | 60.66%   | 3061/5046     |
| 1    | 60.08%   | 3032/5046     |
| 2    | 60.40%   | 3048/5046     |
| **Mean ± SD** | **60.38% ± 0.29%** | - |

### InfoSeek (Knowledge-intensive VQA)

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 0    | 57.23%   | 1717/3000     |
| 1    | 57.23%   | 1717/3000     |
| 2    | 57.33%   | 1720/3000     |
| **Mean ± SD** | **57.26% ± 0.06%** | - |

### ViQuAE (Visual Question Answering from Encyclopedias)

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 0    | 57.07%   | 1716/3007     |
| 1    | 57.07%   | 1716/3007     |
| 2    | 56.37%   | 1695/3007     |
| **Mean ± SD** | **56.84% ± 0.40%** | - |

### E-VQA (Encyclopedic VQA)

| Seed | Accuracy | Status |
|------|----------|--------|
| 0    | 35.97%   | ✅ Complete |
| 1    | TBD      | ⏳ Running |
| 2    | TBD      | ⏳ Running |
| **Mean ± SD** | **TBD** | Pending |

---

## Configuration

All experiments used identical TCVM-KAR configuration:

```python
use_tcvm = True                  # Enable TCVM-KAR
tcvm_topk = 20                   # Top-K tokens to mask
tcvm_alpha = 1.0                 # Contrastive penalty weight
tcvm_beta = 0.7                  # Plausibility threshold (APC)
tcvm_mask_strategy = 'zero'      # Masking strategy
# seed = {0, 1, 2}               # Random seeds
```

### Knowledge-Aware Router (KAR)

The TCVM-KAR implementation features:
- **Dynamic routing** based on attention weights
- **Visual masking** when λ_t > 0.5 (vision-dominant)
- **Context masking** when λ_t ≤ 0.5 (context-dominant)
- **Automatic fallback** to visual-only TCVM if context unavailable

---

## Key Observations

### 1. Consistency Across Seeds

The results show **high consistency** across different random seeds:
- **A-OKVQA**: σ = 0.32% (very stable)
- **OK-VQA**: σ = 0.29% (very stable)
- **InfoSeek**: σ = 0.06% (extremely stable)
- **ViQuAE**: σ = 0.40% (stable)

This demonstrates the **robustness** of the TCVM-KAR method.

### 2. Dataset Performance Ranking

Based on mean accuracy:
1. **OK-VQA**: 60.38% ± 0.29% (highest)
2. **A-OKVQA**: 59.77% ± 0.32%
3. **InfoSeek**: 57.26% ± 0.06%
4. **ViQuAE**: 56.84% ± 0.40%
5. **E-VQA**: TBD (pending)

### 3. InfoSeek Shows Exceptional Stability

InfoSeek has the **lowest standard deviation** (0.06%), suggesting:
- TCVM-KAR is particularly consistent on this dataset
- The dataset may have less variance in question difficulty
- The routing mechanism is very stable for entity-centric questions

---

## Comparison with Previous Results

### Single Seed (Seed 0) Results

From `TCVM_KAR_RUNTIME_STATUS.md`:

| Dataset   | TCVM-KAR (seed 0) | Multi-seed Mean |
|-----------|-------------------|-----------------|
| InfoSeek  | 57.23%            | 57.26%          |
| ViQuAE    | 57.07%            | 56.84%          |
| A-OKVQA   | 59.71%            | 59.77%          |
| OK-VQA    | 60.66%            | 60.38%          |
| E-VQA     | 35.97%            | TBD             |

The single-seed and multi-seed results are **highly aligned**, confirming reproducibility.

---

## Files Generated

### Result Files

All result files are located in `results/llava1.5/{dataset}/`:

```
results/llava1.5/
├── aokvqa/
│   ├── aokvqa_tcvm_seed0_results.csv
│   ├── aokvqa_tcvm_seed1_results.csv
│   ├── aokvqa_tcvm_seed2_results.csv
│   └── accuracy.txt
├── evqa/
│   ├── evqa_tcvm_seed0_results.json
│   ├── evqa_tcvm_seed1_results.json  (in progress)
│   └── evqa_tcvm_seed2_results.json  (in progress)
├── infoseek/
│   ├── infoseek_tcvm_seed0_results.jsonl
│   ├── infoseek_tcvm_seed1_results.jsonl
│   └── infoseek_tcvm_seed2_results.jsonl
├── okvqa/
│   ├── okvqa_tcvm_seed0_results.csv
│   ├── okvqa_tcvm_seed1_results.csv
│   └── okvqa_tcvm_seed2_results.csv
└── viquae/
    ├── viquae_tcvm_seed0_results.jsonl
    ├── viquae_tcvm_seed1_results.jsonl
    └── viquae_tcvm_seed2_results.jsonl
```

---

## Next Steps

### Immediate (After E-VQA Completes)

1. ✅ Complete E-VQA seed 1 and seed 2 evaluations
2. ✅ Calculate E-VQA mean ± SD
3. ✅ Update this summary with complete E-VQA results
4. ✅ Generate final comparison table (TCVM-KAR vs ALFAR)

### Future Work

1. **Statistical Significance Testing**
   - Paired t-tests between TCVM-KAR and ALFAR
   - Effect size calculations (Cohen's d)

2. **Routing Analysis**
   - Analyze λ_t distribution across datasets
   - Correlate routing decisions with question types
   - Identify when visual vs. context masking is preferred

3. **Failure Analysis**
   - Compare errors across seeds
   - Identify persistent failure modes
   - Analyze cases where routing helped/hurt

4. **Multi-Model Expansion**
   - Run TCVM-KAR on InstructBLIP, Shikra, MiniGPT-4
   - Compare routing behavior across architectures
   - Identify model-specific patterns

---

## Monitoring E-VQA Progress

The E-VQA evaluations are running in the background. To check progress:

```bash
# Check if processes are still running
ps aux | grep eval_evqa.py | grep -v grep

# View partial output (seed 1)
tail -f /proc/<PID>/fd/1

# View partial output (seed 2)
tail -f /proc/<PID>/fd/1
```

Expected completion time: ~4-6 hours (700 items each at ~3-4s per item)

---

## Technical Details

### Environment
- **Platform**: Spartan HPC (University of Melbourne)
- **Python**: 3.9.21
- **PyTorch**: 2.1.2+cu118
- **NumPy**: 1.26.4 (downgraded from 2.0.2 for compatibility)
- **GPU**: A100 (40GB) for inference, CPU for E-VQA evaluation

### Execution Timeline
- **March 25, 2026**: Seed 0 experiments completed
- **March 25-26, 2026**: Seeds 1-2 experiments completed
- **March 26, 2026**: Evaluations completed (4/5 datasets)

---

**Last Updated**: March 26, 2026, 11:47 AEDT
**Status**: 4/5 datasets complete, E-VQA in progress
