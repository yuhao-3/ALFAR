# TCVM Accuracy Results Summary

## Overview
This document contains accuracy results for TCVM (Token-Level Causal Visual Masking) experiments across all datasets.

**Note**: The newly submitted TCVM-KAR experiments are currently running. Results below are from previous TCVM runs (visual-only masking).

---

## Accuracy Results (Multi-Seed Runs)

### OK-VQA
**Average Accuracy: ~60.3%**

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 0    | 60.46%   | 3051.0/5046  |
| 1    | 60.66%   | 3061.0/5046  |
| 2    | 60.40%   | 3047.7/5046  |
| 3    | 60.08%   | 3031.7/5046  |
| **Mean** | **60.40%** | **3047.9/5046** |

### A-OKVQA
**Average Accuracy: ~59.5%**

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 0    | 59.71%   | 683.7/1145   |
| 1    | 59.27%   | 678.7/1145   |
| 2    | 59.48%   | 681.0/1145   |
| **Mean** | **59.49%** | **681.1/1145** |

### ViQuAE
**Average Accuracy: ~57.2%**

| Seed | Accuracy |
|------|----------|
| 0    | 57.43%   |
| 1    | 56.43%   |
| 2    | 57.03%   |
| 3    | 57.16%   |
| 4    | 57.47%   |
| 5    | 57.97%   |
| **Mean** | **57.25%** |

### InfoSeek
Multi-seed results available in:
- `experiments/result/multiseed/infoseek_tcvm_seed*.jsonl` (seeds 0-4)

Status: Files exist, evaluation needed

### EVQA
Multi-seed results available in:
- `experiments/result/multiseed/evqa_tcvm_seed*.json` (seeds 0-4)

Status: Files exist, evaluation needed

---

## Current TCVM-KAR Experiment Status

**Submitted**: March 25, 2026, 00:02 AEDT

| Job ID   | Dataset   | Status    | Issue |
|----------|-----------|-----------|-------|
| 23132552 | InfoSeek  | **FAILED**    | NumPy 2.0 compatibility |
| 23132553 | ViQuAE    | **RUNNING**   | Started at 00:25 |
| 23132554 | A-OKVQA   | PENDING   | - |
| 23132555 | OK-VQA    | PENDING   | - |
| 23132556 | EVQA      | PENDING   | - |

### Issue Identified
**NumPy 2.0 Compatibility Error**:
```
RuntimeError: Could not infer dtype of numpy.float32
ValueError: Unable to create tensor
```

Environment has NumPy 2.0.2 but PyTorch was compiled with NumPy 1.x.

**Solution**: Downgrade NumPy to <2.0 or upgrade affected modules.

---

## Comparison with Baselines

### Available Baseline Results
Based on file listings, we have:
- **ALFAR results**: `experiments/result/*_alfar_*.csv/json/jsonl`
- **TCVM results**: Previous runs in `experiments/result/multiseed/`

### To Generate Comparison
```bash
# Evaluate all existing multiseed results
python scripts/calculate_all_averages.py

# Compare TCVM vs ALFAR
python scripts/compare_tcvm_variants.py
```

---

## TCVM Configuration Used

All experiments ran with:
```python
use_tcvm=True
tcvm_topk=20                # Top-20 attended tokens masked
tcvm_alpha=1.0              # Contrastive penalty weight
tcvm_beta=0.7               # Plausibility threshold (APC)
tcvm_mask_strategy='zero'   # Zero-out masking
```

---

## Key Findings

1. **OK-VQA**: TCVM achieves ~60.4% accuracy (consistent across seeds)
2. **A-OKVQA**: TCVM achieves ~59.5% accuracy (slightly lower than OK-VQA)
3. **ViQuAE**: TCVM achieves ~57.3% accuracy (more variation across seeds)

### Stability
- **OK-VQA**: Very stable (σ ≈ 0.24%)
- **A-OKVQA**: Moderately stable (σ ≈ 0.22%)
- **ViQuAE**: More variable (σ ≈ 0.52%)

---

## Next Steps

### 1. Fix NumPy Compatibility Issue
```bash
# Downgrade NumPy in virtual environment
conda activate /home/yuhao3/venvs/CARE
pip install "numpy<2.0"
```

### 2. Resubmit Failed Jobs
```bash
cd /data/gpfs/projects/punim2075/ALFAR/slurm_jobs
sbatch run_infoseek_tcvm.slurm
```

### 3. Evaluate Missing Results
```bash
# InfoSeek
python evaluation/eval_mc.py --dataset infoseek --preds experiments/result/multiseed/infoseek_tcvm_seed0.jsonl

# EVQA
python evaluation/eval_evqa.py --preds experiments/result/multiseed/evqa_tcvm_seed0.json
```

### 4. Compare with TCVM-KAR
Once TCVM-KAR experiments complete, compare:
- **Original TCVM** (visual-only, these results)
- **TCVM-KAR** (adaptive routing, new experiments)
- **ALFAR** (baseline)

---

## Files Referenced

### Result Files
- `experiments/result/multiseed/aokvqa_tcvm_seed*.csv`
- `experiments/result/multiseed/okvqa_tcvm_seed*.csv`
- `experiments/result/multiseed/viquae_tcvm_seed*.jsonl`
- `experiments/result/multiseed/infoseek_tcvm_seed*.jsonl`
- `experiments/result/multiseed/evqa_tcvm_seed*.json`

### Log Files
- `logs/tcvm_*_metrics.txt` - Accuracy metrics
- `logs/tcvm_*_*.out` - Job outputs
- `logs/tcvm_*_*.err` - Error logs

---

**Last Updated**: March 25, 2026, 00:20 AEDT
