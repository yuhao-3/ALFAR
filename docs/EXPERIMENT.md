# ALFAR Experiments Guide

**Version**: 2.0
**Last Updated**: 2026-04-07
**Status**: Active Research

---

## Table of Contents

1. [Overview](#overview)
2. [Experimental Framework](#experimental-framework)
3. [Datasets](#datasets)
4. [Methods](#methods)
5. [Running Experiments](#running-experiments)
6. [Evaluation](#evaluation)
7. [Bucket Analysis](#bucket-analysis)
8. [Results](#results)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

This document provides comprehensive guidance for running experiments with ALFAR (Adaptive Logits Fusion and Attention Reallocation).

### Research Goals

1. **Validate ALFAR effectiveness** on knowledge-intensive VQA tasks
2. **Understand amplification mechanisms** through three-way comparison
3. **Assess robustness** to misleading context

### Supported Models

- ✅ **LLaVA-1.5** (primary model)
- ✅ **InstructBLIP**
- ✅ **MiniGPT-4**
- ✅ **Shikra**

---

## Experimental Framework

### Three-Way Comparison

We evaluate three methods to understand context amplification:

| Method | Context | Amplification | Configuration |
|--------|---------|---------------|---------------|
| **No-Context** | ✗ | ✗ | Parametric knowledge only |
| **Regular MRAG** | ✓ | ✗ | Standard RAG (context in prompt) |
| **ALFAR** | ✓ | ✓ | Attention reallocation + logits fusion |

**Why Three-Way?**
- Tests if context helps: No-Context vs Regular MRAG
- Tests if amplification helps: Regular MRAG vs ALFAR
- Tests amplification safety: ALFAR performance in misleading contexts

### Bucket-Based Analysis

Samples classified by context quality:

| Bucket | Model Correct<br>w/o Context | Context Has<br>Answer | Scenario |
|--------|------------------------------|----------------------|----------|
| **Corrective** | ✗ | ✓ | Context can help |
| **Neutral** | ✓ | ✓ | Context confirms knowledge |
| **Misleading** | ✗ | ✗ | Context lacks answer |
| **Other** | ✓ | ✗ | Context contradicts (excluded) |

---

## Datasets

### Available Benchmarks

| Dataset | Type | Val Size | Image Source | Focus |
|---------|------|----------|--------------|-------|
| **A-OKVQA** | Open-ended | 1,145 | COCO | Knowledge VQA |
| **OK-VQA** | Open-ended | ~5,046 | COCO | Knowledge VQA |
| **InfoSeek** | 4-choice MC | ~3,000 | Wikipedia | Entity knowledge |
| **ViQuAE** | 4-choice MC | ~700 | Wikipedia | Question answering |
| **E-VQA** | Open-ended | ~2,000 | COCO | Explanations |

### Dataset Locations

```
data/
├── eval_data/
│   ├── aokvqa/
│   │   ├── aokvqa_v1p0_val.json
│   │   └── retrieval_indices.json
│   ├── infoseek/
│   ├── viquae/
│   ├── okvqa/
│   └── evqa/
├── retrieval_result/        # Retrieved context indices
├── wiki/                    # Wikipedia corpus
└── images/
    ├── coco/
    │   ├── val2014/
    │   └── val2017/
    └── infoseek/
```

### Data Download

**Images**:
- COCO: Download from [COCO website](https://cocodataset.org/)
- InfoSeek: Download from [InfoSeek repo](https://github.com/open-vision-language/infoseek)

**Knowledge Base**:
- Download from [Google Drive](https://drive.google.com/file/d/18uFkE9SbPnUT9DLBd8DCfed6ah2sDbrJ/view)
- Save to `data/wiki/`

---

## Methods

### 1. No-Context Baseline

**Purpose**: Measure parametric knowledge only

**Configuration**:
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()  # REQUIRED: Fix empty answers

model.generate(
    input_ids,          # Question only, NO context
    images=image,
    # NO images_cd
    # NO att_alpha
    do_sample=True,
    temperature=1.0,
    top_p=1.0
)
```

**Scripts**:
- Multiple Choice: `experiments/eval/no_context_llava.py`
- OKVQA: `experiments/eval/no_context_llava_okvqa.py`

**Expected Results** (A-OKVQA):
- Accuracy: ~46.46%
- Empty answers: 0% (with `evolve_vcd_sampling()`)

---

### 2. Regular MRAG (Standard RAG)

**Purpose**: Test if context alone (without amplification) helps

**Configuration**:
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()  # Same as ALFAR

model.generate(
    input_ids,          # Question + Context
    images=image,
    # NO images_cd
    att_alpha=0.0,      # CRITICAL: Must be 0.0
    do_sample=True,
    temperature=1.0,
    top_p=1.0
)
```

**Scripts**:
- Multiple Choice: `experiments/eval/regular_mrag_llava_mc.py`
- OKVQA: `experiments/eval/regular_mrag_llava_okvqa.py`
- EVQA: `experiments/eval/regular_mrag_llava_evqa.py`

**Critical Notes**:
- ❌ **WRONG**: Omitting `att_alpha` parameter
- ✅ **CORRECT**: `att_alpha=0.0`
- This ensures attention reallocation is disabled

**Expected Results** (A-OKVQA):
- Accuracy: ~46.08% (≈ No-Context)
- Proves context needs amplification

---

### 3. ALFAR (Full Method)

**Purpose**: Evaluate context amplification mechanisms

**Configuration**:
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

model.generate(
    input_ids,                      # Question + Context
    images=image,
    images_cd=input_ids_no_context, # Enable contrastive decoding
    cd_beta=1.0,                    # Contrastive weight
    att_alpha=1.5,                  # Attention reallocation
    question_len=len_question,
    prompt_len=len_prompt,
    context_len=len_context,
    do_sample=True,
    temperature=1.0,
    top_p=1.0
)
```

**Scripts**:
- Multiple Choice: `experiments/eval/alfar_mc_llava.py`
- OKVQA: `experiments/eval/alfar_okvqa_llava.py`
- EVQA: `experiments/eval/alfar_evqa_llava.py`
- Other models: `alfar_mc_instructblip.py`, `alfar_mc_minigpt.py`, `alfar_mc_shikra.py`

**Parameters by Dataset**:

| Dataset | att_alpha | cd_beta | ret_sim | Notes |
|---------|-----------|---------|---------|-------|
| A-OKVQA | 1.5 | 1.0 | 0.9 | Motivation experiments (high amplification) |
| OK-VQA | 0.4 | 0.7 | 0.9 | Standard configuration |
| InfoSeek | 0.2 | 0.7 | Variable | Moderate amplification |
| ViQuAE | 0.2 | 0.7 | Variable | Moderate amplification |
| E-VQA | 0.1 | 0.7 | 1.0 | Minimal (gold evidence) |

**Expected Results** (A-OKVQA):
- Accuracy: ~60.23%
- Gain over Regular MRAG: +14.15%
- Gain over No-Context: +13.77%

---

## Running Experiments

### Environment Setup

```bash
# Activate environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# Verify Python version
python --version  # Should be 3.9.21

# Check GPU availability
nvidia-smi
```

### Interactive Execution (Testing)

**A-OKVQA Example**:
```bash
cd /data/gpfs/projects/punim2075/ALFAR

# No-Context
python experiments/eval/no_context_llava_okvqa.py

# Regular MRAG
python experiments/eval/regular_mrag_llava_okvqa.py

# ALFAR
python experiments/eval/alfar_okvqa_llava.py
```

**InfoSeek Example**:
```bash
python experiments/eval/alfar_mc_llava.py \
    --dataset infoseek \
    --image-folder data/images/infoseek/
```

### SLURM Execution (Production)

**Submit Single Job**:
```bash
sbatch slurm_jobs/run_regular_mrag_aokvqa.slurm
```

**Submit All Motivation Experiments**:
```bash
bash slurm_jobs/run_all_motivation_experiments.sh
```

**Standard SLURM Configuration**:
```bash
#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%j.out
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View real-time logs
tail -f logs/slurm-JOBID.out

# Monitor results directory
watch -n 5 'ls -lh experiments/result/ | tail -20'

# Check specific job details
sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## Evaluation

### Running Evaluation Scripts

**A-OKVQA / OK-VQA**:
```bash
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/aokvqa_alfar_results.csv
```

**InfoSeek / ViQuAE**:
```bash
python evaluation/eval_mc.py \
    --dataset infoseek \
    --preds experiments/result/infoseek_alfar_results.jsonl
```

**E-VQA**:
```bash
python evaluation/eval_evqa.py \
    --preds experiments/result/evqa_alfar_results.json
```

### Result File Formats

**Multiple Choice** (InfoSeek, ViQuAE):
```json
[
    {
        "question_id": "infoseek_val_00000",
        "prediction": "A",
        "answer": "B",
        "correct": false
    }
]
```

**Open-ended** (A-OKVQA, OK-VQA):
```csv
question_id,prediction,answer,correct
aokvqa_val_00000,Paris,Paris,True
aokvqa_val_00001,dog,cat,False
```

---

## Bucket Analysis

### Purpose

Analyze performance by context quality to understand when amplification helps vs harms.

### Running Analysis

```bash
# Three-way bucket analysis
python scripts/bucket_analysis_threeway.py \
    --dataset aokvqa \
    --no_context experiments/result/no_context_aokvqa_results.csv \
    --regular_mrag experiments/result/regular_mrag_aokvqa_results.csv \
    --alfar experiments/result/aokvqa_alfar_results.csv

# Generate visualizations
python scripts/plot_threeway_bucket_analysis.py --dataset aokvqa
```

### Analysis Outputs

**Statistics** (`results/bucket_analysis_threeway/`):
- `{dataset}_bucket_stats.json` - Per-bucket accuracy summary
- `{dataset}_bucket_assignments.json` - Sample-level classifications
- `{dataset}_bucket_samples.json` - Detailed sample information

**Visualizations**:
- `{dataset}_threeway_comparison.png` - Main comparison chart
- `{dataset}_delta_comparison.png` - Improvement visualization

### Understanding Results

**Corrective Bucket**:
- Model wrong without context, context has answer
- ALFAR should significantly outperform both baselines
- Measures context utilization capability

**Neutral Bucket**:
- Model correct without context, context confirms
- Tests parametric knowledge preservation
- ALFAR should minimize performance drop

**Misleading Bucket** (Critical Test):
- Model wrong without context, context lacks answer
- Tests robustness to misleading information
- ALFAR should NOT harm performance vs Regular MRAG

---

## Results

### A-OKVQA Motivation Experiments (n=1145)

#### Overall Performance

| Method | Accuracy | vs No-Context | vs Regular MRAG |
|--------|----------|---------------|-----------------|
| No-Context | 46.46% | - | - |
| Regular MRAG | 46.08% | -0.38% | - |
| **ALFAR** | **60.23%** | **+13.77%** | **+14.15%** |

**Key Finding**: Regular MRAG ≈ No-Context, proving context needs amplification.

#### Bucket-Wise Performance

**1. Corrective Bucket (n=135)**

| Method | Accuracy | Interpretation |
|--------|----------|---------------|
| No-Context | 0.00% | Model has no parametric knowledge |
| Regular MRAG | 53.33% | Context helps somewhat |
| **ALFAR** | **72.59%** | **+19.26% better context utilization** |

**2. Neutral Bucket (n=402)**

| Method | Accuracy | Interpretation |
|--------|----------|---------------|
| No-Context | 100.00% | Model has strong parametric knowledge |
| Regular MRAG | 77.61% | Context disrupts parametric knowledge |
| **ALFAR** | **93.28%** | **+15.67% better preservation** |

**3. Misleading Bucket (n=292)** ⚠️ **CRITICAL TEST**

| Method | Accuracy | Interpretation |
|--------|----------|---------------|
| No-Context | 0.00% | Model lacks knowledge |
| Regular MRAG | 32.19% | Indirect information helps |
| **ALFAR** | **38.36%** | **+6.16% even with misleading context** ✅ |

**Critical Insight**: ALFAR outperforms Regular MRAG even when context is misleading, proving amplification is robust, not blind.

---

## Troubleshooting

### Common Issues

#### Empty Predictions

**Symptom**: Model generates empty strings

**Cause**: VCD sampling issue

**Solution**:
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()  # MUST call before generate()
```

**Verification**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/result/results.csv')
empty = (df['prediction'].str.strip() == '').sum()
print(f'Empty: {empty}/{len(df)} ({empty/len(df)*100:.1f}%)')
"
```

#### CUDA Out of Memory

**Solutions**:
1. Increase SLURM memory: `#SBATCH --mem=128G`
2. Use smaller batch size in code
3. Clear GPU cache: `torch.cuda.empty_cache()`

#### Wrong Regular MRAG Configuration

**Common Mistakes**:
- ❌ Not setting `att_alpha` parameter
- ❌ Setting `att_alpha=None`
- ❌ Setting `images_cd` parameter

**Correct Configuration**:
```python
model.generate(
    ...,
    att_alpha=0.0,  # MUST be 0.0, not None
    # NO images_cd parameter
)
```

**Verification**:
```bash
# Check logs for parameter values
grep "att_alpha" logs/slurm-JOBID.out
grep "images_cd" logs/slurm-JOBID.out
```

#### Incorrect Bucket Classification

**Issue**: Unexpected bucket distributions

**Debug Steps**:
1. Check `bucket_assignments.json` for sample cases
2. Verify context detection logic (keyword matching)
3. Manually inspect samples from each bucket

#### Job Failures

**Check Job Status**:
```bash
sacct -j JOBID --format=JobID,State,ExitCode,Reason
```

**Common Causes**:
- Time limit exceeded → Increase `#SBATCH --time`
- Memory exceeded → Increase `#SBATCH --mem`
- GPU error → Check CUDA compatibility

---

## Experiment Checklist

### Before Running

- [ ] Environment activated
- [ ] Data files exist and verified
- [ ] Images downloaded and in correct location
- [ ] SLURM script parameters configured
- [ ] Output directory exists (`experiments/result/`)
- [ ] Logs directory exists (`logs/`)
- [ ] Previous results backed up (if any)
- [ ] Experiment plan documented in `docs/`

### During Execution

- [ ] Job submitted successfully
- [ ] Monitor job status regularly
- [ ] Check logs for errors
- [ ] Verify GPU utilization
- [ ] Monitor output file growth

### After Completion

- [ ] Results file generated
- [ ] Evaluation script run successfully
- [ ] Accuracy computed and reasonable
- [ ] Bucket analysis completed (if applicable)
- [ ] Visualizations generated
- [ ] **Results documented in `docs/`**
- [ ] Results backed up

---

## Best Practices

### Experiment Workflow

1. **Start Small**: Test on A-OKVQA (smallest dataset) first
2. **Verify Baselines**: Ensure No-Context and Regular MRAG work correctly
3. **Check Sanity**: Verify ALFAR > Regular MRAG ≈ No-Context
4. **Scale Up**: Run on larger datasets (InfoSeek, OK-VQA)
5. **Document Everything**: Save all results and analysis to `docs/`

### Code Organization

- Keep experiment scripts in `experiments/eval/`
- Keep analysis scripts in `scripts/`
- Keep SLURM jobs in `slurm_jobs/`
- **Keep ALL documentation in `docs/`**

### Result Management

- Save results to `experiments/result/` or `results/`
- Use clear naming: `{dataset}_{method}_results.{ext}`
- Never overwrite without backup
- Document result file locations in `docs/`

---

## Advanced Topics

### Multi-Model Experiments

**Supported Models**:
- LLaVA: `alfar_mc_llava.py`, `alfar_okvqa_llava.py`
- InstructBLIP: `alfar_mc_instructblip.py`
- MiniGPT-4: `alfar_mc_minigpt.py`
- Shikra: `alfar_mc_shikra.py`

**Usage**:
```bash
python experiments/eval/alfar_mc_instructblip.py --dataset infoseek
```

### Parameter Sensitivity Analysis

Test different `att_alpha` values:
```python
for alpha in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
    model.generate(..., att_alpha=alpha)
```

Test different `cd_beta` values:
```python
for beta in [0.0, 0.3, 0.5, 0.7, 1.0]:
    model.generate(..., cd_beta=beta)
```

### Ablation Studies

**Test Components Separately**:

1. **Attention Reallocation Only**:
   ```python
   model.generate(..., att_alpha=1.5, cd_beta=0.0)
   ```

2. **Contrastive Decoding Only**:
   ```python
   model.generate(..., att_alpha=0.0, cd_beta=1.0, images_cd=...)
   ```

3. **Both (Full ALFAR)**:
   ```python
   model.generate(..., att_alpha=1.5, cd_beta=1.0, images_cd=...)
   ```

---

## References

### Documentation
- [CLAUDE.md](CLAUDE.md) - Claude assistant instructions
- [Main README](../README.md) - Project overview

### Papers
- ALFAR Paper: [OpenReview](https://openreview.net/pdf/40d94836204a19bf22a4813c820925434476760b.pdf)
- VCD: [GitHub](https://github.com/DAMO-NLP-SG/VCD)
- A-OKVQA: [GitHub](https://github.com/allenai/aokvqa)

### Datasets
- [InfoSeek](https://github.com/open-vision-language/infoseek)
- [ViQuAE](https://github.com/PaulLerner/ViQuAE)
- [E-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)
- [OK-VQA](https://okvqa.allenai.org/index.html)
- [A-OKVQA](https://github.com/allenai/aokvqa)

---

**Version**: 2.0
**Last Updated**: 2026-04-07
**Repository**: https://github.com/Lackel/ALFAR
**Maintained by**: ALFAR Project Team
