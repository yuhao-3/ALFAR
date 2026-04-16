# Claude Assistant Instructions for ALFAR Project

## Project Overview

**ALFAR (Adaptive Logits Fusion and Attention Reallocation)** is a training-free, plug-and-play method for improving knowledge-intensive visual question answering in Multimodal Large Language Models (MLLMs).

### Key Mechanisms
1. **Attention Reallocation**: Adaptively shifts attention from visual tokens to context tokens based on query-context relevance
2. **Logits Fusion**: Decouples and weights parametric vs contextual knowledge to mitigate conflicts

### Paper
- **Title**: "Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation"
- **Authors**: Wenbin An, Jiahao Nie, Feng Tian, Haonan Lin, Mingxiang Cai, Yaqiang Wu, Qianying Wang, Xiaoqin Zhang, Shijian Lu
- **Link**: [OpenReview PDF](https://openreview.net/pdf/40d94836204a19bf22a4813c820925434476760b.pdf)

---

## Documentation Rules

### **CRITICAL**: All Documentation in docs/
- ✅ **All new documentation files MUST be placed in `docs/` directory**
- ✅ **All experiment documentation MUST go in `docs/`**
- ✅ **All analysis results MUST be documented in `docs/`**
- ❌ **NEVER create documentation files outside `docs/`**
- 中文提醒：**新文件放到docs/里** (New files go in docs/)

### File Naming Convention
- Use descriptive names: `docs/[COMPONENT]_[TYPE]_[VERSION].md`
- Examples:
  - `docs/EXPERIMENT.md` - Main experiment documentation
  - `docs/RESULTS_SUMMARY.md` - Results summary
  - `docs/ANALYSIS_[DATASET]_[DATE].md` - Dataset-specific analysis

---

## Repository Structure

Based on [original ALFAR repository](https://github.com/Lackel/ALFAR):

```
ALFAR/
├── data/
│   ├── eval_data/          # Question datasets (5 benchmarks)
│   │   ├── aokvqa/
│   │   ├── infoseek/
│   │   ├── viquae/
│   │   ├── okvqa/
│   │   └── evqa/
│   ├── retrieval_result/   # Retrieved knowledge indices
│   ├── wiki/               # Processed Wikipedia knowledge base
│   └── images/             # Dataset images (COCO, InfoSeek, etc.)
│
├── evaluation/             # Evaluation scripts for all datasets
│   ├── eval_mc.py         # InfoSeek/ViQuAE evaluation
│   ├── eval_okvqa.py      # OKVQA/AOKVQA evaluation
│   └── eval_evqa.py       # EVQA evaluation
│
├── experiments/
│   └── eval/              # Main experiment scripts
│       ├── alfar_mc_llava.py          # ALFAR for MC datasets
│       ├── alfar_okvqa_llava.py       # ALFAR for OKVQA/AOKVQA
│       ├── alfar_evqa_llava.py        # ALFAR for EVQA
│       ├── no_context_llava.py        # No-context baseline
│       ├── regular_mrag_llava_*.py    # Regular MRAG baselines
│       ├── attention.py                # Attention reallocation logic
│       └── vcd_sample.py               # Contrastive decoding (VCD)
│
├── docs/                  # **ALL documentation goes here**
│   ├── CLAUDE.md         # This file - instructions for Claude
│   └── EXPERIMENT.md     # Comprehensive experiment guide
│
├── image/                 # Visual assets (framework diagrams)
├── requirements.txt       # Python dependencies
└── README.md             # Project README
```

---

## Problem Formulation

### Research Questions

1. **Primary**: How can we improve MLLMs' utilization of retrieved knowledge in knowledge-intensive VQA?
2. **Secondary**: Are ALFAR's amplification mechanisms robust to misleading context?

### Three-Way Comparison Framework

We compare three approaches to understand the value of context amplification:

| Method | Context in Prompt | ALFAR Intervention | Purpose |
|--------|------------------|-------------------|---------|
| **No-Context** | ✗ | ✗ | Baseline: parametric knowledge only |
| **Regular MRAG** | ✓ | ✗ | Test if context alone helps |
| **ALFAR** | ✓ | ✓ | Test if amplification helps |

**Key Insight**: This three-way design separates the effects of:
- Having context (No-Context vs Regular MRAG)
- Amplifying context (Regular MRAG vs ALFAR)

---

## Motivation Experiments (A-OKVQA)

### Bucket-Based Analysis Methodology

Samples are classified into buckets based on context quality:

| Bucket | No-Ctx Correct? | Context Has Answer? | Interpretation |
|--------|-----------------|---------------------|----------------|
| **Corrective** | ✗ | ✓ | Context can help, model needs it |
| **Neutral** | ✓ | ✓ | Model already knows, context confirms |
| **Misleading** | ✗ | ✗ | Context lacks answer, may mislead |
| **Other** | ✓ | ✗ | Model correct, context contradicts (excluded) |

### Key Results (n=1145)

**Overall Performance**:

| Method | Accuracy | vs No-Context | vs Regular MRAG |
|--------|----------|---------------|-----------------|
| No-Context | 46.46% | - | - |
| Regular MRAG | 46.08% | -0.38% | - |
| **ALFAR** | **60.23%** | **+13.77%** | **+14.15%** |

**Critical Finding**: Regular MRAG ≈ No-Context, proving **context needs amplification to be useful**.

**Bucket-Wise Performance**:

| Bucket | Regular MRAG | ALFAR | ALFAR Advantage |
|--------|-------------|-------|-----------------|
| Corrective | 53.33% | **72.59%** | **+19.26%** |
| Neutral | 77.61% | **93.28%** | **+15.67%** |
| Misleading | 32.19% | **38.36%** | **+6.16%** ✅ |

**Revised Understanding**:
- ✅ ALFAR outperforms Regular MRAG in ALL buckets, including misleading
- ✅ ALFAR's amplification is beneficial AND robust, not blind
- ✅ Regular MRAG fails to utilize context effectively

---

## ALFAR Parameters by Dataset

| Dataset | att_alpha | ret_sim | cd_beta | Notes |
|---------|-----------|---------|---------|-------|
| InfoSeek | 0.2 | Variable | 0.7 | Moderate attention shift |
| ViQuAE | 0.2 | Variable | 0.7 | Moderate attention shift |
| A-OKVQA | 1.5 | 0.9 | 1.0 | Strong amplification (motivation exps) |
| OK-VQA | 0.4 | 0.9 | 0.7 | Strong attention to context |
| E-VQA | 0.1 | 1.0 | 0.7 | Minimal tuning (gold evidence) |

**Parameter Meanings**:
- `att_alpha`: Attention reallocation strength (0.0 = disabled, higher = more context focus)
- `cd_beta`: Contrastive decoding weight (0.0 = disabled, 1.0 = full weight)
- `ret_sim`: Retrieved knowledge similarity threshold

---

## Environment Setup

### Cluster Configuration
- **Cluster**: Spartan HPC
- **Partition**: gpu-a100
- **GPUs**: NVIDIA A100 (40GB VRAM)
- **Scheduler**: SLURM

### Python Environment
```bash
# Activate virtual environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
```

**Key Dependencies**:
- Python: 3.9.21
- PyTorch: 2.1.2+cu118
- NumPy: 1.26.4 (**NOT 2.0+**)
- CUDA: 11.8.0
- Transformers: Latest compatible
- TensorFlow: Required for E-VQA evaluation

---

## Implementation Details

### Method Configurations

#### 1. No-Context Baseline
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()  # Fix empty answer issue

model.generate(
    input_ids,  # Question only, NO context
    images=image,
    # NO images_cd → No contrastive decoding
    # NO att_alpha → No attention reallocation
    do_sample=True,
    temperature=1.0,
    top_p=1.0
)
```

#### 2. Regular MRAG (Standard RAG)
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()  # Same sampling as ALFAR

model.generate(
    input_ids,  # Question + Context in prompt
    images=image,
    # NO images_cd → No contrastive decoding
    att_alpha=0.0,  # MUST be 0.0 to disable ALFAR
    do_sample=True,
    temperature=1.0,
    top_p=1.0
)
```

#### 3. ALFAR (Full Method)
```python
from vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

model.generate(
    input_ids,  # Question + Context in prompt
    images=image,
    images_cd=input_ids_no_context,  # Enable contrastive decoding
    cd_beta=1.0,
    att_alpha=1.5,  # Enable attention reallocation
    question_len=...,
    prompt_len=...,
    context_len=...,
    do_sample=True,
    temperature=1.0,
    top_p=1.0
)
```

### Critical Implementation Notes

**Must Call `evolve_vcd_sampling()`**:
- Fixes empty answer generation issue
- Required for ALL three methods

**Regular MRAG Configuration**:
- ❌ **WRONG**: No att_alpha parameter
- ✅ **CORRECT**: `att_alpha=0.0`
- This ensures attention reallocation code path is disabled

---

## Datasets

### Available Datasets

| Dataset | Type | Size | Images | Domain |
|---------|------|------|--------|--------|
| **A-OKVQA** | Open-ended | 1,145 val | COCO | Knowledge-intensive VQA |
| **OK-VQA** | Open-ended | ~5,046 val | COCO | Knowledge-intensive VQA |
| **InfoSeek** | Multiple choice | ~3,000 val | Wikipedia | Entity-centric |
| **ViQuAE** | Multiple choice | ~700 val | Wikipedia | Question answering |
| **E-VQA** | Open-ended | ~2,000 val | COCO | Explanation-based |

### Dataset Locations
```
data/eval_data/[dataset]/      # Question data
data/images/[dataset]/          # Images
data/retrieval_result/          # Retrieved context indices
data/wiki/                      # Wikipedia corpus
```

---

## Running Experiments

### Quick Start (A-OKVQA)

**Interactive Testing**:
```bash
cd /data/gpfs/projects/punim2075/ALFAR
python experiments/eval/alfar_okvqa_llava.py
```

**SLURM Production**:
```bash
sbatch slurm_jobs/run_regular_mrag_aokvqa.slurm
```

### Monitoring Jobs

```bash
# Check status
squeue -u $USER

# View logs
tail -f logs/slurm-JOBID.out

# Watch results
watch -n 5 ls -lh experiments/result/
```

### Evaluation

```bash
# For OKVQA/AOKVQA
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/aokvqa_alfar_results.csv

# For InfoSeek/ViQuAE
python evaluation/eval_mc.py \
    --dataset infoseek \
    --preds experiments/result/infoseek_alfar_results.jsonl

# For EVQA
python evaluation/eval_evqa.py \
    --preds experiments/result/evqa_alfar_results.json
```

---

## Analysis Pipeline

### Bucket Analysis Workflow

```
1. Run three methods (No-Context, Regular MRAG, ALFAR)
   ↓
2. Classify samples into buckets based on:
   - No-context correctness
   - Context support for gold answer
   ↓
3. Compute per-bucket accuracy for each method
   ↓
4. Generate visualizations
   ↓
5. Document results in docs/
```

### Analysis Scripts

```bash
# Three-way bucket analysis
python scripts/bucket_analysis_threeway.py \
    --dataset aokvqa \
    --no_context experiments/result/no_context_aokvqa_results.csv \
    --regular_mrag experiments/result/regular_mrag_aokvqa_results.csv \
    --alfar experiments/result/aokvqa_alfar_results.csv

# Generate plots
python scripts/plot_threeway_bucket_analysis.py --dataset aokvqa
```

---

## Common Tasks & Guidelines

### When Running New Experiments

1. **ALWAYS document experiment plan in `docs/` FIRST**
2. Create SLURM job scripts in `slurm_jobs/`
3. Test on A-OKVQA first (smallest, fastest)
4. Monitor jobs actively
5. Document results in `docs/` when complete

### When Analyzing Results

1. Check existing scripts in `scripts/` first
2. Create new analysis scripts if needed
3. Save visualizations to `results/bucket_analysis_threeway/` or similar
4. **Document findings in `docs/`**

### When Debugging

**Empty Predictions**:
- Ensure `evolve_vcd_sampling()` is called

**CUDA OOM**:
- Reduce batch size
- Increase SLURM memory allocation

**Wrong Baseline Results**:
- Verify att_alpha=0.0 for Regular MRAG
- Verify NO images_cd for both No-Context and Regular MRAG
- Check logs for parameter values

---

## Critical Concepts to Remember

### Key Findings

1. **Regular MRAG ≈ No-Context**: Simply having context provides NO benefit
2. **ALFAR's amplification is essential**: +14% comes from amplification mechanisms
3. **ALFAR is robust**: Outperforms in ALL buckets, including misleading (+6.16%)
4. **Three-way analysis is critical**: Separates effects of context vs amplification

### Common Pitfalls

- ❌ Don't confuse Regular MRAG with ALFAR
- ❌ Don't forget `evolve_vcd_sampling()`
- ❌ Don't use att_alpha > 0 for Regular MRAG
- ❌ Don't create docs outside `docs/` directory
- ❌ Don't overwrite results without backup

---

## Workflow Checklist

### Before Running Experiments

- [ ] Environment activated
- [ ] Data files verified
- [ ] SLURM script configured correctly
- [ ] Output directory exists
- [ ] Experiment documented in `docs/`

### After Running Experiments

- [ ] Results generated successfully
- [ ] Evaluation script run
- [ ] Bucket analysis completed (if applicable)
- [ ] Visualizations generated
- [ ] **Results documented in `docs/`**

---

## File Organization

### Results Files
- `experiments/result/` - Main results
- `results/bucket_analysis_threeway/` - Bucket analysis outputs

### Documentation
- **`docs/`** - All documentation (ALWAYS use this)
- Never create markdown files in root or other directories

### Code
- `experiments/eval/` - Experiment scripts
- `scripts/` - Analysis scripts
- `slurm_jobs/` - Job submission scripts

---

## Version History

- **v2.0** (2026-04-07): Reorganized to follow original ALFAR structure, all docs in docs/
- **v1.0** (2026-03-28): Initial version

---

**Last Updated**: 2026-04-07
**Repository**: https://github.com/Lackel/ALFAR
**Maintained by**: ALFAR Project Team
