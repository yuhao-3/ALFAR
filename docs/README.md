# ALFAR Documentation Hub

Complete documentation for ALFAR (Adaptive Logits Fusion and Attention Reallocation) and TCVM-KAR (Token Contrastive Visual Masking with Knowledge-Aware Router).

## Quick Start

**New to the project?** Start here:
1. [ALFAR Technical Documentation](ALFAR_TECHNICAL_DOCUMENTATION.md) - Understand the base ALFAR method
2. [TCVM-KAR Quick Reference](TCVM_KAR_QUICK_REFERENCE.md) - Get started with the new TCVM-KAR model
3. [TCVM-KAR Runtime Status](TCVM_KAR_RUNTIME_STATUS.md) - Check current experiment status

**Running experiments?**
- [TCVM-KAR Experiments Running](TCVM_KAR_EXPERIMENTS_RUNNING.md) - How to run TCVM-KAR
- [Monitoring and Evaluation Guide](MONITORING_AND_EVALUATION_GUIDE.md) - Track job progress
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Fix common issues

---

## Documentation Index

### TCVM-KAR (NEW MODEL) 🚀

**TCVM-KAR** is the latest advancement - a Knowledge-Aware Router that dynamically decides whether to mask visual tokens or context tokens based on attention patterns.

| Document | Description | When to Use |
|----------|-------------|-------------|
| [**TCVM-KAR Quick Reference**](TCVM_KAR_QUICK_REFERENCE.md) | Command cheat sheet and quick tips | Need quick commands or parameter reference |
| [**TCVM-KAR Runtime Status**](TCVM_KAR_RUNTIME_STATUS.md) | Current running jobs and status | Check what's currently running |
| [**TCVM-KAR Experiments Running**](TCVM_KAR_EXPERIMENTS_RUNNING.md) | How to submit and run experiments | Submit new TCVM-KAR jobs |
| [**TCVM-KAR Upgrade Summary**](TCVM_KAR_UPGRADE_SUMMARY.md) | Technical implementation details | Understand how KAR routing works |

**Key Features of TCVM-KAR**:
- **Adaptive Routing**: Dynamically masks visual OR context tokens
- **Lambda Metric**: Uses attention-based routing metric λ_t
- **Intelligent Fallback**: Auto-falls back to visual-only if no context
- **RAG Support**: Better handling of retrieval-augmented generation

### TCVM Implementation

Original TCVM (visual-only masking) documentation:

| Document | Description | When to Use |
|----------|-------------|-------------|
| [**TCVM Implementation Guide**](TCVM_IMPLEMENTATION_GUIDE.md) | Technical guide for TCVM | Understand original TCVM mechanism |
| [**TCVM vs ALFAR Comparison**](TCVM_VS_ALFAR_COMPARISON.md) | Detailed method comparison | Compare different approaches |
| [**TCVM Accuracy Summary**](TCVM_ACCURACY_SUMMARY.md) | Performance metrics | View TCVM results |

### ALFAR (Base Method)

Original ALFAR approach documentation:

| Document | Description | When to Use |
|----------|-------------|-------------|
| [**ALFAR Technical Documentation**](ALFAR_TECHNICAL_DOCUMENTATION.md) | Complete technical guide | Deep dive into ALFAR mechanisms |
| [**Baselines**](BASELINES.md) | Baseline performance metrics | Compare with baselines |

### Experiments & Results

Multi-seed experiment documentation:

| Document | Description | When to Use |
|----------|-------------|-------------|
| [**Multiseed Final Summary**](MULTISEED_FINAL_SUMMARY.md) | Final results across all seeds | View comprehensive results |
| [**Multiseed Results Summary**](MULTISEED_RESULTS_SUMMARY.md) | Statistical analysis | Understand variance/reliability |
| [**Multiseed Experiments**](MULTISEED_EXPERIMENTS.md) | How multi-seed works | Run multi-seed experiments |
| [**Quick Multiseed Guide**](QUICK_MULTISEED_GUIDE.md) | Fast start for multi-seed | Quick multi-seed commands |
| [**Experiment Status**](EXPERIMENT_STATUS.md) | Historical experiment tracking | Review past experiments |
| [**EVQA Status**](EVQA_STATUS.md) | EVQA-specific status | EVQA experiment details |

### Operational Guides

Essential guides for running and monitoring experiments:

| Document | Description | When to Use |
|----------|-------------|-------------|
| [**Monitoring and Evaluation Guide**](MONITORING_AND_EVALUATION_GUIDE.md) | Complete monitoring reference | Monitor jobs, evaluate results |
| [**Troubleshooting Guide**](TROUBLESHOOTING_GUIDE.md) | Fix common problems | Job failed or not working |

---

## Methods Overview

### ALFAR (Adaptive Logits Fusion and Attention Reallocation)

**What it does**:
- Reallocates attention from visual to context tokens based on relevance
- Fuses logits from two decoding branches (with/without context)

**Key parameters**:
- `att_alpha`: Attention reallocation strength (0.1-0.4)
- `cd_beta`: Plausibility threshold (typically 0.7)
- `ret_sim`: Retrieved knowledge similarity (0.9-1.0)

**Best for**: Knowledge-intensive VQA with high-quality retrieval

### TCVM (Token Contrastive Visual Masking)

**What it does**:
- Masks top-K attended visual tokens
- Uses contrastive decoding to penalize hallucinations

**Key parameters**:
- `tcvm_topk`: Number of tokens to mask (typically 20)
- `tcvm_alpha`: Contrastive penalty weight (typically 1.0)
- `tcvm_beta`: Plausibility threshold (typically 0.7)

**Best for**: Reducing visual hallucinations

### TCVM-KAR (TCVM with Knowledge-Aware Router) 🆕

**What it does**:
- Computes routing metric: λ_t = visual_attn / (visual_attn + context_attn)
- If λ_t > 0.5: Masks visual tokens (vision-dominant)
- If λ_t ≤ 0.5: Masks context tokens (context-dominant)
- Automatically adapts to question type and available knowledge

**Key parameters**:
- `use_tcvm`: Enable TCVM-KAR (set to `True`)
- `tcvm_topk`: Number of tokens to mask (typically 20)
- `tcvm_alpha`: Contrastive penalty weight (typically 1.0)
- `tcvm_beta`: Plausibility threshold (typically 0.7)
- `tcvm_mask_strategy`: How to mask (`'zero'` or `'random'`)

**Best for**: Adaptive hallucination mitigation with RAG support

**Advantages over original TCVM**:
- ✅ Handles both visual and context hallucinations
- ✅ Automatically routes based on attention patterns
- ✅ Better for RAG scenarios
- ✅ More robust to question diversity

---

## Common Tasks

### Starting New TCVM-KAR Experiments

```bash
# Submit all 5 datasets
bash slurm_jobs/run_all_tcvm.sh

# Submit specific dataset
sbatch slurm_jobs/run_infoseek_tcvm.slurm
```

See: [TCVM-KAR Experiments Running](TCVM_KAR_EXPERIMENTS_RUNNING.md)

### Monitoring Running Jobs

```bash
# Quick status
squeue -u $USER

# Live monitoring
watch -n 10 'squeue -u $USER'

# Follow output
tail -f logs/tcvm_infoseek_[JOBID].out
```

See: [Monitoring and Evaluation Guide](MONITORING_AND_EVALUATION_GUIDE.md)

### Evaluating Results

```bash
# OKVQA/AOKVQA
python evaluation/eval_okvqa.py --dataset aokvqa --preds experiments/result/aokvqa_tcvm_results.csv

# InfoSeek/ViQuAE
python evaluation/eval_mc.py --dataset infoseek --preds experiments/result/infoseek_tcvm_results.jsonl

# EVQA
python evaluation/eval_evqa.py --preds experiments/result/evqa_tcvm_results.json
```

See: [Monitoring and Evaluation Guide](MONITORING_AND_EVALUATION_GUIDE.md#evaluation-scripts)

### Troubleshooting Failed Jobs

```bash
# Check error log
cat logs/tcvm_[dataset]_[jobid].err

# Search for errors
grep -i "error\|exception" logs/tcvm_[dataset]_[jobid].out

# Check job details
scontrol show job [JOBID]
```

See: [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

### Running Multi-seed Experiments

```bash
# Submit multi-seed job (runs 5 seeds)
sbatch slurm_jobs/run_aokvqa_alfar_multiseed.slurm

# Calculate averages
python scripts/calculate_all_averages.py
```

See: [Quick Multiseed Guide](QUICK_MULTISEED_GUIDE.md)

---

## File Structure Reference

### Core Implementation
```
experiments/
├── eval/
│   ├── alfar_mc_llava.py          # ALFAR for InfoSeek/ViQuAE
│   ├── alfar_okvqa_llava.py       # ALFAR for OKVQA/AOKVQA
│   ├── alfar_evqa_llava.py        # ALFAR for EVQA
│   ├── attention.py                # Attention reallocation
│   ├── vcd_sample.py               # Contrastive decoding + KAR router
│   ├── tcvm_utils.py               # TCVM-KAR utilities
│   └── test_tcvm.py                # TCVM-KAR tests
└── llava/
    └── model/
        └── language_model/
            └── llava_llama.py      # Modified LLaVA model
```

### Evaluation Scripts
```
evaluation/
├── eval_mc.py                      # InfoSeek/ViQuAE evaluation
├── eval_okvqa.py                   # OKVQA/AOKVQA evaluation
└── eval_evqa.py                    # EVQA evaluation
```

### SLURM Job Scripts
```
slurm_jobs/
├── run_all_tcvm.sh                 # Submit all TCVM-KAR jobs
├── run_infoseek_tcvm.slurm        # InfoSeek TCVM-KAR
├── run_viquae_tcvm.slurm          # ViQuAE TCVM-KAR
├── run_aokvqa_tcvm.slurm          # AOKVQA TCVM-KAR
├── run_okvqa_tcvm.slurm           # OKVQA TCVM-KAR
├── run_evqa_tcvm.slurm            # EVQA TCVM-KAR
└── run_*_multiseed.slurm          # Multi-seed variants
```

### Helper Scripts
```
scripts/
├── monitor_tcvm_kar_jobs.sh       # Monitor TCVM-KAR jobs
├── monitor_multiseed_jobs.sh      # Monitor multi-seed jobs
├── calculate_all_averages.py      # Calculate multi-seed averages
└── aggregate_multiseed_results.py # Aggregate results
```

### Data
```
data/
├── eval_data/                      # Question datasets
│   ├── aokvqa/
│   ├── infoseek/
│   ├── viquae/
│   ├── okvqa/
│   └── evqa/
├── retrieval_result/               # Retrieved knowledge indices
│   ├── aokvqa_retrieval.json
│   └── ...
├── wiki/                           # Wikipedia knowledge base
└── images/                         # Dataset images
    ├── coco/
    ├── infoseek/
    └── ...
```

### Results & Logs
```
experiments/result/                 # Result files
├── aokvqa_tcvm_results.csv
├── aokvqa_alfar_seed0.csv
└── ...

logs/                               # Job logs
├── tcvm_infoseek_[JOBID].out
├── tcvm_infoseek_[JOBID].err
└── ...
```

---

## Dataset Configurations

### TCVM-KAR Parameters by Dataset

| Dataset   | tcvm_topk | tcvm_alpha | tcvm_beta | Notes |
|-----------|-----------|------------|-----------|-------|
| InfoSeek  | 20        | 1.0        | 0.7       | Entity-focused, benefits from visual masking |
| ViQuAE    | 20        | 1.0        | 0.7       | Wikipedia-based, balanced routing |
| A-OKVQA   | 20        | 1.0        | 0.7       | Commonsense + external knowledge |
| OK-VQA    | 20        | 1.0        | 0.7       | External knowledge intensive |
| EVQA      | 20        | 1.0        | 0.7       | Gold evidence, less routing needed |

### ALFAR Parameters by Dataset

| Dataset   | att_alpha | ret_sim | cd_beta | Notes |
|-----------|-----------|---------|---------|-------|
| InfoSeek  | 0.2       | Variable| 0.7     | Moderate attention shift |
| ViQuAE    | 0.2       | Variable| 0.7     | Moderate attention shift |
| A-OKVQA   | 0.4       | 0.9     | 0.7     | Strong attention to context |
| OK-VQA    | 0.4       | 0.9     | 0.7     | Strong attention to context |
| E-VQA     | 0.1       | 1.0     | 0.7     | Minimal tuning (gold evidence) |

---

## Performance Benchmarks

### TCVM-KAR Expected Results

Based on preliminary experiments:

| Dataset   | Expected Accuracy | Status |
|-----------|------------------|---------|
| InfoSeek  | TBD              | Running |
| ViQuAE    | TBD              | Queued  |
| A-OKVQA   | TBD              | Queued  |
| OK-VQA    | TBD              | Queued  |
| EVQA      | TBD              | Queued  |

**Note**: Check [TCVM_KAR_RUNTIME_STATUS.md](TCVM_KAR_RUNTIME_STATUS.md) for latest results

### ALFAR Baseline Results

See [MULTISEED_FINAL_SUMMARY.md](MULTISEED_FINAL_SUMMARY.md) for detailed ALFAR results across 5 seeds.

---

## Environment Setup

### Python Environment
- **Python**: 3.9.21
- **PyTorch**: 2.1.2+cu118
- **NumPy**: 1.26.4 (NOT 2.0+)
- **CUDA**: 11.8.0
- **Transformers**: Latest compatible version

### Cluster Environment
- **Cluster**: Spartan HPC
- **Partition**: gpu-a100
- **GPUs**: NVIDIA A100 (40GB VRAM)
- **Scheduler**: SLURM

### Activate Environment
```bash
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate
```

---

## Getting Help

### Common Issues
1. **NumPy 2.0 error**: See [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md#issue-numpy-20-incompatibility--solved)
2. **Job stuck in queue**: See [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md#issue-job-stuck-in-pending-pd-state)
3. **CUDA out of memory**: See [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md#issue-cuda-out-of-memory)
4. **Empty result files**: See [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md#issue-empty-result-files)

### Documentation Priority
1. **Quick Reference**: [TCVM_KAR_QUICK_REFERENCE.md](TCVM_KAR_QUICK_REFERENCE.md)
2. **Current Status**: [TCVM_KAR_RUNTIME_STATUS.md](TCVM_KAR_RUNTIME_STATUS.md)
3. **Troubleshooting**: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
4. **Monitoring**: [MONITORING_AND_EVALUATION_GUIDE.md](MONITORING_AND_EVALUATION_GUIDE.md)

### Support Checklist
Before asking for help:
- [ ] Checked relevant documentation
- [ ] Reviewed error logs
- [ ] Verified environment setup
- [ ] Consulted troubleshooting guide
- [ ] Gathered diagnostic info (job ID, error messages, logs)

---

## Recent Updates

**March 25, 2026**:
- ✅ Fixed NumPy 2.0 compatibility issue
- ✅ Submitted all 5 TCVM-KAR experiments (Jobs 23153037-23153041)
- ✅ Created comprehensive documentation suite
- ✅ Added runtime status tracking
- 🔄 InfoSeek TCVM-KAR currently running

**Documentation Coverage**:
- [x] TCVM-KAR implementation
- [x] Runtime status tracking
- [x] Comprehensive troubleshooting
- [x] Monitoring and evaluation
- [x] Multi-seed experiments
- [x] Baseline comparisons

---

## Contributing

When adding new documentation:
1. Place in `docs/` directory
2. Use descriptive filename with UPPERCASE
3. Update this README index
4. Follow existing markdown format
5. Include last updated date

---

## Quick Links

**Essential Reading**:
- [TCVM-KAR Quick Reference](TCVM_KAR_QUICK_REFERENCE.md) - Start here for TCVM-KAR
- [ALFAR Technical Documentation](ALFAR_TECHNICAL_DOCUMENTATION.md) - Deep dive
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - When things go wrong

**Current Status**:
- [TCVM-KAR Runtime Status](TCVM_KAR_RUNTIME_STATUS.md) - What's running now
- [Experiment Status](EXPERIMENT_STATUS.md) - Historical experiments

**Results & Analysis**:
- [Multiseed Final Summary](MULTISEED_FINAL_SUMMARY.md) - Comprehensive results
- [TCVM vs ALFAR Comparison](TCVM_VS_ALFAR_COMPARISON.md) - Method comparison
- [Baselines](BASELINES.md) - Baseline metrics

---

**Documentation Version**: 2.0
**Last Updated**: March 25, 2026
**Maintained by**: ALFAR Project Team
