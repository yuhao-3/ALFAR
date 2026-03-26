# Multi-Seed Experiments Guide

## Overview

To replicate the statistical analysis from the ALFAR paper (An et al., 2024), we run experiments **multiple times with different random seeds** and compute **mean ± standard deviation**.

This guide explains:
1. Why multiple seeds are needed
2. How to run multi-seed experiments with SLURM
3. How to aggregate results and compute statistics

---

## Why Multiple Seeds?

### Problem: Randomness in Sampling

VLMs use **stochastic sampling** during generation:
- `temperature` > 0 introduces randomness
- `top_p`, `top_k` filtering varies across runs
- Random initialization of certain operations

**Result**: Same model + same input → different outputs across runs

### Solution: Statistical Significance

Run the same experiment **N times** (typically N=3 to 5) with different seeds:
- Seed 0, 1, 2, 3, 4
- Compute **mean** and **standard deviation**
- Report: `Accuracy = 52.34 ± 1.23%`

**Benefits**:
- Shows result stability (low std = robust method)
- Enables statistical significance testing
- Standard practice in ML research

---

## How Random Seeds Are Used

### In the Codebase

All evaluation scripts use `transformers.set_seed()`:

```python
# alfar_okvqa_llava.py (line 178)
from transformers import set_seed

parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seed(args.seed)
```

### What `set_seed()` Does

```python
def set_seed(seed: int):
    random.seed(seed)           # Python random
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU
```

**Effect**: Makes generation **reproducible** for the same seed, but **different** across seeds.

---

## Running Multi-Seed Experiments

### Method 1: SLURM Array Jobs (Recommended)

**Advantage**: Runs all seeds **in parallel** (faster)

#### Step 1: Submit Array Job

```bash
cd /data/gpfs/projects/punim2075/ALFAR/slurm_jobs

# Submit OK-VQA with TCVM (seeds 0-4)
sbatch run_okvqa_tcvm_multiseed.slurm
```

**What happens**:
- SLURM creates **5 jobs** (array tasks 0, 1, 2, 3, 4)
- Each task runs with a different seed
- Jobs run **in parallel** if GPUs available

#### Step 2: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Example output:
# JOBID    PARTITION  NAME                  ST  TIME  NODES
# 23001_0  gpu-a100   tcvm_okvqa_multiseed  R   2:30  1
# 23001_1  gpu-a100   tcvm_okvqa_multiseed  R   2:30  1
# 23001_2  gpu-a100   tcvm_okvqa_multiseed  PD  0:00  1
# 23001_3  gpu-a100   tcvm_okvqa_multiseed  PD  0:00  1
# 23001_4  gpu-a100   tcvm_okvqa_multiseed  PD  0:00  1
```

**Status codes**:
- `R` = Running
- `PD` = Pending (waiting for GPU)
- `CG` = Completing

#### Step 3: Check Outputs

```bash
# View logs (replace JOB_ID with your actual job ID)
tail -f /data/gpfs/projects/punim2075/ALFAR/logs/tcvm_okvqa_seed0_JOB_ID.out

# Check all completed seeds
ls -lht experiments/result/multiseed/okvqa_tcvm_seed*.csv
```

**Expected files**:
```
experiments/result/multiseed/
├── okvqa_tcvm_seed0.csv
├── okvqa_tcvm_seed1.csv
├── okvqa_tcvm_seed2.csv
├── okvqa_tcvm_seed3.csv
└── okvqa_tcvm_seed4.csv

logs/
├── tcvm_okvqa_seed0_metrics.txt
├── tcvm_okvqa_seed1_metrics.txt
├── tcvm_okvqa_seed2_metrics.txt
├── tcvm_okvqa_seed3_metrics.txt
└── tcvm_okvqa_seed4_metrics.txt
```

### Method 2: Sequential Jobs (Alternative)

If you have limited GPU quota, run seeds **sequentially**:

```bash
cd /data/gpfs/projects/punim2075/ALFAR/slurm_jobs

# Submit seeds one by one
for seed in 0 1 2 3 4; do
    sbatch --export=SEED=$seed run_okvqa_tcvm_singleseed.slurm
done
```

---

## Aggregating Results

### Step 1: Run Aggregation Script

After all seeds complete, aggregate results:

```bash
cd /data/gpfs/projects/punim2075/ALFAR

# Aggregate TCVM on OK-VQA
python scripts/aggregate_multiseed_results.py \
    --dataset okvqa \
    --method tcvm \
    --seeds 0 1 2 3 4 \
    --results_dir experiments/result/multiseed
```

### Step 2: View Statistics

**Output**:
```
============================================================
Results for TCVM on OKVQA
============================================================

Metric               Mean       Std        N     Values
------------------------------------------------------------
accuracy             0.5234     ±0.0123     5     [0.5201, 0.5245, 0.5198, 0.5267, 0.5259]
f1                   0.6123     ±0.0089     5     [0.6109, 0.6145, 0.6098, 0.6167, 0.6096]

============================================================
LaTeX-friendly format:
------------------------------------------------------------
accuracy: $52.34 \pm 1.23$
f1: $61.23 \pm 0.89$
============================================================

Statistics saved to: experiments/result/tcvm_okvqa_stats.json
```

### Step 3: Use in Paper

**Reporting format**:
```
OK-VQA Results:
- TCVM:     52.34 ± 1.23%
- ALFAR:    54.67 ± 0.98%
- Baseline: 48.23 ± 1.45%
```

---

## Creating Multi-Seed Scripts for Other Datasets

### Template: A-OKVQA with TCVM

```bash
#!/bin/bash
#SBATCH --job-name=tcvm_aokvqa_multiseed
#SBATCH --account=punim2075
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4  # 5 seeds
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/tcvm_aokvqa_seed%a_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/tcvm_aokvqa_seed%a_%j.err

SEED=$SLURM_ARRAY_TASK_ID

cd /data/gpfs/projects/punim2075/ALFAR
module load CUDA/11.8.0
source ALFAR/bin/activate
cd experiments/eval

python alfar_okvqa_llava.py \
    --dataset aokvqa \
    --image-folder ../../data/images/coco/val2014 \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/multiseed/aokvqa_tcvm_seed${SEED}.csv \
    --use_tcvm \
    --tcvm_topk 20 \
    --tcvm_alpha 1.0 \
    --tcvm_beta 0.7 \
    --seed ${SEED}

cd /data/gpfs/projects/punim2075/ALFAR
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/multiseed/aokvqa_tcvm_seed${SEED}.csv \
    > logs/tcvm_aokvqa_seed${SEED}_metrics.txt
```

### Template: InfoSeek with ALFAR

```bash
#!/bin/bash
#SBATCH --array=0-4
# ... (same SLURM config) ...

SEED=$SLURM_ARRAY_TASK_ID

python alfar_mc_llava.py \
    --dataset infoseek \
    --image-folder ../../data/images/infoseek \
    --model-path ../../models/llava-v1.5-7b \
    --answers-file ../result/multiseed/infoseek_alfar_seed${SEED}.jsonl \
    --cd_beta 0.7 \
    --att_alpha 0.2 \
    --seed ${SEED}
```

---

## Advanced: Comparing Multiple Methods

### Scenario: Compare TCVM vs ALFAR vs Baseline

Run all methods with multiple seeds:

```bash
# TCVM (5 seeds)
sbatch run_okvqa_tcvm_multiseed.slurm

# ALFAR (5 seeds)
sbatch run_okvqa_alfar_multiseed.slurm

# Baseline (5 seeds)
sbatch run_okvqa_baseline_multiseed.slurm
```

Aggregate all results:

```bash
# Aggregate each method
for method in tcvm alfar baseline; do
    python scripts/aggregate_multiseed_results.py \
        --dataset okvqa \
        --method $method \
        --seeds 0 1 2 3 4 \
        --output experiments/result/${method}_okvqa_stats.json
done
```

Create comparison table:

```python
import json

methods = ['baseline', 'alfar', 'tcvm']
for method in methods:
    with open(f'experiments/result/{method}_okvqa_stats.json') as f:
        stats = json.load(f)
        acc = stats['accuracy']
        print(f"{method.upper():10} {acc['mean']*100:.2f} ± {acc['std']*100:.2f}")

# Output:
# BASELINE   48.23 ± 1.45
# ALFAR      54.67 ± 0.98
# TCVM       52.34 ± 1.23
```

---

## FAQ

### Q1: How many seeds should I use?

**Answer**: Typically **3-5 seeds**

- **3 seeds**: Minimum for std computation
- **5 seeds**: Standard in most papers (good balance)
- **10 seeds**: Overkill unless very high variance

### Q2: What if results vary a lot across seeds?

**High standard deviation** (e.g., 52.34 ± 5.67%) suggests:

1. **Sampling temperature too high**: Try reducing `--temperature` (e.g., 0.7 instead of 1.0)
2. **Dataset too small**: More variance on smaller test sets
3. **Method inherently unstable**: May need deterministic components

**Solutions**:
- Use `--temperature 0.7` or even `0.5` for more stable sampling
- Use greedy decoding: `--temperature 0.0` (but less diversity)
- Increase number of seeds to get more reliable estimates

### Q3: Can I cancel a running array job?

**Yes**:

```bash
# Cancel entire array job
scancel JOB_ID

# Cancel specific seed (e.g., seed 3)
scancel JOB_ID_3
```

### Q4: How to resume if some seeds failed?

**Check which seeds completed**:

```bash
ls experiments/result/multiseed/okvqa_tcvm_seed*.csv

# Output shows:
# seed0.csv, seed1.csv, seed4.csv
# Missing: seed2, seed3
```

**Rerun missing seeds only**:

```bash
# Submit only seeds 2 and 3
sbatch --array=2,3 run_okvqa_tcvm_multiseed.slurm
```

### Q5: What if I get "image not found" errors?

**Issue**: The log shows many "Warning: Image not found" messages

**Cause**: Using wrong image folder (e.g., `val2017` instead of `val2014` for OK-VQA)

**Solution**: Check dataset requirements:

| Dataset | Image Folder |
|---------|--------------|
| OK-VQA | `data/images/coco/val2014` |
| A-OKVQA | `data/images/coco/val2014` |
| InfoSeek | `data/images/infoseek` |
| ViQuAE | `data/images/viquae` |
| E-VQA | `data/images/inaturalist2021` |

Fix in SLURM script:
```bash
--image-folder ../../data/images/coco/val2014  # Not val2017!
```

---

## Best Practices

### 1. Always Use Version Control for Scripts

Before submitting large jobs:

```bash
git add slurm_jobs/run_okvqa_tcvm_multiseed.slurm
git commit -m "Add multi-seed script for OK-VQA TCVM"
```

### 2. Test with 1 Seed First

Before running 5 seeds, verify with 1:

```bash
# Test seed 0 only
sbatch --array=0 run_okvqa_tcvm_multiseed.slurm

# If successful, run all seeds
sbatch run_okvqa_tcvm_multiseed.slurm
```

### 3. Monitor Resource Usage

Check GPU memory and utilization:

```bash
# SSH into compute node
ssh gpu-a100-XXX

# Monitor GPU
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

### 4. Keep Logs Organized

Use meaningful output names:

```bash
#SBATCH --output=logs/%x_seed%a_%j.out
#SBATCH --error=logs/%x_seed%a_%j.err

# %x = job name
# %a = array task ID (seed)
# %j = job ID
```

### 5. Document Hyperparameters

Keep a record of settings used:

```bash
# Create a README in results directory
cat > experiments/result/multiseed/README.md <<EOF
# Multi-Seed Results

## Configuration
- Model: LLaVA-1.5-7B
- Method: TCVM
- Seeds: 0, 1, 2, 3, 4
- Hyperparameters:
  - tcvm_topk: 20
  - tcvm_alpha: 1.0
  - tcvm_beta: 0.7
  - temperature: 1.0
  - top_p: 1.0

## Submission
- Date: 2026-03-22
- Job ID: 23001
- Submitted by: yuhaol11
EOF
```

---

## Summary

**Workflow**:

1. **Submit multi-seed job**:
   ```bash
   sbatch run_okvqa_tcvm_multiseed.slurm
   ```

2. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f logs/tcvm_okvqa_seed0_*.out
   ```

3. **Aggregate results**:
   ```bash
   python scripts/aggregate_multiseed_results.py \
       --dataset okvqa --method tcvm --seeds 0 1 2 3 4
   ```

4. **Report statistics**:
   ```
   TCVM on OK-VQA: 52.34 ± 1.23%
   ```

**Key Files**:
- Multi-seed SLURM script: `slurm_jobs/run_okvqa_tcvm_multiseed.slurm`
- Aggregation script: `scripts/aggregate_multiseed_results.py`
- Results: `experiments/result/multiseed/*.csv`
- Statistics: `experiments/result/*_stats.json`

---

## References

- ALFAR paper: An et al., "Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation", arXiv:2406.12718, 2024
- SLURM job arrays: https://slurm.schedmd.com/job_array.html
- HuggingFace set_seed: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.set_seed
