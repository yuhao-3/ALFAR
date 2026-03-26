# Multi-Seed Experiment Status

**Submission Date**: 2026-03-22 22:59:54 AEDT
**Submitted By**: yuhao3
**Total Jobs**: 10 (50 array tasks)

---

## Submitted Jobs

### TCVM Experiments (5 datasets × 5 seeds each)

| Dataset | Job ID | Seeds | Status | Results Path |
|---------|--------|-------|--------|--------------|
| OK-VQA | 23027056 | 0-4 | PENDING | `experiments/result/multiseed/okvqa_tcvm_seed{0-4}.csv` |
| A-OKVQA | 23027057 | 0-4 | PENDING | `experiments/result/multiseed/aokvqa_tcvm_seed{0-4}.csv` |
| InfoSeek | 23027058 | 0-4 | PENDING | `experiments/result/multiseed/infoseek_tcvm_seed{0-4}.jsonl` |
| ViQuAE | 23027059 | 0-4 | PENDING | `experiments/result/multiseed/viquae_tcvm_seed{0-4}.jsonl` |
| E-VQA | 23027060 | 0-4 | PENDING | `experiments/result/multiseed/evqa_tcvm_seed{0-4}.json` |

### ALFAR Experiments (5 datasets × 5 seeds each)

| Dataset | Job ID | Seeds | Status | Results Path |
|---------|--------|-------|--------|--------------|
| OK-VQA | 23027061 | 0-4 | PENDING | `experiments/result/multiseed/okvqa_alfar_seed{0-4}.csv` |
| A-OKVQA | 23027062 | 0-4 | PENDING | `experiments/result/multiseed/aokvqa_alfar_seed{0-4}.csv` |
| InfoSeek | 23027063 | 0-4 | PENDING | `experiments/result/multiseed/infoseek_alfar_seed{0-4}.jsonl` |
| ViQuAE | 23027064 | 0-4 | PENDING | `experiments/result/multiseed/viquae_alfar_seed{0-4}.jsonl` |
| E-VQA | 23027065 | 0-4 | PENDING | `experiments/result/multiseed/evqa_alfar_seed{0-4}.json` |

---

## Monitoring

### Check Job Status

```bash
# All jobs
squeue -u yuhao3

# Specific job (e.g., OK-VQA TCVM)
squeue -j 23027056

# Use monitoring script
bash scripts/monitor_multiseed_jobs.sh
```

### View Logs

```bash
# Live log (replace JOBID with actual job ID from squeue)
tail -f logs/tcvm_okvqa_seed0_23027056_0.out

# Check all logs for a dataset
ls -lht logs/tcvm_okvqa_seed*
```

### Check Results

```bash
# List completed results
ls -lht experiments/result/multiseed/

# Check specific dataset
ls experiments/result/multiseed/okvqa_tcvm_seed*.csv
```

---

## Expected Timeline

| Phase | Time | Description |
|-------|------|-------------|
| **Queuing** | 0-60 min | Jobs wait for GPU availability |
| **Execution** | ~24 hours | Each dataset runs on 1 GPU |
| **Total** | ~24-48 hours | All jobs complete (depends on queue) |

**Note**: Since we have 50 tasks (10 jobs × 5 seeds) competing for GPUs, some will run in parallel while others wait in queue.

---

## After Completion

### Step 1: Verify All Seeds Completed

```bash
# Check TCVM results (should show 5 files per dataset)
ls experiments/result/multiseed/okvqa_tcvm_seed*.csv
ls experiments/result/multiseed/aokvqa_tcvm_seed*.csv
ls experiments/result/multiseed/infoseek_tcvm_seed*.jsonl
ls experiments/result/multiseed/viquae_tcvm_seed*.jsonl
ls experiments/result/multiseed/evqa_tcvm_seed*.json

# Check ALFAR results
ls experiments/result/multiseed/okvqa_alfar_seed*.csv
ls experiments/result/multiseed/aokvqa_alfar_seed*.csv
ls experiments/result/multiseed/infoseek_alfar_seed*.jsonl
ls experiments/result/multiseed/viquae_alfar_seed*.jsonl
ls experiments/result/multiseed/evqa_alfar_seed*.json
```

### Step 2: Aggregate Results

Run aggregation for each dataset:

```bash
cd /data/gpfs/projects/punim2075/ALFAR

# TCVM aggregation
python scripts/aggregate_multiseed_results.py --dataset okvqa --method tcvm --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset aokvqa --method tcvm --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset infoseek --method tcvm --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset viquae --method tcvm --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset evqa --method tcvm --seeds 0 1 2 3 4

# ALFAR aggregation
python scripts/aggregate_multiseed_results.py --dataset okvqa --method alfar --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset aokvqa --method alfar --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset infoseek --method alfar --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset viquae --method alfar --seeds 0 1 2 3 4
python scripts/aggregate_multiseed_results.py --dataset evqa --method alfar --seeds 0 1 2 3 4
```

**Output files**:
- `experiments/result/tcvm_{dataset}_stats.json`
- `experiments/result/alfar_{dataset}_stats.json`

### Step 3: Create Results Table

**Expected Output Format**:

```
Dataset: OK-VQA
============================================================
Method    Accuracy (%)      Std (%)
------------------------------------------------------------
TCVM      52.34             ±1.23
ALFAR     54.67             ±0.98

Dataset: A-OKVQA
============================================================
Method    Accuracy (%)      Std (%)
------------------------------------------------------------
TCVM      48.23             ±1.45
ALFAR     51.89             ±1.12

...
```

---

## Troubleshooting

### Some Seeds Failed?

**Check which completed**:
```bash
# Find missing seeds
ls experiments/result/multiseed/okvqa_tcvm_seed*.csv
# If seed2 and seed3 are missing:
```

**Rerun failed seeds**:
```bash
cd slurm_jobs
sbatch --array=2,3 run_okvqa_tcvm_multiseed.slurm
```

### Job Stuck in Queue?

**Check queue position**:
```bash
squeue -u yuhao3 -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %Q"
```

**Check partition availability**:
```bash
sinfo -p gpu-a100
```

### Out of Memory Errors?

**Check log for OOM**:
```bash
grep -i "out of memory\|OOM\|killed" logs/tcvm_okvqa_seed0_*.out
```

**Solution**: Reduce batch size or request more memory in SLURM script:
```bash
#SBATCH --mem=128G  # Increase from 64G
```

### Image Not Found Warnings?

**Check image paths** in the log:
```bash
grep "Warning: Image not found" logs/tcvm_okvqa_seed0_*.out | head
```

**Common issues**:
- OK-VQA/A-OKVQA: Should use `val2014`, not `val2017`
- Check if image directories exist:
  ```bash
  ls -ld data/images/coco/val2014
  ls -ld data/images/infoseek
  ls -ld data/images/viquae
  ls -ld data/images/inaturalist2021
  ```

---

## Emergency Commands

### Cancel All Jobs

```bash
scancel -u yuhao3
```

### Cancel Specific Method

```bash
# Cancel all TCVM jobs
scancel 23027056 23027057 23027058 23027059 23027060

# Cancel all ALFAR jobs
scancel 23027061 23027062 23027063 23027064 23027065
```

### Cancel Specific Dataset

```bash
# Cancel OK-VQA only (both TCVM and ALFAR)
scancel 23027056 23027061
```

---

## File Structure

```
ALFAR/
├── slurm_jobs/
│   ├── run_okvqa_tcvm_multiseed.slurm
│   ├── run_aokvqa_tcvm_multiseed.slurm
│   ├── run_infoseek_tcvm_multiseed.slurm
│   ├── run_viquae_tcvm_multiseed.slurm
│   ├── run_evqa_tcvm_multiseed.slurm
│   ├── run_okvqa_alfar_multiseed.slurm
│   ├── run_aokvqa_alfar_multiseed.slurm
│   ├── run_infoseek_alfar_multiseed.slurm
│   ├── run_viquae_alfar_multiseed.slurm
│   ├── run_evqa_alfar_multiseed.slurm
│   └── submit_all_multiseed.sh
│
├── experiments/result/multiseed/
│   ├── okvqa_tcvm_seed{0-4}.csv
│   ├── aokvqa_tcvm_seed{0-4}.csv
│   ├── infoseek_tcvm_seed{0-4}.jsonl
│   ├── viquae_tcvm_seed{0-4}.jsonl
│   ├── evqa_tcvm_seed{0-4}.json
│   ├── okvqa_alfar_seed{0-4}.csv
│   ├── aokvqa_alfar_seed{0-4}.csv
│   ├── infoseek_alfar_seed{0-4}.jsonl
│   ├── viquae_alfar_seed{0-4}.jsonl
│   └── evqa_alfar_seed{0-4}.json
│
├── experiments/result/
│   ├── tcvm_okvqa_stats.json
│   ├── tcvm_aokvqa_stats.json
│   ├── tcvm_infoseek_stats.json
│   ├── tcvm_viquae_stats.json
│   ├── tcvm_evqa_stats.json
│   ├── alfar_okvqa_stats.json
│   ├── alfar_aokvqa_stats.json
│   ├── alfar_infoseek_stats.json
│   ├── alfar_viquae_stats.json
│   └── alfar_evqa_stats.json
│
├── logs/
│   ├── tcvm_okvqa_seed{0-4}_*.out
│   ├── tcvm_aokvqa_seed{0-4}_*.out
│   ├── tcvm_infoseek_seed{0-4}_*.out
│   ├── tcvm_viquae_seed{0-4}_*.out
│   ├── tcvm_evqa_seed{0-4}_*.out
│   ├── alfar_okvqa_seed{0-4}_*.out
│   ├── alfar_aokvqa_seed{0-4}_*.out
│   ├── alfar_infoseek_seed{0-4}_*.out
│   ├── alfar_viquae_seed{0-4}_*.out
│   ├── alfar_evqa_seed{0-4}_*.out
│   ├── tcvm_{dataset}_seed{0-4}_metrics.txt
│   ├── alfar_{dataset}_seed{0-4}_metrics.txt
│   └── multiseed_jobs_20260322_225954.txt
│
└── scripts/
    ├── aggregate_multiseed_results.py
    └── monitor_multiseed_jobs.sh
```

---

## Summary

- **10 jobs** submitted (5 TCVM + 5 ALFAR)
- **50 array tasks** total (10 jobs × 5 seeds each)
- **5 datasets**: OK-VQA, A-OKVQA, InfoSeek, ViQuAE, E-VQA
- **2 methods**: TCVM, ALFAR
- **Estimated completion**: 24-48 hours

**Next check**: Tomorrow (2026-03-23) to see first results

**Contact**: yuhao3 (job owner)
