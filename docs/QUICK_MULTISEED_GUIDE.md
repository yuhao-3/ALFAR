# Quick Multi-Seed Experiment Guide

## TL;DR

Run experiments with 5 seeds to compute mean ± std like in the paper.

---

## 1. Submit Multi-Seed Job

```bash
cd /data/gpfs/projects/punim2075/ALFAR/slurm_jobs

# OK-VQA with TCVM (runs seeds 0-4 in parallel)
sbatch run_okvqa_tcvm_multiseed.slurm
```

**What this does**:
- Runs the same experiment 5 times (seeds 0, 1, 2, 3, 4)
- Saves results to: `experiments/result/multiseed/okvqa_tcvm_seed{0,1,2,3,4}.csv`
- Saves metrics to: `logs/tcvm_okvqa_seed{0,1,2,3,4}_metrics.txt`

---

## 2. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live output (replace JOBID with your job ID)
tail -f logs/tcvm_okvqa_seed0_JOBID.out

# Check which seeds completed
ls experiments/result/multiseed/okvqa_tcvm_seed*.csv
```

---

## 3. Aggregate Results

After all 5 seeds complete:

```bash
cd /data/gpfs/projects/punim2075/ALFAR

python scripts/aggregate_multiseed_results.py \
    --dataset okvqa \
    --method tcvm \
    --seeds 0 1 2 3 4
```

**Output**:
```
============================================================
Results for TCVM on OKVQA
============================================================

Metric               Mean       Std        N     Values
------------------------------------------------------------
accuracy             0.5234     ±0.0123     5     [0.5201, 0.5245, 0.5198, 0.5267, 0.5259]

============================================================
LaTeX-friendly format:
------------------------------------------------------------
accuracy: $52.34 \pm 1.23$
============================================================
```

---

## 4. Report in Paper

```
OK-VQA Results:
- TCVM:  52.34 ± 1.23%
```

---

## Troubleshooting

### Some seeds failed?

Check which completed:
```bash
ls experiments/result/multiseed/okvqa_tcvm_seed*.csv
```

Rerun missing seeds (e.g., 2 and 3):
```bash
sbatch --array=2,3 run_okvqa_tcvm_multiseed.slurm
```

### "Image not found" errors?

Check the log:
```bash
grep "Warning: Image not found" logs/tcvm_okvqa_seed0_*.out | head -5
```

Fix image folder path in SLURM script:
- OK-VQA/A-OKVQA: `../../data/images/coco/val2014` (not val2017!)
- InfoSeek: `../../data/images/infoseek`
- ViQuAE: `../../data/images/viquae`

---

## Available Multi-Seed Scripts

Current:
- `run_okvqa_tcvm_multiseed.slurm` - OK-VQA with TCVM

To create for other datasets:
1. Copy `run_okvqa_tcvm_multiseed.slurm`
2. Change `--dataset` and `--image-folder`
3. Update job name and output paths

---

## Commands at a Glance

| Action | Command |
|--------|---------|
| **Submit job** | `sbatch run_okvqa_tcvm_multiseed.slurm` |
| **Check status** | `squeue -u $USER` |
| **View log** | `tail -f logs/tcvm_okvqa_seed0_*.out` |
| **List results** | `ls experiments/result/multiseed/` |
| **Aggregate** | `python scripts/aggregate_multiseed_results.py --dataset okvqa --method tcvm` |
| **Cancel job** | `scancel JOBID` |
| **Rerun seed** | `sbatch --array=2 run_okvqa_tcvm_multiseed.slurm` |

---

## See Full Guide

For detailed explanation, see: `docs/MULTISEED_EXPERIMENTS.md`
