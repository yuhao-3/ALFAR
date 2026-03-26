# TCVM-KAR Multi-Seed Experiments Status

**Last Updated**: March 25, 2026
**Purpose**: Run each model-dataset combination 3 times for statistical reliability
**Goal**: Report mean(SD) for all results

---

## Current Status

### LLaVA 1.5-7B Multi-Seed Experiments

**Submitted**: March 25, 2026, ~21:00 AEDT

| Job Array | Dataset   | Seeds  | Status      | Job IDs |
|-----------|-----------|--------|-------------|---------|
| 23175470  | InfoSeek  | 1-2    | RUNNING/PD  | Array job |
| 23175471  | ViQuAE    | 1-2    | RUNNING/PD  | Array job |
| 23175472  | A-OKVQA   | 1-2    | RUNNING/PD  | Array job |
| 23175473  | OK-VQA    | 1-2    | RUNNING/PD  | Array job |
| 23175474  | E-VQA     | 1-2    | RUNNING/PD  | Array job |

**Total**: 10 jobs (5 datasets × 2 seeds)
**Note**: Seed 0 already completed (March 25, 14:38-15:03)

---

## Results Location

All results saved in: `results/llava1.5/{dataset}/`

### Naming Convention

```
results/llava1.5/{dataset}/{dataset}_tcvm_seed{0,1,2}_results.{ext}
```

**Example**:
```
results/llava1.5/infoseek/infoseek_tcvm_seed0_results.jsonl  ✅
results/llava1.5/infoseek/infoseek_tcvm_seed1_results.jsonl  ⏳
results/llava1.5/infoseek/infoseek_tcvm_seed2_results.jsonl  ⏳
```

---

## Seed 0 Results (Already Complete)

| Dataset   | Accuracy | Correct/Total |
|-----------|----------|---------------|
| InfoSeek  | 57.23%   | 1,717/3,000   |
| ViQuAE    | 57.07%   | 1,716/3,007   |
| A-OKVQA   | 59.71%   | 684/1,145     |
| OK-VQA    | 60.66%   | 3,061/5,046   |
| E-VQA     | 35.97%   | 50/139        |

---

## Expected Results Format

After all seeds complete:

| Dataset   | Seed 0  | Seed 1 | Seed 2 | Mean (SD) |
|-----------|---------|--------|--------|-----------|
| InfoSeek  | 57.23%  | TBD    | TBD    | TBD       |
| ViQuAE    | 57.07%  | TBD    | TBD    | TBD       |
| A-OKVQA   | 59.71%  | TBD    | TBD    | TBD       |
| OK-VQA    | 60.66%  | TBD    | TBD    | TBD       |
| E-VQA     | 35.97%  | TBD    | TBD    | TBD       |

---

## Monitoring Commands

### Check Job Status
```bash
# All jobs
squeue -u $USER

# Specific array
squeue -j 23175470  # InfoSeek
squeue -j 23175471  # ViQuAE
# etc...
```

### Monitor Progress
```bash
# Watch queue
watch -n 10 'squeue -u $USER'

# Check specific seed
tail -f logs/tcvm_infoseek_seed1_*.out
tail -f logs/tcvm_infoseek_seed2_*.out
```

---

## Aggregation After Completion

### Aggregate Single Dataset
```bash
python scripts/aggregate_tcvm_multiseed.py --model llava1.5 --dataset infoseek
```

### Aggregate All Datasets
```bash
python scripts/aggregate_tcvm_multiseed.py --model llava1.5 --all
```

**Output**: Creates `accuracy_multiseed_summary.txt` in each dataset folder with:
- Individual seed results
- Mean (SD) across seeds
- Configuration details

---

## Timeline

| Milestone | Time | Status |
|-----------|------|--------|
| Seed 0 complete | Mar 25, 14:38-15:03 | ✅ Done |
| Seeds 1-2 submitted | Mar 25, ~21:00 | ✅ Done |
| Seeds 1-2 expected completion | ~24-48 hours | ⏳ Running |
| Aggregate results | After all complete | ⏳ Pending |
| Generate final table | After aggregation | ⏳ Pending |

**Expected Completion**: March 26-27, 2026

---

## Configuration (All Seeds)

Consistent TCVM-KAR configuration across all seeds:

```python
use_tcvm = True
tcvm_topk = 20              # Top-K tokens to mask
tcvm_alpha = 1.0            # Contrastive penalty weight
tcvm_beta = 0.7             # Plausibility threshold (APC)
tcvm_mask_strategy = 'zero' # Zero-out masked tokens
# seed = 0, 1, 2            # Only difference
```

---

## Next Steps

### Immediate
1. ⏳ Wait for seeds 1-2 to complete (~24-48 hours)
2. ⏳ Monitor progress with `squeue -u $USER`
3. ⏳ Check logs if any jobs fail

### After Completion
1. Run aggregation script for all datasets
2. Generate final results table with mean(SD)
3. Update TCVM_KAR_RESULTS.md with multi-seed results
4. Plan multi-seed experiments for other models

---

## Other Models (To Do)

After LLaVA 1.5 multi-seed complete, run 3 seeds for:

1. **InstructBLIP** (InfoSeek, ViQuAE)
2. **Shikra** (InfoSeek, ViQuAE)
3. **MiniGPT-4** (InfoSeek, ViQuAE)
4. **LLaVA-NEXT** (InfoSeek, ViQuAE) - if available
5. **Qwen2.5-VL** (InfoSeek, ViQuAE) - if available

**Total additional experiments**: 5 models × 2 datasets × 3 seeds = 30 jobs

---

## File Checklist

Current files created:

- ✅ `slurm_jobs/run_infoseek_tcvm_multiseed.slurm`
- ✅ `slurm_jobs/run_viquae_tcvm_multiseed.slurm`
- ✅ `slurm_jobs/run_aokvqa_tcvm_multiseed.slurm`
- ✅ `slurm_jobs/run_okvqa_tcvm_multiseed.slurm`
- ✅ `slurm_jobs/run_evqa_tcvm_multiseed.slurm`
- ✅ `slurm_jobs/run_all_tcvm_multiseed.sh`
- ✅ `scripts/aggregate_tcvm_multiseed.py`
- ✅ Seed 0 results renamed with `_seed0` suffix

---

## Troubleshooting

### If Jobs Fail

1. Check error logs:
   ```bash
   cat logs/tcvm_{dataset}_seed{1,2}_*.err
   ```

2. Check for CUDA/memory issues:
   ```bash
   grep -i "cuda\|memory" logs/tcvm_{dataset}_seed{1,2}_*.err
   ```

3. Resubmit failed seed:
   ```bash
   # Edit SLURM script to run only failed seed
   sbatch slurm_jobs/run_{dataset}_tcvm_multiseed.slurm
   ```

### If Results Don't Match

Check that seed is actually being set:
```bash
grep "seed" logs/tcvm_{dataset}_seed{1,2}_*.out
```

---

**Document Version**: 1.0
**Last Updated**: March 25, 2026, ~21:00 AEDT
**Status**: Multi-seed experiments running
