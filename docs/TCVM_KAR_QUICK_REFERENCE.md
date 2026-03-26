# TCVM-KAR Quick Reference Card

## ✅ Current Status
**All 5 TCVM-KAR experiments submitted and queued!**

```
Job ID      Dataset      Status
--------    ---------    --------
23132552    InfoSeek     PENDING
23132553    ViQuAE       PENDING
23132554    A-OKVQA      PENDING
23132555    OK-VQA       PENDING
23132556    EVQA         PENDING
```

## 📊 Quick Monitoring

### One-line Status Check
```bash
squeue -u $USER
```

### Detailed Status
```bash
bash scripts/monitor_tcvm_kar_jobs.sh
```

### Watch Live (updates every 10s)
```bash
watch -n 10 'squeue -u $USER'
```

### Tail Specific Job
```bash
tail -f logs/tcvm_infoseek_23132552.out
tail -f logs/tcvm_aokvqa_23132554.out
# ... etc for other datasets
```

## 🎯 What's Different (TCVM → TCVM-KAR)

| Feature | Original TCVM | TCVM-KAR (New) |
|---------|---------------|----------------|
| Masking | Visual tokens only | Visual **OR** Context (adaptive) |
| Routing | None | Lambda_t metric (λ > 0.5 → visual, else context) |
| RAG Support | Penalizes context tokens | Intelligently masks relevant modality |
| Fallback | N/A | Auto-falls back to visual-only if no context |

## 📁 Key Files

### Code
- `experiments/eval/tcvm_utils.py` - TCVM-KAR utilities
- `experiments/eval/vcd_sample.py` - Router implementation
- `experiments/eval/test_tcvm.py` - Unit tests (all passing ✅)

### Documentation
- `docs/TCVM_KAR_UPGRADE_SUMMARY.md` - Implementation details
- `docs/TCVM_KAR_EXPERIMENTS_RUNNING.md` - Experiment guide
- `TCVM_KAR_QUICK_REFERENCE.md` - This file

### Scripts
- `slurm_jobs/run_all_tcvm.sh` - Submit all experiments
- `scripts/monitor_tcvm_kar_jobs.sh` - Monitor progress

## 🔬 After Jobs Complete

### 1. Check Results
```bash
ls -lh experiments/result/*tcvm_results.*
```

### 2. Run Evaluation
```bash
python evaluation/eval_okvqa.py --dataset aokvqa
python evaluation/eval_okvqa.py --dataset okvqa
python evaluation/eval_mc.py --dataset infoseek
python evaluation/eval_mc.py --dataset viquae
python evaluation/eval_evqa.py
```

### 3. Compare with Baselines
- ALFAR results: `experiments/result/*_alfar_*.csv`
- Original TCVM: Previous `*tcvm*` files (backed up if needed)
- TCVM-KAR: Current `*tcvm*` files

## 🛠️ Troubleshooting

### Jobs Not Starting
- **Reason**: Resource contention, check with `squeue`
- **Wait Time**: Can be several hours on busy cluster
- **Check Priority**: Look at REASON column in squeue

### Job Failed
```bash
# Check error log
cat logs/tcvm_DATASET_JOBID.err

# Search for errors in output
grep -i "error" logs/tcvm_DATASET_JOBID.out
```

### Cancel Jobs
```bash
scancel 23132552          # Cancel specific job
scancel -u $USER          # Cancel all your jobs
```

## 📈 Expected Timeline

- **Queue Time**: 0-4 hours (depends on cluster load)
- **Run Time per Job**: 12-24 hours
- **Total Time**: ~1-2 days for all jobs to complete

## 🎓 TCVM-KAR Parameters

```python
use_tcvm=True              # Enable TCVM-KAR
tcvm_topk=20               # Mask top-20 attended tokens
tcvm_alpha=1.0             # Contrastive penalty weight
tcvm_beta=0.7              # Plausibility threshold (APC)
tcvm_mask_strategy='zero'  # Masking strategy
tcvm_debug=False           # Set True to see routing decisions
```

## 📞 Quick Commands Cheat Sheet

```bash
# Monitor
squeue -u $USER                           # Check queue
bash scripts/monitor_tcvm_kar_jobs.sh     # Detailed status
watch -n 10 'squeue -u $USER'             # Live updates

# Logs
tail -f logs/tcvm_DATASET_JOBID.out       # Follow output
grep "KAR Router" logs/tcvm_*.out         # See routing decisions
grep "Accuracy" logs/tcvm_*.out           # Find results

# Results
ls -lh experiments/result/*tcvm*          # List result files
du -h experiments/result/*tcvm*           # Check sizes

# Cancel
scancel JOBID                             # Cancel one job
scancel -u $USER                          # Cancel all jobs

# Job Info
scontrol show job JOBID                   # Detailed job info
sacct -j JOBID                            # Accounting info
```

---

**Last Updated**: March 25, 2026, 00:02 AEDT
**All experiments submitted successfully! ✅**
