# TCVM-KAR Experiments - Running

**Submission Date**: March 25, 2026
**Status**: All experiments submitted and queued

## Submitted Jobs

All 5 TCVM-KAR experiments have been submitted to the SLURM queue:

| Job ID   | Dataset   | Status  | Expected Time | Result File |
|----------|-----------|---------|---------------|-------------|
| 23132552 | InfoSeek  | Pending | ~24 hours     | `experiments/result/infoseek_tcvm_results.jsonl` |
| 23132553 | ViQuAE    | Pending | ~24 hours     | `experiments/result/viquae_tcvm_results.jsonl` |
| 23132554 | A-OKVQA   | Pending | ~24 hours     | `experiments/result/aokvqa_tcvm_results.csv` |
| 23132555 | OK-VQA    | Pending | ~24 hours     | `experiments/result/okvqa_tcvm_results.csv` |
| 23132556 | EVQA      | Pending | ~24 hours     | `experiments/result/evqa_tcvm_results.json` |

## TCVM-KAR Configuration

All experiments use the following TCVM-KAR settings:

```bash
--use_tcvm                    # Enable TCVM-KAR
--tcvm_topk 20                # Top-K tokens to mask
--tcvm_alpha 1.0              # Contrastive penalty weight
--tcvm_beta 0.7               # Plausibility threshold (APC)
--tcvm_mask_strategy zero     # Masking strategy
--seed 0                      # Random seed
```

### Knowledge-Aware Router (KAR)
The upgraded TCVM implementation automatically:
- Extracts visual AND context attention weights
- Computes routing metric: λ_t = Σ(visual_attn) / (Σ(visual_attn) + Σ(context_attn))
- Dynamically routes to mask visual tokens (λ_t > 0.5) OR context tokens (λ_t ≤ 0.5)
- Falls back to visual-only TCVM if context indices are unavailable

## Monitoring Jobs

### Quick Check
```bash
# Check job queue
squeue -u $USER

# Check all job statuses
bash scripts/monitor_tcvm_kar_jobs.sh
```

### Watch Real-time Progress
```bash
# Watch job queue (updates every 10 seconds)
watch -n 10 'squeue -u $USER'

# Tail specific job output
tail -f logs/tcvm_infoseek_23132552.out
tail -f logs/tcvm_viquae_23132553.out
tail -f logs/tcvm_aokvqa_23132554.out
tail -f logs/tcvm_okvqa_23132555.out
tail -f logs/tcvm_evqa_23132556.out
```

### Check Completed Results
```bash
# List all result files
ls -lh experiments/result/*tcvm_results.*

# Check result file sizes
du -h experiments/result/*tcvm_results.*
```

## Job Management

### Cancel Jobs
```bash
# Cancel specific job
scancel 23132552

# Cancel all your jobs
scancel -u $USER
```

### Check Job Details
```bash
# Get detailed job info
scontrol show job 23132552

# Check job accounting (after completion)
sacct -j 23132552 --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

## After Completion

### Evaluate Results
Once jobs complete, run evaluation scripts:

```bash
# A-OKVQA
python evaluation/eval_okvqa.py --dataset aokvqa --preds experiments/result/aokvqa_tcvm_results.csv

# OK-VQA
python evaluation/eval_okvqa.py --dataset okvqa --preds experiments/result/okvqa_tcvm_results.csv

# InfoSeek
python evaluation/eval_mc.py --dataset infoseek --preds experiments/result/infoseek_tcvm_results.jsonl

# ViQuAE
python evaluation/eval_mc.py --dataset viquae --preds experiments/result/viquae_tcvm_results.jsonl

# EVQA
python evaluation/eval_evqa.py --preds experiments/result/evqa_tcvm_results.json
```

### Generate Summary Report
```bash
# Aggregate all TCVM-KAR results
python scripts/aggregate_tcvm_kar_results.py
```

## Expected Outcomes

### What to Compare
After completion, compare TCVM-KAR against:
1. **Original TCVM** (visual-only masking)
2. **ALFAR baseline** (attention-based routing)
3. **Standard generation** (no hallucination mitigation)

### Key Metrics
- **Accuracy**: Overall correctness
- **Routing Analysis**: Distribution of λ_t values
  - How often is vision dominant vs. context dominant?
  - Does routing correlate with question type?
- **Hallucination Reduction**: Improved grounding metrics

### Analysis Scripts
```bash
# Analyze routing patterns (to be created)
python scripts/analyze_kar_routing.py --results experiments/result/aokvqa_tcvm_results.csv

# Compare with baselines
python scripts/compare_tcvm_variants.py
```

## Troubleshooting

### Job Stuck in Pending
- **Priority**: Other jobs may have higher priority
- **Resources**: GPU nodes may be fully utilized
- **Check**: `squeue -u $USER` for reason column

### Job Failed
```bash
# Check error log
cat logs/tcvm_DATASET_JOBID.err

# Check output log for Python errors
grep -i "error\|exception\|traceback" logs/tcvm_DATASET_JOBID.out
```

### Out of Memory (OOM)
- Current allocation: 64GB RAM, 1x A100 GPU
- If OOM occurs, increase `--mem` in SLURM script

### CUDA Errors
```bash
# Check CUDA availability in log
grep "CUDA available" logs/tcvm_DATASET_JOBID.out

# Verify GPU allocation
grep "Allocated GPUs" logs/tcvm_DATASET_JOBID.out
```

## Notes

### Differences from Previous TCVM Runs
The current TCVM-KAR experiments use the upgraded implementation with:
- ✅ Context masking capabilities
- ✅ Dynamic routing based on attention
- ✅ Automatic context index calculation
- ✅ Fallback to visual-only TCVM

Previous results in `experiments/result/*tcvm*` were generated with visual-only TCVM.

### Log File Locations
- **STDOUT**: `logs/tcvm_DATASET_JOBID.out`
- **STDERR**: `logs/tcvm_DATASET_JOBID.err`
- **Results**: `experiments/result/DATASET_tcvm_results.*`

### Resource Allocation
Each job requests:
- **Partition**: gpu-a100
- **Nodes**: 1
- **GPUs**: 1x A100 (40GB VRAM)
- **CPUs**: 8 cores
- **RAM**: 64GB
- **Time**: 24 hours

## Contact & Support

If jobs fail or produce unexpected results:
1. Check log files for errors
2. Verify CUDA and PyTorch compatibility
3. Ensure data files are accessible
4. Check virtual environment activation

For questions about TCVM-KAR implementation, see:
- `docs/TCVM_KAR_UPGRADE_SUMMARY.md`
- `experiments/eval/tcvm_utils.py`
- `experiments/eval/vcd_sample.py`
