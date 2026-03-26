# Monitoring and Evaluation Guide

Complete guide for monitoring SLURM jobs and evaluating results for ALFAR and TCVM-KAR experiments.

## Table of Contents
- [Job Monitoring](#job-monitoring)
- [Progress Tracking](#progress-tracking)
- [Log Analysis](#log-analysis)
- [Result Validation](#result-validation)
- [Evaluation Scripts](#evaluation-scripts)
- [Multi-seed Analysis](#multi-seed-analysis)
- [Performance Comparison](#performance-comparison)

---

## Job Monitoring

### Basic Status Checks

#### Check Job Queue
```bash
# View all your jobs
squeue -u $USER

# Formatted output
squeue -u $USER --format="%.10i %.12j %.8T %.10M %.6D %R"
```

**Output columns**:
- `JOBID`: Job identifier
- `NAME`: Job name
- `ST`: State (R=Running, PD=Pending, CG=Completing, F=Failed)
- `TIME`: Elapsed time
- `NODES`: Number of nodes
- `NODELIST(REASON)`: Node name or reason for pending

#### Check Specific Job Details
```bash
# Detailed job information
scontrol show job [JOBID]

# Job start/end times
squeue -u $USER --start

# Job accounting info
sacct -j [JOBID] --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

### Live Monitoring

#### Watch Job Queue (Auto-refresh)
```bash
# Update every 10 seconds
watch -n 10 'squeue -u $USER'

# Update every 30 seconds with formatting
watch -n 30 'squeue -u $USER --format="%.10i %.15j %.8T %.10M %.10l %R"'
```

**Stop watching**: Press `Ctrl+C`

#### Monitor Multiple Jobs
```bash
# Custom monitoring script
bash scripts/monitor_tcvm_kar_jobs.sh

# Watch specific jobs
watch -n 10 'squeue -j 23153037,23153038,23153039,23153040,23153041'
```

### Job States Reference

| State | Code | Meaning | Action |
|-------|------|---------|--------|
| PENDING | PD | Waiting to start | Wait or check reason |
| RUNNING | R | Currently executing | Monitor progress |
| COMPLETING | CG | Finishing up | Wait for completion |
| COMPLETED | CD | Finished successfully | Check results |
| FAILED | F | Terminated with error | Check logs |
| CANCELLED | CA | Manually cancelled | Resubmit if needed |
| TIMEOUT | TO | Exceeded time limit | Increase time, resubmit |
| OUT_OF_MEMORY | OOM | Ran out of memory | Increase RAM, resubmit |

---

## Progress Tracking

### Follow Real-time Output

#### Tail Output Logs
```bash
# Follow output log (STDOUT)
tail -f logs/tcvm_infoseek_23153037.out

# Follow error log (STDERR)
tail -f logs/tcvm_infoseek_23153037.err

# Follow both simultaneously (in separate terminals)
tail -f logs/tcvm_infoseek_23153037.out &
tail -f logs/tcvm_infoseek_23153037.err
```

#### Monitor Progress Bars
```bash
# Watch for tqdm progress indicators
tail -f logs/tcvm_infoseek_23153037.out | grep --line-buffered "%"
```

**Example output**:
```
Processing: 45%|████▌     | 1350/3007 [1:12:34<1:27:45, 3.18s/it]
```

### Check Job Progress Without Tailing

```bash
# Last 20 lines
tail -20 logs/tcvm_infoseek_23153037.out

# Search for specific progress indicators
grep -i "processing\|progress\|%" logs/tcvm_infoseek_23153037.out | tail -5

# Check for completion
grep -i "completed\|finished\|done" logs/tcvm_infoseek_23153037.out
```

### Estimate Time to Completion

```bash
# Check elapsed time
squeue -j [JOBID] --format="%.10i %.8T %.10M %.10l"

# Extract processing speed from logs
grep "it/s\|s/it" logs/tcvm_infoseek_23153037.out | tail -1
```

**Calculate remaining time**:
```
Examples/s = 3.18 seconds per example
Remaining = (Total - Current) × Examples/s
          = (3007 - 1350) × 3.18
          = 5268 seconds ≈ 1h 28m
```

---

## Log Analysis

### Search Logs for Specific Information

#### Find Errors
```bash
# Search for errors in output
grep -i "error\|exception\|failed" logs/tcvm_infoseek_23153037.out

# Search for errors in stderr
grep -i "error\|exception\|traceback" logs/tcvm_infoseek_23153037.err

# Show context (5 lines before/after)
grep -i -C 5 "error" logs/tcvm_infoseek_23153037.err
```

#### Find TCVM-KAR Routing Decisions
```bash
# If --tcvm_debug is enabled
grep "KAR Router" logs/tcvm_infoseek_23153037.out

# Count visual vs context masking
grep "KAR Router.*VISUAL" logs/tcvm_infoseek_23153037.out | wc -l
grep "KAR Router.*CONTEXT" logs/tcvm_infoseek_23153037.out | wc -l
```

#### Find Accuracy/Performance Metrics
```bash
# Search for accuracy
grep -i "accuracy\|score\|metric" logs/tcvm_infoseek_23153037.out

# Extract final metrics
grep -A 10 "Final results\|Evaluation complete" logs/tcvm_infoseek_23153037.out
```

### Analyze Job Performance

#### Check Resource Usage
```bash
# Memory usage (MaxRSS = peak RAM)
sacct -j [JOBID] --format=JobID,MaxRSS,MaxVMSize,AveRSS

# GPU utilization (if job is running)
ssh [node] nvidia-smi

# Example:
ssh spartan-gpgpu121 nvidia-smi
```

#### Check Time Usage
```bash
# Elapsed time and time limit
sacct -j [JOBID] --format=JobID,Elapsed,Timelimit,State

# Start and end times
sacct -j [JOBID] --format=JobID,Start,End,Elapsed
```

### Common Log Patterns

#### Successful Start
```bash
grep -A 5 "Starting TCVM evaluation" logs/tcvm_infoseek_23153037.out
```
**Expected**:
```
Starting TCVM evaluation on InfoSeek dataset...
TCVM Configuration:
  - Top-K: 20 visual tokens
  - Alpha: 1.0 (contrastive penalty)
  - Beta: 0.7 (plausibility threshold)
```

#### Successful Completion
```bash
tail -20 logs/tcvm_infoseek_23153037.out
```
**Expected**:
```
Results saved to experiments/result/infoseek_tcvm_results.jsonl
Evaluation completed successfully
End time: [timestamp]
```

---

## Result Validation

### Check Result Files Exist

```bash
# List all result files
ls -lh experiments/result/

# Check specific method results
ls -lh experiments/result/*tcvm*
ls -lh experiments/result/*alfar*

# Find empty result files (problematic)
find experiments/result -name "*_results.*" -size 0
```

### Validate Result File Contents

#### For CSV Files (OKVQA, AOKVQA)
```bash
# Check file size
ls -lh experiments/result/aokvqa_tcvm_results.csv

# Count lines (should match dataset size)
wc -l experiments/result/aokvqa_tcvm_results.csv

# View first few lines
head -5 experiments/result/aokvqa_tcvm_results.csv

# Verify columns
head -1 experiments/result/aokvqa_tcvm_results.csv
```

**Expected columns**: `question_id,answer,prediction` or similar

#### For JSONL Files (InfoSeek, ViQuAE)
```bash
# Count entries
wc -l experiments/result/infoseek_tcvm_results.jsonl

# Validate JSON format
python3.9 -c "
import json
with open('experiments/result/infoseek_tcvm_results.jsonl') as f:
    lines = [json.loads(line) for line in f]
print(f'Valid JSONL with {len(lines)} entries')
print(f'Keys: {list(lines[0].keys())}')
"

# View first entry
head -1 experiments/result/infoseek_tcvm_results.jsonl | python3.9 -m json.tool
```

#### For JSON Files (EVQA)
```bash
# Validate JSON
python3.9 -m json.tool experiments/result/evqa_tcvm_results.json > /dev/null && echo "Valid JSON"

# Check structure
python3.9 -c "
import json
data = json.load(open('experiments/result/evqa_tcvm_results.json'))
print(f'Type: {type(data)}')
print(f'Length: {len(data)}')
if isinstance(data, dict):
    print(f'Sample keys: {list(data.keys())[:5]}')
"
```

### Quick Result Summary

```bash
# Count predictions per dataset
echo "Result file sizes:"
for file in experiments/result/*_results.*; do
    if [ -f "$file" ]; then
        name=$(basename $file)
        size=$(wc -l < "$file" 2>/dev/null || echo "N/A")
        echo "  $name: $size lines"
    fi
done
```

---

## Evaluation Scripts

### Running Evaluation

#### OKVQA / A-OKVQA
```bash
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# A-OKVQA
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/aokvqa_tcvm_results.csv

# OK-VQA
python evaluation/eval_okvqa.py \
    --dataset okvqa \
    --preds experiments/result/okvqa_tcvm_results.csv
```

**Output**: Accuracy score
```
Accuracy: 65.32%
```

#### InfoSeek / ViQuAE (Multiple Choice)
```bash
# InfoSeek
python evaluation/eval_mc.py \
    --dataset infoseek \
    --preds experiments/result/infoseek_tcvm_results.jsonl

# ViQuAE
python evaluation/eval_mc.py \
    --dataset viquae \
    --preds experiments/result/viquae_tcvm_results.jsonl
```

**Output**: Accuracy score
```
Accuracy: 42.15%
```

#### E-VQA
```bash
python evaluation/eval_evqa.py \
    --preds experiments/result/evqa_tcvm_results.json
```

**Output**: Multiple metrics
```
Accuracy: 58.73%
F1 Score: 62.14%
```

### Batch Evaluation

#### Evaluate All TCVM-KAR Results
```bash
# Create batch evaluation script
for dataset in aokvqa okvqa; do
    echo "Evaluating $dataset..."
    python evaluation/eval_okvqa.py \
        --dataset $dataset \
        --preds experiments/result/${dataset}_tcvm_results.csv \
        > logs/tcvm_${dataset}_metrics.txt
done

for dataset in infoseek viquae; do
    echo "Evaluating $dataset..."
    python evaluation/eval_mc.py \
        --dataset $dataset \
        --preds experiments/result/${dataset}_tcvm_results.jsonl \
        > logs/tcvm_${dataset}_metrics.txt
done

echo "Evaluating EVQA..."
python evaluation/eval_evqa.py \
    --preds experiments/result/evqa_tcvm_results.json \
    > logs/tcvm_evqa_metrics.txt
```

#### Check All Metrics
```bash
# View all metric files
for file in logs/*tcvm*metrics.txt; do
    echo "=== $(basename $file) ==="
    cat $file
    echo ""
done
```

### Save Evaluation Outputs

```bash
# Save to timestamped log
timestamp=$(date +%Y%m%d_%H%M%S)
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/aokvqa_tcvm_results.csv \
    | tee logs/aokvqa_tcvm_eval_${timestamp}.txt

# Save all evaluations
bash scripts/generate_missing_metrics.sh
```

---

## Multi-seed Analysis

### Check Multi-seed Results

```bash
# List all seed results for a dataset/method
ls -lh experiments/result/aokvqa_alfar_seed*.csv

# Check which seeds are complete
for seed in 0 1 2 3 4; do
    file="experiments/result/aokvqa_alfar_seed${seed}.csv"
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo "Seed $seed: ✓ ($(wc -l < $file) lines)"
    else
        echo "Seed $seed: ✗ MISSING or EMPTY"
    fi
done
```

### Evaluate All Seeds

```bash
# Evaluate each seed
for seed in 0 1 2 3 4; do
    echo "Evaluating seed $seed..."
    python evaluation/eval_okvqa.py \
        --dataset aokvqa \
        --preds experiments/result/aokvqa_alfar_seed${seed}.csv \
        > logs/alfar_aokvqa_seed${seed}_metrics.txt
done
```

### Calculate Average Across Seeds

```bash
# Extract accuracies
for seed in 0 1 2 3 4; do
    acc=$(grep "Accuracy:" logs/alfar_aokvqa_seed${seed}_metrics.txt | awk '{print $2}' | tr -d '%')
    echo "$acc"
done | awk '{sum+=$1; n++} END {print "Average: " sum/n "%"}'
```

### Use Aggregation Script

```bash
# Aggregate multi-seed results
python scripts/aggregate_multiseed_results.py \
    --method alfar \
    --dataset aokvqa \
    --seeds 5

# Or calculate all averages
python scripts/calculate_all_averages.py
```

**Example output**:
```
ALFAR A-OKVQA Results (5 seeds):
Seed 0: 65.32%
Seed 1: 65.18%
Seed 2: 65.45%
Seed 3: 65.21%
Seed 4: 65.39%
-------------------
Mean: 65.31%
Std:  0.10%
```

---

## Performance Comparison

### Compare Methods

```bash
# Compare TCVM-KAR vs ALFAR on same dataset
echo "=== A-OKVQA Comparison ==="
echo -n "ALFAR: "
grep "Accuracy:" logs/alfar_aokvqa_metrics.txt
echo -n "TCVM-KAR: "
grep "Accuracy:" logs/tcvm_aokvqa_metrics.txt
```

### Compare Across Datasets

```bash
# Create comparison table
echo "Method | InfoSeek | ViQuAE | A-OKVQA | OK-VQA | EVQA"
echo "-------|----------|--------|---------|--------|------"

for method in alfar tcvm; do
    echo -n "$method | "
    for dataset in infoseek viquae aokvqa okvqa evqa; do
        acc=$(grep "Accuracy:" logs/${method}_${dataset}_metrics.txt 2>/dev/null | awk '{print $2}' | tr -d '%')
        echo -n "${acc:-N/A} | "
    done
    echo ""
done
```

### View Baseline Comparisons

```bash
# Check documented baselines
cat docs/BASELINES.md

# View TCVM vs ALFAR comparison
cat docs/TCVM_VS_ALFAR_COMPARISON.md

# View experiment summaries
cat docs/MULTISEED_FINAL_SUMMARY.md
```

### Generate Comparison Report

```bash
# Custom comparison script
python scripts/compare_methods.py \
    --methods alfar tcvm \
    --datasets aokvqa okvqa infoseek viquae evqa \
    --output comparison_report.md
```

---

## Automated Monitoring Scripts

### Monitor TCVM-KAR Jobs

```bash
# Use provided monitoring script
bash scripts/monitor_tcvm_kar_jobs.sh
```

**Example output**:
```
=== TCVM-KAR Job Monitor ===
Time: 2026-03-25 16:00:00

Job 23153037 (InfoSeek): RUNNING
  Node: spartan-gpgpu121
  Elapsed: 00:45:23
  Progress: 45% (1350/3007)

Job 23153038 (ViQuAE): PENDING (Resources)
Job 23153039 (A-OKVQA): PENDING (Priority)
Job 23153040 (OK-VQA): PENDING (Priority)
Job 23153041 (EVQA): PENDING (Priority)
```

### Monitor Multi-seed Experiments

```bash
# Check multi-seed job status
bash scripts/monitor_multiseed_jobs.sh
```

### Create Custom Monitor

```bash
# Simple custom monitor script
#!/bin/bash
while true; do
    clear
    echo "=== Job Status ==="
    squeue -u $USER --format="%.10i %.15j %.8T %.10M %R"

    echo ""
    echo "=== Latest Log Activity ==="
    tail -3 logs/tcvm_*_23153037.out 2>/dev/null | grep "%"

    sleep 30
done
```

**Save as**: `scripts/my_monitor.sh`
**Run**: `bash scripts/my_monitor.sh`

---

## Quick Reference Commands

### Essential Monitoring
```bash
# Job status
squeue -u $USER

# Live follow
tail -f logs/tcvm_infoseek_23153037.out

# Check completion
tail -20 logs/tcvm_infoseek_23153037.out | grep -i "completed\|finished"
```

### Essential Evaluation
```bash
# Run evaluation
python evaluation/eval_okvqa.py --dataset aokvqa --preds experiments/result/aokvqa_tcvm_results.csv

# Check results
ls -lh experiments/result/*tcvm*

# View metrics
cat logs/tcvm_aokvqa_metrics.txt
```

### Essential Validation
```bash
# Check for empty files
find experiments/result -name "*tcvm*" -size 0

# Validate result count
wc -l experiments/result/aokvqa_tcvm_results.csv

# Check logs for errors
grep -i "error" logs/tcvm_aokvqa_*.err
```

---

## Best Practices

### 1. Regular Monitoring
- Check job status every few hours
- Monitor at least once daily for long-running jobs
- Set up email notifications for job completion (optional)

### 2. Log Management
- Periodically archive old logs: `mkdir -p logs/archive && mv logs/old_*.out logs/archive/`
- Keep recent logs for debugging
- Clean up empty or failed job logs

### 3. Result Backup
```bash
# Backup results before re-running
mkdir -p experiments/result/backup_$(date +%Y%m%d)
cp experiments/result/*tcvm* experiments/result/backup_$(date +%Y%m%d)/
```

### 4. Systematic Evaluation
- Evaluate immediately after job completion
- Save evaluation outputs to log files
- Document any anomalies or unexpected results

### 5. Progress Documentation
- Keep a running log of experiments
- Note any changes to parameters or configuration
- Record job IDs and completion dates

---

**Last Updated**: March 25, 2026
**Covers**: ALFAR, TCVM, TCVM-KAR on Spartan HPC
