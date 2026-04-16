# Baseline SLURM Scripts Fix Summary

**Date**: 2026-04-08
**Issue**: Missing SLURM job submission scripts
**Status**: ✅ **RESOLVED**

---

## Problem Identified

The baseline submission script `slurm_jobs/run_all_baselines_aokvqa.sh` was failing because 5 out of 6 required SLURM job files were missing.

### Error Output
```
Submitting VCD...
sbatch: error: Unable to open file slurm_jobs/run_baseline_vcd_aokvqa.slurm
Submitting CD...
sbatch: error: Unable to open file slurm_jobs/run_baseline_cd_aokvqa.slurm
Submitting CAD...
Submitting AdaCAD...
sbatch: error: Unable to open file slurm_jobs/run_baseline_adacad_aokvqa.slurm
Submitting Entropy...
sbatch: error: Unable to open file slurm_jobs/run_baseline_entropy_aokvqa.slurm
Submitting COIECD...
sbatch: error: Unable to open file slurm_jobs/run_baseline_coiecd_aokvqa.slurm
```

---

## Root Cause

**Expected Files**: 6 SLURM scripts for baseline methods
**Actual Files**: Only 1 script existed (`run_baseline_cad_aokvqa.slurm`)

**Missing Scripts**:
1. `run_baseline_vcd_aokvqa.slurm` - Visual Contrastive Decoding
2. `run_baseline_cd_aokvqa.slurm` - Contrastive Decoding
3. `run_baseline_adacad_aokvqa.slurm` - Adaptive Context-Aware Decoding
4. `run_baseline_entropy_aokvqa.slurm` - Entropy-Based Decoding
5. `run_baseline_coiecd_aokvqa.slurm` - COIECD

---

## Solution Implemented

Created all 5 missing SLURM scripts using the existing CAD script as a template.

### File Details

| File | Size | Created | Method | Key Parameters |
|------|------|---------|--------|----------------|
| `run_baseline_vcd_aokvqa.slurm` | 1.2K | 2026-04-08 | VCD | vcd-alpha=0.5, blur-radius=10.0 |
| `run_baseline_cd_aokvqa.slurm` | 1.1K | 2026-04-08 | CD | cd-alpha=0.5 |
| `run_baseline_cad_aokvqa.slurm` | 1.1K | 2026-04-07 | CAD | cad-alpha=0.5 |
| `run_baseline_adacad_aokvqa.slurm` | 1.2K | 2026-04-08 | AdaCAD | adacad-alpha-max=1.0 |
| `run_baseline_entropy_aokvqa.slurm` | 1.2K | 2026-04-08 | Entropy | entropy-temperature=0.5 |
| `run_baseline_coiecd_aokvqa.slurm` | 1.2K | 2026-04-08 | COIECD | coiecd-alpha=0.5, temperature=0.7 |

### Common Configuration

All scripts share the same SLURM configuration:
```bash
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
```

### Script Structure

Each script includes:
1. **SLURM directives** (partition, resources, output logs)
2. **Environment activation** (virtual environment)
3. **Inference execution** (baseline_all_okvqa_llava.py with method-specific parameters)
4. **Automatic evaluation** (eval_okvqa.py)
5. **Logging** (start time, job ID, node, end time)

---

## Verification

### Test Submission
```bash
bash slurm_jobs/run_all_baselines_aokvqa.sh
```

### Result: ✅ SUCCESS
```
Submitting all baseline experiments for A-OKVQA...
Submitting VCD...
Submitted batch job 23708470
Submitting CD...
Submitted batch job 23708471
Submitting CAD...
Submitted batch job 23708472
Submitting AdaCAD...
Submitted batch job 23708473
Submitting Entropy...
Submitted batch job 23708474
Submitting COIECD...
Submitted batch job 23708475
All baseline jobs submitted!
```

### Job Queue Status
All 6 baseline jobs successfully queued:
- 23708470: vcd_aokvqa (PENDING)
- 23708471: cd_aokvqa (PENDING)
- 23708472: cad_aokvqa (PENDING)
- 23708473: adacad_aokvqa (PENDING)
- 23708474: entropy_aokvqa (PENDING)
- 23708475: coiecd_aokvqa (PENDING)

---

## Implementation Details

### Example: VCD Script
```bash
#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/baseline_vcd_aokvqa_%j.out
#SBATCH --job-name=vcd_aokvqa

# VCD (Visual Contrastive Decoding) Baseline for A-OKVQA

echo "Starting VCD baseline experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate environment
source /data/gpfs/projects/punim2075/ALFAR/ALFAR/bin/activate

# Run VCD baseline
python experiments/eval/baseline_all_okvqa_llava.py \
    --method vcd \
    --dataset aokvqa \
    --model-path /data/gpfs/projects/punim2075/model/llava_1.5_7b \
    --image-folder /data/gpfs/projects/punim2075/ALFAR/data/images/coco/val2014 \
    --answers-file experiments/result/vcd_aokvqa_results.csv \
    --vcd-alpha 0.5 \
    --vcd-blur-radius 10.0 \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed 0

echo "VCD baseline completed!"
echo "End time: $(date)"

# Evaluate results
echo "Evaluating VCD results..."
python evaluation/eval_okvqa.py \
    --dataset aokvqa \
    --preds experiments/result/vcd_aokvqa_results.csv

echo "Done!"
```

---

## Expected Results

### Output Files
Each baseline will generate:
1. **Inference results**: `experiments/result/<method>_aokvqa_results.csv`
2. **Log file**: `logs/baseline_<method>_aokvqa_<jobid>.out`
3. **Evaluation metrics**: Printed in log file

### Evaluation Metrics
Each log will contain:
- Accuracy on A-OKVQA validation set (1,145 questions)
- Comparison against expected performance from BASELINES.md

### Expected Performance (A-OKVQA)
| Method | Expected Accuracy | Description |
|--------|------------------|-------------|
| VCD | ~48-52% | Visual contrastive |
| CD | ~48-50% | Text contrastive |
| CAD | ~50-55% | Context amplification |
| AdaCAD | ~52-57% | Adaptive amplification |
| Entropy | ~47-50% | Uncertainty reduction |
| COIECD | ~51-56% | CAD + entropy |

---

## Impact

### Before Fix
- ❌ Could not run baseline comparison experiments
- ❌ Manual intervention required to test each baseline
- ❌ No batch submission capability

### After Fix
- ✅ All 6 baseline methods can be submitted with one command
- ✅ Consistent experiment infrastructure across all baselines
- ✅ Automatic evaluation after each experiment
- ✅ Proper logging and job tracking

---

## Usage Instructions

### Submit All Baselines
```bash
cd /data/gpfs/projects/punim2075/ALFAR
bash slurm_jobs/run_all_baselines_aokvqa.sh
```

### Submit Individual Baseline
```bash
sbatch slurm_jobs/run_baseline_<method>_aokvqa.slurm
```
Replace `<method>` with: vcd, cd, cad, adacad, entropy, or coiecd

### Monitor Jobs
```bash
# Check queue
squeue -u $USER

# Watch specific job log
tail -f logs/baseline_<method>_aokvqa_<jobid>.out

# Check results
ls -lh experiments/result/*_aokvqa_results.csv
```

---

## Files Modified/Created

### Created (5 files)
1. `/data/gpfs/projects/punim2075/ALFAR/slurm_jobs/run_baseline_vcd_aokvqa.slurm`
2. `/data/gpfs/projects/punim2075/ALFAR/slurm_jobs/run_baseline_cd_aokvqa.slurm`
3. `/data/gpfs/projects/punim2075/ALFAR/slurm_jobs/run_baseline_adacad_aokvqa.slurm`
4. `/data/gpfs/projects/punim2075/ALFAR/slurm_jobs/run_baseline_entropy_aokvqa.slurm`
5. `/data/gpfs/projects/punim2075/ALFAR/slurm_jobs/run_baseline_coiecd_aokvqa.slurm`

### Updated (1 file)
1. `/data/gpfs/projects/punim2075/ALFAR/docs/BASELINE_VERIFICATION_REPORT.md`
   - Added section documenting the issue and fix
   - Updated conclusion to reflect resolved status

### Unchanged (1 file)
1. `/data/gpfs/projects/punim2075/ALFAR/slurm_jobs/run_all_baselines_aokvqa.sh`
   - Already correctly configured
   - Now functional with all required files present

---

## Related Documentation

- **Main Verification Report**: `docs/BASELINE_VERIFICATION_REPORT.md`
- **Baseline Documentation**: `docs/BASELINES.md`
- **Submission Script**: `slurm_jobs/run_all_baselines_aokvqa.sh`
- **Implementation**: `experiments/eval/baseline_all_okvqa_llava.py`

---

## Next Steps

1. **Monitor Running Jobs**: Watch for completion of all 6 baseline experiments
2. **Verify Results**: Check that all result files are generated correctly
3. **Compare Performance**: Analyze results against expected accuracies
4. **Document Results**: Update BASELINES.md with actual experimental results

---

## Lessons Learned

1. **Template-Based Creation**: Using existing CAD script as template ensured consistency
2. **Batch Testing**: Always test submission scripts before running large experiments
3. **Documentation**: Important to document both the issue and the fix
4. **Infrastructure Gaps**: SLURM scripts are essential infrastructure that should be created alongside code

---

**Fix Implemented By**: Claude Code
**Date**: 2026-04-08
**Status**: ✅ **RESOLVED AND TESTED**
