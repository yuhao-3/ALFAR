#!/bin/bash
# Master script to run all motivation experiments
#
# This script submits jobs for:
# 1. No-context inference on InfoSeek
# 2. No-context inference on A-OKVQA
# 3. Bucket analysis (runs after inference completes)

echo "========================================="
echo "ALFAR Motivation Experiment Submission"
echo "========================================="
echo ""

cd /data/gpfs/projects/punim2075/ALFAR/slurm_jobs

# Submit no-context inference jobs
echo "Submitting no-context inference jobs..."
INFOSEEK_JOB=$(sbatch --parsable run_no_context_infoseek.slurm)
AOKVQA_JOB=$(sbatch --parsable run_no_context_aokvqa.slurm)

echo "  ✓ InfoSeek no-context job: $INFOSEEK_JOB"
echo "  ✓ A-OKVQA no-context job: $AOKVQA_JOB"
echo ""

# Submit bucket analysis job with dependency on inference jobs
echo "Submitting bucket analysis job (waits for inference to complete)..."
ANALYSIS_JOB=$(sbatch --parsable --dependency=afterok:$INFOSEEK_JOB:$AOKVQA_JOB run_bucket_analysis.slurm)
echo "  ✓ Bucket analysis job: $ANALYSIS_JOB"
echo ""

echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Job status:"
echo "  - InfoSeek no-context: $INFOSEEK_JOB"
echo "  - A-OKVQA no-context: $AOKVQA_JOB"
echo "  - Bucket analysis: $ANALYSIS_JOB (depends on above)"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: /data/gpfs/projects/punim2075/ALFAR/logs/"
echo ""
echo "Expected completion:"
echo "  - No-context inference: ~4-6 hours"
echo "  - Bucket analysis: ~10-15 minutes"
echo ""
echo "Final outputs will be in:"
echo "  - results/bucket_analysis/"
echo "  - figures/"
