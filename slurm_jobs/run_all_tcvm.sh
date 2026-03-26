#!/bin/bash
# Master script to submit all TCVM evaluation jobs
# Usage: bash run_all_tcvm.sh [dataset1 dataset2 ...]
# If no arguments provided, submits all datasets

echo "========================================="
echo "TCVM Evaluation Job Submission"
echo "========================================="

# Change to slurm_jobs directory
cd "$(dirname "$0")"

# Define all datasets
ALL_DATASETS=("infoseek" "viquae" "aokvqa" "okvqa" "evqa")

# Use command line arguments if provided, otherwise use all datasets
if [ $# -eq 0 ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$@")
fi

echo "Datasets to run: ${DATASETS[*]}"
echo "========================================="

# Track submitted jobs
SUBMITTED_JOBS=()

# Submit jobs for each dataset
for dataset in "${DATASETS[@]}"; do
    script="run_${dataset}_tcvm.slurm"

    if [ -f "$script" ]; then
        echo "Submitting: $script"
        JOB_ID=$(sbatch "$script" | awk '{print $4}')

        if [ $? -eq 0 ]; then
            echo "  ✓ Job $JOB_ID submitted successfully"
            SUBMITTED_JOBS+=("$JOB_ID:$dataset")
        else
            echo "  ✗ Failed to submit $script"
        fi
    else
        echo "  ✗ Script not found: $script"
    fi
    echo ""
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "Total jobs submitted: ${#SUBMITTED_JOBS[@]}"

if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "Job IDs:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        IFS=':' read -r job_id dataset <<< "$job"
        echo "  - $dataset: $job_id"
    done

    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Check specific job output:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        IFS=':' read -r job_id dataset <<< "$job"
        echo "  tail -f ../logs/tcvm_${dataset}_${job_id}.out"
    done
fi

echo "========================================="
