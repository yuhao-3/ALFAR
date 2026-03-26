#!/bin/bash
# Submit all TCVM-KAR multi-seed experiments (seeds 1-2, seed 0 already done)

echo "========================================="
echo "TCVM-KAR Multi-seed Job Submission"
echo "========================================="
echo "Submitting seeds 1-2 for all datasets"
echo "(Seed 0 already completed)"
echo "========================================="

# Submit array jobs for each dataset
echo "Submitting InfoSeek (seeds 1-2)..."
JOB_INFOSEEK=$(sbatch slurm_jobs/run_infoseek_tcvm_multiseed.slurm | awk '{print $4}')

echo "Submitting ViQuAE (seeds 1-2)..."
JOB_VIQUAE=$(sbatch slurm_jobs/run_viquae_tcvm_multiseed.slurm | awk '{print $4}')

echo "Submitting A-OKVQA (seeds 1-2)..."
JOB_AOKVQA=$(sbatch slurm_jobs/run_aokvqa_tcvm_multiseed.slurm | awk '{print $4}')

echo "Submitting OK-VQA (seeds 1-2)..."
JOB_OKVQA=$(sbatch slurm_jobs/run_okvqa_tcvm_multiseed.slurm | awk '{print $4}')

echo "Submitting E-VQA (seeds 1-2)..."
JOB_EVQA=$(sbatch slurm_jobs/run_evqa_tcvm_multiseed.slurm | awk '{print $4}')

echo "========================================="
echo "Summary"
echo "========================================="
echo "Submitted job arrays:"
echo "  - InfoSeek: $JOB_INFOSEEK (seeds 1-2)"
echo "  - ViQuAE: $JOB_VIQUAE (seeds 1-2)"
echo "  - A-OKVQA: $JOB_AOKVQA (seeds 1-2)"
echo "  - OK-VQA: $JOB_OKVQA (seeds 1-2)"
echo "  - E-VQA: $JOB_EVQA (seeds 1-2)"
echo ""
echo "Total: 10 jobs (5 datasets × 2 seeds)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "========================================="
