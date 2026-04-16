#!/bin/bash
# Submit all baseline experiments for A-OKVQA

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to the project root (one level up from slurm_jobs)
cd "$SCRIPT_DIR/.." || exit 1

echo "Working directory: $(pwd)"
echo "Submitting all baseline experiments for A-OKVQA..."

# VCD - Visual Contrastive Decoding
echo "Submitting VCD..."
sbatch slurm_jobs/run_baseline_vcd_aokvqa.slurm

# CD - Contrastive Decoding
echo "Submitting CD..."
sbatch slurm_jobs/run_baseline_cd_aokvqa.slurm

# CAD - Context-Aware Decoding
echo "Submitting CAD..."
sbatch slurm_jobs/run_baseline_cad_aokvqa.slurm

# AdaCAD - Adaptive Context-Aware Decoding
echo "Submitting AdaCAD..."
sbatch slurm_jobs/run_baseline_adacad_aokvqa.slurm

# Entropy - Entropy-based Decoding
echo "Submitting Entropy..."
sbatch slurm_jobs/run_baseline_entropy_aokvqa.slurm

# COIECD - Contextual Information-Entropy Constraint Decoding
echo "Submitting COIECD..."
sbatch slurm_jobs/run_baseline_coiecd_aokvqa.slurm

echo "All baseline jobs submitted!"
echo "Monitor with: squeue -u \$USER"
