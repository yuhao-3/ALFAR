#!/bin/bash
#SBATCH --job-name=download_missing_infoseek
#SBATCH --account=punim2075
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/download_missing_infoseek_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/download_missing_infoseek_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR

# Run the download script
bash scripts/download_missing_infoseek.sh

echo "========================================="
echo "End time: $(date)"
