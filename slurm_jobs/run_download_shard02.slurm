#!/bin/bash
#SBATCH --job-name=download_shard02
#SBATCH --partition=sapphire
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/gpfs/projects/punim2075/ALFAR/logs/download_shard02_%j.out
#SBATCH --error=/data/gpfs/projects/punim2075/ALFAR/logs/download_shard02_%j.err

echo "========================================="
echo "Downloading OVEN shard02.tar"
echo "Starting at: $(date)"
echo "========================================="

# Navigate to project directory
cd /data/gpfs/projects/punim2075/ALFAR

# Activate environment
source ALFAR/bin/activate

# Clean up any incomplete downloads
rm -f data/images/oven_temp/.cache/huggingface/download/*shard02*.incomplete

# Download shard02
echo "Starting shard02 download..."
huggingface-cli download ychenNLP/oven \
    --repo-type dataset \
    --local-dir data/images/oven_temp \
    --include "shard02.tar"

# Check if download succeeded
if [ -f "data/images/oven_temp/shard02.tar" ]; then
    size=$(du -h data/images/oven_temp/shard02.tar | cut -f1)
    echo "========================================="
    echo "Download complete!"
    echo "File: shard02.tar"
    echo "Size: $size"
    echo "Finished at: $(date)"
    echo "========================================="
else
    echo "ERROR: Download failed!"
    exit 1
fi
