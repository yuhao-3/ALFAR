#!/bin/bash
#SBATCH --job-name=infoseek_download
#SBATCH --account=punim2075
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/infoseek_download_%j.out
#SBATCH --error=logs/infoseek_download_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR
source ALFAR/bin/activate

# Install huggingface-cli if needed
pip install -q huggingface-hub

TARGET_DIR="data/images/infoseek_images"
mkdir -p "$TARGET_DIR"

echo "Downloading OVEN dataset shards for InfoSeek..."
echo "Target: shards 00, 04, 05 (approximately 2,855 missing images)"
echo "========================================="

# Download specific shards from OVEN dataset
huggingface-cli download ychenNLP/oven \
    --repo-type dataset \
    --local-dir data/images/oven_download \
    --include "*.tar"

echo "Extracting downloaded shards..."

# Extract shards 00, 04, 05
for shard_num in 00 04 05; do
    echo "Processing shard $shard_num..."
    mkdir -p "$TARGET_DIR/$shard_num"

    # Find and extract matching tar files
    find data/images/oven_download -name "*${shard_num}*.tar" 2>/dev/null | while read tarfile; do
        echo "  Extracting: $(basename $tarfile)"
        tar -xf "$tarfile" -C "$TARGET_DIR/$shard_num/" 2>/dev/null || true
    done
done

# Count total images
total=$(find "$TARGET_DIR" -type f \( -name "*.jpg" -o -name "*.JPEG" -o -name "*.png" \) | wc -l)

echo "========================================="
echo "Download complete!"
echo "Total images in infoseek_images: $total"
echo "Location: $TARGET_DIR"
echo "End time: $(date)"
echo "========================================="

# Check for missing images
if [ -f missing_infoseek_images.txt ]; then
    echo "Checking remaining missing images..."
    missing_count=0
    while read img_id; do
        # Check if image exists
        found=$(find "$TARGET_DIR" -name "${img_id}.*" 2>/dev/null | wc -l)
        if [ $found -eq 0 ]; then
            ((missing_count++))
        fi
    done < missing_infoseek_images.txt
    echo "Still missing: $missing_count images"
fi
