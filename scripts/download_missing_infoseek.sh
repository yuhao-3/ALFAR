#!/bin/bash
# Download missing InfoSeek images from OVEN dataset shards 00, 04, 05

set -e

echo "========================================="
echo "Downloading Missing InfoSeek Images"
echo "Shard 00: 33 images"
echo "Shard 04: 869 images"
echo "Shard 05: 1,953 images"
echo "Total: 2,855 missing images"
echo "========================================="

cd /data/gpfs/projects/punim2075/ALFAR

# Check if authenticated
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "ERROR: Not authenticated with Hugging Face!"
    echo "Please run: huggingface-cli login"
    echo "Then request access to: https://huggingface.co/datasets/ychenNLP/oven"
    exit 1
fi

echo "✓ Authenticated"

# Remove corrupted tar files
echo "Removing corrupted tar files..."
rm -f data/images/oven_download/.cache/huggingface/download/shard*.tar

TARGET_DIR="data/images/infoseek_images"
TEMP_DIR="data/images/oven_temp"
mkdir -p "$TARGET_DIR" "$TEMP_DIR"

# Download specific shards
echo "Downloading shard 00, 04, 05..."
huggingface-cli download ychenNLP/oven \
    --repo-type dataset \
    --local-dir "$TEMP_DIR" \
    --include "shard00.tar" "shard04.tar" "shard05.tar"

# Extract shards
for shard in 00 04 05; do
    echo "Extracting shard${shard}.tar..."
    mkdir -p "$TARGET_DIR/$shard"

    if [ -f "$TEMP_DIR/shard${shard}.tar" ]; then
        tar -xf "$TEMP_DIR/shard${shard}.tar" -C "$TARGET_DIR/$shard/" || echo "Warning: Some files may not have extracted"
        echo "✓ Shard $shard extracted"
    else
        echo "⚠ Shard${shard}.tar not found"
    fi
done

# Count images
total=$(find "$TARGET_DIR" -type f \( -name "*.jpg" -o -name "*.JPEG" -o -name "*.png" \) 2>/dev/null | wc -l)

echo "========================================="
echo "Extraction complete!"
echo "Total images: $total"
echo "========================================="

# Check which images are still missing
echo "Checking for remaining missing images..."
still_missing=0
found=0

while IFS= read -r img_id; do
    if find "$TARGET_DIR" -name "${img_id}.*" 2>/dev/null | grep -q .; then
        ((found++))
    else
        ((still_missing++))
    fi
done < missing_infoseek_images.txt

echo "Found: $found images"
echo "Still missing: $still_missing images"
echo "========================================="
