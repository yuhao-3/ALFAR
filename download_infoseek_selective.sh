#!/bin/bash
#
# Selective download of InfoSeek images from OVEN dataset
# Only downloads the 3 shards needed for 2,855 images
#

set -e

# Configuration
ROOT_DIR="/data/gpfs/projects/punim2075/ALFAR"
TARGET_DIR="$ROOT_DIR/data/images/infoseek_images"
TEMP_DIR="$ROOT_DIR/data/images/oven_temp"

# OVEN Hugging Face repository
HF_REPO="ychenNLP/oven"
HF_URL="https://huggingface.co/datasets/$HF_REPO/resolve/main"

# Shards we need based on analysis
# Shard 00: 33 images
# Shard 04: 869 images
# Shard 05: 1,953 images

echo "=================================="
echo "InfoSeek Selective Image Download"
echo "=================================="
echo ""
echo "Target: 2,855 images from 3 shards"
echo "  - Shard 00: 33 images"
echo "  - Shard 04: 869 images"
echo "  - Shard 05: 1,953 images"
echo ""

# Create directories
mkdir -p "$TARGET_DIR"
mkdir -p "$TEMP_DIR"

cd "$ROOT_DIR"
source ALFAR/bin/activate

# Function to download and extract a shard
download_shard() {
    local shard_num=$1
    local shard_padded=$(printf "%02d" $shard_num)

    echo ""
    echo "=================================="
    echo "Processing Shard $shard_padded"
    echo "=================================="

    # Create shard target directory
    mkdir -p "$TARGET_DIR/$shard_padded"

    # Try different possible file patterns for OVEN dataset
    # The exact structure depends on how the dataset is organized
    local patterns=(
        "images/shard_${shard_padded}.tar"
        "images/${shard_padded}.tar"
        "oven_${shard_padded}.tar"
        "shard${shard_padded}.tar"
    )

    local downloaded=0
    for pattern in "${patterns[@]}"; do
        echo "Trying to download: $pattern"

        if wget -q --spider "${HF_URL}/${pattern}" 2>/dev/null; then
            echo "✅ Found: $pattern"
            echo "Downloading..."

            wget -c "${HF_URL}/${pattern}" -O "$TEMP_DIR/shard_${shard_padded}.tar"

            if [ $? -eq 0 ]; then
                echo "✅ Downloaded successfully"
                echo "Extracting to $TARGET_DIR/$shard_padded/"

                tar -xf "$TEMP_DIR/shard_${shard_padded}.tar" -C "$TARGET_DIR/$shard_padded/" 2>/dev/null || true

                # Count extracted images
                local count=$(find "$TARGET_DIR/$shard_padded/" -type f \( -name "*.jpg" -o -name "*.JPEG" -o -name "*.png" \) | wc -l)
                echo "✅ Extracted $count images"

                # Clean up tar file
                rm -f "$TEMP_DIR/shard_${shard_padded}.tar"

                downloaded=1
                break
            fi
        fi
    done

    if [ $downloaded -eq 0 ]; then
        echo "❌ Could not find shard $shard_padded in any expected location"
        echo "   Will try using huggingface-cli instead..."

        # Fallback to huggingface-cli
        huggingface-cli download "$HF_REPO" \
            --repo-type dataset \
            --local-dir "$TEMP_DIR/hf_shard_${shard_padded}" \
            --include "**/*${shard_padded}*.tar*" \
            2>&1 | grep -v "Fetching"

        # Extract any tar files found
        find "$TEMP_DIR/hf_shard_${shard_padded}" -name "*.tar*" -exec tar -xf {} -C "$TARGET_DIR/$shard_padded/" \; 2>/dev/null || true

        # Count extracted images
        local count=$(find "$TARGET_DIR/$shard_padded/" -type f \( -name "*.jpg" -o -name "*.JPEG" -o -name "*.png" \) | wc -l)
        echo "Extracted $count images using huggingface-cli"
    fi
}

# Download each required shard
echo "Starting downloads..."
echo ""

read -p "Download Shard 00 (33 images)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_shard 0
fi

read -p "Download Shard 04 (869 images)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_shard 4
fi

read -p "Download Shard 05 (1,953 images)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_shard 5
fi

# Count total images
echo ""
echo "=================================="
echo "Download Summary"
echo "=================================="

total_images=$(find "$TARGET_DIR" -type f \( -name "*.jpg" -o -name "*.JPEG" -o -name "*.png" \) | wc -l)
echo "Total images in target directory: $total_images"

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Download complete!"
echo "Images location: $TARGET_DIR"
