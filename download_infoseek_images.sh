#!/bin/bash
# Download InfoSeek images from Hugging Face OVEN dataset

set -e

echo "========================================="
echo "Downloading InfoSeek Images from Hugging Face"
echo "========================================="

# Install huggingface_hub if not available
source ALFAR/bin/activate
pip install -q huggingface_hub

# Create target directory
mkdir -p data/images/infoseek_images_new

cd data/images/infoseek_images_new

echo "Downloading OVEN image dataset..."
echo "This may take a while (dataset is large)..."

# Download using huggingface-cli
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

# Download the OVEN dataset
snapshot_download(
    repo_id="ychenNLP/oven",
    repo_type="dataset",
    local_dir="oven_data",
    allow_patterns=["*.tar", "*.jsonl"],
    max_workers=4
)

print("Download complete!")
EOF

echo "========================================="
echo "Extracting images..."

# Find and extract tar files
find oven_data -name "*.tar" -exec tar -xf {} \;

echo "========================================="
echo "Download and extraction complete!"
ls -lh

cd /data/gpfs/projects/punim2075/ALFAR
