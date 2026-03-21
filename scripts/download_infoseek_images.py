#!/usr/bin/env python3
"""
Download only the specific OVEN images needed for InfoSeek evaluation.
This avoids downloading the entire 243GB OVEN dataset.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import subprocess

def load_required_images(json_file):
    """Load the list of required images from infoseek_mc.json"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = set()
    for item in data:
        images.add(item['image'])

    return sorted(images)

def group_by_shard(images):
    """Group images by their OVEN shard prefix (first 2 digits)"""
    shards = defaultdict(list)

    for img in images:
        # Extract shard number from oven_XXNNNNNN.jpg -> XX
        if img.startswith('oven_'):
            shard_id = img.split('_')[1][:2]
            shards[shard_id].append(img)

    return dict(shards)

def download_oven_shard(shard_id, output_dir):
    """Download a specific OVEN shard from Hugging Face"""
    print(f"\n{'='*60}")
    print(f"Downloading OVEN shard {shard_id}...")
    print(f"{'='*60}")

    shard_dir = output_dir / f"shard_{shard_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Try to download the shard tar file
    # The OVEN dataset structure may vary, let's try common patterns
    patterns = [
        f"oven_{shard_id}.tar",
        f"shard_{shard_id}.tar",
        f"{shard_id}.tar",
        f"oven_images_{shard_id}.tar.gz",
    ]

    cmd = [
        "huggingface-cli", "download",
        "ychenNLP/oven",
        "--repo-type", "dataset",
        "--local-dir", str(shard_dir),
        "--include", f"*{shard_id}*"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Shard {shard_id} downloaded successfully")
        return shard_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download shard {shard_id}")
        print(f"Error: {e.stderr}")
        return None

def extract_images(shard_dir, required_images, target_dir):
    """Extract only the required images from a shard"""
    print(f"\nExtracting {len(required_images)} images from {shard_dir.name}...")

    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    # Find tar files in shard directory
    tar_files = list(shard_dir.glob("*.tar")) + list(shard_dir.glob("*.tar.gz"))

    if not tar_files:
        print(f"⚠️  No tar files found in {shard_dir}")
        return 0

    for tar_file in tar_files:
        print(f"Processing {tar_file.name}...")

        # Create a list of images to extract
        image_list = ' '.join([f"*/{img}" for img in required_images])

        # Extract specific images
        cmd = f"tar -xf {tar_file} -C {target_dir} {image_list} 2>/dev/null || true"
        subprocess.run(cmd, shell=True)

    # Count extracted images
    for img in required_images:
        if (target_dir / img).exists():
            extracted += 1

    print(f"✅ Extracted {extracted}/{len(required_images)} images")
    return extracted

def main():
    # Paths
    root = Path(__file__).parent
    json_file = root / "data/eval_data/mc/infoseek_mc.json"
    output_dir = root / "data/images/infoseek_download_temp"
    target_dir = root / "data/images/infoseek_images"

    print("="*60)
    print("InfoSeek Selective Image Downloader")
    print("="*60)

    # Load required images
    print(f"\n1. Loading required images from {json_file}...")
    required_images = load_required_images(json_file)
    print(f"   Found {len(required_images)} unique images needed")

    # Group by shard
    print(f"\n2. Analyzing shard distribution...")
    shards = group_by_shard(required_images)
    for shard_id, imgs in sorted(shards.items()):
        print(f"   Shard {shard_id}: {len(imgs)} images")

    # Check what we already have
    print(f"\n3. Checking existing images...")
    existing = 0
    missing_by_shard = defaultdict(list)

    for shard_id, imgs in shards.items():
        for img in imgs:
            img_path = target_dir / shard_id / img
            if img_path.exists():
                existing += 1
            else:
                # Try alternate extensions
                base = img.rsplit('.', 1)[0]
                found = False
                for ext in ['.jpg', '.JPEG', '.png', '.PNG']:
                    if (target_dir / shard_id / f"{base}{ext}").exists():
                        existing += 1
                        found = True
                        break
                if not found:
                    missing_by_shard[shard_id].append(img)

    print(f"   ✅ Already have: {existing}/{len(required_images)} images")
    print(f"   ❌ Missing: {len(required_images) - existing} images")

    if len(required_images) - existing == 0:
        print("\n🎉 All required images are already downloaded!")
        return

    # Download missing shards
    print(f"\n4. Downloading missing shards...")
    for shard_id, missing_imgs in sorted(missing_by_shard.items()):
        if not missing_imgs:
            continue

        print(f"\n   Shard {shard_id}: {len(missing_imgs)} images needed")
        response = input(f"   Download shard {shard_id}? (y/n): ")

        if response.lower() != 'y':
            print(f"   Skipped shard {shard_id}")
            continue

        shard_dir = download_oven_shard(shard_id, output_dir)
        if shard_dir:
            extracted = extract_images(shard_dir, missing_imgs, target_dir / shard_id)
            print(f"   Extracted {extracted} images to {target_dir / shard_id}")

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)

if __name__ == "__main__":
    main()
