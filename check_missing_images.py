#!/usr/bin/env python3
"""
Check which InfoSeek images are missing
"""
import json
import os
from pathlib import Path
from collections import defaultdict

# Load InfoSeek questions
print("Loading InfoSeek questions...")
with open('data/eval_data/mc/infoseek_mc.json', 'r') as f:
    questions = json.load(f)

# Get all needed images
needed_images = set()
for q in questions:
    img = q['image']
    # Remove extension to get base name
    base = img.split('.')[0]
    needed_images.add(base)

print(f"Total unique images needed: {len(needed_images)}")

# Check in infoseek_images directory
image_dir = Path('data/images/infoseek_images')
found_images = set()
missing_images = []

# Walk through all subdirectories
for img_base in needed_images:
    # Check both .jpg and .JPEG extensions
    found = False
    for ext in ['.jpg', '.JPEG', '.JPG', '.jpeg']:
        # Search in all subdirectories
        for subdir in image_dir.glob('*'):
            if subdir.is_dir():
                img_path = subdir / f"{img_base}{ext}"
                if img_path.exists():
                    found_images.add(img_base)
                    found = True
                    break
            if found:
                break
        if found:
            break

    if not found:
        missing_images.append(img_base)

print(f"\nFound: {len(found_images)}/{len(needed_images)}")
print(f"Missing: {len(missing_images)}/{len(needed_images)}")

if missing_images:
    print(f"\nFirst 20 missing images:")
    for img in missing_images[:20]:
        print(f"  - {img}")

    # Save full list
    with open('missing_infoseek_images.txt', 'w') as f:
        for img in sorted(missing_images):
            f.write(f"{img}\n")
    print(f"\nFull list saved to: missing_infoseek_images.txt")
else:
    print("\nAll InfoSeek images are present! ✓")
