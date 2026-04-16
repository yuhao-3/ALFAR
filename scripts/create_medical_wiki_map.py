"""
Create wiki_map files for medical Wikipedia
Matches the format of data/wiki/wiki_map.npy and wiki_map_17k.npy
"""

import numpy as np
import argparse
from pathlib import Path

def create_wiki_map(wiki_file, output_file):
    """
    Create wiki_map from wiki_with_image file

    Format: {index: page_id, ...}
    where index is 0, 1, 2, ... and page_id is the Wikipedia page ID
    """
    print(f"Loading wiki data from {wiki_file}...")
    wiki_data = np.load(wiki_file, allow_pickle=True).item()

    print(f"Creating wiki_map for {len(wiki_data)} articles...")

    # Create mapping: index -> page_id
    wiki_map = {}
    for idx, page_id in enumerate(sorted(wiki_data.keys(), key=lambda x: int(x))):
        wiki_map[idx] = page_id

    print(f"Saving wiki_map to {output_file}...")
    np.save(output_file, wiki_map)

    print(f"✓ Created wiki_map with {len(wiki_map)} entries")
    return wiki_map

def main():
    parser = argparse.ArgumentParser(description='Create wiki_map files for medical Wikipedia')
    parser.add_argument('--wiki-file', type=str,
                       default='data/wiki/medical_wiki_with_image.npy',
                       help='Input wiki_with_image.npy file')
    parser.add_argument('--output', type=str,
                       default='data/wiki/medical_wiki_map.npy',
                       help='Output wiki_map file')
    args = parser.parse_args()

    wiki_file = Path(args.wiki_file)
    if not wiki_file.exists():
        print(f"Error: Wiki file not found: {wiki_file}")
        return

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    wiki_map = create_wiki_map(wiki_file, output_file)

    print(f"\n✓ Wiki map files created successfully!")
    print(f"  Input:  {wiki_file}")
    print(f"  Output: {output_file}")
    print(f"  Entries: {len(wiki_map)}")
    print(f"\nFormat: {{index: page_id, ...}}")
    print(f"Sample entries:")
    for i in range(min(5, len(wiki_map))):
        print(f"  {i}: {wiki_map[i]}")

if __name__ == '__main__':
    main()
