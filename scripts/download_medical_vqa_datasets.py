"""
Download Medical VQA Datasets (VQA-RAD and PathVQA)
These datasets can be integrated with the medical Wikipedia knowledge base
"""

import os
import json
from pathlib import Path
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_vqarad(output_dir):
    """
    Download VQA-RAD dataset from Hugging Face

    Dataset: flaviagiammarino/vqa-rad
    - 2,248 question-answer pairs
    - 315 radiology images
    - Topics: radiology, medical imaging
    """
    print("\n" + "="*60)
    print("Downloading VQA-RAD (Radiology VQA)")
    print("="*60)

    output_path = Path(output_dir) / 'vqarad'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download from Hugging Face
        dataset = load_dataset('flaviagiammarino/vqa-rad')

        # Save splits
        for split in dataset.keys():
            print(f"\nProcessing {split} split...")
            split_data = dataset[split]

            # Save as JSON
            output_file = output_path / f'{split}.json'
            data_list = []

            for idx, item in enumerate(tqdm(split_data, desc=f"Processing {split}")):
                data_list.append({
                    'question_id': idx,
                    'image_id': item.get('image_id', f'img_{idx}'),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'question_type': item.get('question_type', 'open'),
                    'answer_type': item.get('answer_type', 'OPEN')
                })

            with open(output_file, 'w') as f:
                json.dump(data_list, f, indent=2)

            print(f"  Saved {len(data_list)} samples to {output_file}")

            # Save images if available
            images_dir = output_path / 'images'
            images_dir.mkdir(exist_ok=True)

            if 'image' in split_data.features:
                print(f"  Saving images...")
                for idx, item in enumerate(tqdm(split_data, desc="Saving images")):
                    image = item['image']
                    image_id = item.get('image_id', f'img_{idx}')
                    image_path = images_dir / f'{image_id}.jpg'
                    image.save(image_path)

        # Save dataset info
        info = {
            'dataset': 'VQA-RAD',
            'description': 'Visual Question Answering on Radiology Images',
            'total_questions': sum(len(dataset[split]) for split in dataset.keys()),
            'splits': list(dataset.keys()),
            'source': 'https://huggingface.co/datasets/flaviagiammarino/vqa-rad',
            'license': 'CC0 1.0 Universal'
        }

        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ VQA-RAD downloaded successfully!")
        print(f"  Location: {output_path}")
        print(f"  Total questions: {info['total_questions']}")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading VQA-RAD: {e}")
        return False

def download_pathvqa(output_dir):
    """
    Download PathVQA dataset from Hugging Face

    Dataset: flaviagiammarino/path-vqa
    - 32,799 questions
    - 4,998 pathology images
    - Topics: pathology, histopathology
    """
    print("\n" + "="*60)
    print("Downloading PathVQA (Pathology VQA)")
    print("="*60)

    output_path = Path(output_dir) / 'pathvqa'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download from Hugging Face
        dataset = load_dataset('flaviagiammarino/path-vqa')

        # Save splits
        for split in dataset.keys():
            print(f"\nProcessing {split} split...")
            split_data = dataset[split]

            # Save as JSON
            output_file = output_path / f'{split}.json'
            data_list = []

            for idx, item in enumerate(tqdm(split_data, desc=f"Processing {split}")):
                data_list.append({
                    'question_id': idx,
                    'image_id': item.get('image_id', f'img_{idx}'),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'question_type': item.get('question_type', 'open')
                })

            with open(output_file, 'w') as f:
                json.dump(data_list, f, indent=2)

            print(f"  Saved {len(data_list)} samples to {output_file}")

            # Save images if available
            images_dir = output_path / 'images'
            images_dir.mkdir(exist_ok=True)

            if 'image' in split_data.features:
                print(f"  Saving images...")
                for idx, item in enumerate(tqdm(split_data, desc="Saving images")):
                    image = item['image']
                    image_id = item.get('image_id', f'img_{idx}')
                    image_path = images_dir / f'{image_id}.jpg'
                    image.save(image_path)

        # Save dataset info
        info = {
            'dataset': 'PathVQA',
            'description': 'Visual Question Answering on Pathology Images',
            'total_questions': sum(len(dataset[split]) for split in dataset.keys()),
            'splits': list(dataset.keys()),
            'source': 'https://huggingface.co/datasets/flaviagiammarino/path-vqa',
            'license': 'MIT',
            'paper': 'https://arxiv.org/abs/2003.10286'
        }

        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ PathVQA downloaded successfully!")
        print(f"  Location: {output_path}")
        print(f"  Total questions: {info['total_questions']}")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading PathVQA: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download Medical VQA Datasets')
    parser.add_argument('--output-dir', type=str,
                       default='data/eval_data/medical',
                       help='Output directory for datasets')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['vqarad', 'pathvqa', 'all'],
                       default=['all'],
                       help='Which datasets to download')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['vqarad', 'pathvqa']

    results = {}

    if 'vqarad' in datasets_to_download:
        results['vqarad'] = download_vqarad(output_dir)

    if 'pathvqa' in datasets_to_download:
        results['pathvqa'] = download_pathvqa(output_dir)

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for dataset, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {dataset.upper()}: {status}")

    print(f"\n✓ All downloads complete!")
    print(f"  Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review downloaded datasets in {output_dir}")
    print(f"  2. Generate retrieval indices using medical Wikipedia")
    print(f"  3. Run ALFAR experiments on medical VQA datasets")

if __name__ == '__main__':
    main()
