"""
Download Medical Wikipedia and convert to ALFAR format
Matches the structure of data/wiki/wiki_with_image.npy
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Medical categories from WikiProject Medicine
MEDICAL_CATEGORIES = [
    'Category:Medicine',
    'Category:Diseases_and_disorders',
    'Category:Anatomy',
    'Category:Pharmacology',
    'Category:Medical_treatments',
    'Category:Medical_procedures',
    'Category:Diagnostic_medicine',
    'Category:Medical_equipment',
    'Category:Medical_signs',
    'Category:Medical_tests',
    'Category:Drugs',
    'Category:Pathology',
    'Category:Surgery',
    'Category:Medical_specialties',
]

def get_category_members(category, limit=500):
    """Get all article titles in a category"""
    url = "https://en.wikipedia.org/w/api.php"
    articles = []

    headers = {
        'User-Agent': 'ALFAR Medical Wikipedia Downloader/1.0 (Research Project; Contact: research@example.org)'
    }

    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmlimit': limit,
        'cmtype': 'page',
        'format': 'json'
    }

    try:
        while True:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'query' not in data:
                break

            members = data['query']['categorymembers']
            for member in members:
                if member['ns'] == 0:  # Main namespace only
                    articles.append((member['title'], member['pageid']))

            # Check for continuation
            if 'continue' not in data:
                break
            params['cmcontinue'] = data['continue']['cmcontinue']
            time.sleep(0.1)

    except Exception as e:
        print(f"Error fetching {category}: {e}")
        return []

    return articles

def get_article_content(title, pageid):
    """Get article content and intro"""
    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        'User-Agent': 'ALFAR Medical Wikipedia Downloader/1.0 (Research Project; Contact: research@example.org)'
    }

    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,
        'exintro': False,
        'format': 'json'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        pages = data['query']['pages']
        page_id = str(pageid)

        if page_id not in pages:
            return None

        page = pages[page_id]
        full_text = page.get('extract', '')

        if not full_text:
            return None

        # Get intro (first paragraph) as summary
        if '\n\n' in full_text:
            intro = full_text.split('\n\n')[0]
        else:
            intro = full_text[:500]

        return {
            'wikipedia_summary': intro,
            'wikipedia_content': full_text
        }
    except Exception as e:
        print(f"Error fetching content for {title}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download Medical Wikipedia and convert to ALFAR format')
    parser.add_argument('--output', type=str, default='data/wiki/medical_wiki_with_image.npy',
                       help='Output .npy file path')
    parser.add_argument('--max-articles', type=int, default=10000,
                       help='Maximum number of articles to download')
    parser.add_argument('--checkpoint-every', type=int, default=100,
                       help='Save checkpoint every N articles')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint file
    checkpoint_path = output_path.parent / (output_path.stem + '_checkpoint.json')

    print(f"Collecting articles from {len(MEDICAL_CATEGORIES)} medical categories...")

    # Collect all unique article titles
    all_articles = {}  # title -> pageid
    for category in tqdm(MEDICAL_CATEGORIES, desc="Fetching category members"):
        try:
            articles = get_category_members(category, limit=500)
            for title, pageid in articles:
                all_articles[title] = pageid
            print(f"  {category}: {len(articles)} articles")
        except Exception as e:
            print(f"  Error with {category}: {e}")

    print(f"\nTotal unique medical articles: {len(all_articles)}")

    # Limit if requested
    if args.max_articles and len(all_articles) > args.max_articles:
        items = list(all_articles.items())[:args.max_articles]
        all_articles = dict(items)
        print(f"Limiting to {args.max_articles} articles")

    # Load checkpoint if exists
    medical_wiki = {}
    downloaded_titles = set()

    if checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                downloaded_titles = set(checkpoint_data.get('downloaded', []))
                print(f"Found {len(downloaded_titles)} previously downloaded articles")

                # Load partial .npy if exists
                if output_path.exists():
                    medical_wiki = np.load(output_path, allow_pickle=True).item()
                    print(f"Loaded {len(medical_wiki)} articles from existing .npy file")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Download article content
    print(f"\nDownloading {len(all_articles) - len(downloaded_titles)} remaining articles...")
    downloaded = len(downloaded_titles)
    failed = 0

    for title, pageid in tqdm(all_articles.items(), desc="Downloading articles"):
        # Skip if already downloaded
        if title in downloaded_titles:
            continue

        article_data = get_article_content(title, pageid)

        if article_data:
            # Use pageid as key to match ALFAR format
            medical_wiki[str(pageid)] = article_data
            downloaded_titles.add(title)
            downloaded += 1
        else:
            failed += 1

        # Rate limiting
        time.sleep(0.1)

        # Save checkpoint
        if downloaded % args.checkpoint_every == 0:
            print(f"\n  Checkpoint: Downloaded {downloaded}, Failed: {failed}")
            print(f"  Saving to {output_path}...")

            # Save .npy file
            np.save(output_path, medical_wiki)

            # Save checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump({'downloaded': list(downloaded_titles)}, f)

    # Final save
    print(f"\n✓ Download complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Total articles in database: {len(medical_wiki)}")

    print(f"\nSaving final file to {output_path}...")
    np.save(output_path, medical_wiki)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n✓ Medical Wikipedia saved in ALFAR format!")
    print(f"  File: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
    print(f"\nFormat structure:")
    print(f"  {{")
    print(f"    'page_id': {{")
    print(f"      'wikipedia_summary': '...',")
    print(f"      'wikipedia_content': '...'")
    print(f"    }}, ...")
    print(f"  }}")

if __name__ == '__main__':
    main()
