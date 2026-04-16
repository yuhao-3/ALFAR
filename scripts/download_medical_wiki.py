"""
Download Medical Wikipedia Articles
Downloads articles from WikiProject Medicine categories using Wikipedia API
"""

import json
import time
import requests
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

    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmlimit': limit,
        'cmtype': 'page',  # Only pages, not subcategories
        'format': 'json'
    }

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        if 'query' not in data:
            break

        members = data['query']['categorymembers']
        for member in members:
            if member['ns'] == 0:  # Main namespace only
                articles.append(member['title'])

        # Check for continuation
        if 'continue' not in data:
            break
        params['cmcontinue'] = data['continue']['cmcontinue']
        time.sleep(0.1)  # Be nice to Wikipedia servers

    return articles

def get_article_content(title):
    """Get article content, extract, and intro"""
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts|pageprops',
        'explaintext': True,  # Plain text, no HTML
        'exintro': False,  # Get full article
        'format': 'json'
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        pages = data['query']['pages']
        page_id = list(pages.keys())[0]

        if page_id == '-1':  # Page doesn't exist
            return None

        page = pages[page_id]

        # Get full content
        full_text = page.get('extract', '')

        # Get intro (first paragraph) as summary
        if '\n\n' in full_text:
            intro = full_text.split('\n\n')[0]
        else:
            intro = full_text[:500]  # First 500 chars as fallback

        return {
            'title': title,
            'page_id': page_id,
            'summary': intro,
            'content': full_text
        }
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download Medical Wikipedia articles')
    parser.add_argument('--output', type=str, default='data/medical_wiki/articles',
                       help='Output directory for articles')
    parser.add_argument('--max-articles', type=int, default=10000,
                       help='Maximum number of articles to download (default: 10000)')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                       help='Specific categories to download (default: all medical)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select categories
    categories = args.categories if args.categories else MEDICAL_CATEGORIES

    print(f"Collecting articles from {len(categories)} medical categories...")

    # Collect all unique article titles
    all_articles = set()
    for category in tqdm(categories, desc="Fetching category members"):
        try:
            articles = get_category_members(category, limit=500)
            all_articles.update(articles)
            print(f"  {category}: {len(articles)} articles")
        except Exception as e:
            print(f"  Error with {category}: {e}")

    print(f"\nTotal unique medical articles: {len(all_articles)}")

    # Limit if requested
    if args.max_articles and len(all_articles) > args.max_articles:
        all_articles = list(all_articles)[:args.max_articles]
        print(f"Limiting to {args.max_articles} articles")

    # Download article content
    print(f"\nDownloading {len(all_articles)} articles...")
    downloaded = 0
    failed = 0

    for title in tqdm(list(all_articles), desc="Downloading articles"):
        # Check if already downloaded
        safe_title = title.replace('/', '_').replace(':', '_')
        output_file = output_dir / f"{safe_title}.json"

        if output_file.exists():
            downloaded += 1
            continue

        article_data = get_article_content(title)

        if article_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)
            downloaded += 1
        else:
            failed += 1

        # Rate limiting
        time.sleep(0.1)

        # Save progress every 100 articles
        if downloaded % 100 == 0:
            print(f"\n  Downloaded: {downloaded}, Failed: {failed}")

    print(f"\n✓ Download complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

    # Save article list
    article_list_file = output_dir.parent / 'article_titles.txt'
    with open(article_list_file, 'w') as f:
        for title in sorted(all_articles):
            f.write(f"{title}\n")
    print(f"  Article list saved to: {article_list_file}")

if __name__ == '__main__':
    main()
