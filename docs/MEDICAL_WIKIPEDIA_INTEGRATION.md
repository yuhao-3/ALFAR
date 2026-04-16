# Medical Wikipedia Integration Guide for ALFAR

## Overview
This guide explains how to integrate Medical Wikipedia articles as a drop-in replacement for the current Wikipedia knowledge base in ALFAR. Medical Wikipedia uses the **exact same format** as the existing wiki data, requiring **zero code changes**.

---

## Why Medical Wikipedia?

### Advantages
- **Identical Format**: Same structure as `data/wiki/wiki_with_image.npy`
- **Zero Code Changes**: Drop-in replacement for existing wiki
- **Comprehensive**: ~93,420 medical articles from WikiProject Medicine
- **Free & Open**: Public domain, no licensing issues
- **Quality**: Curated by WikiProject Medicine editors
- **Coverage**: Medicine, anatomy, medications, diseases, procedures, sanitation

### Content Categories
- Diseases and conditions
- Anatomy and physiology
- Medications and drugs
- Medical procedures
- Diagnostic tests
- Medical equipment
- Public health and sanitation

---

## Data Sources

### Option 1: Wikipedia Dumps (Recommended)
- **Source**: Official Wikipedia database dumps
- **URL**: https://dumps.wikimedia.org/enwiki/
- **Format**: XML dump files
- **Size**: ~20 GB compressed (full Wikipedia), ~500 MB for medical subset
- **Latest**: Monthly dumps available

### Option 2: WikiMed Offline (Quick Start)
- **Source**: Kiwix WikiMed application
- **URL**: https://www.kiwix.org/en/
- **Format**: ZIM file (compressed Wikipedia)
- **Size**: 1.27 GB
- **Content**: Pre-filtered medical articles only

### Option 3: Hugging Face Datasets
- **Source**: Pre-processed Wikipedia datasets
- **URL**: https://huggingface.co/datasets/wikipedia
- **Format**: Parquet/JSON
- **Advantage**: Already cleaned and structured

---

## Current ALFAR Wikipedia Format

Based on code inspection (`experiments/eval/alfar_mc_llava.py:71`):

```python
knowledge_base = np.load('data/wiki/wiki_with_image.npy', allow_pickle=True).item()

# Structure:
{
    'article_id_1': {
        'wikipedia_summary': 'Short summary of the article...',
        'wikipedia_content': 'Full article text content...'
    },
    'article_id_2': {
        'wikipedia_summary': '...',
        'wikipedia_content': '...'
    },
    ...
}
```

### Usage Pattern
```python
# In ALFAR code:
context = knowledge_base[know_index]['wikipedia_summary']
# Or for longer context:
context = knowledge_base[know_index]['wikipedia_content']
```

---

## Implementation Approach

### Step 1: Download Medical Wikipedia Articles

**Method A: Using Wikipedia Dumps**
```bash
# Download latest English Wikipedia dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Or use the provided script
python scripts/download_medical_wiki.py
```

**Method B: Using WikiMed Kiwix**
```bash
# Download WikiMed ZIM file
wget https://download.kiwix.org/zim/wikipedia_en_medicine.zim

# Extract using kiwix-tools
pip install kiwix-tools
```

### Step 2: Extract Medical Articles

Use WikiProject Medicine category tags to filter:
- Category:Medicine
- Category:Anatomy
- Category:Pharmacology
- Category:Medical equipment
- Category:Diseases
- Template:WikiProject_Medicine

```python
# Extraction script provided in scripts/extract_medical_wikipedia.py
python scripts/extract_medical_wikipedia.py \
    --dump enwiki-latest-pages-articles.xml.bz2 \
    --output data/medical_wiki/
```

### Step 3: Generate Summary and Content

For each article:
1. **Wikipedia Summary**: Extract first paragraph or use article lead section
2. **Wikipedia Content**: Full article text (excluding references, navigation)

```python
# Processing script provided in scripts/process_medical_wiki.py
python scripts/process_medical_wiki.py \
    --input data/medical_wiki/articles/ \
    --output data/wiki/medical_wiki_with_image.npy
```

### Step 4: Replace Existing Wiki

```bash
# Backup current wiki
cp data/wiki/wiki_with_image.npy data/wiki/wiki_with_image_backup.npy

# Replace with medical wiki
cp data/wiki/medical_wiki_with_image.npy data/wiki/wiki_with_image.npy

# Update wiki_map if needed
python scripts/create_wiki_map.py --input data/wiki/wiki_with_image.npy
```

### Step 5: Generate Retrieval Indices

Pre-compute retrieval indices for your medical VQA datasets:
```bash
# For VQA-RAD
python scripts/generate_retrieval_indices.py \
    --dataset vqarad \
    --wiki data/wiki/wiki_with_image.npy \
    --output data/retrieval_result/vqarad_indices_50.npy

# For PathVQA
python scripts/generate_retrieval_indices.py \
    --dataset pathvqa \
    --wiki data/wiki/wiki_with_image.npy \
    --output data/retrieval_result/pathvqa_indices_50.npy
```

---

## Article Identification Methods

### Method 1: Category-based (Recommended)
```python
# Articles in these categories:
medical_categories = [
    'Category:Medicine',
    'Category:Anatomy',
    'Category:Diseases and disorders',
    'Category:Pharmacology',
    'Category:Medical treatments',
    'Category:Medical procedures',
    'Category:Diagnostic medicine',
    'Category:Medical equipment',
]
```

### Method 2: WikiProject Template-based
```python
# Articles with WikiProject Medicine template
if '{{WikiProject Medicine' in article_text:
    is_medical = True
```

### Method 3: Machine Learning (from research paper)
- Research shows 93,420 medical articles identified using ML
- Paper: "Developing an automated mechanism to identify medical articles from wikipedia"
- Uses 7 semantic groups classification

---

## Directory Structure

```
data/
├── wiki/
│   ├── wiki_with_image.npy              # Original general Wikipedia (backup)
│   ├── medical_wiki_with_image.npy      # New medical Wikipedia
│   ├── wiki_map.npy                      # Article ID mapping
│   └── wiki_map_17k.npy                  # Smaller mapping
│
├── medical_wiki/                         # Intermediate files
│   ├── raw/                              # Raw XML dumps
│   ├── articles/                         # Extracted article JSON
│   └── processed/                        # Processed with summaries
│
├── retrieval_result/
│   ├── vqarad_indices_50.npy            # VQA-RAD retrieval indices
│   ├── vqarad_distance_50.npy           # Similarity scores
│   ├── pathvqa_indices_50.npy           # PathVQA retrieval indices
│   └── pathvqa_distance_50.npy          # Similarity scores
│
└── eval_data/
    └── medical/                          # Medical VQA datasets
        ├── vqarad/
        └── pathvqa/
```

---

## Expected Data Statistics

### Medical Wikipedia Corpus
- **Total articles**: ~93,420 (as of 2020)
- **Total words**: Estimated 50-100 million
- **Average article length**: 500-1000 words
- **Categories covered**: 7 main semantic groups
  1. Diseases
  2. Drugs
  3. Anatomy
  4. Symptoms
  5. Medical procedures
  6. Medical equipment
  7. Health-related topics

### File Sizes
- `medical_wiki_with_image.npy`: ~2-5 GB (depending on processing)
- Retrieval indices: ~1-2 MB per dataset
- Raw Wikipedia dump: ~20 GB compressed

---

## Validation Steps

### 1. Check Format Compatibility
```python
import numpy as np

# Load medical wiki
med_wiki = np.load('data/wiki/medical_wiki_with_image.npy', allow_pickle=True).item()

# Verify structure
sample_key = list(med_wiki.keys())[0]
assert 'wikipedia_summary' in med_wiki[sample_key]
assert 'wikipedia_content' in med_wiki[sample_key]
print("✓ Format compatible with ALFAR")
```

### 2. Verify Medical Content
```python
# Check for medical terminology
medical_terms = ['disease', 'treatment', 'diagnosis', 'symptom', 'patient']
sample_text = med_wiki[sample_key]['wikipedia_content'].lower()

medical_count = sum(1 for term in medical_terms if term in sample_text)
print(f"Medical terms found: {medical_count}/{len(medical_terms)}")
```

### 3. Test Retrieval
```python
# Test with sample medical query
from experiments.eval.alfar_mc_llava import *

# Should retrieve relevant medical articles
test_query = "What is pneumonia?"
# Run retrieval and verify results
```

---

## Scripts Provided

### 1. `scripts/download_medical_wiki.py`
Downloads latest Wikipedia dump or WikiMed ZIM file

### 2. `scripts/extract_medical_wikipedia.py`
Extracts medical articles from Wikipedia dump using category filters

### 3. `scripts/process_medical_wiki.py`
Processes articles into ALFAR-compatible format with summaries

### 4. `scripts/create_wiki_map.py`
Creates index mapping for efficient retrieval

### 5. `scripts/generate_retrieval_indices.py`
Pre-computes retrieval indices for medical VQA datasets

---

## Comparison: General vs Medical Wikipedia

| Aspect | General Wikipedia | Medical Wikipedia |
|--------|-------------------|-------------------|
| **Articles** | ~6 million | ~93,420 |
| **Domain** | All topics | Medical only |
| **Terminology** | General audience | Medical + patient-friendly |
| **Depth** | Variable | Detailed for medical topics |
| **References** | Mixed quality | Often cite medical journals |
| **Use Case** | General VQA | Medical VQA (VQA-RAD, PathVQA) |
| **File Size** | 9.4 GB | ~2-5 GB |

---

## Integration with Medical VQA Datasets

### VQA-RAD (Radiology)
- **Questions**: Clinical questions about radiology images
- **Relevant Wiki Articles**: Anatomy, diseases visible on X-rays/CT/MRI
- **Example**: "What organ is enlarged?" → Retrieve articles about heart, liver, etc.

### PathVQA (Pathology)
- **Questions**: Questions about pathology images (microscopy)
- **Relevant Wiki Articles**: Cellular biology, diseases, histology
- **Example**: "What type of cancer is shown?" → Retrieve cancer-related articles

### MIMIC-Diff-VQA
- **Questions**: Differences between two medical images
- **Relevant Wiki Articles**: Disease progression, treatment effects
- **Example**: "What changed between these X-rays?" → Retrieve disease/treatment articles

---

## Advanced Options

### Option A: Hybrid Wikipedia (Medical + General)
Keep both for comprehensive coverage:
```python
# Load both
general_wiki = np.load('data/wiki/wiki_with_image_backup.npy', allow_pickle=True).item()
medical_wiki = np.load('data/wiki/medical_wiki_with_image.npy', allow_pickle=True).item()

# Merge (medical articles override general if duplicates)
combined_wiki = {**general_wiki, **medical_wiki}
np.save('data/wiki/hybrid_wiki.npy', combined_wiki)
```

### Option B: Multi-Source Medical KB
Combine multiple medical sources:
```python
sources = {
    'wikipedia': medical_wiki,
    'wikidoc': wikidoc_data,
    'medqa': textbook_data
}
# Combine and deduplicate
```

### Option C: Specialized Subsets
Create domain-specific subsets:
- Radiology Wiki (for VQA-RAD)
- Pathology Wiki (for PathVQA)
- Clinical Wiki (for MIMIC)

---

## Troubleshooting

### Issue 1: Article Count Too Low
**Solution**: Adjust category filters or use broader medical categories

### Issue 2: Format Mismatch
**Solution**: Verify numpy dict structure matches exactly

### Issue 3: Retrieval Quality Poor
**Solution**:
- Regenerate embeddings with medical-specific model (BioBERT, PubMedBERT)
- Increase number of retrieved articles (top-50 → top-100)

### Issue 4: Memory Issues
**Solution**: Process articles in batches, use memory-mapped numpy arrays

---

## Performance Expectations

### Retrieval Accuracy
- Medical articles should be more relevant for medical VQA
- Expected improvement: 10-20% in context relevance
- Better terminology matching with clinical questions

### ALFAR Performance
- Should see improvement on medical VQA benchmarks
- Especially for knowledge-intensive medical questions
- Parametric knowledge (LLaVA-Med) + Medical context → Better accuracy

---

## Next Steps

1. ✅ Choose download method (dump vs Kiwix)
2. ✅ Run extraction scripts
3. ✅ Validate format compatibility
4. ✅ Generate retrieval indices
5. ✅ Test with medical VQA datasets
6. ✅ Evaluate performance improvements

---

## Resources

### Documentation
- Wikipedia Dumps: https://dumps.wikimedia.org/
- Kiwix: https://www.kiwix.org/
- WikiProject Medicine: https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Medicine

### Research Papers
- "Developing an automated mechanism to identify medical articles from wikipedia" (PMC7357526)
- "Medical Wikis Dedicated to Clinical Practice: A Systematic Review" (PMC4392552)

### Tools
- `mwparserfromhell`: Python library for parsing Wikipedia markup
- `wikipediaapi`: Python wrapper for Wikipedia API
- `kiwix-tools`: Extract from ZIM files

---

**Last Updated**: 2026-04-16
**Purpose**: Medical Wikipedia integration for ALFAR medical VQA
