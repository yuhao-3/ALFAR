# Medical Wikipedia Download Status

**Date**: 2026-04-16
**Status**: In Progress

## Overview
Downloading medical Wikipedia articles to create a medical-domain knowledge base for ALFAR, matching the format of the existing `data/wiki/wiki_with_image.npy` file.

## Progress

### Data Collection
- **Categories Scanned**: 14 medical categories
- **Unique Articles Found**: 1,755
- **Target Download**: 1,000 articles (limited for initial testing)
- **Current Progress**: ~82/1,000 articles downloaded
- **Download Speed**: ~1.65 articles/second
- **Estimated Time**: ~10 minutes total

### Category Breakdown
| Category | Articles Found |
|----------|----------------|
| Medical equipment | 331 |
| Medical signs | 320 |
| Medical treatments | 249 |
| Medical tests | 205 |
| Anatomy | 135 |
| Surgery | 124 |
| Pathology | 106 |
| Pharmacology | 75 |
| Medical specialties | 68 |
| Drugs | 57 |
| Medical procedures | 51 |
| Medicine | 43 |
| Diseases and disorders | 31 |
| Diagnostic medicine | 0 |

## File Structure

### Output Files
```
data/wiki/
├── medical_wiki_with_image.npy              # Medical Wikipedia (in progress)
└── medical_wiki_with_image_checkpoint.json  # Checkpoint for resuming
```

### Planned Files
```
data/wiki/
├── medical_wiki_with_image.npy    # Main medical wiki data
└── medical_wiki_map.npy           # Index mapping (to be created)
```

## Data Format

Matching the format of `data/wiki/wiki_with_image.npy`:

```python
{
    'page_id': {
        'wikipedia_summary': 'First paragraph or intro section...',
        'wikipedia_content': 'Full article text...'
    },
    ...
}
```

Where `page_id` is the Wikipedia page ID (e.g., '12345').

## Scripts Created

### 1. `scripts/download_and_convert_medical_wiki.py`
- Downloads medical Wikipedia articles using Wikipedia API
- Converts to ALFAR-compatible .npy format
- Features:
  - Checkpoint support (resume on interruption)
  - User-Agent headers (avoid 403 errors)
  - Rate limiting (0.1s between requests)
  - Progress tracking with tqdm

**Usage**:
```bash
python scripts/download_and_convert_medical_wiki.py \
    --max-articles 10000 \
    --checkpoint-every 50
```

### 2. `scripts/create_medical_wiki_map.py`
- Creates wiki_map files from wiki_with_image file
- Format: `{index: page_id, ...}`
- Matches structure of `wiki_map.npy` and `wiki_map_17k.npy`

**Usage**:
```bash
python scripts/create_medical_wiki_map.py \
    --wiki-file data/wiki/medical_wiki_with_image.npy \
    --output data/wiki/medical_wiki_map.npy
```

## Next Steps

1. ✅ Download 1,000 medical articles (in progress)
2. ⏳ Verify data format compatibility
3. ⏳ Create wiki_map file
4. ⏳ Test with ALFAR retrieval system
5. ⏳ Expand to full medical corpus (~10,000 articles)
6. ⏳ Evaluate on medical VQA datasets (VQA-RAD, PathVQA)

## Technical Notes

### Wikipedia API Access
- **Issue**: Initial attempts blocked with 403 Forbidden errors
- **Solution**: Added proper User-Agent headers
- **Rate Limit**: 0.1s delay between requests to be respectful

### Memory Management
- Checkpoint every 50 articles to avoid data loss
- Can resume from checkpoint if interrupted
- .npy format for efficient storage and loading

### Data Quality
- Only main namespace articles (ns=0)
- Full article text extracted
- First paragraph used as summary
- Plain text format (no HTML/markup)

## Comparison: General vs Medical Wikipedia

| Aspect | General Wikipedia | Medical Wikipedia |
|--------|-------------------|-------------------|
| **Articles** | ~6 million | ~1,755 (current scan) |
| **Domain** | All topics | Medical only |
| **File Size** | 9.4 GB | ~1-2 MB (1,000 articles) |
| **Use Case** | General VQA | Medical VQA |
| **Categories** | All | 14 medical categories |

## Resources

- **Original Docs**: `docs/MEDICAL_WIKIPEDIA_INTEGRATION.md`
- **Wikipedia API**: https://en.wikipedia.org/w/api.php
- **WikiProject Medicine**: https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Medicine

---

**Last Updated**: 2026-04-16 15:18 AEDT
**Maintained by**: ALFAR Project Team
